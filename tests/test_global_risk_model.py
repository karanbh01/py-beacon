# tests/test_global_risk_model.py
"""BN-419: the global factor model, with country, sector and currency
factors fitted on weekly returns, and the analysis choosing it for a run
whose names span more than one country.

The fit is checked against returns planted from known factor returns: 400
names, two years of weeks and a little noise, so each fitted return lands
close to the planted one. The analysis runs on a synthetic run of 60 names in
four countries (one with too few names for a factor of its own), three
sectors and three currencies, over three years of sessions.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.panels import FUND_WEIGHTS, INDEX_WEIGHTS, NAME_RETURNS, REFERENCE
from beacon.analysis.reports.options import describe
from beacon.analysis.risk.model import FACTOR, RebalanceRiskModels, RiskModelSettings
from beacon.exceptions import CalculationError
from beacon.risk.model import MARKET_FACTOR
from beacon.risk.model.global_fit import (
    COUNTRY,
    SECTOR,
    assign_groups,
    fit_global_model,
    group_dummies,
    weekly_returns,
)

NAMES = [f"N{i:03d}" for i in range(400)]
WEEKS = 104
NOISE = 0.002


def planted(seed: int = 0) -> dict:
    """Base-currency returns made from known factor returns, with the
    country and sector returns meeting the cap-weighted constraints."""
    rng = np.random.default_rng(seed)
    country = pd.Series(rng.choice(["US", "DE", "JP"], len(NAMES), p=[0.5, 0.3, 0.2]),
                        index=NAMES)
    sector = pd.Series(rng.choice(["A", "B", "C"], len(NAMES)), index=NAMES)
    caps = pd.Series(rng.lognormal(10.0, 1.0, len(NAMES)), index=NAMES)
    countries, sectors = group_dummies(country, COUNTRY), group_dummies(sector, SECTOR)
    styles = pd.DataFrame({"size": rng.normal(size=len(NAMES))}, index=NAMES)
    exposures = pd.concat([countries, sectors, styles], axis=1)

    def constrained(dummies: pd.DataFrame) -> np.ndarray:
        sizes = caps.to_numpy() @ dummies.to_numpy()
        found = rng.normal(0.0, 0.02, (WEEKS, dummies.shape[1]))

        return found - (found @ sizes / sizes.sum())[:, None]

    truth = {MARKET_FACTOR: rng.normal(0.002, 0.02, WEEKS), "size": rng.normal(0.0, 0.01, WEEKS)}
    truth.update(dict(zip(countries.columns, constrained(countries).T, strict=True)))
    truth.update(dict(zip(sectors.columns, constrained(sectors).T, strict=True)))
    local = (truth[MARKET_FACTOR][:, None]
             + sum(np.outer(truth[column], exposures[column]) for column in exposures.columns)
             + rng.normal(0.0, NOISE, (WEEKS, len(NAMES))))
    currency = country.map({"DE": "EUR", "JP": "JPY"}).dropna()
    fx = pd.DataFrame(rng.normal(0.0, 0.01, (WEEKS, 2)), columns=["EUR", "JPY"])
    moves = np.column_stack([fx[currency[name]].to_numpy() if name in currency.index
                             else np.zeros(WEEKS) for name in NAMES])
    base = pd.DataFrame((1.0 + local) * (1.0 + moves) - 1.0, columns=NAMES)

    return {"base": base, "exposures": exposures, "caps": caps, "currency": currency, "fx": fx,
            "truth": truth, "blocks": [list(countries.columns), list(sectors.columns)]}


@pytest.fixture(scope="module")
def world():
    return planted()


@pytest.fixture(scope="module")
def fitted(world):
    return fit_global_model(world["base"], world["exposures"], world["blocks"], world["caps"],
                            world["currency"], world["fx"])


class TestGroups:

    def test_a_thin_country_joins_its_region(self):
        country = pd.Series({"A": "US", "B": "US", "C": "NZ", "D": "AU"})
        region = pd.Series({"A": "Americas", "B": "Americas", "C": "Pacific", "D": "Pacific"})

        assert assign_groups(country, region, 2).to_dict() == {
            "A": "US", "B": "US", "C": "Pacific (other)", "D": "Pacific (other)"}

    def test_without_regions_nothing_is_merged(self):
        country = pd.Series({"A": "US", "B": None})

        assert assign_groups(country, None, 5).to_dict() == {"A": "US", "B": "Unknown"}

    def test_one_dummy_column_a_group(self):
        dummies = group_dummies(pd.Series({"A": "US", "B": "DE", "C": "US"}), COUNTRY)

        assert list(dummies.columns) == ["country:DE", "country:US"]
        assert dummies.sum(axis=1).tolist() == [1.0, 1.0, 1.0]


class TestWeekly:

    def test_days_compound_into_weeks_ending_on_friday(self):
        days = pd.bdate_range("2024-01-01", periods=10)
        daily = pd.DataFrame({"A": [0.01] * 10, "B": [0.01] * 9 + [np.nan]}, index=days)

        weekly = weekly_returns(daily)

        assert list(weekly.index) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")]
        assert weekly["A"].tolist() == pytest.approx([1.01 ** 5 - 1.0] * 2)
        assert weekly["B"].iloc[0] == pytest.approx(1.01 ** 5 - 1.0)
        assert np.isnan(weekly["B"].iloc[1])


class TestFit:

    def test_the_planted_factor_returns_are_recovered(self,
                                                      world,
                                                      fitted):
        for factor, truth in world["truth"].items():
            error = np.abs(fitted.factor_returns[factor].to_numpy() - truth).max()

            assert error < 10 * NOISE, factor

    def test_the_constraints_hold(self,
                                  world,
                                  fitted):
        for block in world["blocks"]:
            sizes = world["caps"].to_numpy() @ world["exposures"][block].to_numpy()

            found = fitted.factor_returns[block].to_numpy() @ sizes

            assert np.abs(found).max() < 1e-12 * sizes.sum()

    def test_currency_factors_take_the_currencies_own_returns(self,
                                                              world,
                                                              fitted):
        for code in ("EUR", "JPY"):
            np.testing.assert_array_equal(fitted.factor_returns[f"currency:{code}"],
                                          world["fx"][code])

        assert fitted.exposures.loc[world["currency"].index[0]].filter(like="currency:").sum() == 1

    def test_each_constraint_takes_one_dimension(self,
                                                 fitted):
        assert fitted.dimensions == len(fitted.factor_names) - 2

    def test_a_cap_weighted_portfolio_has_no_country_or_sector_risk(self,
                                                                    world,
                                                                    fitted):
        weights = (world["caps"] / world["caps"].sum()).to_dict()

        contributions = fitted.decompose_active_risk(weights, {}).factor_contributions

        assert abs(contributions.filter(like="country:").sum()) < 1e-12
        assert abs(contributions.filter(like="sector:").sum()) < 1e-12

    def test_a_one_country_portfolio_carries_country_risk(self,
                                                          world,
                                                          fitted):
        index = (world["caps"] / world["caps"].sum()).to_dict()
        germany = world["caps"][world["exposures"]["country:DE"] == 1.0]
        held = (germany / germany.sum()).to_dict()

        found = fitted.decompose_active_risk(held, index)

        assert found.reconciles()
        assert found.factor_contributions.filter(like="country:").sum() > 0.0
        assert found.factor_contributions.filter(like="currency:").sum() > 0.0

    def test_specific_risk_is_the_noise(self,
                                        fitted):
        weekly = np.sqrt(fitted.specific_variance / fitted.periods_per_year)

        assert weekly.mean() == pytest.approx(NOISE, rel=0.1)

    def test_without_currencies_there_are_no_currency_factors(self,
                                                              world):
        found = fit_global_model(world["base"], world["exposures"], world["blocks"],
                                 world["caps"])

        assert not [name for name in found.factor_names if name.startswith("currency:")]

    def test_refused(self,
                     world):
        with pytest.raises(CalculationError, match="do not cover"):
            fit_global_model(world["base"].drop(columns=NAMES[0]), world["exposures"], [])
        with pytest.raises(CalculationError, match="no returns for the currencies"):
            fit_global_model(world["base"], world["exposures"], [], None, world["currency"],
                             world["fx"].drop(columns="JPY"))
        with pytest.raises(CalculationError, match="at least 2 complete"):
            fit_global_model(world["base"].iloc[:1], world["exposures"], [])


# The synthetic run: 60 names over three years of sessions.
DAYS = pd.bdate_range("2021-01-01", "2023-12-29")
REBALANCES = [pd.Timestamp("2022-07-01"), pd.Timestamp("2023-07-03")]
RUN_NAMES = [f"R{i:02d}" for i in range(60)]
COUNTRIES = ["US"] * 30 + ["DE"] * 20 + ["NZ"] * 4 + ["JP"] * 6
REGIONS = {"US": "Americas", "DE": "Europe", "NZ": "Asia Pacific", "JP": "Asia Pacific"}
LISTINGS = ["US"] * 40 + ["DE"] * 20
CURRENCY_OF = {"US": "USD", "DE": "EUR", "NZ": "NZD", "JP": "JPY"}


def synthetic_run(countries: list[str] = COUNTRIES) -> RunData:
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (len(DAYS), len(RUN_NAMES))),
                           index=DAYS, columns=RUN_NAMES)
    reference = pd.concat([pd.DataFrame({
        "IDENTIFIER": RUN_NAMES, "COUNTRY_DOMICILE": countries,
        "COUNTRY_LISTING": LISTINGS, "REGION": [REGIONS[code] for code in countries],
        "SECTOR": [("A", "B", "C")[i % 3] for i in range(len(RUN_NAMES))]},
        index=pd.DatetimeIndex([day] * len(RUN_NAMES))) for day in REBALANCES])
    weights = pd.DataFrame(1.0 / len(RUN_NAMES), index=DAYS, columns=RUN_NAMES)
    codes = sorted({CURRENCY_OF[code] for code in countries} - {"USD"})
    rates = pd.DataFrame(np.cumprod(1.0 + rng.normal(0.0, 0.005, (len(DAYS), len(codes))),
                                    axis=0), index=DAYS, columns=codes)
    currencies = pd.DataFrame({"CURRENCY": [CURRENCY_OF[code] for code in countries]},
                              index=RUN_NAMES)
    nav = pd.Series(1e6, index=DAYS)

    return RunData(currency="USD", returns=nav.pct_change().fillna(0.0), nav=nav, units=None,
                   initial_capital=1e6, periods_per_year=252, risk_free_rate=0.0,
                   tables={REFERENCE: reference, FUND_WEIGHTS: weights, INDEX_WEIGHTS: weights,
                           NAME_RETURNS: returns, "currencies": currencies, "rates": rates})


def styles(day: pd.Timestamp,
           names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"size": np.linspace(-1.0, 1.0, len(names))}, index=names)


def models(run: RunData | None = None,
           **settings) -> RebalanceRiskModels:
    return RebalanceRiskModels(run or synthetic_run(), RiskModelSettings(kind=FACTOR, **settings),
                               exposures=styles)


class TestTheAnalysis:

    def test_names_in_several_countries_get_the_global_model(self):
        model = models().at(REBALANCES[1])

        assert model.periods_per_year == 52
        assert {"country:US", "country:DE", "country:Asia Pacific (other)", "sector:A",
                "currency:EUR", "currency:NZD", "currency:JPY", "size"} <= set(model.factor_names)
        assert "currency:USD" not in model.factor_names

    def test_it_is_fitted_on_the_weeks_before_the_rebalance(self):
        model = models(weeks=52).at(REBALANCES[0])
        weeks = model.factor_returns.index

        assert len(weeks) == 52
        assert all(day.day_of_week == 4 for day in weeks)
        assert weeks[-1] < REBALANCES[0]

    def test_names_in_one_country_keep_the_styles_model(self):
        run = synthetic_run(["US"] * len(RUN_NAMES))
        found = models(run)

        assert not found.is_global(REBALANCES[0])
        assert found.at(REBALANCES[0]).factor_names == [MARKET_FACTOR, "size"]

    def test_styles_only_when_chosen(self):
        assert models(structure="styles").at(REBALANCES[0]).factor_names == [MARKET_FACTOR,
                                                                              "size"]

    def test_the_country_is_read_from_the_field_chosen(self):
        model = models(country_field="COUNTRY_LISTING", minimum_names=1).at(REBALANCES[1])

        assert sorted(name for name in model.factor_names
                      if name.startswith("country:")) == ["country:DE", "country:US"]

    def test_each_part_can_be_turned_off(self):
        model = models(sectors=False, currencies=False, weekly=False).at(REBALANCES[0])

        assert model.periods_per_year == 252
        assert not [name for name in model.factor_names
                    if name.startswith(("sector:", "currency:"))]
        assert "country:DE" in model.factor_names

    def test_a_rebalance_without_the_weeks_before_it_has_no_history(self):
        found = models(weeks=104)

        assert not found.has_history(REBALANCES[0])
        assert found.has_history(REBALANCES[1])
        with pytest.raises(CalculationError, match="fewer than the 104"):
            found.at(REBALANCES[0])

    @pytest.mark.parametrize(("settings", "message"),
                             [({"structure": "world"}, "Unknown structure"),
                              ({"weeks": 4}, "weeks is a whole number"),
                              ({"minimum_names": 0}, "minimum_names is a whole number")])
    def test_settings_refused(self,
                              settings,
                              message):
        with pytest.raises(ValueError, match=message):
            RiskModelSettings(**settings)

    def test_the_choices_are_report_options(self):
        section = next(item for item in describe() if item["name"] == "risk_model")
        found = {item["name"]: item["default"] for item in section["parameters"]}

        assert found == {"window": 252, "structure": "auto", "country_field": "COUNTRY_DOMICILE",
                         "minimum_names": 10, "sectors": True, "currencies": True,
                         "weekly": True, "weeks": 104}
