#!/bin/bash
# scripts/test_chunks.sh — run the test suite in memory-safe chunks (BN-159).
#
# Why this exists: the full suite in one pytest process gets killed on this
# class of machine when memory is tight (a resident data job can leave well
# under 1 GB free), so the suite runs as chunks of 8 files, each a fresh
# process. This script is that workflow as a tool rather than session lore,
# with two additions:
#
#   * it checks free RAM first and runs chunks TWO-WIDE when there is
#     headroom, serial when the machine is under pressure;
#   * it skips the `slow` tests (the example notebooks, 3-7 minutes of real
#     Jupyter kernels) by default — pass --all before pushing.
#
# Usage:
#   scripts/test_chunks.sh          # routine: not slow, adaptive width
#   scripts/test_chunks.sh --all    # everything, e.g. before a push
#   scripts/test_chunks.sh --serial # force one-wide regardless of RAM
set -u

cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/Scripts:$PATH"

CHUNK=8
MARK='not slow'
FORCE_SERIAL=0
# Two-wide only with real headroom: each pytest process can peak at a couple
# of GB (synthetic panels, server fixtures), and the point of chunking is to
# never be the reason the machine starts paging.
# Overridable for testing the parallel path itself; the default is the
# operational judgement.
MIN_FREE_GB_FOR_PARALLEL=${TEST_CHUNKS_PARALLEL_GB:-8}

for arg in "$@"; do
  case "$arg" in
    --all) MARK='' ;;
    --serial) FORCE_SERIAL=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

free_gb() {
  # KB of free physical memory, via PowerShell (works from Git Bash).
  local kb
  kb=$(powershell -NoProfile -Command \
    "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" 2>/dev/null \
    | tr -d '[:space:]')
  [ -n "$kb" ] || { echo 0; return; }
  echo $((kb / 1024 / 1024))
}

WIDTH=1
if [ "$FORCE_SERIAL" -eq 0 ]; then
  FREE=$(free_gb)
  if [ "$FREE" -ge "$MIN_FREE_GB_FOR_PARALLEL" ]; then
    WIDTH=2
  fi
  echo "free RAM: ${FREE} GB -> running ${WIDTH}-wide"
else
  echo "forced serial"
fi

FILES=($(ls tests/test_*.py | sort))
TOTAL=${#FILES[@]}
LOGDIR=$(mktemp -d)
FAILED=0
PASS_TOTAL=0

run_chunk() {
  # $1 = offset; writes log, prints one summary line, returns pytest's exit.
  local i=$1
  local log="$LOGDIR/chunk-$i.log"
  local args=(-q --no-cov -p no:randomly)
  [ -n "$MARK" ] && args+=(-m "$MARK")
  python -u -m pytest "${FILES[@]:i:CHUNK}" "${args[@]}" >"$log" 2>&1
  local code=$?
  local tail_line
  tail_line=$(grep -E "passed|failed|error" "$log" | tail -1)
  printf '=== files %2d-%2d  %s\n' "$((i + 1))" "$((i + CHUNK))" "$tail_line"
  # pytest exits 5 when a chunk deselects everything (all slow) — not a failure.
  [ "$code" -eq 5 ] && code=0
  if [ "$code" -ne 0 ]; then
    echo "--- failures in files $((i + 1))-$((i + CHUNK)):"
    grep -E "^FAILED|^ERROR" "$log" | head -20
  fi
  return "$code"
}

i=0
while [ "$i" -lt "$TOTAL" ]; do
  if [ "$WIDTH" -eq 2 ] && [ $((i + CHUNK)) -lt "$TOTAL" ]; then
    run_chunk "$i" &
    first=$!
    run_chunk $((i + CHUNK)) &
    second=$!
    wait "$first" || FAILED=1
    wait "$second" || FAILED=1
    i=$((i + 2 * CHUNK))
  else
    run_chunk "$i" || FAILED=1
    i=$((i + CHUNK))
  fi
done

PASS_TOTAL=$(cat "$LOGDIR"/chunk-*.log | grep -oE "[0-9]+ passed" \
  | awk '{s+=$1} END {print s}')
echo
if [ "$FAILED" -ne 0 ]; then
  echo "SUITE FAILED (${PASS_TOTAL:-0} passed elsewhere; logs in $LOGDIR)"
  exit 1
fi
[ -n "$MARK" ] && SKIPPED=" (slow tests skipped — run with --all before pushing)"
echo "SUITE GREEN: ${PASS_TOTAL:-0} passed${SKIPPED:-}"
rm -rf "$LOGDIR"
