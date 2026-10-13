# src/beacon/plot/tables/script.py
"""
The script a table's HTML carries: it sorts the rows when a column's heading
is clicked, keeping each row group together, and blends a heatmap's colours.
"""
# BN-302.
SCRIPT = r"""
(function () {
  var t = document.getElementById("TABLE_ID");
  if (!t) return;
  var blend = t.getAttribute("data-blend"), grouped = t.getAttribute("data-blocks") === "1";
  var stripes = t.getAttribute("data-stripes") === "1";
  function rows(body) { return Array.prototype.slice.call(body.querySelectorAll("tr.r")); }
  function parts() {
    var bodies = Array.prototype.slice.call(t.tBodies).map(rows);
    bodies = bodies.filter(function (r) { return r.length; });
    return grouped ? bodies : [[].concat.apply([], bodies)];
  }
  function mix(a, b) { return b ? "color-mix(in srgb, " + a + ", " + b + ")" : a; }
  function fill(cell) { return cell ? cell.getAttribute("data-fill") : null; }
  function shadeLine(cells, degrees) {
    cells.forEach(function (cell, i) {
      var me = fill(cell);
      if (!me) return;
      cell.style.background = "linear-gradient(" + degrees + "deg, " + mix(me, fill(cells[i - 1])) +
        " 0%, " + me + " 50%, " + mix(me, fill(cells[i + 1])) + " 100%)";
    });
  }
  function rgba(text) {
    var parts = text.match(/[\d.]+/g).map(Number);
    return parts.length > 3 ? parts : parts.concat([1]);
  }
  // A block blended across and down at once, painted pixel by pixel at its
  // real size: a browser draws a stretched tiny image without smoothing.
  // Each colour sits at its cell's middle; a missing cell lends its place
  // its nearest neighbour's colour and is left blank.
  function shadeBlock(grid) {
    var ratio = window.devicePixelRatio || 1;
    var first = grid[0][0].getBoundingClientRect();
    var last = grid[grid.length - 1][grid[0].length - 1].getBoundingClientRect();
    var left = first.left, top = first.top;
    var width = Math.max(1, Math.round((last.right - left) * ratio));
    var height = Math.max(1, Math.round((last.bottom - top) * ratio));
    var xs = grid[0].map(function (cell) {
      var box = cell.getBoundingClientRect();
      return ((box.left + box.right) / 2 - left) * ratio;
    });
    var ys = grid.map(function (cells) {
      var box = cells[0].getBoundingClientRect();
      return ((box.top + box.bottom) / 2 - top) * ratio;
    });
    var colours = grid.map(function (cells) {
      return cells.map(function (cell) { return rgba(cell.getAttribute("data-mix")); });
    });
    function between(places, at) {
      var i = 0;
      while (i < places.length - 2 && at > places[i + 1]) i++;
      if (places.length < 2) return [0, 0, 0];
      var share = (at - places[i]) / ((places[i + 1] - places[i]) || 1);
      return [i, i + 1, Math.min(1, Math.max(0, share))];
    }
    var canvas = document.createElement("canvas");
    canvas.width = width; canvas.height = height;
    var pen = canvas.getContext("2d"), image = pen.createImageData(width, height);
    for (var y = 0; y < height; y++) {
      var down = between(ys, y + 0.5);
      for (var x = 0; x < width; x++) {
        var across = between(xs, x + 0.5), at = (y * width + x) * 4;
        for (var k = 0; k < 4; k++) {
          var high = colours[down[0]][across[0]][k] * (1 - across[2]) +
            colours[down[0]][across[1]][k] * across[2];
          var low = colours[down[1]][across[0]][k] * (1 - across[2]) +
            colours[down[1]][across[1]][k] * across[2];
          var value = high * (1 - down[2]) + low * down[2];
          image.data[at + k] = k === 3 ? Math.round(value * 255) : Math.round(value);
        }
      }
    }
    pen.putImageData(image, 0, 0);
    var url = "url(" + canvas.toDataURL() + ")";
    grid.forEach(function (cells) {
      cells.forEach(function (cell) {
        if (!fill(cell)) return;
        var box = cell.getBoundingClientRect();
        cell.style.background = url + " " + (left - box.left) + "px " + (top - box.top) +
          "px / " + width / ratio + "px " + height / ratio + "px no-repeat";
      });
    });
  }
  function shade() {
    parts().forEach(function (block) {
      var grid = block.map(function (row) {
        return Array.prototype.slice.call(row.querySelectorAll("td.h"));
      });
      if (!grid.length || !grid[0].length) return;
      if (blend === "columns") grid.forEach(function (cells) { shadeLine(cells, 90); });
      if (blend === "rows") grid[0].forEach(function (_, x) {
        shadeLine(grid.map(function (cells) { return cells[x]; }), 180);
      });
      if (blend === "both") shadeBlock(grid);
    });
  }
  function stripe() {
    if (!stripes) return;
    parts().forEach(function (block) {
      block.forEach(function (row, i) { row.classList.toggle("s", i % 2 === 1); });
    });
  }
  function value(row, column) {
    var text = row.querySelectorAll("td.v")[column].getAttribute("data-v");
    return text === "" ? null : Number(text);
  }
  var direction = {};
  Array.prototype.slice.call(t.querySelectorAll("th.c")).forEach(function (head) {
    head.addEventListener("click", function () {
      var column = Number(head.getAttribute("data-col"));
      var way = direction[column] = direction[column] === -1 ? 1 : -1;
      Array.prototype.slice.call(t.querySelectorAll("th.c")).forEach(function (other) {
        other.removeAttribute("aria-sort");
      });
      head.setAttribute("aria-sort", way > 0 ? "ascending" : "descending");
      Array.prototype.slice.call(t.tBodies).forEach(function (body) {
        // A row group's heading spans its rows from the first, so it moves
        // to whichever row is first now.
        var label = body.querySelector("td.rg");
        var sorted = rows(body).sort(function (a, b) {
          var x = value(a, column), y = value(b, column);
          if (x === null) return y === null ? 0 : 1;
          if (y === null) return -1;
          return (x - y) * way;
        });
        sorted.forEach(function (row) { body.appendChild(row); });
        if (label && sorted.length) sorted[0].insertBefore(label, sorted[0].firstChild);
      });
      stripe();
      shade();
    });
  });
  stripe();
  shade();
})();
"""
