"""Round-trip precision tests for the harness's R<->Python data exchange
(review round 1 on PR #36, findings 1-3): every test here calls Rscript for
real, so all are marked ``external``.
"""

import json
import random
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from driver import _desanitize, write_csv

pytestmark = pytest.mark.external


def _bits(x: float) -> str:
    """A double's raw bytes, as hex: two Python floats are the identical
    IEEE 754 value exactly when this string matches, unlike `==`, which
    treats -0.0 as 0.0 and any NaN as unequal to any other NaN."""
    return struct.pack("<d", x).hex()


def _random_doubles(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return (
        [rng.uniform(-1, 1) for _ in range(n // 3)]
        + [rng.uniform(-1e8, 1e8) for _ in range(n // 3)]
        + [
            rng.gauss(0, 1) * 10 ** rng.randint(-15, 15)
            for _ in range(n - 2 * (n // 3))
        ]
    )


class TestJsonDigitsPrecision:
    """fit_earth.R's and blackbox.R's write_result must round-trip a double
    exactly: digits = I(17) (significant digits), not digits = NA, which
    jsonlite 2.0.0 rounds to 15 significant digits."""

    def _round_trip(
        self, values: list[float], tmp_path: Path, digits: str
    ) -> list[float]:
        payload_path = tmp_path / "values.json"
        out_path = tmp_path / "out.json"
        payload_path.write_text(json.dumps(values))
        script = tmp_path / "roundtrip.R"
        script.write_text(f"""
suppressMessages(library(jsonlite))
x <- fromJSON("{payload_path}")
write_json(x, "{out_path}", digits = {digits}, auto_unbox = TRUE)
""")
        subprocess.run(
            ["Rscript", str(script)], check=True, capture_output=True, text=True
        )
        return json.loads(out_path.read_text())

    def test_digits_i17_round_trips_60000_doubles_exactly(self, tmp_path):
        values = _random_doubles(60000, seed=0)
        back = self._round_trip(values, tmp_path, digits="I(17)")
        mismatches = [
            i
            for i, (a, b) in enumerate(zip(values, back, strict=True))
            if _bits(a) != _bits(b)
        ]
        assert mismatches == [], f"{len(mismatches)} of {len(values)} changed"

    def test_digits_na_changes_at_least_one_value(self, tmp_path):
        # A negative control: shows the test above is not vacuous, and pins
        # the exact regression a digits = I(17) -> digits = NA mutation
        # would reintroduce.
        values = _random_doubles(2000, seed=1)
        back = self._round_trip(values, tmp_path, digits="NA")
        mismatches = [
            i
            for i, (a, b) in enumerate(zip(values, back, strict=True))
            if _bits(a) != _bits(b)
        ]
        assert mismatches, "expected digits = NA to change at least one value"


class TestCsvHexFloatPrecision:
    """driver.write_csv's numeric columns must reach R's read.csv exactly:
    checked directly (VALIDATION_PLAN.md's harness section, and driver.py's
    write_csv docstring), %.17g decimal text does not round-trip exactly on
    a build with no long double (about 1 in 3 values off by 1-3 ulps);
    hex floats (Python's float.hex()) do.
    """

    def _read_back_bits(self, csv_path: Path, tmp_path: Path) -> list[str]:
        out_path = tmp_path / "bits.json"
        script = tmp_path / "read_bits.R"
        script.write_text(f"""
suppressMessages(library(jsonlite))
d <- read.csv("{csv_path}")
x <- d[["x0"]]
bits <- vapply(x, function(v) {{
  con <- rawConnection(raw(0), "r+")
  writeBin(v, con)
  seek(con, 0)
  b <- readBin(con, "raw", n = 8)
  close(con)
  paste(as.character(b), collapse = "")
}}, character(1))
write_json(bits, "{out_path}", auto_unbox = TRUE)
""")
        subprocess.run(
            ["Rscript", str(script)], check=True, capture_output=True, text=True
        )
        return json.loads(out_path.read_text())

    def test_hex_float_csv_round_trips_4000_doubles_exactly(self, tmp_path):
        values = np.array(_random_doubles(4000, seed=2))
        csv_path = tmp_path / "d.csv"
        write_csv(csv_path, {"x0": values})
        r_bits = self._read_back_bits(csv_path, tmp_path)
        expected_bits = [_bits(v) for v in values]
        mismatches = [
            i
            for i, (a, b) in enumerate(zip(expected_bits, r_bits, strict=True))
            if a != b
        ]
        assert mismatches == [], f"{len(mismatches)} of {len(values)} changed"

    def test_percent_17g_decimal_text_does_not_round_trip_exactly(self, tmp_path):
        # A negative control: the format driver.write_csv used to use, and
        # the exact regression a hex-float -> %.17g mutation would
        # reintroduce (VALIDATION_PLAN.md's harness section: R here has no
        # long double, so read.csv's decimal parsing is not exact).
        values = _random_doubles(2000, seed=3)
        csv_path = tmp_path / "d.csv"
        csv_path.write_text("x0\n" + "\n".join(f"{v:.17g}" for v in values) + "\n")
        r_bits = self._read_back_bits(csv_path, tmp_path)
        expected_bits = [_bits(v) for v in values]
        mismatches = [a != b for a, b in zip(expected_bits, r_bits, strict=True)]
        assert any(mismatches), "expected %.17g decimal text to lose precision here"


class TestNaAndNanStayDistinct:
    """R's NA and NaN both print as bare tokens jsonlite's default JSON
    encoding cannot tell apart from each other or from a missing value
    (VALIDATION_PLAN.md's harness section); na = "string" keeps the tokens
    distinct in the file, and driver._desanitize must not re-merge them.
    """

    def test_na_and_nan_round_trip_as_distinguishable_python_values(self, tmp_path):
        out_path = tmp_path / "out.json"
        script = tmp_path / "na_nan.R"
        script.write_text(f"""
suppressMessages(library(jsonlite))
write_json(list(a = NA_real_, b = NaN, c = 1.5), "{out_path}",
           digits = I(17), auto_unbox = TRUE, na = "string")
""")
        subprocess.run(
            ["Rscript", str(script)], check=True, capture_output=True, text=True
        )
        raw = json.loads(out_path.read_text())
        assert raw == {"a": "NA", "b": "NaN", "c": 1.5}  # distinct in the file already
        result = _desanitize(raw)
        assert result["a"] is None
        assert isinstance(result["b"], float) and result["b"] != result["b"]  # NaN
        assert result["c"] == 1.5
        assert result["a"] is not result["b"]
