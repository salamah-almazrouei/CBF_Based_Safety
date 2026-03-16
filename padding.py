"""Pad a trajectory by holding its last sample for extra time.

Only the first column (time) is incremented during padding. Every other column
stays equal to the final row of the input trajectory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path("/Users/salamahalmazrouei/Desktop/DataSet/Third_Experiment/test2.csv")
DEFAULT_OUTPUT = Path("/Users/salamahalmazrouei/Desktop/DataSet/Third_Experiment/test_padded2.csv")
DEFAULT_HOLD_SECONDS = 5.0


TIME_COLUMN_CANDIDATES = ("t", "time", "time_s", "timestamp")


def infer_dt(times: np.ndarray) -> float:
    """Use the last time increment so padding continues at the current dt."""
    if times.size < 2:
        raise ValueError("Need at least 2 rows to infer dt from the time column.")

    dt = float(times[-1] - times[-2])
    if dt <= 0.0:
        raise ValueError("Time column must be strictly increasing.")
    return dt


def find_time_column(header: list[str]) -> int:
    lowered = [h.strip().lower() for h in header]
    for name in TIME_COLUMN_CANDIDATES:
        if name in lowered:
            return lowered.index(name)
    return 0


def load_csv(csv_path: Path) -> tuple[str, np.ndarray, int]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]

    if len(rows) < 3:
        raise ValueError("CSV must contain a header and at least 2 data rows.")

    header_fields = [cell.strip() for cell in rows[0]]
    header = ",".join(header_fields)
    time_col = find_time_column(header_fields)
    data = np.asarray([[float(x) for x in row] for row in rows[1:]], dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < 2:
        raise ValueError("CSV must contain at least 2 data rows.")
    return header, data, time_col


def pad_last_point(data: np.ndarray, hold_seconds: float, time_col: int = 0) -> tuple[np.ndarray, float]:
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError("Input trajectory must be a 2D array with at least one column.")
    if hold_seconds <= 0.0:
        raise ValueError("hold_seconds must be greater than 0.")
    if not (0 <= time_col < data.shape[1]):
        raise ValueError("time_col is out of bounds.")

    dt = infer_dt(data[:, time_col])
    n_extra = int(np.round(hold_seconds / dt))
    if n_extra <= 0:
        return np.array(data, dtype=float, copy=True), dt

    out = np.array(data, dtype=float, copy=True)
    last_row = out[-1].copy()
    extra = np.tile(last_row, (n_extra, 1))
    extra[:, time_col] = out[-1, time_col] + dt * np.arange(1, n_extra + 1, dtype=float)
    return np.vstack((out, extra)), dt


def main() -> None:
    parser = argparse.ArgumentParser(description="Pad a trajectory by holding its last row.")
    parser.add_argument("--in_csv", type=Path, default=DEFAULT_INPUT, help="Input trajectory CSV.")
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUTPUT, help="Output padded CSV.")
    parser.add_argument(
        "--hold_seconds",
        type=float,
        default=DEFAULT_HOLD_SECONDS,
        help="Extra duration to append while holding the final row.",
    )
    args = parser.parse_args()

    header, data, time_col = load_csv(args.in_csv)
    padded, dt = pad_last_point(data, hold_seconds=args.hold_seconds, time_col=time_col)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out_csv, padded, delimiter=",", header=header, comments="")

    print(
        f"Saved padded trajectory to: {args.out_csv}\n"
        f"Time column index: {time_col}\n"
        f"Detected dt: {dt:.6f} s\n"
        f"Original rows: {data.shape[0]} | Padded rows: {padded.shape[0]}"
    )


if __name__ == "__main__":
    main()
