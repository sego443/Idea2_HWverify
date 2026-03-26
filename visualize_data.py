#!/usr/bin/env python3
"""Save quick-look QC plots for processed OpenBCI CSV files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from brainflow.data_filter import DataFilter


DEFAULT_PROCESSED_DIR = Path("outputs/processed_csv")
DEFAULT_OUTPUT_DIR = Path("outputs/processed_qc")


def visualize_first_seconds(
    input_file: str | Path,
    seconds_to_show: int = 5,
    sampling_rate: int = 250,
    channels: int = 8,
    title: Optional[str] = None,
):
    """Load processed CSV and build a figure for the first N seconds of EXG data."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到处理好的数据文件 {input_path}")

    data = DataFilter.read_file(str(input_path))
    points = min(sampling_rate * seconds_to_show, data.shape[1])
    exg_data = [data[i][:points] for i in range(1, channels + 1)]

    fig, axes = plt.subplots(channels, 1, figsize=(12, 10), sharex=True)
    time_axis = np.arange(points) / sampling_rate
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    for i in range(channels):
        ch_data = exg_data[i]
        axes[i].plot(time_axis, ch_data, color=colors[i % len(colors)], linewidth=1.0)
        axes[i].set_ylabel(f"Ch {i + 1}\n(uV)", fontsize=10)
        axes[i].grid(True, linestyle="--", alpha=0.6)
        y_min, y_max = np.min(ch_data), np.max(ch_data)
        margin = max((y_max - y_min) * 0.1, 10)
        axes[i].set_ylim([y_min - margin, y_max + margin])

    axes[-1].set_xlabel("Time (Seconds)", fontsize=12)
    if title:
        plt.suptitle(title, fontsize=16, y=0.92)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    return fig, axes


def qc_name_for(processed_path: str | Path) -> str:
    return f"{Path(processed_path).stem}_first5s.png"


def save_visualization(
    input_file: str | Path,
    output_file: str | Path,
    seconds_to_show: int = 5,
    sampling_rate: int = 250,
    channels: int = 8,
    title: Optional[str] = None,
) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = visualize_first_seconds(
        input_file=input_file,
        seconds_to_show=seconds_to_show,
        sampling_rate=sampling_rate,
        channels=channels,
        title=title,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def visualize_session(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seconds_to_show: int = 5,
    sampling_rate: int = 250,
    channels: int = 8,
) -> list[Path]:
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    files = sorted(processed_path.glob("*_processed.csv"))
    if not files:
        raise FileNotFoundError(f"{processed_path} 中没有找到 *_processed.csv")

    saved: list[Path] = []
    for file_path in files:
        title = f"{file_path.stem}: Filtered Time Series (First {seconds_to_show} Sec)"
        saved.append(
            save_visualization(
                input_file=file_path,
                output_file=output_path / qc_name_for(file_path),
                seconds_to_show=seconds_to_show,
                sampling_rate=sampling_rate,
                channels=channels,
                title=title,
            )
        )
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save QC plots for processed OpenBCI CSV files")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing *_processed.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write QC PNG files",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=5,
        help="How many leading seconds to visualize",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=250,
        help="Sampling rate used for the x axis",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=8,
        help="How many EXG channels to draw",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = visualize_session(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        seconds_to_show=args.seconds,
        sampling_rate=args.sampling_rate,
        channels=args.channels,
    )
    print(f"Done. Wrote {len(saved)} QC plots to: {args.output_dir}")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
