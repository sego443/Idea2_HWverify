#!/usr/bin/env python3
"""Process OpenBCI raw recordings into BrainFlow-filtered numeric CSV files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes


DEFAULT_SESSION_DIR = Path("data/OpenBCISession_2026-03-25_19-51-25")
DEFAULT_OUTPUT_DIR = Path("outputs/processed_csv")


def clean_raw_data(input_file: Path, temp_file: Path) -> None:
    """Strip comments, headers, and non-numeric fields from a raw OpenBCI text file."""
    with input_file.open("r", encoding="utf-8") as f_in, temp_file.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.startswith("%") or line.strip().lower().startswith("sample"):
                continue

            clean_parts: list[str] = []
            for part in line.strip().split(","):
                value = part.strip()
                try:
                    float(value)
                except ValueError:
                    continue
                clean_parts.append(value)

            if clean_parts:
                f_out.write(",".join(clean_parts) + "\n")


def processed_name_for(input_path: Path) -> str:
    return f"{input_path.stem}_processed.csv"


def process_playback_file(
    input_file: str | Path,
    output_file: str | Path,
    bandpass: Optional[Sequence[float]] = (5.0, 50.0),
    bandstop: Optional[Sequence[float]] = None,
    env: Optional[str] = "both",
    disable_bandpass: bool = False,
    disable_env: bool = False,
    temp_file: str | Path | None = None,
) -> Path:
    """Replicate the GUI playback pipeline and write a numeric CSV."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} 不存在")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(temp_file) if temp_file else output_path.with_suffix(".clean_tmp.csv")

    clean_raw_data(input_path, temp_path)
    try:
        data = DataFilter.read_file(str(temp_path))
        sampling_rate = 250
        exg_channels = range(1, 9)

        if bandstop:
            low, high = bandstop
            for channel in exg_channels:
                DataFilter.perform_bandstop(
                    data[channel], sampling_rate, low, high, 4, FilterTypes.BUTTERWORTH.value, 0
                )

        if (not disable_bandpass) and bandpass:
            low, high = bandpass
            for channel in exg_channels:
                DataFilter.perform_bandpass(
                    data[channel], sampling_rate, low, high, 4, FilterTypes.BUTTERWORTH.value, 0
                )

        if not disable_env and env:
            env_list: list[int] = []
            if env == "both":
                env_list = [NoiseTypes.FIFTY.value, NoiseTypes.SIXTY.value]
            elif env == "50":
                env_list = [NoiseTypes.FIFTY.value]
            elif env == "60":
                env_list = [NoiseTypes.SIXTY.value]

            for noise in env_list:
                for channel in exg_channels:
                    DataFilter.remove_environmental_noise(data[channel], sampling_rate, noise)

        DataFilter.write_file(data, str(output_path), "w")
        return output_path
    finally:
        if temp_path.exists():
            os.remove(temp_path)


def process_session(
    session_dir: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    bandpass: Optional[Sequence[float]] = (5.0, 50.0),
    bandstop: Optional[Sequence[float]] = None,
    env: Optional[str] = "both",
    disable_bandpass: bool = False,
    disable_env: bool = False,
) -> list[Path]:
    session_path = Path(session_dir)
    output_path = Path(output_dir)
    files = sorted(session_path.glob("OpenBCI-RAW-*_*.txt"))
    if not files:
        raise FileNotFoundError(f"{session_path} 中没有找到 OpenBCI-RAW-*_*.txt")

    processed_files: list[Path] = []
    for raw_path in files:
        processed_files.append(
            process_playback_file(
                input_file=raw_path,
                output_file=output_path / processed_name_for(raw_path),
                bandpass=bandpass,
                bandstop=bandstop,
                env=env,
                disable_bandpass=disable_bandpass,
                disable_env=disable_env,
            )
        )
    return processed_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-process OpenBCI raw recordings into numeric CSV files")
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=DEFAULT_SESSION_DIR,
        help="Directory containing OpenBCI-RAW-*_*.txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write processed CSV files",
    )
    parser.add_argument(
        "--bandpass",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=[5.0, 50.0],
        help="Bandpass Hz range (default: 5 50). Use --no-bandpass to disable.",
    )
    parser.add_argument(
        "--no-bandpass",
        action="store_true",
        help="Disable bandpass filtering.",
    )
    parser.add_argument(
        "--bandstop",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Optional bandstop Hz range.",
    )
    parser.add_argument(
        "--env",
        choices=["50", "60", "both"],
        default="both",
        help="Environmental noise removal mode.",
    )
    parser.add_argument(
        "--no-env",
        action="store_true",
        help="Disable environmental noise removal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = process_session(
        session_dir=args.session_dir,
        output_dir=args.output_dir,
        bandpass=args.bandpass,
        bandstop=args.bandstop,
        env=args.env,
        disable_bandpass=args.no_bandpass,
        disable_env=args.no_env,
    )
    print(f"Done. Wrote {len(processed)} processed files to: {args.output_dir}")
    for path in processed:
        print(f"- {path}")


if __name__ == "__main__":
    main()
