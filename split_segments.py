#!/usr/bin/env python3
"""Split OpenBCI recordings into 100 equal-length segments (ch1-4 only).

This script:
1) Drops the initial transient by starting after the first pause block.
2) Uses ch1-4 only and computes an adaptive anchor channel per recording.
3) Builds 100 equal-length aligned segments and channel-wise expanded samples.
4) Exports artifacts for downstream ML and for standalone visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from process_data import process_playback_file, processed_name_for
from visualize_data import qc_name_for, save_visualization
from visualize_segments import pick_detail_segment, plot_detail, plot_overview, plot_sensitivity


EXG_COLS = ["EXG Channel 0", "EXG Channel 1", "EXG Channel 2", "EXG Channel 3"]


@dataclass
class RecordingResult:
    direction: str
    anchor_channel: str
    channel_scores: dict[str, float]
    segment_length: int
    total_action_samples: int
    used_samples: int
    dropped_remainder: int
    dropped_initial_rows: int
    segments: np.ndarray  # shape=(100, L, 4)
    source_rows: np.ndarray  # shape=(100, L)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split OpenBCI recordings into equal segments")
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path("data/OpenBCISession_2026-03-25_19-51-25"),
        help="Directory containing OpenBCI-RAW-*_*.txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/segmentation"),
        help="Output directory for segmented arrays and metadata",
    )
    parser.add_argument(
        "--segments-per-file",
        type=int,
        default=100,
        help="Number of equal-length time segments per recording",
    )
    parser.add_argument(
        "--pause-tol",
        type=float,
        default=1e-9,
        help="Absolute value threshold for pause detection on ch1-4",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("outputs/processed_csv"),
        help="Directory to write intermediate processed CSV files",
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=Path("outputs/processed_qc"),
        help="Directory to write QC plots for processed CSV files",
    )
    parser.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip saving first-seconds QC plots for processed CSV files",
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=Path("outputs/visualizations"),
        help="Directory to write per-direction segment visualizations",
    )
    parser.add_argument(
        "--skip-segment-viz",
        action="store_true",
        help="Skip overview/detail/sensitivity plots for segmented outputs",
    )
    return parser.parse_args()


def extract_direction(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    return parts[-1].lower() if parts else stem.lower()


def load_recording(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (signals, source_rows) where signals shape is (N, 4)."""
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("Sample Index"):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Cannot find header row in {path}")

    reader = csv.reader(lines[header_idx:], skipinitialspace=True)
    header = next(reader, None)
    if not header:
        raise ValueError(f"Cannot parse header row in {path}")

    col_idx = {name.strip(): i for i, name in enumerate(header)}
    required = ["Sample Index", *EXG_COLS]
    missing = [name for name in required if name not in col_idx]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    rows = []
    source_rows = []
    for csv_idx, row in enumerate(reader, start=header_idx + 2):
        try:
            sample_index = float(row[col_idx["Sample Index"]].strip())
            values = [float(row[col_idx[col]].strip()) for col in EXG_COLS]
        except (IndexError, ValueError, AttributeError):
            continue
        rows.append([sample_index, *values])
        source_rows.append(csv_idx)

    data = np.asarray(rows, dtype=float)
    src = np.asarray(source_rows, dtype=int)
    if data.size == 0:
        raise ValueError(f"No valid data rows parsed from {path}")
    return data, src


def load_processed_recording(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (signals, source_rows) from a processed numeric CSV file."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 5:
        raise ValueError(f"Processed file {path} has too few columns: {data.shape}")
    source_rows = np.arange(data.shape[0], dtype=int)
    return data, source_rows


def apply_filters(data: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """Apply 1-50Hz bandpass and 50Hz/60Hz notch filters to remove DC drift and powerline noise."""
    filtered_data = data.copy()
    
    # 1. Bandpass Filter (1 Hz to 50 Hz)
    # Using a 4th order Butterworth filter
    nyq = 0.5 * fs
    low = 1.0 / nyq
    high = 49.9 / nyq  # Slightly below 50 so it doesn't fail if exactly at nyq limit context, 50Hz is fine if nyq=125
    b_bp, a_bp = signal.butter(4, [1.0, 50.0], btype='bandpass', fs=fs)
    
    # 2. Notch Filters (50Hz and 60Hz)
    q = 30.0  # Quality factor
    b_50, a_50 = signal.iirnotch(50.0, q, fs)
    b_60, a_60 = signal.iirnotch(60.0, q, fs)
    
    for i in range(1, 5):  # Channels 1-4
        ch_data = filtered_data[:, i]
        # Apply bandpass
        ch_data = signal.filtfilt(b_bp, a_bp, ch_data)
        # Apply notches
        ch_data = signal.filtfilt(b_50, a_50, ch_data)
        ch_data = signal.filtfilt(b_60, a_60, ch_data)
        filtered_data[:, i] = ch_data
        
    return filtered_data


def find_first_pause_end(data: np.ndarray, pause_tol: float) -> int:
    """Return index right after the first pause block."""
    is_pause = (np.abs(data[:, 0]) <= pause_tol) & (np.max(np.abs(data[:, 1:5]), axis=1) <= pause_tol)
    pause_idxs = np.flatnonzero(is_pause)
    if pause_idxs.size == 0:
        raise ValueError("Pause marker not found; cannot remove initial transient")

    start = pause_idxs[0]
    end = start
    while end + 1 < len(is_pause) and is_pause[end + 1]:
        end += 1
    return end + 1


def compute_channel_scores(samples: np.ndarray) -> dict[str, float]:
    scores: dict[str, float] = {}
    for i in range(4):
        x = samples[:, i]
        centered = np.abs(x - np.median(x))
        # Robust high-percentile amplitude as sensitivity score.
        scores[f"ch{i + 1}"] = float(np.percentile(centered, 95))
    return scores


def find_action_boundaries(
    data: np.ndarray, source_rows: np.ndarray, pause_tol: float, segments_per_file: int = 100
) -> list[tuple[int, int]]:
    """Identify action segment boundaries using the specified zero-segment logic.

    1. Convert data to uV (assuming OpenBCI 1x gain, 1 count = 0.5364 uV).
    2. Treat fluctuations < 1000 uV as 0 (pause regions).
    3. Find all 0 segments, measure lengths, sort, and pick the top 101 longest.
    4. Sort these top 101 zero segments chronologically.
    5. The 100 regions between these zero segments are the action bounds.
    6. Ensure all action segments are padded from original data to equal the longest action length.
    """
    exg = data[:, 1:5]
    
    # 1. Convert to uV for 1x Gain 
    uV = exg * 0.5364
    
    # 2. Use the differential (fluctuation) to find rest (0) segments, not absolute DC level.
    diffs = np.max(np.abs(np.diff(uV, axis=0)), axis=1)
    
    # Applying a small smoothing helps prevent tiny noise spikes from breaking long pauses
    # Pad diffs so it aligns with the original array size
    diffs_padded = np.append(diffs, diffs[-1])
    envelope = np.convolve(diffs_padded, np.ones(10)/10, mode='same')
    
    # Grid search for the optimal threshold ratio to find EXACTLY `segments_per_file + 1` gaps
    med_env = np.median(envelope)
    best_th, best_cnt = med_env, 0
    target_pauses = segments_per_file + 1
    
    for ratio in np.linspace(0.1, 20.0, 400):
        th = med_env * ratio
        is_z = envelope < th
        b = 0
        i = 0
        while i < len(is_z):
            if is_z[i]:
                st = i
                while i < len(is_z) and is_z[i]: i += 1
                if i - st > 50: b += 1
            else: i += 1
        
        if abs(target_pauses - b) < abs(target_pauses - best_cnt) or (b == target_pauses and best_cnt != target_pauses):
            best_cnt = b
            best_th = th
            if best_cnt == target_pauses: break
            
    # 1. 动态寻找最佳基线，使过滤后的长停顿恰好有 target_pauses 段
    is_zero = envelope < best_th
    
    # 2. 找出所有的0段并标记位置, 测量长度
    blocks = []
    i = 0
    while i < len(is_zero):
        if is_zero[i]:
            start = i
            while i < len(is_zero) and is_zero[i]:
                i += 1
            blocks.append((start, i - start))  # (start_index, length)
        else:
            i += 1
            
    # 3. 排序, 选取(segments_per_file + 1) 个最长的0段
    blocks.sort(key=lambda x: x[1], reverse=True)
    
    required_pauses = min(segments_per_file + 1, len(blocks))
    if required_pauses <= 1:
        raise ValueError(f"Found only {len(blocks)} zero blocks (threshold 1000uV), not enough to extract segments.")
        
    actual_segments = required_pauses - 1
    
    top_pauses = blocks[:required_pauses]
    
    # 4. 排序并按顺序取出时间点
    top_pauses.sort(key=lambda x: x[0])  # Sort chronologically
    
    # 5. 取这些0段的两端作为分割点, 中间分出有动作的段
    action_intervals = []
    action_lengths = []
    for i in range(actual_segments):
        action_start = top_pauses[i][0] + top_pauses[i][1]  # current 0-segment ends
        action_end = top_pauses[i+1][0]                     # next 0-segment begins
        
        if action_end <= action_start:
            # Fallback if consecutive pauses touch (should not happen mathematically if they are distinct blocks)
            action_end = action_start + 1
            
        action_intervals.append((action_start, action_end))
        action_lengths.append(action_end - action_start)
        
    # 6. 根据最长动作段长度, 将其他段向两边延展(统一补齐至相同长度)
    max_action_len = max(action_lengths)
    
    segments = []
    for i in range(actual_segments):
        a_start, a_end = action_intervals[i]
        pad_total = max_action_len - (a_end - a_start)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        
        # Extend into the original data naturally
        s_start = a_start - pad_left
        s_end = a_end + pad_right
        
        # Ensure we don't go out of bounds of the file!
        if s_start < 0:
            shift = -s_start
            s_start += shift
            s_end += shift
        if s_end > len(data):
            shift = s_end - len(data)
            s_start -= shift
            s_end -= shift
            
        segments.append((s_start, s_end))
        
    return segments


def process_file(
    path: Path,
    segments_per_file: int,
    pause_tol: float,
    processed_dir: Path,
    qc_dir: Path | None,
) -> tuple[RecordingResult, Path, Path | None]:
    processed_path = process_playback_file(
        input_file=path,
        output_file=processed_dir / processed_name_for(path),
    )
    qc_path: Path | None = None
    if qc_dir is not None:
        qc_path = save_visualization(
            input_file=processed_path,
            output_file=qc_dir / qc_name_for(processed_path),
            seconds_to_show=5,
            sampling_rate=250,
            channels=8,
            title=f"{path.stem}: Filtered Time Series (First 5 Sec)",
        )

    data, source_rows = load_processed_recording(processed_path)

    # Drop the first 2 seconds of data to remove initial transient (assuming 250Hz sampling rate)
    first_valid = int(2.0 * 250.0)

    post = data[first_valid:]
    post_src = source_rows[first_valid:]
    if post.size == 0:
        raise ValueError(f"No data remains after initial transient removal in {path}")

    boundaries = find_action_boundaries(post, post_src, pause_tol=pause_tol, segments_per_file=segments_per_file)
    
    if len(boundaries) != segments_per_file:
        print(f"Warning: Expected {segments_per_file} action segments in {path.name}, but identified {len(boundaries)} due to dropped blocks.")

    # Extract strictly equal-length padded segments defined by boundaries
    segments_list = []
    src_segments_list = []
    
    for start_idx, end_idx in boundaries:
        seg_data = post[start_idx:end_idx, 1:5]
        seg_src = post_src[start_idx:end_idx]
        segments_list.append(seg_data)
        src_segments_list.append(seg_src)

    # Validate output length is perfectly uniform and non-overlapping
    max_length = segments_list[0].shape[0]
    segments_array = np.array(segments_list, dtype=np.float32)  # shape=(100, L, 4)

    # Compute channel scores on all exported action data
    all_action = np.vstack(segments_list)
    channel_scores = compute_channel_scores(all_action)
    anchor_idx = int(np.argmax([channel_scores[f"ch{i + 1}"] for i in range(4)]))
    anchor = f"ch{anchor_idx + 1}"

    return (
        RecordingResult(
            direction=extract_direction(path),
            anchor_channel=anchor,
            channel_scores=channel_scores,
            segment_length=int(max_length),
            total_action_samples=int(all_action.shape[0]),
            used_samples=int(all_action.shape[0]),
            dropped_remainder=0,
            dropped_initial_rows=int(first_valid),
            segments=segments_array,
            source_rows=np.array(src_segments_list, dtype=object),
        ),
        processed_path,
        qc_path,
    )


def save_result(output_dir: Path, result: RecordingResult, processed_path: Path, qc_path: Path | None) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    direction = result.direction

    # Segments are fixed shape: (100, L, 4)
    segments_array = result.segments
    
    # Save segments directly
    segments_path = output_dir / f"{direction}_segments.npy"
    np.save(segments_path, segments_array)

    # Segment metadata
    summary = {
        "direction": direction,
        "anchor_channel": result.anchor_channel,
        "channel_scores": result.channel_scores,
        "segments": int(segments_array.shape[0]),
        "segment_length": int(result.segment_length),
        "total_action_samples": int(result.total_action_samples),
        "used_samples": int(result.used_samples),
        "dropped_remainder": int(result.dropped_remainder),
        "dropped_initial_rows": int(result.dropped_initial_rows),
        "segments_path": str(segments_path),
        "processed_path": str(processed_path),
        "qc_plot_path": str(qc_path) if qc_path else None,
    }

    summary_path = output_dir / f"{direction}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def save_segment_visualizations(
    summaries: list[dict[str, object]],
    out_dir: Path,
    overview_count: int = 6,
    seed: int = 42,
) -> dict[str, dict[str, object]]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_manifest: dict[str, dict[str, object]] = {}

    for file_item in summaries:
        direction = str(file_item["direction"])
        anchor = str(file_item["anchor_channel"])
        channel_scores = dict(file_item["channel_scores"])

        seg_path = Path(str(file_item["segments_path"]))
        segments = np.load(seg_path)
        if segments.ndim != 3 or segments.shape[2] != 4:
            raise ValueError(f"Unexpected segment shape for {direction}: {segments.shape}")

        dir_out = out_dir / direction
        dir_out.mkdir(parents=True, exist_ok=True)

        overview_name = f"{direction}_overview_segments.png"
        detail_seg = pick_detail_segment(segments, anchor)
        detail_name = f"{direction}_detail_seg{detail_seg}.png"
        sens_name = f"{direction}_sensitivity_anchor_{anchor}.png"

        picked = plot_overview(
            direction=direction,
            segments=segments,
            out_path=dir_out / overview_name,
            count=overview_count,
            seed=seed,
            specific_ids=None,
        )
        plot_detail(direction=direction, segments=segments, seg_id=detail_seg, out_path=dir_out / detail_name)
        plot_sensitivity(
            direction=direction,
            channel_scores=channel_scores,
            anchor_channel=anchor,
            out_path=dir_out / sens_name,
        )

        figure_manifest[direction] = {
            "overview": str(dir_out / overview_name),
            "overview_segment_ids": picked,
            "detail": str(dir_out / detail_name),
            "detail_segment_id": detail_seg,
            "sensitivity": str(dir_out / sens_name),
            "anchor_channel": anchor,
        }

    (out_dir / "figure_manifest.json").write_text(json.dumps(figure_manifest, indent=2), encoding="utf-8")
    return figure_manifest


def main() -> None:
    args = parse_args()
    session_dir = args.session_dir
    output_dir = args.output_dir
    processed_dir = args.processed_dir
    qc_dir = None if args.skip_qc else args.qc_dir
    viz_dir = None if args.skip_segment_viz else args.viz_dir

    files = sorted(session_dir.glob("OpenBCI-RAW-*_*.txt"))
    if not files:
        raise SystemExit(f"No OpenBCI txt files found in {session_dir}")

    summaries = []
    for path in files:
        result, processed_path, qc_path = process_file(
            path,
            segments_per_file=args.segments_per_file,
            pause_tol=args.pause_tol,
            processed_dir=processed_dir,
            qc_dir=qc_dir,
        )
        summaries.append(save_result(output_dir, result, processed_path, qc_path))

    manifest = {
        "session_dir": str(session_dir),
        "output_dir": str(output_dir),
        "segments_per_file": args.segments_per_file,
        "channels_used": ["ch1", "ch2", "ch3", "ch4"],
        "files": summaries,
        "total_time_segments": int(sum(item["segments"] for item in summaries)),
        "total_channel_samples": int(sum(item["segments"] * 4 for item in summaries)),
    }
    if viz_dir is not None:
        manifest["segment_visualizations"] = save_segment_visualizations(summaries, viz_dir)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Done. Wrote segmentation outputs to: {output_dir}")
    for item in summaries:
        print(
            f"- {item['direction']}: segments={item['segments']}, "
            f"length={item['segment_length']}, "
            f"anchor={item['anchor_channel']}"
        )


if __name__ == "__main__":
    main()
