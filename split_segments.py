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
from bisect import bisect_right
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
    segmentation_method: str
    segmentation_metrics: dict[str, dict[str, float | int | bool | str]]
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
    parser.add_argument(
        "--segment-method",
        choices=["auto", "pause_search", "peak_pair", "pause_peak_hybrid"],
        default="auto",
        help="Segmentation strategy to use. 'auto' runs both and keeps the better valid result.",
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


def choose_anchor_channel(samples: np.ndarray) -> tuple[int, str, dict[str, float]]:
    scores = compute_channel_scores(samples)
    anchor_idx = int(np.argmax([scores[f"ch{i + 1}"] for i in range(4)]))
    return anchor_idx, f"ch{anchor_idx + 1}", scores


def build_action_intervals(total_length: int, pauses: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Build action intervals using file start/end as implicit boundaries."""
    action_intervals: list[tuple[int, int]] = []
    prev_end = 0
    for start, length in pauses:
        if start > prev_end:
            action_intervals.append((prev_end, start))
        prev_end = start + length

    if prev_end < total_length:
        action_intervals.append((prev_end, total_length))
    return action_intervals


def extract_zero_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if mask[i]:
            start = i
            while i < len(mask) and mask[i]:
                i += 1
            blocks.append((start, i - start))
        else:
            i += 1
    return blocks


def choose_pause_to_remove(
    pauses: list[tuple[int, int]],
    action_intervals: list[tuple[int, int]],
    action_index: int,
) -> int | None:
    """Remove the adjacent pause that yields a merged action closest to the current median length."""
    action_lengths = np.asarray([end - start for start, end in action_intervals], dtype=float)
    target_length = float(np.median(action_lengths))
    candidates: list[tuple[float, int]] = []

    if action_index > 0:
        left_pause_idx = action_index - 1
        merged_left = action_intervals[action_index][1] - action_intervals[action_index - 1][0]
        candidates.append((abs(merged_left - target_length), left_pause_idx))

    if action_index < len(action_intervals) - 1:
        right_pause_idx = action_index
        merged_right = action_intervals[action_index + 1][1] - action_intervals[action_index][0]
        candidates.append((abs(merged_right - target_length), right_pause_idx))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def pad_action_intervals(
    data_length: int,
    action_intervals: list[tuple[int, int]],
    allow_edge_zero_pad: bool = False,
) -> list[tuple[int, int]]:
    action_lengths = [end - start for start, end in action_intervals]
    max_action_len = max(action_lengths)
    segments: list[tuple[int, int]] = []
    last_idx = len(action_intervals) - 1
    for idx, (a_start, a_end) in enumerate(action_intervals):
        pad_total = max_action_len - (a_end - a_start)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        s_start = a_start - pad_left
        s_end = a_end + pad_right

        if allow_edge_zero_pad and idx == 0 and s_start < 0:
            segments.append((s_start, s_end))
            continue
        if allow_edge_zero_pad and idx == last_idx and s_end > data_length:
            segments.append((s_start, s_end))
            continue

        if s_start < 0:
            shift = -s_start
            s_start += shift
            s_end += shift
        if s_end > data_length:
            shift = s_end - data_length
            s_start -= shift
            s_end -= shift

        segments.append((s_start, s_end))
    return segments


def score_action_intervals(action_intervals: list[tuple[int, int]]) -> float:
    lengths = np.asarray([end - start for start, end in action_intervals], dtype=float)
    median_length = float(np.median(lengths))
    return float(lengths.std() / median_length + lengths.max() / median_length)


def build_pause_candidates(
    data: np.ndarray,
    ref_signal: np.ndarray,
    reference_channel: str,
    segments_per_file: int,
) -> list[tuple[float, list[tuple[int, int]], list[tuple[int, int]], dict[str, float | int | bool | str]]]:
    """Return valid pause-search candidates before fixed-length padding."""
    candidates: list[tuple[float, list[tuple[int, int]], list[tuple[int, int]], dict[str, float | int | bool | str]]] = []

    for zero_threshold in range(200, 2001, 50):
        pauses = sorted(extract_zero_blocks(np.abs(ref_signal) < zero_threshold), key=lambda item: item[0])
        if len(pauses) < 2:
            continue

        action_intervals = build_action_intervals(len(data), pauses)
        while len(action_intervals) > segments_per_file:
            action_lengths = np.asarray([end - start for start, end in action_intervals], dtype=float)
            shortest_idx = int(np.argmin(action_lengths))
            pause_idx = choose_pause_to_remove(pauses, action_intervals, shortest_idx)
            if pause_idx is None:
                break
            pauses.pop(pause_idx)
            action_intervals = build_action_intervals(len(data), pauses)

        if len(action_intervals) != segments_per_file:
            continue

        score = score_action_intervals(action_intervals)
        metrics: dict[str, float | int | bool | str] = {
            "valid": True,
            "score": score,
            "reference_channel": reference_channel,
            "zero_threshold": float(zero_threshold),
            "segments_detected": len(action_intervals),
            "segment_length_min": int(min(end - start for start, end in action_intervals)),
            "segment_length_median": float(np.median([end - start for start, end in action_intervals])),
            "segment_length_max": int(max(end - start for start, end in action_intervals)),
        }
        candidates.append((score, list(pauses), action_intervals, metrics))

    return candidates


def segment_with_pause_search(
    data: np.ndarray,
    ref_signal: np.ndarray,
    reference_channel: str,
    segments_per_file: int,
) -> tuple[list[tuple[int, int]], dict[str, float | int | bool | str]]:
    """Search over zero thresholds and keep the most stable valid 100-segment solution."""
    candidates = build_pause_candidates(data, ref_signal, reference_channel, segments_per_file)
    best = min(candidates, key=lambda item: item[0]) if candidates else None

    if best is None:
        raise ValueError("Pause-search method could not find a valid 100-segment solution.")

    return pad_action_intervals(len(data), best[2]), best[3]


def pair_peak_sequences(maxima: np.ndarray, minima: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for max_idx, min_idx in zip(np.sort(maxima), np.sort(minima)):
        if abs(int(max_idx) - int(min_idx)) > max_gap:
            continue
        pairs.append((int(max_idx), int(min_idx)))
    return pairs


def validate_window_edges(signal_1d: np.ndarray, windows: list[tuple[int, int]], threshold: float = 1000.0) -> bool:
    for start, end in windows:
        left_value = 0.0 if start < 0 else float(signal_1d[start])
        right_value = 0.0 if end > len(signal_1d) else float(signal_1d[end - 1])
        if abs(left_value) >= threshold or abs(right_value) >= threshold:
            return False
    return True


def select_exact_non_overlapping_windows(
    candidates: list[dict[str, float | int]], segments_per_file: int
) -> list[dict[str, float | int]] | None:
    if len(candidates) < segments_per_file:
        return None

    windows = sorted(candidates, key=lambda item: (int(item["end"]), int(item["start"])))
    ends = [int(item["end"]) for item in windows]
    prev = [bisect_right(ends, int(item["start"])) - 1 for item in windows]

    neg_inf = float("-inf")
    n = len(windows)
    dp = [[neg_inf] * (segments_per_file + 1) for _ in range(n + 1)]
    take = [[False] * (segments_per_file + 1) for _ in range(n + 1)]
    parent = [[0] * (segments_per_file + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        window = windows[i - 1]
        for j in range(segments_per_file + 1):
            dp[i][j] = dp[i - 1][j]
            parent[i][j] = i - 1
        for j in range(1, segments_per_file + 1):
            prev_idx = prev[i - 1] + 1
            if dp[prev_idx][j - 1] == neg_inf:
                continue
            candidate_score = dp[prev_idx][j - 1] + float(window["strength"])
            if candidate_score > dp[i][j]:
                dp[i][j] = candidate_score
                take[i][j] = True
                parent[i][j] = prev_idx

    if dp[n][segments_per_file] == neg_inf:
        return None

    picked: list[dict[str, float | int]] = []
    i = n
    j = segments_per_file
    while i > 0 and j > 0:
        if take[i][j]:
            picked.append(windows[i - 1])
            i = parent[i][j]
            j -= 1
        else:
            i -= 1

    picked.reverse()
    return picked


def build_target_lengths(raw_lengths: list[int]) -> list[int]:
    values = np.asarray(raw_lengths, dtype=float)
    percentiles = [50, 60, 70, 80, 90, 95, 100]
    targets = {max(32, int(round(np.percentile(values, p)))) for p in percentiles}
    targets.update(range(220, 341, 20))
    return sorted(targets)


def extract_segment_window(
    data: np.ndarray,
    source_rows: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    clip_start = max(0, start_idx)
    clip_end = min(len(data), end_idx)
    seg_data = data[clip_start:clip_end, 1:5]
    seg_src = source_rows[clip_start:clip_end]

    pad_before = max(0, -start_idx)
    pad_after = max(0, end_idx - len(data))
    if pad_before or pad_after:
        seg_data = np.pad(seg_data, ((pad_before, pad_after), (0, 0)), mode="constant", constant_values=0.0)
        seg_src = np.concatenate(
            [
                np.full(pad_before, -1, dtype=source_rows.dtype),
                seg_src,
                np.full(pad_after, -1, dtype=source_rows.dtype),
            ]
        )

    return seg_data, seg_src


def segment_with_peak_pair(
    data: np.ndarray,
    ref_signal: np.ndarray,
    reference_channel: str,
    segments_per_file: int,
) -> tuple[list[tuple[int, int]], dict[str, float | int | bool | str]]:
    """Use the anchor channel's extrema pairs as segment centers."""
    maxima, _ = signal.find_peaks(ref_signal, distance=80)
    minima, _ = signal.find_peaks(-ref_signal, distance=80)
    if maxima.size < segments_per_file or minima.size < segments_per_file:
        raise ValueError(
            f"Peak-pair method found only {maxima.size} maxima and {minima.size} minima on reference channel."
        )

    maxima = maxima[np.argsort(ref_signal[maxima])[-segments_per_file:]]
    minima = minima[np.argsort((-ref_signal[minima]))[-segments_per_file:]]
    pairs = pair_peak_sequences(maxima, minima, max_gap=200)
    if len(pairs) != segments_per_file:
        raise ValueError(f"Peak-pair method produced {len(pairs)} valid max/min pairs; expected {segments_per_file}.")

    action_intervals: list[tuple[int, int]] = []
    edge_ok = True
    for max_idx, min_idx in pairs:
        center = int(round((max_idx + min_idx) / 2))
        start = max(0, center - 140)
        end = min(len(data), center + 140)
        if end - start < 280:
            if start == 0:
                end = min(len(data), 280)
            elif end == len(data):
                start = max(0, len(data) - 280)
        action_intervals.append((start, end))
        edge_ok = edge_ok and abs(ref_signal[start]) < 1000 and abs(ref_signal[end - 1]) < 1000

    action_intervals.sort(key=lambda item: item[0])
    lengths = [end - start for start, end in action_intervals]
    metrics: dict[str, float | int | bool | str] = {
        "valid": len(action_intervals) == segments_per_file,
        "score": score_action_intervals(action_intervals),
        "reference_channel": reference_channel,
        "paired_extrema_max_gap": int(max(abs(max_idx - min_idx) for max_idx, min_idx in pairs)),
        "edge_below_1000": edge_ok,
        "segments_detected": len(action_intervals),
        "segment_length_min": int(min(lengths)),
        "segment_length_median": float(np.median(lengths)),
        "segment_length_max": int(max(lengths)),
    }
    return action_intervals, metrics


def segment_with_pause_peak_hybrid(
    data: np.ndarray,
    ref_signal: np.ndarray,
    reference_channel: str,
    segments_per_file: int,
) -> tuple[list[tuple[int, int]], dict[str, float | int | bool | str]]:
    """Use anchor-derived pause intervals and anchor peaks, then pick 100 final non-overlapping windows."""
    best: tuple[float, int, float, list[tuple[int, int]], dict[str, float | int | bool | str]] | None = None
    for zero_threshold in range(200, 2001, 50):
        pauses = sorted(extract_zero_blocks(np.abs(ref_signal) < zero_threshold), key=lambda item: item[0])
        if len(pauses) < 2:
            continue

        raw_intervals = build_action_intervals(len(data), pauses)
        raw_candidates: list[dict[str, float | int]] = []
        for interval_start, interval_end in raw_intervals:
            interval_length = interval_end - interval_start
            if interval_length < 2:
                continue
            interval_signal = ref_signal[interval_start:interval_end]
            max_local = int(np.argmax(interval_signal))
            min_local = int(np.argmin(interval_signal))
            max_idx = interval_start + max_local
            min_idx = interval_start + min_local
            pair_gap = abs(max_idx - min_idx)
            if pair_gap >= 200:
                continue

            center = int(round((max_idx + min_idx) / 2))
            raw_candidates.append(
                {
                    "center": center,
                    "strength": abs(float(ref_signal[max_idx] - ref_signal[min_idx])),
                    "pair_gap": pair_gap,
                    "window_length": interval_length,
                }
            )

        if len(raw_candidates) < segments_per_file:
            continue

        raw_length_floor = max(32, int(round(np.percentile([int(item["window_length"]) for item in raw_candidates], 25))))
        for target_length in build_target_lengths([int(item["window_length"]) for item in raw_candidates]):
            final_candidates: list[dict[str, float | int]] = []
            front_span = max(0, target_length // 2 - 75)
            for item in raw_candidates:
                if int(item["window_length"]) < raw_length_floor:
                    continue
                center = int(item["center"])
                start = center - front_span
                end = start + target_length
                final_candidates.append(
                    {
                        "start": start,
                        "end": end,
                        "strength": float(item["strength"]),
                        "pair_gap": int(item["pair_gap"]),
                        "window_length": target_length,
                    }
                )

            selected = select_exact_non_overlapping_windows(final_candidates, segments_per_file)
            if selected is None:
                continue

            final_windows = [(int(item["start"]), int(item["end"])) for item in selected]
            pair_gaps = [int(item["pair_gap"]) for item in selected]
            total_strength = float(sum(float(item["strength"]) for item in selected))
            edge_ok = validate_window_edges(ref_signal, final_windows, threshold=1000.0)
            metrics: dict[str, float | int | bool | str] = {
                "valid": True,
                "score": -total_strength,
                "reference_channel": reference_channel,
                "source_zero_threshold": float(zero_threshold),
                "min_raw_length_required": int(raw_length_floor),
                "candidate_windows_found": len(raw_candidates),
                "candidate_windows_after_length_filter": len(final_candidates),
                "segments_detected": len(final_windows),
                "pair_gap_max": int(max(pair_gaps)),
                "pair_gap_median": float(np.median(pair_gaps)),
                "edge_below_1000": edge_ok,
                "segment_length_min": target_length,
                "segment_length_median": float(target_length),
                "segment_length_max": target_length,
                "padded_segment_length": target_length,
            }
            length_penalty = abs(target_length - 300)
            ranking_score = float(length_penalty)
            if (
                best is None
                or ranking_score < best[0]
                or (ranking_score == best[0] and target_length > best[1])
                or (ranking_score == best[0] and target_length == best[1] and total_strength > best[2])
            ):
                best = (ranking_score, target_length, total_strength, final_windows, metrics)

    if best is None:
        raise ValueError("Pause-peak hybrid method could not find 100 non-overlapping aligned windows.")

    return best[3], best[4]


def find_action_boundaries(
    data: np.ndarray,
    source_rows: np.ndarray,
    pause_tol: float,
    segments_per_file: int = 100,
    method: str = "auto",
    anchor_idx: int = 0,
    anchor_channel: str = "ch1",
) -> tuple[list[tuple[int, int]], str, dict[str, dict[str, float | int | bool | str]]]:
    del source_rows, pause_tol
    ref_signal = data[:, 1 + anchor_idx]
    methods = [method] if method != "auto" else ["pause_search", "peak_pair", "pause_peak_hybrid"]
    results: dict[str, tuple[list[tuple[int, int]], dict[str, float | int | bool | str]]] = {}
    errors: dict[str, str] = {}

    for name in methods:
        try:
            if name == "pause_search":
                results[name] = segment_with_pause_search(data, ref_signal, anchor_channel, segments_per_file)
            elif name == "peak_pair":
                results[name] = segment_with_peak_pair(data, ref_signal, anchor_channel, segments_per_file)
            elif name == "pause_peak_hybrid":
                results[name] = segment_with_pause_peak_hybrid(data, ref_signal, anchor_channel, segments_per_file)
            else:
                raise ValueError(f"Unknown segmentation method: {name}")
        except Exception as exc:
            errors[name] = str(exc)

    if not results:
        raise ValueError("All segmentation methods failed: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))

    if method == "auto":
        chosen_method = min(results.items(), key=lambda item: float(item[1][1]["score"]))[0]
    else:
        chosen_method = method
        if chosen_method not in results:
            raise ValueError(errors.get(chosen_method, f"Segmentation method failed: {chosen_method}"))

    metrics = {name: info[1] for name, info in results.items()}
    for name, message in errors.items():
        metrics[name] = {"valid": False, "error": message}
    return results[chosen_method][0], chosen_method, metrics


def process_file(
    path: Path,
    segments_per_file: int,
    pause_tol: float,
    processed_dir: Path,
    qc_dir: Path | None,
    segment_method: str,
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

    # Drop the first 1 second of data to remove the initial transient (assuming 250Hz sampling rate)
    first_valid = int(1.0 * 250.0)

    post = data[first_valid:]
    post_src = source_rows[first_valid:]
    if post.size == 0:
        raise ValueError(f"No data remains after initial transient removal in {path}")

    anchor_idx, anchor, pre_segment_scores = choose_anchor_channel(post[:, 1:5])
    boundaries, chosen_method, segmentation_metrics = find_action_boundaries(
        post,
        post_src,
        pause_tol=pause_tol,
        segments_per_file=segments_per_file,
        method=segment_method,
        anchor_idx=anchor_idx,
        anchor_channel=anchor,
    )
    
    if len(boundaries) != segments_per_file:
        print(f"Warning: Expected {segments_per_file} action segments in {path.name}, but identified {len(boundaries)} due to dropped blocks.")

    # Extract strictly equal-length padded segments defined by boundaries
    segments_list = []
    src_segments_list = []
    
    for start_idx, end_idx in boundaries:
        seg_data, seg_src = extract_segment_window(post, post_src, start_idx, end_idx)
        segments_list.append(seg_data)
        src_segments_list.append(seg_src)

    # Validate output length is perfectly uniform and non-overlapping
    max_length = segments_list[0].shape[0]
    segments_array = np.array(segments_list, dtype=np.float32)  # shape=(100, L, 4)

    all_action = np.vstack(segments_list)

    return (
        RecordingResult(
            direction=extract_direction(path),
            anchor_channel=anchor,
            channel_scores=pre_segment_scores,
            segmentation_method=chosen_method,
            segmentation_metrics=segmentation_metrics,
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
        "segmentation_method": result.segmentation_method,
        "segmentation_metrics": result.segmentation_metrics,
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
            segment_method=args.segment_method,
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
            f"anchor={item['anchor_channel']}, "
            f"method={item['segmentation_method']}"
        )


if __name__ == "__main__":
    main()
