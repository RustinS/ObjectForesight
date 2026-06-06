#!/usr/bin/env python3
"""
Extract context frames from GCS videos for visualization samples.
Downloads videos from GCS, extracts context frames, saves to context/ subfolder.
"""

import csv
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

GCS_BUCKET = "gs://your-bucket/manip_data"
SAMPLES_DIR = Path("./outputs/fig_samples_sup2")
CSV_PATH = SAMPLES_DIR / "selected_samples.csv"


def parse_csv():
    """Parse CSV and group samples by video_id for efficient downloading."""
    video_to_samples = defaultdict(list)

    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            slug = row["slug"]
            frame_ids = list(map(int, row["frame_ids"].split()))
            # First 3 frames are context frames
            context_frames = frame_ids[:3]
            video_to_samples[video_id].append({
                "slug": slug,
                "context_frames": context_frames,
            })

    return video_to_samples


def download_video(video_id: str, dest_path: str) -> bool:
    """Download video from GCS."""
    gcs_path = f"{GCS_BUCKET}/{video_id}/action.mp4"
    cmd = ["gsutil", "-q", "cp", gcs_path, dest_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to download {gcs_path}: {result.stderr}")
        return False
    return True


def extract_frame(video_path: str, frame_num: int, output_path: str) -> bool:
    """Extract a single frame from video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"select=eq(n\\,{frame_num})",
        "-vframes", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to extract frame {frame_num}: {result.stderr}")
        return False
    return True


def process_video(video_id: str, samples: list):
    """Download video and extract context frames for all samples using it."""
    print(f"\nProcessing video: {video_id} ({len(samples)} samples)")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "action.mp4")

        # Download video
        if not download_video(video_id, video_path):
            return

        # Process each sample
        for sample in samples:
            slug = sample["slug"]
            context_frames = sample["context_frames"]
            sample_dir = SAMPLES_DIR / slug
            context_dir = sample_dir / "context"

            # Skip if context folder already exists and has all frames
            if context_dir.exists():
                existing = list(context_dir.glob("*.png"))
                if len(existing) >= 3:
                    print(f"  Skipping {slug} (context folder already complete)")
                    continue

            context_dir.mkdir(exist_ok=True)

            print(f"  Extracting frames for {slug}: {context_frames}")
            for frame_num in context_frames:
                output_path = context_dir / f"context_{frame_num}.png"
                if output_path.exists():
                    continue
                extract_frame(video_path, frame_num, str(output_path))

    print(f"  Done with {video_id}")


def main():
    video_to_samples = parse_csv()
    print(f"Found {len(video_to_samples)} unique videos for {sum(len(s) for s in video_to_samples.values())} samples")

    for video_id, samples in video_to_samples.items():
        process_video(video_id, samples)

    print("\nDone!")


if __name__ == "__main__":
    main()
