import argparse
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test_expert", choices=["train", "test", "test_silent", "test_expert"])
    parser.add_argument("--video_dir", type=str, default="../data/videos/")
    parser.add_argument("--resolution", type=int, default=360, choices=[144, 240, 360, 480, 720, 1080], help="Resolution for video download (e.g., 360, 720, 1080)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip downloading videos if they already exist in the target directory")
    parser.add_argument("--silence_errors", action="store_true", help="Silence errors")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use for downloading videos")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for downloading videos")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to download (for debug runs). If None, downloads all videos.")
    return parser.parse_args()


def download_video(row, video_dir, args):
    """Download a single video. Returns tuple (video_id, status, video_url) where status is 'success', 'failed', or 'skipped'."""
    video_id = row.video_id
    video_url = row.video_url
    video_path = os.path.join(video_dir, f"{video_id}.mp4")

    # Skip if video already exists
    if os.path.exists(video_path) and args.skip_existing:
        return (video_id, 'skipped', video_url)

    # Construct yt-dlp command
    cmd_args = [
        "yt-dlp",
        "-S", "vcodec:h264,res,acodec:m4a",  # Quicktime compatible; h264 decodes faster as well
        "-f", f"bestvideo[height<={args.resolution}]+bestaudio/best[height<={args.resolution}]",
        "-o", video_path,
        #"--quiet",
        "--concurrent-fragments", str(args.threads),
        video_url
    ]
    if args.silence_errors:
        cmd_args.extend([
            "--no-warnings", "--ignore-errors"
        ])

    # Execute command and check for errors
    result = subprocess.run(cmd_args, capture_output=True)

    if result.returncode == 0:
        return (video_id, 'success', video_url)
    else:
        return (video_id, 'failed', video_url)


def main(args):
    # Setup
    video_dir = os.path.join(args.video_dir, f"{args.resolution}p")
    os.makedirs(video_dir, exist_ok=True)
    df = load_dataset("rghermi/sf20k", split=args.split).to_pandas()
    df = df[['video_id', 'video_url']].drop_duplicates()
    if args.max_videos is not None:
        df = df.head(args.max_videos)
    total_videos = len(df)

    # Initialize counts
    downloaded_count = 0
    failed_count = 0
    skipped_count = 0
    failed_videos = {}

    # Parallel download using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all download tasks
        future_to_row = {
            executor.submit(download_video, row, video_dir, args): row
            for _, row in df.iterrows()
        }

        # Process completed downloads with progress bar
        with tqdm(total=total_videos, desc="Downloading Videos") as pbar:
            for future in as_completed(future_to_row):
                video_id, status, video_url = future.result()
                
                if status == 'success':
                    downloaded_count += 1
                elif status == 'failed':
                    failed_count += 1
                    failed_videos[video_id] = video_url
                elif status == 'skipped':
                    skipped_count += 1
                
                pbar.update(1)

    # --- Final Logging ---
    print("\n" + "="*50)
    print("📊 DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Target Resolution:  {args.resolution}p")
    print(f"Total Videos in Split: {total_videos}")
    print("-" * 25)
    print(f"✅ Successfully Downloaded: {downloaded_count}")
    print(f"⏩ Skipped (already exist): {skipped_count}")
    print(f"❌ Failed to Download:     {failed_count}")
    print("-" * 25)
    if failed_videos:
        print("Failed video IDs and URLs:")
        for video_id, url in failed_videos.items():
            print(f"  - {video_id}: {url}")
    print("="*50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
