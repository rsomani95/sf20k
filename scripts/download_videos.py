import argparse
import subprocess
import os
import threading
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global lock for updating counters safely across threads
lock = threading.Lock()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test_expert", choices=["train", "test", "test_silent", "test_expert"])
    parser.add_argument("--video_dir", type=str, default="../data/videos/")
    parser.add_argument("--resolution", type=int, default=360, choices=[144, 240, 360, 480, 720, 1080])
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing videos")
    # New argument for workers
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel downloads")
    return parser.parse_args()

def download_single_video(row, args, video_dir):
    """
    Helper function to download one video. 
    Returns: (status_code, video_id, video_url)
    status_code: 0=success, 1=failed, 2=skipped
    """
    video_id = row.video_id
    video_url = row.video_url
    video_path = os.path.join(video_dir, f"{video_id}.mp4")

    # Check skip
    if os.path.exists(video_path) and args.skip_existing:
        return 2, video_id, video_url

    # Construct yt-dlp command with ARIA2 optimizations
    cmd_args = [
        "yt-dlp",
        "-f", f"bestvideo[height<={args.resolution}]+bestaudio/best[height<={args.resolution}]",
        "-o", video_path,
        "--merge-output-format", "mp4",
        "--no-warnings",
        "--ignore-errors",
        # --- SPEED OPTIMIZATIONS ---
        "--external-downloader", "aria2c",
        "--external-downloader-args", "aria2c:-j 4 -x 4 -s 4 -k 1M",
        # ---------------------------
        video_url
    ]
    
    # Capture output to avoid console spam in parallel mode
    result = subprocess.run(cmd_args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    if result.returncode == 0:
        return 0, video_id, video_url
    else:
        return 1, video_id, video_url

def main(args):
    # Setup
    video_dir = os.path.join(args.video_dir, f"{args.resolution}p")
    os.makedirs(video_dir, exist_ok=True)
    
    print("Loading dataset...")
    df = load_dataset("rghermi/sf20k", split=args.split).to_pandas()
    df = df[['video_id', 'video_url']].drop_duplicates()
    total_videos = len(df)

    # Initialize counts
    downloaded_count = 0
    failed_count = 0
    skipped_count = 0
    failed_videos = {}

    print(f"Starting download with {args.workers} workers using aria2c...")

    # ThreadPool execution
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = [executor.submit(download_single_video, row, args, video_dir) for _, row in df.iterrows()]
        
        # Process as they complete with TQDM
        for future in tqdm(as_completed(futures), total=total_videos, desc="Downloading"):
            status, vid, url = future.result()
            
            if status == 0:
                downloaded_count += 1
            elif status == 1:
                failed_count += 1
                failed_videos[vid] = url
            elif status == 2:
                skipped_count += 1

    # --- Final Logging ---
    print("\n" + "="*50)
    print("📊 DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Target Resolution:  {args.resolution}p")
    print(f"Total Videos:       {total_videos}")
    print("-" * 25)
    print(f"✅ Successfully Downloaded: {downloaded_count}")
    print(f"⏩ Skipped (already exist): {skipped_count}")
    print(f"❌ Failed to Download:      {failed_count}")
    print("-" * 25)
    
    if failed_videos:
        print(f"Failed count: {len(failed_videos)}")
        # Optional: Save failed to a file instead of printing thousands of lines
        with open("failed_downloads.txt", "w") as f:
            for video_id, url in failed_videos.items():
                f.write(f"{video_id},{url}\n")
        print("Failed videos saved to failed_downloads.txt")
    print("="*50)

if __name__ == "__main__":
    args = parse_args()
    main(args)