import argparse
import subprocess
import os
import shutil
import concurrent.futures
from tqdm import tqdm
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test_expert", choices=["train", "test", "test_silent", "test_expert"])
    parser.add_argument("--video_dir", type=str, default="../data/videos/")
    parser.add_argument("--resolution", type=int, default=360, choices=[144, 240, 360, 480, 720, 1080], help="Resolution for video download (e.g., 360, 720, 1080)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip downloading videos if they already exist in the target directory")
    
    # Performance & Error Handling
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel download workers")
    parser.add_argument("--no_aria2c", action="store_true", help="Disable aria2c external downloader (use native yt-dlp)")
    parser.add_argument("--suppress_errors", action="store_true", help="Suppress download errors and warnings (fail silently)")
    
    return parser.parse_args()


def check_dependencies(no_aria2c):
    """Checks if aria2c is installed if requested."""
    # If no_aria2c is True, we don't need to check anything
    if no_aria2c:
        return

    # Otherwise, we expect aria2c to be available
    if shutil.which('aria2c') is None:
        raise Exception(
            "\n\n❌ 'aria2c' is not installed/found on your system.\n"
            "It is highly recommended for faster downloads.\n"
            "  - To install on Ubuntu/Debian: sudo apt install aria2c\n"
            "  - To install on MacOS: brew install aria2c\n"
            "  - To bypass this error and use the slower native downloader, use the flag: --no_aria2c\n"
        )


def download_single_video(args_pack):
    """
    Worker function for parallel execution.
    args_pack: (video_id, video_url, video_dir, resolution, suppress_errors, use_aria2c)
    """
    video_id, video_url, video_dir, resolution, suppress_errors, use_aria2c = args_pack
    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    
    # Construct yt-dlp command
    cmd_args = [
        "yt-dlp",
        "-f", f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]",
        "-o", video_path,
        "--merge-output-format", "mp4",
        "-N", "4",  # Download multiple fragments of the same video in parallel
    ]

    # Add aria2c if enabled
    if use_aria2c:
        cmd_args.extend([
            "--downloader", "aria2c",
            "--downloader-args", "aria2c:-j 4 -x 4 -s 4 -k 1M"
        ])

    # Error handling flags
    if suppress_errors:
        cmd_args.extend(["--no-warnings", "--ignore-errors"])
    
    try:
        # Capture output to prevent terminal spam in parallel mode
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return True, video_id, video_url, None
        else:
            error_msg = result.stderr if result.stderr else "Unknown Error"
            return False, video_id, video_url, error_msg

    except Exception as e:
        return False, video_id, video_url, str(e)


def main(args):
    # 1. Check Dependencies
    check_dependencies(args.no_aria2c)

    # 2. Setup
    # Change: video_dir / split / res / file.ext
    video_dir = os.path.join(args.video_dir, args.split, f"{args.resolution}p")
    os.makedirs(video_dir, exist_ok=True)
    
    print(f"Loading dataset split '{args.split}'...")
    df = load_dataset("rghermi/sf20k", split=args.split).to_pandas()
    df = df[['video_id', 'video_url']].drop_duplicates()
    total_videos = len(df)

    # 3. Prepare Tasks
    download_tasks = []
    skipped_count = 0
    
    # Simplify logic for enabling aria2c
    use_aria2c = not args.no_aria2c
    
    print("Preparing download queue...")
    for _, row in df.iterrows():
        video_id = row.video_id
        video_path = os.path.join(video_dir, f"{video_id}.mp4")

        # Check if exists
        if args.skip_existing and os.path.exists(video_path):
            skipped_count += 1
            continue
            
        # Add to queue: (video_id, video_url, video_dir, resolution, suppress_errors, use_aria2c)
        download_tasks.append((
            video_id, 
            row.video_url, 
            video_dir, 
            args.resolution, 
            args.suppress_errors, 
            use_aria2c
        ))

    # 4. Execute Downloads
    downloaded_count = 0
    failed_count = 0
    failed_videos = {} # video_id -> (url, error)

    print("\n🚀 Starting downloads")
    print(f"   - Workers: {args.workers}")
    print(f"   - Aria2c:  {'Enabled' if use_aria2c else 'Disabled'}")
    print(f"   - Output:  {video_dir}")
    print(f"   - Total:   {total_videos}")
    print(f"   - Queued:  {len(download_tasks)} (Skipped {skipped_count})\n")

    if download_tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            futures = [executor.submit(download_single_video, task) for task in download_tasks]
            
            # Process as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading"):
                success, vid_id, vid_url, error_msg = future.result()
                
                if success:
                    downloaded_count += 1
                else:
                    failed_count += 1
                    failed_videos[vid_id] = (vid_url, error_msg)

    print("\n" + "="*60)
    print("📊 DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Target Resolution:  {args.resolution}p")
    print(f"Total in Split:     {total_videos}")
    print("-" * 30)
    print(f"✅ Downloaded:       {downloaded_count}")
    print(f"⏩ Skipped:          {skipped_count}")
    print(f"❌ Failed:           {failed_count}")
    print("-" * 30)
    
    if failed_videos:
        print("\n❌ Failed Videos:")
        for vid_id, (url, error) in failed_videos.items():
            print(f"  • {vid_id}: {url}")
            if not args.suppress_errors:
                # Indent error message
                err_lines = error.strip().split('\n')
                # Take last few lines of error for brevity
                short_err = '\n    '.join(err_lines[-3:]) 
                print(f"    Error: ...\n    {short_err}")
    print("="*60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
