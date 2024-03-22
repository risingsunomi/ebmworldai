import cv2
import srt
import argparse
import sys
from captures import Captures

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('build_video_dataset')

def update_progress(frame_count: int, total_frames: int):
    # Calculate the progress percentage
    progress = (frame_count / total_frames) * 100
    
    # Update the progress bar in the terminal
    sys.stdout.write(f"\rProgress: [{int(progress)}%] [{frame_count} / {total_frames}] [{'=' * int(progress // 2)}{' ' * (50 - int(progress // 2))} frames]")
    sys.stdout.flush()

def save_frame_to_db(db_captures, frame, subtitle):
    try:
        db_captures.insert_pframe_tensor(frame, subtitle)
    except Exception as err:
        logger.error(f"Insert of pframe failed\n{err}")
        raise

def main(video_path: str, srt_path: str=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Open the SRT file
    if srt_path:
        with open(srt_path, 'r') as f:
            subtitles = list(srt.parse(f.read()))
    else:
        subtitles = None

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # create a instance of the capture model
    db_captures = Captures()

    # Initialize variables for progress tracking
    frame_count = 0
    cap_frame_count = 0

    if subtitles:
        subtitle_index = 0

    # Iterate through the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        update_progress(frame_count, total_frames)
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
        subtitle = ""

        if subtitles:
            # Check if the current frame is within the subtitle timestamps
            if subtitle_index < len(subtitles) and current_time >= subtitles[subtitle_index].start.total_seconds():
                subtitle = subtitles[subtitle_index].content
                if current_time >= subtitles[subtitle_index].end.total_seconds():
                    subtitle_index += 1

        save_frame_to_db(
            db_captures,
            frame,
            subtitle
        )

        cap_frame_count += 1

        cv2.imshow("Captured Frame", frame)

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()

    print(f"Frame capturing complete. Captured {cap_frame_count} frames with text")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="build_video_dataset",
        description="video and closed caption dataset builder"
    )
    
    arg_parser.add_argument(
        "--video", 
        type=str,
        help="path to video to capture"
    )

    arg_parser.add_argument(
        "--srt", 
        type=str, 
        help="path to .srt file for video"
    )
    
    args = arg_parser.parse_args()

    if args.video:
        print(f"loading video {args.video}")
        main(args.video, args.srt)
    else:
        print("Please specify the video path")