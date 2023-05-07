import os
import cv2
import glob
import logging
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from imagestyler.transfer import DepthAwareStyleTransfer
from imagestyler.split import split_video_to_frames
from imagestyler.join import get_frame_rate, images_to_video, sorting_function


def main(input_video_path, style_image_1, style_image_2, output_path, max_dim=512):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    tqdm.write = logger.info

    def process_image(creator, content_img, style_img, style_img_2):
        art = creator(os.path.join(content_path, content_img), style_img, style_img_2)
        output_path = os.path.join(output_dir, content_img)
        art.save(output_path)
        tqdm.write(f"Saved {content_img} to {output_path}")

    tqdm.write("Starting the process...")

    # Step 1: Split video into frames
    tqdm.write("Split video into frames...")
    frame_output_dir = os.path.join(output_path, "frames")
    split_video_to_frames(input_video_path, frame_output_dir, max_dim=max_dim)

    # Step 2: Process images using DepthAwareStyleTransfer
    tqdm.write("Starting style transfer of frames")

    creator = DepthAwareStyleTransfer(max_dim=128)
    content_path = frame_output_dir
    output_dir = os.path.join(output_path, "processed_frames")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    content_images = sorted(os.listdir(content_path))

    # Set the number of worker threads according to your requirements.
    # You can use os.cpu_count() to get the number of CPU cores in your system.
    num_workers = os.cpu_count()

    # Using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for content_img in tqdm(content_images, desc="Processing images"):
            executor.submit(
                process_image, creator, content_img, style_image_1, style_image_2
            )

    # Step 3: Convert processed images to video
    tqdm.write("Converting frames into video.")
    processed_image_folder = output_dir
    output_video = os.path.join(output_path, "output_video.mp4")
    frame_rate = get_frame_rate(input_video_path)
    images_to_video(
        processed_image_folder,
        output_video,
        fps=frame_rate,
        sorted_function=sorting_function,
    )

    tqdm.write(
        f"Process completed successfully. Video can be found here: {output_video}"
    )


def orchestrate():
    parser = argparse.ArgumentParser(
        description="Depth-aware style transfer for videos."
    )
    parser.add_argument("-i", "--input_video", help="Path to the input video file.")
    parser.add_argument(
        "-sf", "--style_img1", help="Path to the foreground style image."
    )
    parser.add_argument(
        "-sb", "--style_img2", help="Path to the background style image."
    )
    parser.add_argument("-o", "--output_path", help="Path to the output directory.")
    parser.add_argument("-md", "--max_dim", help="Maximum size of the output images")

    args = parser.parse_args()
    main(
        args.input_video,
        args.style_img1,
        args.style_img2,
        args.output_path,
        args.max_dim,
    )
