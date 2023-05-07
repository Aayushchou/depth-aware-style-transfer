import cv2
import os
import glob


def images_to_video(image_folder, output_video, fps=30, sorted_function=None):
    # Get the list of image file paths
    image_files = glob.glob(os.path.join(image_folder, "*"))

    # Sort the image files using the optional sorting function
    if sorted_function:
        image_files = sorted(image_files, key=sorted_function)

    # Read the first image to get the dimensions
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape

    # Create a VideoWriter object
    video_writer = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    # Write the frames to the video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()


def sorting_function(file):
    # This example sorts files by their numeric value in the filename assuming they are named like 'frame0001.png'
    return int(os.path.basename(file).split(".")[0][5:])


def get_frame_rate(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video file.")
        return None

    fps = int(video.get(cv2.CAP_PROP_FPS))
    video.release()
    return fps
