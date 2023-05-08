import cv2
import os


def split_video_to_frames(video_path, output_dir, max_dim):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video is opened successfully
    if not video.isOpened():
        print("Error: Could not open the video file.")
        return

    def resize_frame(frame, max_dim):
        height, width, _ = frame.shape
        scale_factor = max_dim / max(height, width)
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        return cv2.resize(frame, (512, 264))

    frame_number = 0
    while True:
        # Read the next frame from the video
        ret, frame = video.read()

        # Break the loop if we've reached the end of the video
        if not ret:
            break

        # Resize the frame
        resized_frame = resize_frame(frame, max_dim)

        # Save the current frame as an image
        frame_filename = os.path.join(output_dir, f"frame{frame_number:04d}.png")
        cv2.imwrite(frame_filename, resized_frame)
        frame_number += 1

    # Release the video object and close all windows
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    split_video_to_frames(
        "data/input/video/1am_trimmed.mp4", output_dir="data/input/imgs", max_dim=512
    )
