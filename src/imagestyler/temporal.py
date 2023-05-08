import os
import cv2
import numpy as np


def warp_image_with_optical_flow(img1, img2, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    img2_warped = cv2.remap(img2, flow, None, cv2.INTER_LINEAR)
    return img2_warped


def estimate_optical_flow(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        img1_gray, img2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow


def apply_temporal_consistency(prev_stylized, prev_frame, current_frame):
    flow = estimate_optical_flow(prev_frame, current_frame)
    warped_stylized = warp_image_with_optical_flow(prev_stylized, current_frame, flow)
    return warped_stylized


def blend_consecutive_frames(frames, smoothness):
    blended_frame = np.zeros_like(frames[0], dtype=np.float32)
    frame_count = len(frames)
    weights = np.array(
        [smoothness ** (frame_count - i - 1) for i in range(frame_count)]
    )
    normalized_weights = weights / weights.sum()

    for frame, weight in zip(frames, normalized_weights):
        blended_frame += frame * weight

    return blended_frame.astype(np.uint8)


def smooth_stylized_frames(
    content_frames_path,
    stylized_frames_path,
    output_dir,
    n_consecutive_frames,
    smoothness,
):
    os.makedirs(output_dir, exist_ok=True)

    content_frame_files = sorted(os.listdir(content_frames_path))
    stylized_frame_files = sorted(os.listdir(stylized_frames_path))

    prev_frame = None
    prev_stylized = None
    prev_stylized_frames = []

    for frame_number, (content_frame_file, stylized_frame_file) in enumerate(
        zip(content_frame_files, stylized_frame_files)
    ):
        content_frame_path = os.path.join(content_frames_path, content_frame_file)
        stylized_frame_path = os.path.join(stylized_frames_path, stylized_frame_file)

        current_frame = cv2.cvtColor(cv2.imread(content_frame_path), cv2.COLOR_BGR2RGB)
        stylized_frame = cv2.cvtColor(
            cv2.imread(stylized_frame_path), cv2.COLOR_BGR2RGB
        )

        if prev_frame is not None and prev_stylized is not None:
            warped_stylized = apply_temporal_consistency(
                prev_stylized, prev_frame, current_frame
            )
        else:
            warped_stylized = None

        if warped_stylized is not None:
            warped_stylized = cv2.resize(
                warped_stylized,
                (stylized_frame.shape[1], stylized_frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            stylized_frame = cv2.addWeighted(
                warped_stylized, 0.2, stylized_frame, 0.8, 0
            )

        prev_stylized_frames.append(stylized_frame)
        if len(prev_stylized_frames) > n_consecutive_frames:
            prev_stylized_frames.pop(0)

        if len(prev_stylized_frames) > 1:
            blended_stylized_frame = blend_consecutive_frames(
                prev_stylized_frames, smoothness
            )
        else:
            blended_stylized_frame = stylized_frame

        output_frame_path = os.path.join(output_dir, f"frame{frame_number:04d}.png")
        cv2.imwrite(
            output_frame_path, cv2.cvtColor(blended_stylized_frame, cv2.COLOR_RGB2BGR)
        )

        prev_frame = current_frame
        prev_stylized = stylized_frame


if __name__ == "__main__":
    # Example usage
    content_frames_path = "data/input/imgs"
    stylized_frames_path = "data/output/test/processed_frames"
    output_dir = "data/output/test/smoothed_frames"
    n_consecutive_frames = 1
    smoothness = 0.3
    smooth_stylized_frames(
        content_frames_path,
        stylized_frames_path,
        output_dir,
        n_consecutive_frames,
        smoothness,
    )
