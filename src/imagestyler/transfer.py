import cv2
from typing import Tuple
import numpy as np
import PIL.Image
import tensorflow as tf
import tensorflow_hub as hub

import torch

from imagestyler.utils import load_img, tensor_to_image


class StyleTransfer:
    """
    A class to generate stylized images using image style transfer.

    Attributes:
        content_img (List[tf.Tensor]): A list of content images as TensorFlow tensors.
        style_img (List[tf.Tensor]): A list of style images as TensorFlow tensors.
    """

    def __init__(self, content_img: str, style_img: str, max_dim: int):
        """
        Initializes the StyleTransfer class with content and style images.

        Args:
            content_path (str): Path to the directory containing content images.
            style_path (str): Path to the directory containing style images.
            max_dim (int): Maximum dimension for resizing images.
        """
        self.model = hub.load(
            "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
        )

        self.content_img = load_img(content_img, max_dim=max_dim)
        self.style_img = self.load_img(style_img, apply_max_dim=False)

    def __call__(self) -> PIL.Image.Image:
        """
        Generates stylized images by applying style transfer to content images.

        Returns:
            List[PIL.Image.Image]: A list of stylized PIL.Image objects.
        """

        stylized_image = self.model(
            tf.constant(self.content_img), tf.constant(self.style_img)
        )[0]
        stylized_image = tensor_to_image(stylized_image[0])
        return stylized_image


class DepthAwareStyleTransfer:
    def __init__(
        self,
        max_dim: int,
        model_type: str = "DPT_Hybrid",
        alpha: float = 0.8,
        blur_kernel_size: Tuple[int, int] = (15, 15),
        blur_sigma: int = 3,
    ) -> None:
        """
        Initialize the DepthAwareStyleTransfer class.

        Args:
            content_img (str): Path to the content image.
            style_img (str): Path to the first style image.
            style_img_2 (str): Path to the second style image.
            max_dim (int): Maximum dimension of the output image.
            model_type (str, optional): Type of depth model. Default is "DPT_Hybrid".
            alpha (float, optional): Blending factor for depth. Default is 0.99.
            blur_kernel_size (Tuple[int, int], optional): Kernel size for Gaussian blur. Default is (15, 15).
            blur_sigma (int, optional): Sigma value for Gaussian blur. Default is 5.
        """
        self.alpha = alpha
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.model = hub.load(
            "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
        )
        self.depth_model = torch.hub.load("intel-isl/MiDaS", model_type)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.depth_model.to(device)
        self.depth_model.eval()
        self.max_dim = max_dim

    def __call__(self, content_img, style_img, style_img_2) -> PIL.Image.Image:
        """
        Perform the depth-aware style transfer.

        Returns:
            PIL.Image.Image: The depth-aware style transferred image.
        """
        content_img = load_img(content_img, max_dim=self.max_dim)
        style_img = load_img(style_img, max_dim=self.max_dim)
        style_img_2 = load_img(style_img_2, apply_max_dim=False)
        content_pil = tensor_to_image(content_img)

        stylized_image = self.model(tf.constant(content_img), tf.constant(style_img))[0]
        stylized_image = tensor_to_image(stylized_image[0])
        stylized_image = stylized_image.resize(content_pil.size, PIL.Image.BILINEAR)

        stylized_image2 = self.model(
            tf.constant(content_img), tf.constant(style_img_2)
        )[0]
        stylized_image2 = tensor_to_image(stylized_image2[0])
        stylized_image2 = stylized_image2.resize(content_pil.size, PIL.Image.BILINEAR)

        content_depth = self.estimate_depth(content_pil)

        depth_aware_image = self.blend_images_based_on_depth(
            stylized_image, stylized_image2, content_depth
        )
        return depth_aware_image

    def estimate_depth(self, img: PIL.Image.Image) -> PIL.Image.Image:
        """
        Estimate the depth map of an input image.

        Args:
            img (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The depth map as a PIL image object.
        """
        img_pil = img.resize((384, 384))
        img_np = np.array(img_pil) / 255.0
        img_torch = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0)

        with torch.no_grad():
            depth = self.depth_model(img_torch)
            depth = depth.squeeze().cpu().numpy()

        # Normalize the depth array to the range [0, 255]
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth_pil = PIL.Image.fromarray(np.uint8(depth_normalized))

        return depth_pil

    def blend_images_based_on_depth(
        self,
        content_img: PIL.Image.Image,
        stylized_img: PIL.Image.Image,
        content_depth: PIL.Image.Image,
    ) -> PIL.Image.Image:
        """
        Blend two images based on the depth map of the content image.

        Args:
            content_img (PIL.Image.Image): The content image.
            stylized_img (PIL.Image.Image): The stylized image.
            content_depth (PIL.Image.Image): The depth map of the content image.

        Returns:
            PIL.Image.Image: The blended image.
        """
        content_np = np.array(content_img).astype(np.float32) / 255.0
        stylized_np = np.array(stylized_img).astype(np.float32) / 255.0

        content_depth_resized_t = np.array(
            content_depth.resize(content_img.size, resample=PIL.Image.BILINEAR)
        )

        content_depth_resized = content_depth_resized_t / 255.0

        stylized_blurred_np = cv2.GaussianBlur(
            stylized_np, self.blur_kernel_size, self.blur_sigma
        )
        blended_np = np.zeros_like(content_np)

        for c in range(3):
            blended_np[..., c] = content_np[..., c] * (
                self.alpha * content_depth_resized
            ) + stylized_blurred_np[..., c] * (1 - self.alpha * content_depth_resized)

        blended_pil = PIL.Image.fromarray(np.uint8(blended_np * 255))

        return blended_pil
