import tensorflow as tf
import PIL.Image
import numpy as np


def tensor_to_image(tensor: tf.Tensor) -> PIL.Image.Image:
    """
    Converts a TensorFlow tensor to a PIL.Image object.

    Args:
        tensor (tf.Tensor): The input TensorFlow tensor.

    Returns:
        PIL.Image.Image: The resulting PIL.Image object.
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img: str, max_dim=None, apply_max_dim=True) -> tf.Tensor:
    """
    Loads an image from a file and preprocesses it as a TensorFlow tensor.

    Args:
        path_to_img (str): The file path of the image.

    Returns:
        tf.Tensor: The preprocessed image as a TensorFlow tensor.
    """
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if apply_max_dim:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        scale = max_dim / max(shape)
        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)

    img = img[tf.newaxis, :]
    return img
