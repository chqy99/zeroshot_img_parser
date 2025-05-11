import io
import base64
import numpy as np
from PIL import Image

class ImageConverter:
    """
    Simplified static methods for converting between PIL, base64, and numpy formats,
    and saving images to disk.

    Methods:
    - pil_to_numpy(img, mode='RGB') -> np.ndarray
    - base64_to_numpy(b64, mode='RGB') -> np.ndarray
    - numpy_to_pil(arr, mode='RGB') -> PIL.Image
    - numpy_to_base64(arr, mode='RGB', fmt='PNG') -> str
    - save_numpy(arr, path, mode='RGB', fmt='PNG') -> None
    - save_pil(img, path, fmt='PNG') -> None
    - save_base64(b64, path, mode='RGB', fmt='PNG') -> None
    """

    @staticmethod
    def pil_to_numpy(img: Image.Image, mode: str = 'RGB') -> np.ndarray:
        """
        Convert a PIL Image to a numpy array in the specified mode.
        """
        return np.asarray(img.convert(mode))

    @staticmethod
    def base64_to_numpy(b64: str, mode: str = 'RGB') -> np.ndarray:
        """
        Decode a base64 string and convert to numpy array.
        """
        data = base64.b64decode(b64)
        buf = io.BytesIO(data)
        img = Image.open(buf).convert(mode)
        return np.asarray(img)

    @staticmethod
    def numpy_to_pil(arr: np.ndarray, mode: str = 'RGB') -> Image.Image:
        """
        Convert a numpy array to a PIL Image in the specified mode.
        """
        return Image.fromarray(arr.astype('uint8'), mode)

    @staticmethod
    def numpy_to_base64(arr: np.ndarray, mode: str = 'RGB', fmt: str = 'PNG') -> str:
        """
        Convert a numpy array to a base64-encoded string.
        """
        img = Image.fromarray(arr.astype('uint8'), mode)
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    @staticmethod
    def save_numpy(arr: np.ndarray, path: str, mode: str = 'RGB', fmt: str = 'PNG') -> None:
        """
        Save a numpy array as an image file.
        """
        img = ImageConverter.numpy_to_pil(arr, mode)
        img.save(path, format=fmt)

    @staticmethod
    def save_pil(img: Image.Image, path: str, fmt: str = 'PNG') -> None:
        """
        Save a PIL Image to disk.
        """
        img.save(path, format=fmt)

    @staticmethod
    def save_base64(b64: str, path: str, mode: str = 'RGB', fmt: str = 'PNG') -> None:
        """
        Decode a base64 string and save as an image file.
        """
        arr = ImageConverter.base64_to_numpy(b64, mode)
        ImageConverter.save_numpy(arr, path, mode, fmt)
