import io
import base64
from PIL import Image
import numpy as np
import image_tools

class MultiFormatImage:
    """
    A unified image container that can be:
      - Initialized from PIL.Image, numpy.ndarray, or base64 string
      - Loaded from local file via `load(path)`
      - Converted to one of three formats via `get(format)` with caching
      - Saved to a local file via `save(path)`
      - One-step converted between formats via `convert(data, input_fmt, output_fmt)`

    Supported modes: 'RGB', 'L' (grayscale)
    Supported formats for `get()` and `convert()`: 'numpy', 'pil', 'base64'
    """
    def __init__(self, data, mode='RGB', fmt='PNG'):
        """
        Initialize from supported input types and store canonical numpy array in given mode.

        Args:
            data: PIL.Image.Image, numpy.ndarray, or base64-encoded string
            mode: 'RGB' for color or 'L' for grayscale
            fmt: Format for base64 encoding and PIL save (default 'PNG')
        """
        self.mode = mode
        self.fmt = fmt

        # Load into PIL and convert to desired mode
        if isinstance(data, Image.Image):
            img = data.convert(self.mode)
            arr = np.array(img)
        elif isinstance(data, np.ndarray):
            arr = data
        elif isinstance(data, str):  # base64
            decoded = base64.b64decode(data)
            buf = io.BytesIO(decoded)
            img = Image.open(buf).convert(self.mode)
            arr = np.array(img)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Initialize cache with numpy as canonical format
        self._data = {'numpy': arr}
        self.H, self.W = arr.shape[:2]
        self.C = 1 if arr.ndim == 2 else arr.shape[2]

    @classmethod
    def load(cls, path, mode='RGB', fmt='PNG'):
        """
        Load an image from a local file path and return MultiFormatImage.
        """
        img = Image.open(path)
        return cls(img, mode=mode, fmt=fmt)

    def get(self, fmt='numpy'):
        """
        Retrieve image in desired format, using cache if available;
        otherwise convert from numpy and cache.

        Args:
            fmt: Target format: 'numpy', 'pil', or 'base64'
        Returns:
            Image in requested format.
        """
        fmt = fmt.lower()
        if fmt in self._data:
            return self._data[fmt]

        base = self._data['numpy']
        if fmt == 'pil':
            result = Image.fromarray(base, mode=self.mode)
        elif fmt == 'base64':
            img = Image.fromarray(base, mode=self.mode)
            buf = io.BytesIO()
            img.save(buf, format=self.fmt)
            result = base64.b64encode(buf.getvalue()).decode('utf-8')
        elif fmt == 'numpy':
            result = base
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

        self._data[fmt] = result
        return result

    def save(self, path):
        """
        Save the image to a local file path using the specified format.
        """
        img = self.get('pil')
        img.save(path, format=self.fmt)

    def shape(self):
        """Return (H, W, C)"""
        return (self.H, self.W, self.C)

    def __repr__(self):
        return f"<MultiFormatImage shape=({self.H}, {self.W}, {self.C}) mode={self.mode}>"

import ast
import uuid
from typing import Optional, Dict

class ImageAnnotation:
    def __init__(self, image, id: Optional[str] = None, metadata: Dict[str, any] = {}):
        self.image_format = MultiFormatImage(image)
        self.id = id if id is not None else str(uuid.uuid4())
        self.meta_data = metadata
        self.meta_data["height"] = self.image_format.H
        self.meta_data["width"] = self.image_format.W
        self.meta_data["channel"] = self.image_format.C
        self.meta_data["describe"] = ""
        self.labels = np.zeros((self.image_format.H, self.image_format.W), dtype=np.uint8)
        self.classes_num = 0
        if "classes_num" in metadata:
            self.classes_num = metadata["classes_num"]
        self.bboxs = []
        if "bboxs" in metadata:
            self.bboxs = ast.literal_eval(metadata["bboxs"])
        self.classes_describe = []
        if "classes_describe" in metadata:
            self.classes_describe = ast.literal_eval(metadata["classes_describe"])
        self.filepath = ""
        if "filepath" in metadata:
            self.filepath = metadata["filepath"]

    def add_one_class(self, mask: np.ndarray, describe: str):
        self.classes_num += 1
        self.labels[mask > 0] = self.classes_num
        self.bboxs.append(image_tools.MaskHandler.mask_to_bbox(mask))
        self.classes_describe.append(describe)

    def visualize_annotation(self):
        visual_mask = image_tools.ImageVisualizer.visualize_masks(self.classes_num, self.labels)
        return visual_mask, self.bboxs

    def collect_meta_data(self):
        meta_data = self.meta_data
        meta_data["classes_num"] = self.classes_num
        meta_data["bboxs"] = str(self.bboxs)
        meta_data["classes_describe"] = str(self.classes_describe)
        self.meta_data["filepath"] = self.filepath

    def get_chromadb_item(self):
        # 如果文件不存在，按当前时间保存
        if self.filepath == "":
            from datetime import datetime
            now = datetime.now()
            time_str = now.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
            self.save(time_str)

        self.collect_meta_data()
        return self.id, self.meta_data, self.image_format.get()

    def save(self, filename):
        from xc_config import _xc_image_dir
        filepath = _xc_image_dir + "/" + filename + ".png"
        # TODO: label 文件是否保存
        self.image_format.save(filepath)
        self.filepath = filepath
