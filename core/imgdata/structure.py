import base64
import cv2
from dataclasses import dataclass, field
from typing import Optional, Callable, Literal, List, Dict, Any
import numpy as np
from .storage_handler import FileHandler, EmbeddingHandler
import uuid
import datetime
from io import BytesIO
from PIL import Image


def np_to_base64(img: np.ndarray, format: str = "PNG") -> str:
    """将 numpy 图像转为 base64 字符串"""
    pil_img = Image.fromarray(img.astype("uint8"))
    buffer = BytesIO()
    pil_img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@dataclass
class BBox:
    x1: int | float
    y1: int | float
    x2: int | float
    y2: int | float

    def to_list(self) -> List:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @staticmethod
    def from_dict(data: dict) -> "BBox":
        return BBox(**data)

    def crop(self, image: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = map(int, [self.x1, self.y1, self.x2, self.y2])
        return image[y1:y2, x1:x2]

    def crop_float(self, image: np.ndarray) -> np.ndarray:
        center = ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
        size = (abs(self.x2 - self.x1), abs(self.y2 - self.y1))
        return cv2.getRectSubPix(
            image, patchSize=(int(size[0]), int(size[1])), center=center
        )

    def area(self) -> int | float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @classmethod
    def mask_to_bbox(cls, mask: np.ndarray) -> "BBox | None":
        rows = np.where(np.any(mask, axis=1))[0]
        cols = np.where(np.any(mask, axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return None
        return cls(
            x1=float(cols[0]), y1=float(rows[0]), x2=float(cols[-1]), y2=float(rows[-1])
        )


@dataclass
class ImageParseUnit:
    bbox: BBox  # 必要的位置信息
    source_module: str  # 解析的来源

    score: Optional[float] = None  # 可选：置信度
    text: Optional[str] = None  # 可选：描述（英文）
    mask: Optional[np.ndarray] = None  # 可选：精细区域
    selected_label: Optional[str] = None  # 可选：来自 label list 的匹配项
    metadata: Dict[str, Any] = field(default_factory=dict)  # 拓展字段

    # 延迟绑定 image
    image: Optional[np.ndarray] = field(default=None)
    bbox_image: Optional[np.ndarray] = field(default=None)
    bbox_image_embedding: Optional[np.ndarray] = field(default=None)
    mask_image: Optional[np.ndarray] = field(default=None)
    mask_image_embedding: Optional[np.ndarray] = field(default=None)
    storage_dict: Dict[str, Any] = field(default_factory=dict)

    def get_bbox_image(self) -> np.ndarray:
        if self.bbox_image is None:
           self.bbox_image = self.bbox.crop(self.image)
        return self.bbox_image

    def get_mask_image(self) -> Optional[np.ndarray]:
        if self.mask is None:
            x1, y1, x2, y2 = map(int, (self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2))
            cropped_img = self.image[y1:y2, x1:x2]
            cropped_mask = self.mask[y1:y2, x1:x2].astype(np.uint8)
            self.mask_image = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)
        return self.mask_image

    def get_bbox_image_embedding(self, handler: Optional[EmbeddingHandler] = None) -> np.ndarray:
        if self.bbox_image_embedding is None:
            handler = handler or EmbeddingHandler.get_default()
            self.bbox_image_embedding = handler.get_embedding(self.get_bbox_image())
        return self.bbox_image_embedding

    def get_mask_image_embedding(self, handler: Optional[EmbeddingHandler] = None) -> Optional[np.ndarray]:
        if self.mask is None:
            return None
        if self.mask_image_embedding is None:
            handler = handler or EmbeddingHandler.get_default()
            mask_img = self.get_mask_image()
            if mask_img is not None:
                self.mask_image_embedding = handler.get_embedding(mask_img)
        return self.mask_image_embedding

    def save_bbox_image(self, handler: Optional[FileHandler] = None, path: Optional[str] = None) -> str:
        handler = handler or FileHandler.get_default()
        if "bbox_image_path" not in self.storage_dict:
            img = self.get_bbox_image()
            path = path or f"{self.source_module}_bbox.png"
            saved_path = handler.save_image(img, path)
            self.storage_dict["bbox_image_path"] = saved_path
        return self.storage_dict["bbox_image_path"]

    def save_mask_image(self, handler: Optional[FileHandler] = None, path: Optional[str] = None) -> Optional[str]:
        if self.mask is None:
            return None
        handler = handler or FileHandler.get_default()
        if "mask_image_path" not in self.storage_dict:
            mask_img = self.get_mask_image()
            if mask_img is not None:
                path = path or f"{self.source_module}_mask.png"
                saved_path = handler.save_image(mask_img, path)
                self.storage_dict["mask_image_path"] = saved_path
        return self.storage_dict.get("mask_image_path")

    def to_dict(self, filter_base64: List[str] = []) -> dict:
        result = {
            "bbox": self.bbox.to_dict(),
            "source_module": self.source_module,
            "score": self.score,
            "text": self.text,
            "selected_label": self.selected_label,
            "metadata": self.metadata,
            "storage_dict": self.storage_dict,
        }

        if "bbox_image" in filter_base64 and self.get_bbox_image() is not None:
            result["bbox_image"] = np_to_base64(self.get_bbox_image())

        if "mask_image" in filter_base64 and self.get_mask_image() is not None:
            result["mask_image"] = np_to_base64(self.get_mask_image())

        return result

    # def from_dict(self):
    #     pass

    # def to_storage(self):
    #     pass

    # def from_storage(self):
    #     pass


class ImageParseItem:
    multi_parse: List[ImageParseUnit] = field(default_factory=list)

    def to_dict(self, filter_base64: List[str] = []) -> dict:
        return {
            "multi_parse": [unit.to_dict(filter_base64=filter_base64) for unit in self.multi_parse]
        }

@dataclass
class ImageParseResult:
    image: np.ndarray
    items: List[ImageParseUnit | ImageParseItem] = field(default_factory=list)
    masks: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    bboxs_image: Optional[np.ndarray] = None
    bbox_image_embedding: Optional[np.ndarray] = field(default=None)
    mask_image: Optional[np.ndarray] = field(default=None)
    mask_image_embedding: Optional[np.ndarray] = field(default=None)
    storage_dict: Optional[Dict[str, Any]] = field(default_factory=None)


    def get_bboxs_image(self) -> np.ndarray:
        if self.bboxs_image is None:
            h, w = self.image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for item in self.items:
                x1, y1, x2, y2 = map(
                    int, (item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2)
                )
                mask[y1:y2, x1:x2] = 1
            self.bboxs_image = self.image * (mask[..., None] > 0)
        return self.bboxs_image

    def get_masks_image(self) -> Optional[np.ndarray]:
        if self.masks_image is None:
            if not any(item.mask is not None for item in self.items):
                return None
            h, w = self.image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for item in self.items:
                if item.mask is not None:
                    mask |= item.mask.astype(np.uint8) > 0
            self.masks = mask
            self.masks_image = self.image * (mask[..., None] > 0)
        return self.masks_image

    def to_dict(self, filter: List[str] = []) -> Dict[str, Any]:
        result = {
            "metadata": self.metadata,
            "items": [item.to_dict(filter=filter) for item in self.items],
        }
        if "image" in filter:
            result["image"] = np_to_base64(self.image)
        if "bboxs_image" in filter:
            result["bboxs_image"] = np_to_base64(self.get_bboxs_image())
        if "masks_image" in filter and self.get_masks_image() is not None:
            result["masks_image"] = np_to_base64(self.get_masks_image())
        return result
