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

    @staticmethod
    def compute_iou(box1: "BBox", box2: "BBox") -> float:
        """
        计算两个 BBox 之间的 IoU（交并比）
        """
        x_left = max(box1.x1, box2.x1)
        y_top = max(box1.y1, box2.y1)
        x_right = min(box1.x2, box2.x2)
        y_bottom = min(box1.y2, box2.y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = box1.area()
        box2_area = box2.area()
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area


# 初始化时必须提供的参数: image, source_module, score, bbox
# 模型增强只提供 label、text、metadata 这些内容理解，不能改变 bbox,mask 这些位置信息
# 因为有模型增强，所以 source_module, score 这些要是列表类型
# 解析类，无序列化操作
@dataclass
class ImageParseItem:
    image: np.ndarray
    source_module: List[str]
    score: List[float]
    bbox: BBox
    type: Literal["ocr", "icon", "instance", "region"] = "region"
    mask: Optional[np.ndarray] = None
    label: Optional[str] = None
    text: Optional[str] = None
    bbox_image: np.ndarray = None
    mask_image: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.source_module, list):
            self.source_module = [self.source_module]
        if not isinstance(self.score, list):
            self.score = [self.score]

    def enrich(self, source_module: str, score: float, **kwargs):
        self.source_module.append(source_module)
        self.score.append(score)
        for key, value in kwargs.items():
            if key in self.__dataclass_fields__:
                setattr(self, key, value)
            else:
                self.metadata[key] = value

    def get_bbox_image(self) -> np.ndarray:
        if self.bbox_image is None:
            self.bbox_image = self.bbox.crop(self.image)
        return self.bbox_image

    def get_mask_image(self) -> Optional[np.ndarray]:
        if self.mask_image is None and self.mask is not None:
            # 裁剪图片和 mask 到 bbox 范围
            cropped_img = self.bbox.crop(self.image)
            # 裁剪 mask（假设 mask 是与 image 同尺寸的二值掩码）
            x1, y1, x2, y2 = map(int, [self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2])
            cropped_mask = self.mask[y1:y2, x1:x2].astype(np.uint8)
            # 对裁剪后的图片应用掩码
            self.mask_image = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)
        return self.mask_image

    def to_dict(self, filter: List[str] = []) -> Dict[str, Any]:
        result = {
            "source_module": self.source_module,
            "score": self.score,
            "type": self.type,
            "label": self.label,
            "text": self.text,
            "bbox": (
                self.bbox.to_dict() if hasattr(self.bbox, "to_dict") else str(self.bbox)
            ),
            "metadata": self.metadata,
        }
        if "image" in filter:
            result["image"] = np_to_base64(self.image)
        if "mask" in filter and self.mask is not None:
            result["mask"] = np_to_base64(self.mask * 255)
        if "bbox_image" in filter:
            bbox_img = self.get_bbox_image()
            result["bbox_image"] = np_to_base64(bbox_img)
        if "mask_image" in filter and self.get_mask_image() is not None:
            result["mask_image"] = np_to_base64(self.get_mask_image())
        return result


# 各个子项bbox的覆盖区域当作一个大的mask，再取原图像的mask_image，记为 bboxs_image
# 各个子项mask的覆盖区域当作一个大的mask，再取原图像的mask_image，记为 masks_image
# 解析类无序列化操作
@dataclass
class ImageParseResult:
    image: np.ndarray
    items: List[ImageParseItem] = field(default_factory=list)
    bboxs_image: Optional[np.ndarray] = None
    masks: Optional[np.ndarray] = None
    masks_image: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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


class IDGenerator:
    def __init__(
        self,
        prefix: str = "item",
        use_date: bool = True,
        use_uuid: bool = True,
        counter: bool = False,
    ):
        self.prefix = prefix
        self.use_date = use_date
        self.use_uuid = use_uuid
        self.counter = counter
        self._counter_value = 0

    def next_id(self) -> str:
        parts = [self.prefix]

        if self.use_date:
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            parts.append(date_str)

        if self.counter:
            parts.append(f"{self._counter_value:04d}")
            self._counter_value += 1

        if self.use_uuid:
            parts.append(uuid.uuid4().hex[:8])

        return "_".join(parts)


default_item_id_gen = IDGenerator(prefix="item")
default_result_id_gen = IDGenerator(prefix="result")


# 解析与存储分离
# 存储相关，包括: id, embedding, bbox_image_path, mask_path, mask_image_path
# 整体图片放在 ImageParseResultStorage 中
# 子项只存储 "bbox_image", "mask", "mask_image"
@dataclass
class ImageParseItemStorage(ImageParseItem):
    id: str = None
    embedding: Optional[np.ndarray] = None
    bbox_image_path: Optional[str] = None
    mask_path: Optional[str] = None
    mask_image_path: Optional[str] = None
    saved: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.id is None:
            from .image_data import default_item_id_gen

            self.id = default_item_id_gen.next_id()

    @classmethod
    def from_parse_item(
        cls, item: ImageParseItem, id: Optional[str] = None
    ) -> "ImageParseItemStorage":
        return cls(
            id=id or item.metadata.get("id"),
            image=item.image,
            source_module=item.source_module,
            score=item.score,
            bbox=item.bbox,
            type=item.type,
            mask=item.mask,
            label=item.label,
            text=item.text,
            bbox_image=item.bbox_image,
            mask_image=item.mask_image,
            metadata=item.metadata.copy(),
        )

    def get_embedding(
        self, embedding_handler: Optional[EmbeddingHandler] = None
    ) -> np.ndarray:
        if self.embedding is None:
            self.embedding = (
                embedding_handler or EmbeddingHandler.get_default()
            ).get_embedding(self.get_bbox_image())
        return self.embedding

    def save_image(
        self,
        file_handler: Optional[FileHandler] = None,
        storage_filter: List[str] = ["bbox_image", "mask", "mask_image"],
    ):
        fh = file_handler or FileHandler.get_default()

        if self.bbox_image_path is None and "bbox_image" in storage_filter:
            self.bbox_image_path = fh.save_image(
                self.get_bbox_image(), f"{self.id}_bbox.png"
            )
        if (
            self.mask_path is None
            and "mask" in storage_filter
            and self.mask is not None
        ):
            self.mask_path = fh.save_image(self.mask, f"{self.id}_mask.png")
        if self.mask_image_path is None and "mask_image" in storage_filter:
            mask_img = self.get_mask_image()
            if mask_img is not None:
                self.mask_image_path = fh.save_image(
                    mask_img, f"{self.id}_mask_image.png"
                )

        self.saved = True

    def to_dict(self) -> dict:
        if not self.saved:
            raise RuntimeError("图像未保存，请先调用 save_image()")

        data = {
            "id": self.id,
            "source_module": self.source_module,
            "score": self.score,
            "type": self.type,
            "label": self.label,
            "text": self.text,
            "metadata": self.metadata,
            "bbox_image_path": self.bbox_image_path,
            "mask_path": self.mask_path,
            "mask_image_path": self.mask_image_path,
            "bbox": self.bbox.to_dict(),
            "embedding": (
                self.embedding.tolist() if self.embedding is not None else None
            ),
        }

        return data

    @classmethod
    def from_dict(
        cls,
        data: dict,
        file_handler: Optional[FileHandler] = None,
        image: Optional[np.ndarray] = None,
    ) -> "ImageParseItemStorage":
        fh = file_handler or FileHandler.get_default()

        bbox_image = (
            fh.load_image(data.get("bbox_image_path"))
            if data.get("bbox_image_path")
            else None
        )
        mask = fh.load_image(data.get("mask_path")) if data.get("mask_path") else None
        mask_image = (
            fh.load_image(data.get("mask_image_path"))
            if data.get("mask_image_path")
            else None
        )
        embedding = (
            np.array(data["embedding"]) if data.get("embedding") is not None else None
        )

        return cls(
            id=data["id"],
            image=image,
            source_module=data.get("source_module", []),
            score=data.get("score", []),
            bbox=BBox.from_dict(data["bbox"]),
            type=data.get("type", "region"),
            label=data.get("label"),
            text=data.get("text"),
            metadata=data.get("metadata", {}),
            bbox_image=bbox_image,
            mask=mask,
            mask_image=mask_image,
            embedding=embedding,
            saved=True,
        )


@dataclass
class ImageParseResultStorage(ImageParseResult):
    id: str = None
    image_path: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    bboxs_image_path: Optional[str] = None
    masks_path: Optional[str] = None
    masks_image_path: Optional[str] = None
    saved: bool = False
    items: List[ImageParseItemStorage] = field(default_factory=list)

    def __post_init__(self):
        if self.id is None:
            self.id = default_result_id_gen.next_id()

    @classmethod
    def from_parse_result(
        cls, result: ImageParseResult, id: Optional[str] = None
    ) -> "ImageParseResultStorage":
        return cls(
            id=id or default_result_id_gen.next_id(),
            image=result.image,
            items=[
                ImageParseItemStorage.from_parse_item(item) for item in result.items
            ],
            metadata=result.metadata.copy(),
        )

    def get_embedding(self, embedding_handler: Optional[EmbeddingHandler] = None):
        if self.embedding is None:
            self.embedding = (
                embedding_handler or EmbeddingHandler.get_default()
            ).get_embedding(self.image)
        return self.embedding

    def save_image(
        self,
        file_handler: Optional[FileHandler] = None,
        storage_filter: List[str] = ["image", "bboxs_image", "masks", "masks_image"],
    ):
        fh = file_handler or FileHandler.get_default()

        if self.image_path is None and "image" in storage_filter:
            self.image_path = fh.save_image(self.image, f"{self.id}_image.png")

        if self.bboxs_image_path is None and "bboxs_image" in storage_filter:
            bboxs_img = self.get_bboxs_image()
            if bboxs_img is not None:
                self.bboxs_image_path = fh.save_image(
                    bboxs_img, f"{self.id}_bboxs_image.png"
                )

        if (
            self.masks_path is None
            and "masks" in storage_filter
            and self.masks is not None
        ):
            self.masks_path = fh.save_image(self.masks, f"{self.id}_masks.png")

        if self.masks_image_path is None and "masks_image" in storage_filter:
            masks_img = self.get_masks_image()
            if masks_img is not None:
                self.masks_image_path = fh.save_image(
                    masks_img, f"{self.id}_masks_image.png"
                )

        self.saved = True

    def to_dict(self) -> dict:
        if not self.saved:
            raise RuntimeError("图像未保存，请先调用 save_image()")

        return {
            "id": self.id,
            "items": [item.to_dict() for item in self.items],
            "metadata": self.metadata,
            "embedding": (
                self.embedding.tolist() if self.embedding is not None else None
            ),
            "image_path": self.image_path,
            "bboxs_image_path": self.bboxs_image_path,
            "masks_path": self.masks_path,
            "masks_image_path": self.masks_image_path,
        }

    @classmethod
    def from_dict(
        cls, data: dict, file_handler: Optional[FileHandler] = None
    ) -> "ImageParseResultStorage":
        fh = file_handler or FileHandler.get_default()
        image = (
            fh.load_image(data.get("image_path")) if data.get("image_path") else None
        )
        embedding = np.array(data["embedding"]) if data.get("embedding") else None

        items = [
            ImageParseItemStorage.from_dict(item, file_handler=fh, image=image)
            for item in data.get("items", [])
        ]

        return cls(
            id=data.get("id"),
            image=image,
            items=items,
            metadata=data.get("metadata"),
            embedding=embedding,
            image_path=data.get("image_path"),
            bboxs_image_path=data.get("bboxs_image_path"),
            masks_path=data.get("masks_path"),
            masks_image_path=data.get("masks_image_path"),
            saved=True,
        )


import json


class ImageParseStorageHelper:

    @staticmethod
    def convert_to_storage(
        result: ImageParseResult,
        id: Optional[str] = None,
    ) -> ImageParseResultStorage:
        """从 ImageParseResult 转换为带有 ID 的 ImageParseResultStorage"""
        return ImageParseResultStorage.from_parse_result(result, id=id)

    @staticmethod
    def save_all_images(
        storage_result: ImageParseResultStorage,
        file_handler: Optional[FileHandler] = None,
        item_filter: list = ["bbox_image", "mask", "mask_image"],
        result_filter: list = ["image", "bboxs_image", "masks", "masks_image"],
    ):
        """保存主图和每个子项的图像"""
        file_handler = file_handler or FileHandler.get_default()

        # 保存主图和整体图像
        storage_result.save_image(
            file_handler=file_handler, storage_filter=result_filter
        )

        # 保存每个子项
        for item in storage_result.items:
            item.save_image(file_handler=file_handler, storage_filter=item_filter)

    @staticmethod
    def compute_embeddings(
        storage_result: ImageParseResultStorage,
        embedding_handler: Optional[EmbeddingHandler] = None,
    ):
        """为主图和所有子项生成 embedding"""
        embedding_handler = embedding_handler or EmbeddingHandler.get_default()

        # 主图 embedding
        storage_result.get_embedding(embedding_handler)

        # 每个子项 embedding
        for item in storage_result.items:
            item.get_embedding(embedding_handler)

    @staticmethod
    def to_json_dict(storage_result: ImageParseResultStorage) -> dict:
        """转换为可以序列化为 JSON 的 dict"""
        return storage_result.to_dict()

    @staticmethod
    def save_to_json(
        storage_result: ImageParseResultStorage,
        output_path: str,
    ):
        """将 storage_result 保存为 JSON 文件"""
        data = ImageParseStorageHelper.to_json_dict(storage_result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def process_full_pipeline(
        cls,
        result: ImageParseResult,
        output_json_path: str,
        id: Optional[str] = None,
    ):
        """完整流程：解析类 → 存储类 → 保存图像 → 计算 embedding → 导出 JSON"""
        storage_result = cls.convert_to_storage(result, id=id)
        cls.save_all_images(storage_result)
        cls.compute_embeddings(storage_result)
        cls.save_to_json(storage_result, output_json_path)
        return storage_result
