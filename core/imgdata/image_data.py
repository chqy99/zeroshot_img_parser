import base64
import cv2
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, Literal
import numpy as np
import os
from io import BytesIO
from PIL import Image
import datetime
import uuid

# ----------------------------- ID Generator -----------------------------


class IDGenerator:
    _instance = None

    def __init__(
        self,
        prefix: str = "img",
        use_date: bool = True,
        use_uuid: bool = True,
        counter: bool = False,
    ):
        self.prefix = prefix
        self.use_date = use_date
        self.use_uuid = use_uuid
        self.counter = counter
        self._counter_value = 0

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def next_id(self, suffix: Optional[str] = None) -> str:
        parts = [self.prefix]

        if self.use_date:
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            parts.append(date_str)

        if self.counter:
            parts.append(f"{self._counter_value:04d}")
            self._counter_value += 1

        if self.use_uuid:
            parts.append(uuid.uuid4().hex[:8])

        if suffix:
            parts.append(suffix)

        return "_".join(parts)


def np_to_base64(img: np.ndarray, format: str = "PNG") -> str:
    """Convert a NumPy image to a base64 string."""
    pil_img = Image.fromarray(img.astype("uint8"))
    buffer = BytesIO()
    pil_img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_np(b64_str: str) -> np.ndarray:
    """Convert a base64 string back to a NumPy image."""
    buffer = BytesIO(base64.b64decode(b64_str))
    pil_img = Image.open(buffer).convert("RGB")
    return np.array(pil_img)


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
    """
    Represents a single semantic unit parsed from an image by a specific module.

    This unit contains positional, textual, visual, and optional mask-level information
    about a region within an image. It may also store cached images and structured storage info.

    Attributes:
        bbox (BBox): The bounding box specifying the location of the parsed region.
        source_module (str): The name of the module or algorithm that produced this unit.

        mask (Optional[np.ndarray]): Optional binary mask (same shape as image) indicating precise region shape.
        image (Optional[np.ndarray]): Original full image (in-memory, not stored unless needed).
        bbox_image (Optional[np.ndarray]): Cropped image corresponding to the bounding box.
        mask_image (Optional[np.ndarray]): Cropped masked image from the region defined by bbox + mask.

        type (Optional[str]): Optional type tag (e.g., "button", "icon", "textline").
        text (Optional[str]): Free-form description or caption, often from OCR or CLIP.
        label (Optional[str]): A selected label from a predefined label set (e.g., matching output).
        score (Optional[float]): Confidence score provided by the parsing model, if available.

        metadata (Dict[str, Any]): Additional non-core metadata such as timestamps, tags, flags, or source info.
        storage_dict (Dict[str, Any]): Reserved for system-level storage and indexing, containing keys such as:
            - 'uid': Unique identifier assigned to this unit (used across storage/index systems).
            - 'image_path': Local file path of the original image.
            - 'bbox_image_path': Local path to the cropped bbox image.
            - 'mask_image_path': Local path to the masked bbox image.
            - 'vector_id': ID of this unit in the vector database (e.g., Faiss, Milvus).
    """

    bbox: BBox
    source_module: str

    mask: Optional[np.ndarray] = None
    image: Optional[np.ndarray] = field(default=None)
    bbox_image: Optional[np.ndarray] = field(default=None)
    mask_image: Optional[np.ndarray] = field(default=None)

    type: Optional[str] = None
    text: Optional[str] = None
    label: Optional[str] = None
    score: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)
    storage_dict: Dict[str, Any] = field(default_factory=dict)
    uid: Optional[str] = None

    def get_uid(self) -> str:
        if not self.uid:
            self.uid = IDGenerator.instance().next_id("unit")
        return self.uid

    def get_bbox_image(self) -> np.ndarray:
        """
        Returns the cropped region of the image defined by the bounding box.
        If not already cached, it computes and stores the result.
        """
        if self.bbox_image is None:
            self.bbox_image = self.bbox.crop(self.image)
        return self.bbox_image

    def get_mask_image(self) -> Optional[np.ndarray]:
        """
        Returns the image region within the bounding box with the mask applied.
        If not already cached, it computes and stores the result.
        Returns None if mask is not available.
        """
        if self.mask_image is None and self.mask is not None:
            x1, y1, x2, y2 = map(
                int, (self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2)
            )
            cropped_img = self.image[y1:y2, x1:x2]
            cropped_mask = self.mask[y1:y2, x1:x2].astype(np.uint8)
            self.mask_image = cv2.bitwise_and(
                cropped_img, cropped_img, mask=cropped_mask
            )
        return self.mask_image

    def to_dict(self, image_filter: Optional[list] = None) -> dict:
        """
        Serializes the object to a dictionary.
        NumPy image arrays are converted to base64-encoded strings for selected fields.
        image_filter: list of field names (e.g. ["image", "bbox_image", "mask_image", "mask"])
        """
        image_filter = image_filter or ["bbox_image", "mask_image", "mask"]
        d = {
            "bbox": self.bbox.to_dict(),
            "source_module": self.source_module,
            "score": self.score,
            "type": self.type,
            "text": self.text,
            "label": self.label,
            "metadata": self.metadata,
            "storage_dict": self.storage_dict,
            "uid": self.uid,
        }
        # Handle ndarray fields
        if "bbox_image" in image_filter:
            d["bbox_image"] = (
                np_to_base64(self.get_bbox_image())
                if self.get_bbox_image() is not None
                else None
            )
        if "mask_image" in image_filter:
            d["mask_image"] = (
                np_to_base64(self.get_mask_image())
                if self.get_mask_image() is not None
                else None
            )
        if "mask" in image_filter:
            d["mask"] = (
                np_to_base64(self.mask.astype(np.uint8))
                if self.mask is not None
                else None
            )
        if "image" in image_filter:
            d["image"] = np_to_base64(self.image) if self.image is not None else None
        return d

    @classmethod
    def from_dict(
        cls, data: dict, image_filter: Optional[list] = None
    ) -> "ImageParseUnit":
        """
        Deserializes an ImageParseUnit from a dictionary.
        image_filter: list of field names to decode from base64 (e.g. ["image", "bbox_image", "mask_image", "mask"])
        """
        image_filter = image_filter or ["bbox_image", "mask_image", "mask"]
        obj = cls(
            bbox=BBox.from_dict(data["bbox"]),
            source_module=data["source_module"],
            score=data.get("score"),
            type=data.get("type"),
            text=data.get("text"),
            label=data.get("label"),
            metadata=data.get("metadata", {}),
            storage_dict=data.get("storage_dict", {}),
            uid=data.get("uid"),
        )
        # Handle ndarray fields
        if "bbox_image" in image_filter and data.get("bbox_image"):
            obj.bbox_image = base64_to_np(data["bbox_image"])
        if "mask_image" in image_filter and data.get("mask_image"):
            obj.mask_image = base64_to_np(data["mask_image"])
        if "mask" in image_filter and data.get("mask"):
            obj.mask = base64_to_np(data["mask"])
        if "image" in image_filter and data.get("image"):
            obj.image = base64_to_np(data["image"])
        return obj

    def to_vector_record(self) -> dict:
        """
        Return a lightweight dict for vector DB storage (no image fields).
        """
        return self.to_dict([])

    def enrich_text(
        self,
        source_module: str,
        score: float,
        text: Optional[str],
        overwrite: bool = False,
    ):
        """
        Enriches the `text` field of the parsing unit.

        Parameters:
            source_module (str): Module name providing the text.
            score (float): Confidence score for the enrichment.
            text (Optional[str]): Text content to assign.
            overwrite (bool): If True, overwrite existing `text`.

        Side Effects:
            - Updates `self.text` if it's missing or `overwrite=True`.
            - Logs enrichment info into metadata.
        """
        if text is None:
            return
        if self.text is None or overwrite:
            self.text = text
            self.metadata["text_enriched_by"] = source_module
            self.metadata[source_module + "_text_score"] = score

    def enrich_label(
        self,
        source_module: str,
        score: float,
        label: Optional[str],
        overwrite: bool = False,
    ):
        """
        Enriches the `label` field of the parsing unit.

        Parameters:
            source_module (str): Module name providing the label.
            score (float): Confidence score for the enrichment.
            label (Optional[str]): Label content to assign.
            overwrite (bool): If True, overwrite existing `label`.

        Side Effects:
            - Updates `self.label` if it's missing or `overwrite=True`.
            - Logs enrichment info into metadata.
        """
        if label is None:
            return
        if self.label is None or overwrite:
            self.label = label
            self.metadata["label_enriched_by"] = source_module
            self.metadata[source_module + "_label_score"] = score

    def save_image(self, base_dir: str, image_filter: Optional[list] = None):
        """
        Save selected image fields to disk using PIL. Update storage_dict with file paths.
        image_filter: list of field names to save (e.g. ["mask", "mask_image", "bbox_image"])
        Default: ["bboxs_image"]
        """
        image_filter = image_filter or ["bbox_image"]
        os.makedirs(base_dir, exist_ok=True)
        if not self.uid:
            self.get_uid()
        for field in image_filter:
            arr = getattr(self, field, None)
            if arr is not None:
                file_path = os.path.join(base_dir, f"{self.uid}_{field}.png")
                img = Image.fromarray(arr.astype("uint8"))
                img.save(file_path)
                self.storage_dict[f"{field}_path"] = file_path

    def load_image(self, image_filter: Optional[list] = None):
        """
        Load selected image fields from disk using storage_dict and PIL.
        image_filter: list of field names to load (e.g. ["mask", "mask_image", "bbox_image"])
        Default: ["bboxs_image"]
        """
        image_filter = image_filter or ["bbox_image"]
        for field in image_filter:
            path = self.storage_dict.get(f"{field}_path")
            if path:
                arr = np.array(Image.open(path).convert("RGB"))
                setattr(self, field, arr)


# # ----------------------------- Group Type -----------------------------

# GroupType = Literal[
#     # ------ ① Spatial / Geometric Relations ------
#     "same_spatial",     # Occupying the same or nearly identical position.
#     "overlap",          # Regions that partially intersect.
#     "contain",          # One region entirely contains another.
#     "adjacent",         # Positioned side-by-side or in close proximity.
#     "align",            # Aligned along the same axis (horizontal or vertical).

#     # ------ ② Structural / Layout-Based Groupings ------
#     "sequence",         # Ordered items (e.g., rows, form fields, timeline).
#     "parent_child",     # Nested or hierarchical UI blocks (e.g., label + input).
#     "functional_block", # Belonging to the same logical UI module (e.g., login form).
#     "text_flow",        # Flow of text across lines or regions (e.g., paragraph).
#     "repetition",       # Visually repeated units (e.g., list items, cards).

#     # ------ ③ Semantic / Appearance-Based Groupings ------
#     "same_semantics",   # Conceptually similar roles (e.g., all are "buttons").
#     "grouped_by_label", # Same assigned label/class name.
#     "grouped_by_style", # Visual similarity in font, iconography, or shape.
#     "grouped_by_color", # Similar color scheme or background color.

#     # ------ ④ Fallback ------
#     "unknown"           # Type could not be inferred.
# ]


# @dataclass
# class ImageParseGroup:
#     """
#     Represents a semantic group composed of ImageParseUnits and/or nested ImageParseGroups.

#     This class enables higher-level semantic structure by grouping related visual regions,
#     either based on spatial relations (e.g. alignment), functional logic (e.g. layout grouping),
#     or inferred semantics (e.g. buttons with same label).

#     Attributes:
#         items (List[Union[ImageParseUnit, ImageParseGroup]]):
#             The atomic or nested members that form this group.

#         type (GroupType):
#             The logic or rule that defines how items are grouped.
#             See `GroupType` for allowed values.

#         group_text (Optional[str]):
#             A high-level semantic summary of the group.
#             Example: "Settings Panel", "Form Section", or "Action Buttons".

#         spatial_text (Optional[str]):
#             A spatially aware or layout-derived textual description.
#             Example: "Top-right aligned menu", "Left column entries".

#         metadata (Dict[str, Any]):
#             Custom annotations for reasoning or rules behind grouping.
#             May include flags, sources, parser notes, or user tags.

#         storage_dict (Dict[str, Any]):
#             A system-reserved dictionary used to store group-related identifiers,
#             such as:
#                 - 'uids': list of unit IDs belonging to the group.
#                 - 'group_vector_id': ID for the group embedding in vector DB.
#     """

#     items: List[Union["ImageParseUnit", "ImageParseGroup"]]
#     type: GroupType = "unknown"

#     group_text: Optional[str] = None
#     spatial_text: Optional[str] = None

#     metadata: Dict[str, Any] = field(default_factory=dict)
#     storage_dict: Dict[str, Any] = field(default_factory=dict)
#     uid: Optional[str] = None

#     def get_uid(self) -> str:
#         if not self.uid:
#             self.uid = _id_generator.next_id("group")
#         return self.uid


@dataclass
class ImageParseResult:
    """
    Represents the full result of image parsing, including atomic units, groups, and visual summaries.

    Attributes:
        image (np.ndarray): The original input image to be parsed.
        units (List[ImageParseUnit]): Flat list of atomic visual parsing units.
        groups (List[ImageParseGroup]): Optional list of higher-level grouped semantics.
        summary_text (Optional[str]): High-level summary of the scene (e.g., from caption model or layout parser).
        bboxs_image (Optional[np.ndarray]): Visualization: bounding boxes rendered over the original image.
        masks (Optional[np.ndarray]): Combined binary mask from all available unit masks.
        masks_image (Optional[np.ndarray]): Visualization: masked regions rendered on the original image.
        metadata (Dict[str, Any]): Optional metadata (e.g., parser source, timestamp, settings).
        storage_dict (Dict[str, Any]): Stores image/embedding UID or related persistent info.
    """

    image: np.ndarray

    units: List["ImageParseUnit"] = field(default_factory=list)
    # groups: List["ImageParseGroup"] = field(default_factory=list)

    summary_text: Optional[str] = None

    bboxs_image: Optional[np.ndarray] = None
    masks: Optional[np.ndarray] = None
    masks_image: Optional[np.ndarray] = None

    metadata: Dict[str, Any] = field(default_factory=dict)
    storage_dict: Dict[str, Any] = field(default_factory=dict)
    uid: Optional[str] = None

    def get_uid(self) -> str:
        if not self.uid:
            self.uid = IDGenerator.instance().next_id("result")
        return self.uid

    def get_bboxs_image(self) -> np.ndarray:
        """
        Returns an image with all bounding box regions highlighted.
        Lazily computed and cached.

        Returns:
            np.ndarray: The original image with regions (from bbox) visually extracted.
        """
        if self.bboxs_image is None:
            h, w = self.image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for item in self.units:
                x1, y1, x2, y2 = map(
                    int, (item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2)
                )
                mask[y1:y2, x1:x2] = 1
            self.bboxs_image = self.image * (mask[..., None] > 0)
        return self.bboxs_image

    def get_masks(self) -> Optional[np.ndarray]:
        """
        Returns a merged binary mask from all unit-level masks.
        Lazily computed and cached.

        Returns:
            Optional[np.ndarray]: A binary mask where any unit-level mask is active.
        """
        if self.masks is None:
            if not any(item.mask is not None for item in self.units):
                return None
            h, w = self.image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for item in self.units:
                if item.mask is not None:
                    mask |= item.mask.astype(np.uint8) > 0
            self.masks = mask
        return self.masks

    def get_masks_image(self) -> Optional[np.ndarray]:
        """
        Returns an image with only the masked regions from all units.
        Lazily computed and cached.

        Returns:
            Optional[np.ndarray]: Masked image showing only the semantic regions.
        """
        if self.masks_image is None:
            mask = self.get_masks()
            if mask is None:
                return None
            self.masks_image = self.image * (mask[..., None] > 0)
        return self.masks_image

    def to_dict(
        self,
        image_filter: Optional[list] = None,
        unit_image_filter: Optional[list] = None,
    ) -> dict:
        """
        Serialize the result, including units (with filter), and optionally image fields.
        """
        image_filter = image_filter or ["image"]
        d = {
            "units": [u.to_dict(image_filter=unit_image_filter) for u in self.units],
            "summary_text": self.summary_text,
            "metadata": self.metadata,
            "storage_dict": self.storage_dict,
            "uid": self.uid,
        }
        if "image" in image_filter and self.image is not None:
            d["image"] = np_to_base64(self.image)
        if "bboxs_image" in image_filter and self.bboxs_image is not None:
            d["bboxs_image"] = np_to_base64(self.bboxs_image)
        if "masks" in image_filter and self.masks is not None:
            d["masks"] = np_to_base64(self.masks)
        if "masks_image" in image_filter and self.masks_image is not None:
            d["masks_image"] = np_to_base64(self.masks_image)
        return d

    @classmethod
    def from_dict(
        cls,
        data: dict,
        image_filter: Optional[list] = None,
        unit_image_filter: Optional[list] = None,
    ) -> "ImageParseResult":
        """
        Deserialize from dict, including units and optionally image fields.
        """
        image_filter = image_filter or ["image"]
        image = (
            base64_to_np(data["image"])
            if "image" in image_filter and data.get("image")
            else None
        )
        obj = cls(
            image=image,
            units=[
                ImageParseUnit.from_dict(u, image_filter=unit_image_filter)
                for u in data.get("units", [])
            ],
            summary_text=data.get("summary_text"),
            metadata=data.get("metadata", {}),
            storage_dict=data.get("storage_dict", {}),
            uid=data.get("uid"),
        )
        if "bboxs_image" in image_filter and data.get("bboxs_image"):
            obj.bboxs_image = base64_to_np(data["bboxs_image"])
        if "masks" in image_filter and data.get("masks"):
            obj.masks = base64_to_np(data["masks"])
        if "masks_image" in image_filter and data.get("masks_image"):
            obj.masks_image = base64_to_np(data["masks_image"])
        return obj

    def to_vector_records(self) -> dict:
        """
        Return a dict for vector DB storage: result-level info + unit uid list.
        """
        return {
            "uid": self.get_uid(),
            "summary_text": self.summary_text,
            "metadata": self.metadata,
            "storage_dict": self.storage_dict,
            "unit_uids": [u.get_uid() for u in self.units],
        }

    def save_image(self, base_dir: str, image_filter: Optional[list] = None):
        """
        Save selected result-level images to disk using PIL. Update storage_dict with file paths.
        image_filter: list of field names to save (e.g. ["image", "bboxs_image", "masks", "masks_image"])
        Default: ["image", "bboxs_image"]
        """
        image_filter = image_filter or ["image", "bboxs_image"]
        os.makedirs(base_dir, exist_ok=True)
        if not self.uid:
            self.get_uid()
        for field in image_filter:
            arr = getattr(self, field, None)
            if arr is not None:
                file_path = os.path.join(base_dir, f"{self.uid}_{field}.png")
                img = Image.fromarray(arr.astype("uint8"))
                img.save(file_path)
                self.storage_dict[f"{field}_path"] = file_path

    def load_image(self, image_filter: Optional[list] = None):
        """
        Load selected result-level images from disk using storage_dict and PIL.
        image_filter: list of field names to load (e.g. ["image", "bboxs_image", "masks", "masks_image"])
        Default: ["image", "bboxs_image"]
        """
        image_filter = image_filter or ["image", "bboxs_image"]
        for field in image_filter:
            path = self.storage_dict.get(f"{field}_path")
            if path:
                arr = np.array(Image.open(path).convert("RGB"))
                setattr(self, field, arr)
