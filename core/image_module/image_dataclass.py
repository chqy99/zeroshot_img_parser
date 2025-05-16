from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
import uuid

@dataclass
class BoundingBox:
    """
    Represents a rectangular region relative to the original image.
    Coordinates: (x_min, y_min, x_max, y_max).
    """
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def width(self) -> int:
        return self.x_max - self.x_min

    def height(self) -> int:
        return self.y_max - self.y_min


@dataclass
class ImageInstance:
    """
    Stores information and analysis results for a standalone image or region.
    No position information is stored here.
    """
    image: np.ndarray
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    height: int = field(init=False)
    width: int = field(init=False)
    channel: int = field(init=False)
    key: str = ""
    description: Optional[str] = None
    mask: Optional[np.ndarray] = None
    mask_image: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.image.ndim == 2:
            h, w = self.image.shape
            c = 1
        else:
            h, w, c = self.image.shape
        self.height = h
        self.width = w
        self.channel = c

    def set_mask(self, mask: np.ndarray) -> None:
        """
        Assign a binary mask and generate a masked image view.
        """
        self.mask = mask
        if mask.dtype != bool:
            mask = mask.astype(bool)
        self.mask_image = np.where(mask[..., None], self.image, 0)

@dataclass
class ImageAnalysisResult:
    """
    Aggregates analysis of an original image and its sub-instances with relative positions.
    """
    original: ImageInstance
    instances: List[ImageInstance] = field(default_factory=list)
    bbox_list: List[BoundingBox] = field(default_factory=list)

    @property
    def instance_num(self) -> int:
        return len(self.instances)

    def add_instance(self, inst: ImageInstance, bbox: BoundingBox) -> None:
        """
        Add a detected sub-instance and its relative bounding box.
        """
        self.instances.append(inst)
        self.bbox_list.append(bbox)

    def summary(self) -> str:
        """
        Return a concise summary of the analysis, including relative positions.
        """
        lines = [
            f"Original: {self.original.id}, size={self.original.width}x{self.original.height}",
        ]
        for idx, (inst, bbox) in enumerate(zip(self.instances, self.bbox_list), 1):
            key = inst.key or "<[Unnamed]>"
            coords = f"({bbox.x_min},{bbox.y_min},{bbox.x_max},{bbox.y_max})"
            lines.append(
                f"#{idx}: id={inst.id}, key={key}, bbox={coords}"
            )
        return "\n".join(lines)
