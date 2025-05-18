from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

@dataclass
class ImageInstance:
    """
    Stores information and analysis results for a standalone image or region.
    No position information is stored here.
    """
    image: np.ndarray
    height: int = field(init=False)
    width: int = field(init=False)
    channel: int = field(init=False)
    class_name: Optional[str] = None
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
    bbox_list: List[list] = field(default_factory=list)

    @property
    def instance_num(self) -> int:
        return len(self.instances)

    def add_instance(self, inst: ImageInstance, bbox: list) -> None:
        """
        Add a detected sub-instance and its relative bounding box.
        """
        self.instances.append(inst)
        self.bbox_list.append(bbox)
