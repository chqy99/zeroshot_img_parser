from abc import ABC, abstractmethod
from typing import Optional, TypeVar, ClassVar
import numpy as np

T = TypeVar("T", bound="FileHandler")


class FileHandler(ABC):
    _default: ClassVar[Optional["FileHandler"]] = None

    @abstractmethod
    def save_image(self, image: np.ndarray, filename: str, **kwargs) -> str:
        pass

    @abstractmethod
    def load_image(self, path: str, **kwargs) -> np.ndarray:
        pass

    @classmethod
    def init_default(cls: type[T], handler: T):
        cls._default = handler

    @classmethod
    def get_default(cls: type[T]) -> T:
        if cls._default is None:
            raise RuntimeError("FileHandler 默认实例未初始化，请调用 init_default")
        return cls._default


T = TypeVar("T", bound="EmbeddingHandler")


class EmbeddingHandler(ABC):
    _default: ClassVar[Optional["EmbeddingHandler"]] = None

    @abstractmethod
    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def init_default(cls: type[T], handler: T):
        cls._default = handler

    @classmethod
    def get_default(cls: type[T]) -> T:
        if cls._default is None:
            raise RuntimeError("EmbeddingHandler 默认实例未初始化，请调用 init_default")
        return cls._default
