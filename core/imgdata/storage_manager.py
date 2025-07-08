import os
import json
import numpy as np
from abc import ABC, abstractmethod

# ------------------------ 装饰器 ------------------------

def storage_field(save: bool = True):
    """用于标记字段是否需要参与存储。"""
    def wrapper(f):
        f._storage_field = True
        f._storage_field_save = save
        return f
    return wrapper


# ------------------------ Storage Handler 抽象类 ------------------------

class StorageHandler(ABC):
    @abstractmethod
    def save(self, value, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def can_handle(self, value):
        pass

    @abstractmethod
    def can_handle_by_name(self, type_name: str):
        pass


# ------------------------ Numpy Handler 示例 ------------------------

class NumpyHandler(StorageHandler):
    def save(self, value, path):
        np.save(path, value)

    def load(self, path):
        return np.load(path, allow_pickle=True)

    def can_handle(self, value):
        return isinstance(value, np.ndarray)

    def can_handle_by_name(self, type_name: str):
        return type_name == 'ndarray'


# ------------------------ Storage Config ------------------------

class StorageConfig:
    def __init__(self):
        self.ignore_fields = set()
        self.default_save_types = {np.ndarray}


# ------------------------ Storage Manager ------------------------

class StorageManager:
    def __init__(self, base_dir, config=None):
        self.base_dir = base_dir
        self.config = config or StorageConfig()
        self.registry = {}  # class -> include/exclude field names
        self.handlers = []

    def register_handler(self, handler: StorageHandler):
        self.handlers.append(handler)

    def register_storage_fields(self, cls, include=None, exclude=None):
        self.registry[cls] = {
            'include': set(include or []),
            'exclude': set(exclude or [])
        }

    def _get_fields_to_store(self, obj):
        cls = obj.__class__
        reg = self.registry.get(cls, {'include': set(), 'exclude': set()})
        include, exclude = reg['include'], reg['exclude']
        fields = []

        for attr in dir(obj):
            if attr.startswith('_') or attr in exclude:
                continue
            if include and attr not in include:
                continue

            val = getattr(obj, attr, None)
            if callable(val):
                continue

            # 优先处理装饰器标注
            prop = getattr(cls, attr, None)
            if hasattr(prop, '_storage_field'):
                if prop._storage_field_save:
                    fields.append(attr)
                continue

            # 类型匹配自动检测
            if any(isinstance(val, t) for t in self.config.default_save_types):
                fields.append(attr)

        return list(set(fields))  # 去重

    def save(self, obj, obj_id):
        cls_name = obj.__class__.__name__
        save_dir = os.path.join(self.base_dir, cls_name, obj_id)
        os.makedirs(save_dir, exist_ok=True)

        if not hasattr(obj, 'storage_dict'):
            obj.storage_dict = {}

        metadata = {}
        for field in self._get_fields_to_store(obj):
            val = getattr(obj, field)
            handler = next((h for h in self.handlers if h.can_handle(val)), None)
            if not handler:
                continue

            filename = f"{field}.dat" if not isinstance(val, np.ndarray) else f"{field}.npy"
            path = os.path.join(save_dir, filename)
            handler.save(val, path)
            obj.storage_dict[field] = path
            metadata[field] = {
                "path": path,
                "type": type(val).__name__,
            }

        # 保存 metadata
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, cls, obj_id):
        cls_name = cls.__name__
        save_dir = os.path.join(self.base_dir, cls_name, obj_id)
        meta_path = os.path.join(save_dir, "metadata.json")

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        obj = cls.__new__(cls)
        obj.storage_dict = {}

        for field, meta in metadata.items():
            path = meta["path"]
            type_name = meta["type"]
            handler = next((h for h in self.handlers if h.can_handle_by_name(type_name)), None)
            if handler:
                val = handler.load(path)
                setattr(obj, field, val)
                obj.storage_dict[field] = path

        return obj
