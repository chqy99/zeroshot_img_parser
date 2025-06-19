import os
import torch
import yaml

from paddleocr import PaddleOCR
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotImageClassification,
)

from sam2.build_sam import build_sam2  # 你的sam2加载接口


class LazyModel:
    """
    惰性加载代理，访问时才真正加载模型。
    """
    def __init__(self, loader_func, device):
        self._loader_func = loader_func
        self._device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            print(f"[LazyModel] 正在加载模型...")
            self._model = self._loader_func()
            # 设备迁移在外层统一处理，这里不转设备
            if hasattr(self._model, 'to'):
                self._model = self._model.to(self._device)
            print(f"[LazyModel] 模型加载完成。")
        return self._model

    def __getattr__(self, attr):
        model = self._load_model()
        return getattr(model, attr)


class ModelLoader:
    def __init__(self, config_path=None, device: str = None):
        config_path = config_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../configs/model_config.yaml"
        )
        with open(config_path, 'r') as f:
            self.models_cfg = yaml.safe_load(f)

        self.device = device or self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}

    def get_model(self, name):
        if name in self.models:
            return self.models[name]

        if name not in self.models_cfg:
            raise ValueError(f"模型 '{name}' 未在配置文件中定义")

        model_cfg = self.models_cfg[name]
        preload = model_cfg.get("preload", False)

        loader_name = f"_load_{name}"
        if not hasattr(self, loader_name):
            raise ValueError(f"模型 '{name}' 未定义加载方法")

        loader_func = lambda: getattr(self, loader_name)(model_cfg)

        if preload:
            print(f"[ModelLoader] 预加载模型 '{name}' 到设备 {self.device}")
            model = loader_func()
            if hasattr(model, 'to'):
                model = model.to(self.device)
            self.models[name] = model
            return model
        else:
            print(f"[ModelLoader] 返回惰性加载代理模型 '{name}'")
            lazy_model = LazyModel(loader_func, self.device)
            self.models[name] = lazy_model
            return lazy_model

    # 各模型加载函数，仅负责加载，不转设备
    def _load_paddleocr(self, cfg):
        return PaddleOCR(lang="ch", use_angle_cls=True,
                         det_model_dir=cfg.get("det"),
                         rec_model_dir=cfg.get("rec"),
                         cls_model_dir=cfg.get("cls"))  # use_gpu由device统一控制

    def _load_sam2(self, cfg):
        model_cfg = cfg.get("model_cfg")
        checkpoint = cfg.get("checkpoint")
        if model_cfg is None or checkpoint is None:
            raise ValueError("sam2模型加载需要 model_cfg 和 checkpoint 路径")
        model = build_sam2(model_cfg, checkpoint)
        return model

    def _load_clip(self, cfg):
        processor_path = cfg.get("processor")
        model_path = cfg.get("model")
        if processor_path is None or model_path is None:
            raise ValueError("clip模型加载需要 processor 和 model 路径")

        processor = AutoProcessor.from_pretrained(processor_path)
        model = AutoModelForZeroShotImageClassification.from_pretrained(model_path)
        return {"processor": processor, "model": model}
