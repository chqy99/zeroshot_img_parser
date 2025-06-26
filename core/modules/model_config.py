import os
import yaml
import asyncio


class LazyModel:
    def __init__(self, loader_func):
        self._loader_func = loader_func
        self._model = None

    def _load_model(self):
        if self._model is None:
            print("[LazyModel] 正在同步加载模型...")
            self._model = self._loader_func()
            print("[LazyModel] 同步加载完成。")
        return self._model

    def __getattr__(self, attr):
        return getattr(self._load_model(), attr)

    def __getitem__(self, key):
        return self._load_model()[key]


class AsyncLazyModel:
    def __init__(self, loader_func):
        self._loader_func = loader_func
        self._model = None
        self._lock = asyncio.Lock()

    async def _load_model_async(self):
        async with self._lock:
            if self._model is None:
                print("[AsyncLazyModel] 正在异步加载模型...")
                self._model = await asyncio.to_thread(self._loader_func)
                print("[AsyncLazyModel] 异步加载完成。")
        return self._model

    def __getattr__(self, attr):
        raise RuntimeError("请使用 await model.get() 获取模型后再调用属性")

    def __getitem__(self, item):
        raise RuntimeError("请使用 await model.get() 获取模型后再调用属性")

    async def get(self):
        return await self._load_model_async()


class ModelLoader:
    _loaders = {}

    @classmethod
    def register_loader(cls, name):
        """注册模型加载函数的装饰器"""
        def wrapper(fn):
            cls._loaders[name] = fn
            return fn
        return wrapper

    def __init__(self, config_path=None, device="cuda"):
        config_path = config_path or os.path.join(
            os.path.dirname(__file__), "../../configs/model_config.yaml"
        )
        with open(config_path, "r") as f:
            self.models_cfg = yaml.safe_load(f)

        self.device = device
        self.models = {}

    def _get_loader_func(self, name):
        if name not in self.models_cfg:
            raise ValueError(f"模型 '{name}' 未在配置中定义")
        if name not in self._loaders:
            raise ValueError(f"模型 '{name}' 未注册加载函数")
        return lambda: self._loaders[name](self.models_cfg[name], self.device)

    def get_model(self, name):
        if name in self.models:
            return self.models[name]

        cfg = self.models_cfg[name]
        preload = cfg.get("preload", False)
        loader_func = self._get_loader_func(name)

        if preload:
            print(f"[ModelLoader] 同步预加载模型 '{name}'")
            model = loader_func()
            self.models[name] = model
        else:
            print(f"[ModelLoader] 同步懒加载代理模型 '{name}'")
            model = LazyModel(loader_func)
            self.models[name] = model

        return model

    async def async_get_model(self, name):
        if name in self.models:
            return self.models[name]

        cfg = self.models_cfg[name]
        preload = cfg.get("preload", False)
        loader_func = self._get_loader_func(name)

        if preload:
            print(f"[ModelLoader] 异步预加载模型 '{name}'")
            model = await asyncio.to_thread(loader_func)
            self.models[name] = model
        else:
            print(f"[ModelLoader] 异步懒加载代理模型 '{name}'")
            model = AsyncLazyModel(loader_func)
            self.models[name] = model

        return model
