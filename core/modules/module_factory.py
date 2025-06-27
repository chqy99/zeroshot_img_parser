# core/modules/module_factory.py

class ModuleFactory:
    _registry = {}

    @classmethod
    def register_module(cls, name):
        def decorator(fn):
            cls._registry[name] = fn
            return fn
        return decorator

    @classmethod
    def get_module(cls, name):
        if name not in cls._registry:
            raise ValueError(f"Module '{name}' not registered.")
        return cls._registry[name]()
