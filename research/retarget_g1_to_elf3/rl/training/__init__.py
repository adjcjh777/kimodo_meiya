from .gpu_selector import select_gpus

__all__ = ["train", "select_gpus"]


def __getattr__(name):
    if name == "train":
        from .train import train
        return train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
