__all__ = ["ELF3TrackingEnv"]


def __getattr__(name):
    if name == "ELF3TrackingEnv":
        from .elf3_env import ELF3TrackingEnv
        return ELF3TrackingEnv
    raise AttributeError(name)
