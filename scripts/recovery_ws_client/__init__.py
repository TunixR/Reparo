from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import ClientConfig
    from .runner import RecoveryWsRunner

__all__ = ["ClientConfig", "RecoveryWsRunner"]


def __getattr__(name: str) -> Any:
    # Lazy so submodules such as `keys` import without pyautogui, which needs a
    # display at import time (headless CI).
    if name == "ClientConfig":
        from .config import ClientConfig

        return ClientConfig
    if name == "RecoveryWsRunner":
        from .runner import RecoveryWsRunner

        return RecoveryWsRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
