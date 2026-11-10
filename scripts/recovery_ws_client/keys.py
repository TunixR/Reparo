"""Key name normalization for keyboard actions.

pyautogui silently ignores key names it does not know, so ``hotkey('ctrl minus')``
presses only ``ctrl`` and still looks successful. Agents use many spellings for the
same key, so names are mapped to pyautogui's and anything left unknown is rejected.

Kept free of pyautogui imports so it can be unit-tested headless.
"""

from __future__ import annotations

from collections.abc import Callable

KEY_ALIASES: dict[str, str] = {
    # Symbols spelled out
    "minus": "-",
    "dash": "-",
    "hyphen": "-",
    "plus": "+",
    "equal": "=",
    "equals": "=",
    "comma": ",",
    "period": ".",
    "dot": ".",
    "slash": "/",
    "backslash": "\\",
    "semicolon": ";",
    "quote": "'",
    "apostrophe": "'",
    "backtick": "`",
    "grave": "`",
    "tilde": "~",
    # Modifiers
    "control": "ctrl",
    "ctl": "ctrl",
    "lctrl": "ctrlleft",
    "rctrl": "ctrlright",
    "lshift": "shiftleft",
    "rshift": "shiftright",
    "lalt": "altleft",
    "ralt": "altright",
    "altgr": "altright",
    "option": "alt",
    "opt": "alt",
    "cmd": "win",
    "command": "win",
    "meta": "win",
    "windows": "win",
    "super": "win",
    # Navigation and editing
    "arrowleft": "left",
    "arrowright": "right",
    "arrowup": "up",
    "arrowdown": "down",
    "leftarrow": "left",
    "rightarrow": "right",
    "uparrow": "up",
    "downarrow": "down",
    "page_up": "pageup",
    "page_down": "pagedown",
    "pgup": "pageup",
    "pgdown": "pagedown",
    "spacebar": "space",
    "print_screen": "printscreen",
    "prtsc": "printscreen",
    "caps_lock": "capslock",
    "num_lock": "numlock",
    "scroll_lock": "scrolllock",
    "ins": "insert",
    "escape": "esc",
    "bksp": "backspace",
}


def split_hotkey(hotkey: str) -> list[str]:
    """Split a hotkey string on spaces, ``+`` or ``-`` joining two key names.

    ``"ctrl -"``, ``"ctrl+-"``, ``"ctrl+shift+s"`` and ``"ctrl + shift + s"`` all
    work; a ``+`` or ``-`` that is itself the key (``"ctrl +"``, ``"ctrl+-"``) is kept.
    """
    parts: list[str] = []
    chunks = hotkey.split()
    for position, chunk in enumerate(chunks):
        if chunk == "+" and 0 < position < len(chunks) - 1:
            continue  # "ctrl + shift + s": a spaced "+" between keys joins them
        if len(chunk) == 1:
            parts.append(chunk)
            continue
        current = ""
        for index, char in enumerate(chunk):
            is_separator = (
                char in "+-"
                and current != ""
                and index + 1 < len(chunk)
                and current[-1] not in "+-"
            )
            if is_separator:
                parts.append(current)
                current = ""
            else:
                current += char
        if current:
            parts.append(current)
    return parts


def normalize_key(key: str) -> str:
    """Map an agent's key name to pyautogui's name (lowercase, aliases resolved)."""
    if len(key) == 1:
        return key
    lowered = key.lower()
    return KEY_ALIASES.get(lowered, lowered)


def resolve_keys(raw: str | list[str], is_valid: Callable[[str], bool]) -> list[str]:
    """Normalize keys and raise ``ValueError`` naming any key pyautogui would ignore."""
    keys = split_hotkey(raw) if isinstance(raw, str) else list(raw)
    if not keys:
        raise ValueError("requires one or more keys")
    normalized = [normalize_key(k) for k in keys]
    invalid = [
        original
        for original, key in zip(keys, normalized, strict=True)
        if not is_valid(key)
    ]
    if invalid:
        raise ValueError(
            f"unknown key name(s): {', '.join(repr(k) for k in invalid)}. "
            "Use pyautogui names, e.g. ctrl, shift, alt, win, enter, esc, tab, space, "
            "backspace, delete, up, down, left, right, pageup, pagedown, home, end, "
            "f1-f12, or the character itself such as '-', '+', 'a'."
        )
    return normalized
