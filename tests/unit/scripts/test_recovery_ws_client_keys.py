import pytest

from scripts.recovery_ws_client.keys import normalize_key, resolve_keys, split_hotkey

# A slice of pyautogui's key names, so the tests run without a display.
VALID = {
    "ctrl",
    "shift",
    "alt",
    "win",
    "esc",
    "enter",
    "pagedown",
    "left",
    "s",
    "c",
    "-",
    "+",
    "=",
    " ",
    "space",
    "f5",
}


def is_valid(key: str) -> bool:
    return key in VALID


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ctrl -", ["ctrl", "-"]),
        ("ctrl +", ["ctrl", "+"]),
        ("ctrl+-", ["ctrl", "-"]),
        ("ctrl++", ["ctrl", "+"]),
        ("ctrl+shift+s", ["ctrl", "shift", "s"]),
        ("ctrl + shift + s", ["ctrl", "shift", "s"]),
        ("ctrl shift +", ["ctrl", "shift", "+"]),
        ("ctrl-c", ["ctrl", "c"]),
        ("  ctrl   c ", ["ctrl", "c"]),
        ("-", ["-"]),
    ],
)
def test_split_hotkey(raw: str, expected: list[str]) -> None:
    assert split_hotkey(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ctrl minus", ["ctrl", "-"]),
        ("Ctrl Plus", ["ctrl", "+"]),
        ("control shift s", ["ctrl", "shift", "s"]),
        ("cmd -", ["win", "-"]),
        ("CTRL+MINUS", ["ctrl", "-"]),
        ("escape", ["esc"]),
        ("page_down", ["pagedown"]),
        ("arrowleft", ["left"]),
        ("alt f5", ["alt", "f5"]),
    ],
)
def test_resolve_keys_normalizes_aliases(raw: str, expected: list[str]) -> None:
    assert resolve_keys(raw, is_valid) == expected


def test_resolve_keys_rejects_unknown_names_instead_of_skipping_them() -> None:
    with pytest.raises(ValueError, match="'zoomout'"):
        resolve_keys("ctrl zoomout", is_valid)


def test_resolve_keys_requires_a_key() -> None:
    with pytest.raises(ValueError, match="one or more keys"):
        resolve_keys("   ", is_valid)


def test_single_characters_keep_their_case() -> None:
    assert normalize_key("S") == "S"
    assert normalize_key(" ") == " "
