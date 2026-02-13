"""Helpers for resolving input resize settings."""

from typing import Tuple


def get_resize_hw(args) -> Tuple[int, int]:
    """
    Resolve input height/width from CLI args.

    Priority:
      1) --resize_h/--resize_w when both are provided
      2) legacy --resize (square)
    """
    resize = int(getattr(args, "resize", 128))
    resize_h = getattr(args, "resize_h", None)
    resize_w = getattr(args, "resize_w", None)

    if resize_h is None and resize_w is None:
        return resize, resize

    if resize_h is None or resize_w is None:
        raise ValueError("Both --resize_h and --resize_w must be set together.")

    return int(resize_h), int(resize_w)
