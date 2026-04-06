"""FEN detection from chess board images."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from PIL import Image


@dataclass
class FenResult:
    """Result of FEN detection."""
    fen: str
    orientation: str  # 'white' or 'black' perspective
    board_detected: bool


def image_to_fen(
    img: Union[str, Path, Image.Image],
    auto_rotate: bool = True
) -> FenResult:
    """Extract FEN from a chess board image.

    Args:
        img: Image path or PIL Image
        auto_rotate: Auto-rotate image and board orientation

    Returns:
        FenResult with detected FEN and metadata
    """
    # Import here to avoid issues with cairo
    from chess_diagram_to_fen import get_fen

    if isinstance(img, (str, Path)):
        img = Image.open(img)

    result = get_fen(
        img=img,
        game="chess",
        auto_rotate_image=auto_rotate,
        auto_rotate_board=auto_rotate
    )

    return FenResult(
        fen=result.fen,
        orientation=result.orientation if hasattr(result, 'orientation') else 'white',
        board_detected=True
    )
