"""Chess bot - main interface for FEN detection and analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

from .fen import image_to_fen, FenResult
from .analysis import analyze_fen, AnalysisResult


@dataclass
class BotResult:
    """Complete result from the chess bot."""
    fen: str
    orientation: str
    eval_cp: int
    eval_str: str
    best_move: str
    best_move_san: str
    continuation: list[str]
    continuation_san: list[str]
    mate: Optional[int]


def analyze(
    img: Union[str, Path],
    engine_path: str = "stockfish",
    depth: int = 20,
    turn: str = "w"
) -> BotResult:
    """Analyze a chess position from an image.

    Full pipeline: detect board → extract FEN → analyze with Stockfish.

    Args:
        img: Path to image file
        engine_path: Path to Stockfish binary
        depth: Search depth (default 20)
        turn: Whose turn, 'w' or 'b' (default 'w')

    Returns:
        BotResult with FEN, evaluation, and best moves

    Example:
        >>> result = analyze("screenshot.png")
        >>> print(f"Position: {result.fen}")
        >>> print(f"Evaluation: {result.eval_str}")
        >>> print(f"Best move: {result.best_move_san}")
    """
    # Detect FEN
    fen_result = image_to_fen(img)

    # Analyze position
    analysis = analyze_fen(fen_result.fen, engine_path, depth, turn)

    return BotResult(
        fen=fen_result.fen,
        orientation=fen_result.orientation,
        eval_cp=analysis.eval_cp,
        eval_str=analysis.eval_str,
        best_move=analysis.best_move,
        best_move_san=analysis.best_move_san,
        continuation=analysis.continuation,
        continuation_san=analysis.continuation_san,
        mate=analysis.mate
    )


def format_result(result: BotResult) -> str:
    """Format bot result as human-readable string.

    Args:
        result: BotResult from analyze()

    Returns:
        Formatted string for display
    """
    lines = [
        f"FEN: {result.fen}",
        f"Eval: {result.eval_str}",
        f"Best: {result.best_move_san}",
    ]

    if result.continuation_san:
        lines.append(f"Line: {' '.join(result.continuation_san[:6])}")

    return "\n".join(lines)
