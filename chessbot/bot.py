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


def format_result(result: BotResult, turn: Optional[str] = None) -> str:
    """Format bot result as human-readable string.

    Args:
        result: BotResult from analyze()
        turn: Whose turn, 'w' or 'b'. If None, shows both.

    Returns:
        Formatted string for display
    """
    # Show just the position part of FEN (without turn info)
    fen_position = result.fen.split()[0] if result.fen else result.fen
    lines = [
        f"FEN: {fen_position}",
    ]

    if turn is None:
        # Show both perspectives
        from .analysis import analyze_fen
        white_analysis = analyze_fen(result.fen, turn="w")
        black_analysis = analyze_fen(result.fen, turn="b")

        # Format line continuation with move numbers
        white_line = format_line(white_analysis.continuation_san, is_white=True)
        black_line = format_line(black_analysis.continuation_san, is_white=False)

        lines.append(f"White to move: {white_analysis.eval_str}, best: {white_analysis.best_move_san}, line: {white_line}")
        lines.append(f"Black to move: {black_analysis.eval_str}, best: {black_analysis.best_move_san}, line: {black_line}")
    else:
        lines.append(f"Eval: {result.eval_str}")
        lines.append(f"Best: {result.best_move_san}")

        if result.continuation_san:
            is_white = turn == "w"
            lines.append(f"Line: {format_line(result.continuation_san, is_white)}")

    return "\n".join(lines)


def format_line(moves: list[str], is_white: bool) -> str:
    """Format a line of moves with proper move numbers.

    Args:
        moves: List of moves in SAN format
        is_white: True if white to move, False if black

    Returns:
        Formatted line like "1. e4 e5 2. Nf3 Nc6" or "1... e5 2. Nf3 Nc6"
    """
    if not moves:
        return ""

    result = []
    move_num = 1

    for i, move in enumerate(moves):
        if is_white:
            # White to move first
            if i % 2 == 0:
                # White's turn - add move number
                result.append(f"{move_num}. {move}")
            else:
                # Black's turn - just the move
                result.append(move)
                move_num += 1
        else:
            # Black to move first
            if i == 0:
                # First move - "1... move"
                result.append(f"1... {move}")
            elif i % 2 == 1:
                # White's turn - add move number
                result.append(f"{move_num}. {move}")
                move_num += 1
            else:
                # Black's subsequent moves - just the move
                result.append(move)

    return " ".join(result)
