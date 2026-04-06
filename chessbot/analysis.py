"""Chess position analysis using Stockfish engine."""

import chess
import chess.engine
from typing import Optional
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Result of position analysis."""
    fen: str
    eval_cp: int  # centipawns (positive = white advantage)
    eval_str: str  # human-readable (e.g., "+1.5" or "M3")
    best_move: str  # UCI format
    best_move_san: str  # SAN format
    continuation: list[str]  # best moves for both sides (UCI)
    continuation_san: list[str]  # best moves (SAN)
    mate: Optional[int]  # mate in N moves (or None)


def analyze_fen(
    fen: str,
    engine_path: str = "stockfish",
    depth: int = 20,
    turn: str = "w"
) -> AnalysisResult:
    """Analyze a chess position and return evaluation and best moves.

    Args:
        fen: FEN string (position only, e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        engine_path: Path to Stockfish binary
        depth: Search depth
        turn: Whose turn, 'w' or 'b'

    Returns:
        AnalysisResult with evaluation and best moves
    """
    # Add/override turn in FEN
    parts = fen.split()
    if len(parts) == 1:
        # Just position, add everything
        fen = f"{fen} {turn} - - 0 1"
    else:
        # Full FEN, override turn
        parts[1] = turn
        fen = " ".join(parts)

    board = chess.Board(fen)

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))

        score = info["score"].relative
        best_move = info.get("pv", [None])[0]

        # Get best move SAN before modifying board
        best_move_san = board.san(best_move) if best_move else None

        # Get continuation (principal variation)
        pv = info.get("pv", [])

        # If mate, show full line; otherwise show 6 moves
        is_mate = score.is_mate()
        num_moves = len(pv) if is_mate else min(6, len(pv))

        continuation = [move.uci() for move in pv[:num_moves]]
        continuation_san = []
        for move in pv[:num_moves]:
            continuation_san.append(board.san(move))
            board.push(move)  # Apply move to board for next SAN

        # Format evaluation
        if score.is_mate():
            mate_in = score.mate()
            eval_str = f"M{abs(mate_in)}" if mate_in > 0 else f"-M{abs(mate_in)}"
            cp = 10000 if mate_in > 0 else -10000
        else:
            cp = score.score()
            eval_str = f"+{cp/100:.1f}" if cp >= 0 else f"{cp/100:.1f}"
            mate_in = None

        return AnalysisResult(
            fen=fen,
            eval_cp=cp,
            eval_str=eval_str,
            best_move=best_move.uci() if best_move else None,
            best_move_san=best_move_san,
            continuation=continuation,
            continuation_san=continuation_san,
            mate=mate_in
        )


def analyze_from_image(
    img_path: str,
    engine_path: str = "stockfish",
    depth: int = 20,
    turn: str = "w"
) -> AnalysisResult:
    """Full pipeline: detect board, predict FEN, analyze position.

    Args:
        img_path: Path to image file
        engine_path: Path to Stockfish binary
        depth: Search depth
        turn: Whose turn, 'w' or 'b'

    Returns:
        AnalysisResult with FEN and analysis
    """
    from .fen import image_to_fen

    fen_result = image_to_fen(img_path)
    return analyze_fen(fen_result.fen, engine_path, depth, turn)
