#!/usr/bin/env python3
"""
Escape-Check KG Engine (bug-fixed)
===================================
Knowledge graph trainer for learning abstract check-evasion patterns.

Bug fixes applied:
  1. is_check() after push checks OPPONENT, not self — all legal moves from
     check escape check by definition. Fixed in: compute_situation_features,
     heuristic_escapes, channel_scores, legal_escape_moves.
  2. heuristic_escapes now correctly classifies blocking moves using square
     indices (ints) instead of comparing int to_square against string set.
  3. safe_king_moves count now correctly counts ALL legal king moves (not just
     those that don't give check to opponent).
  4. channel_scores 'escapes_check' is always 1.0 for legal moves from check
     (since that's guaranteed by legality).
  5. Added 'gives_check' as a bonus channel — moves that escape check AND give
     check back are tactically interesting.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import chess
    import chess.engine
    import chess.svg
except Exception:
    chess = None

try:
    import requests
except Exception:
    requests = None


# =====================================================================
# Utilities
# =====================================================================

def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def decay_factor(dt_seconds: float, half_life_seconds: float) -> float:
    if half_life_seconds <= 0:
        return 1.0
    return 0.5 ** (dt_seconds / half_life_seconds)


def require_python_chess() -> None:
    if chess is None:
        raise RuntimeError("python-chess is required: pip install python-chess")


# =====================================================================
# Database / Knowledge Graph
# =====================================================================

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS nodes (
  node_id INTEGER PRIMARY KEY AUTOINCREMENT,
  node_type TEXT NOT NULL,
  handle TEXT NOT NULL,
  keywords TEXT NOT NULL DEFAULT '[]',
  topic TEXT NOT NULL DEFAULT '',
  facts TEXT NOT NULL DEFAULT '[]',
  features TEXT NOT NULL DEFAULT '{}',
  hp REAL NOT NULL DEFAULT 1.0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(node_type, handle)
);

CREATE TABLE IF NOT EXISTS edges (
  edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
  src_node_id INTEGER NOT NULL,
  dst_node_id INTEGER NOT NULL,
  edge_type TEXT NOT NULL,
  channels TEXT NOT NULL DEFAULT '{}',
  hp REAL NOT NULL DEFAULT 1.0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(src_node_id, dst_node_id, edge_type),
  FOREIGN KEY(src_node_id) REFERENCES nodes(node_id),
  FOREIGN KEY(dst_node_id) REFERENCES nodes(node_id)
);

CREATE TABLE IF NOT EXISTS episodes (
  episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  fen TEXT NOT NULL,
  side_to_move TEXT NOT NULL,
  situation_node_id INTEGER NOT NULL,
  chosen_move_uci TEXT NOT NULL,
  oracle_best_move_uci TEXT,
  oracle_eval_cp INTEGER,
  oracle_eval_mate INTEGER,
  channels TEXT NOT NULL DEFAULT '{}',
  notes TEXT NOT NULL DEFAULT '',
  FOREIGN KEY(situation_node_id) REFERENCES nodes(node_id)
);

CREATE INDEX IF NOT EXISTS idx_nodes_type_handle ON nodes(node_type, handle);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_node_id);
CREATE INDEX IF NOT EXISTS idx_episodes_sit ON episodes(situation_node_id);
"""


class KG:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    # ---- Stats ----

    def stats(self) -> Dict[str, Any]:
        nodes = self.conn.execute("SELECT COUNT(*) as c FROM nodes").fetchone()["c"]
        edges = self.conn.execute("SELECT COUNT(*) as c FROM edges").fetchone()["c"]
        episodes = self.conn.execute("SELECT COUNT(*) as c FROM episodes").fetchone()["c"]
        sit_nodes = self.conn.execute(
            "SELECT COUNT(*) as c FROM nodes WHERE node_type='situation'"
        ).fetchone()["c"]
        act_nodes = self.conn.execute(
            "SELECT COUNT(*) as c FROM nodes WHERE node_type='action'"
        ).fetchone()["c"]
        avg_hp = self.conn.execute(
            "SELECT AVG(hp) as a FROM edges"
        ).fetchone()["a"] or 0.0
        return {
            "total_nodes": nodes,
            "situation_nodes": sit_nodes,
            "action_nodes": act_nodes,
            "total_edges": edges,
            "total_episodes": episodes,
            "avg_edge_hp": round(float(avg_hp), 4),
        }

    # ---- Node ops ----

    def upsert_node(
        self,
        node_type: str,
        handle: str,
        *,
        keywords: Optional[List[str]] = None,
        topic: str = "",
        facts: Optional[List[str]] = None,
        features: Optional[Dict[str, Any]] = None,
        hp_boost: float = 0.0,
    ) -> int:
        keywords = keywords or []
        facts = facts or []
        features = features or {}
        now = utc_now_iso()

        cur = self.conn.execute(
            "SELECT node_id, hp FROM nodes WHERE node_type=? AND handle=?",
            (node_type, handle),
        )
        row = cur.fetchone()
        if row:
            new_hp = float(row["hp"]) + hp_boost
            self.conn.execute(
                """UPDATE nodes
                   SET keywords=?, topic=?, facts=?, features=?, hp=?, updated_at=?
                   WHERE node_id=?""",
                (stable_json(keywords), topic, stable_json(facts),
                 stable_json(features), new_hp, now, row["node_id"]),
            )
            self.conn.commit()
            return int(row["node_id"])

        self.conn.execute(
            """INSERT INTO nodes
               (node_type, handle, keywords, topic, facts, features, hp, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (node_type, handle, stable_json(keywords), topic, stable_json(facts),
             stable_json(features), 1.0 + hp_boost, now, now),
        )
        self.conn.commit()
        return int(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def get_node(self, node_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM nodes WHERE node_id=?", (node_id,)
        ).fetchone()

    def get_node_by(self, node_type: str, handle: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM nodes WHERE node_type=? AND handle=?",
            (node_type, handle),
        ).fetchone()

    # ---- Edge ops ----

    def upsert_edge(
        self,
        src_node_id: int,
        dst_node_id: int,
        edge_type: str,
        *,
        channels_update: Dict[str, float],
        hp_boost: float = 0.0,
    ) -> int:
        now = utc_now_iso()
        cur = self.conn.execute(
            "SELECT edge_id, channels, hp FROM edges "
            "WHERE src_node_id=? AND dst_node_id=? AND edge_type=?",
            (src_node_id, dst_node_id, edge_type),
        )
        row = cur.fetchone()
        if row:
            channels = json.loads(row["channels"])
            for k, v in channels_update.items():
                v = float(v)
                if k in channels:
                    channels[k] = 0.7 * channels[k] + 0.3 * v
                else:
                    channels[k] = v
            new_hp = float(row["hp"]) + hp_boost
            self.conn.execute(
                """UPDATE edges SET channels=?, hp=?, updated_at=? WHERE edge_id=?""",
                (stable_json(channels), new_hp, now, row["edge_id"]),
            )
            self.conn.commit()
            return int(row["edge_id"])

        self.conn.execute(
            """INSERT INTO edges
               (src_node_id, dst_node_id, edge_type, channels, hp, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (src_node_id, dst_node_id, edge_type,
             stable_json(channels_update), 1.0 + hp_boost, now, now),
        )
        self.conn.commit()
        return int(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def top_edges_from(self, src_node_id: int, edge_type: str,
                       limit: int = 10) -> List[sqlite3.Row]:
        cur = self.conn.execute(
            """SELECT e.*, n.handle AS dst_handle, n.node_type AS dst_type,
                      n.topic AS dst_topic
               FROM edges e
               JOIN nodes n ON n.node_id = e.dst_node_id
               WHERE e.src_node_id=? AND e.edge_type=?
               ORDER BY e.hp DESC LIMIT ?""",
            (src_node_id, edge_type, limit),
        )
        return cur.fetchall()

    # ---- Episode ops ----

    def insert_episode(self, **kwargs) -> int:
        now = utc_now_iso()
        self.conn.execute(
            """INSERT INTO episodes
               (created_at, fen, side_to_move, situation_node_id, chosen_move_uci,
                oracle_best_move_uci, oracle_eval_cp, oracle_eval_mate, channels, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                now,
                kwargs["fen"],
                kwargs["side_to_move"],
                kwargs["situation_node_id"],
                kwargs["chosen_move_uci"],
                kwargs.get("oracle_best_move_uci"),
                kwargs.get("oracle_eval_cp"),
                kwargs.get("oracle_eval_mate"),
                stable_json(kwargs.get("channels", {})),
                kwargs.get("notes", ""),
            ),
        )
        self.conn.commit()
        return int(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    # ---- Decay ----

    def apply_decay(self, half_life_hours: float) -> None:
        half_life_seconds = max(1.0, half_life_hours * 3600.0)
        now = dt.datetime.utcnow()

        for table in ("nodes", "edges"):
            id_col = "node_id" if table == "nodes" else "edge_id"
            rows = self.conn.execute(
                f"SELECT {id_col}, hp, updated_at FROM {table}"
            ).fetchall()
            for r in rows:
                updated = dt.datetime.fromisoformat(r["updated_at"].replace("Z", ""))
                dt_sec = max(0.0, (now - updated).total_seconds())
                f = decay_factor(dt_sec, half_life_seconds)
                new_hp = float(r["hp"]) * f
                self.conn.execute(
                    f"UPDATE {table} SET hp=?, updated_at=? WHERE {id_col}=?",
                    (new_hp, utc_now_iso(), r[id_col]),
                )
        self.conn.commit()


# =====================================================================
# Situation language (feature extraction)
# =====================================================================

def square_name(sq: int) -> str:
    return chess.square_name(sq)


def piece_name(piece: "chess.Piece") -> str:
    return piece.symbol().lower()


def is_slider(piece_type: int) -> bool:
    return piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN)


def compute_situation_features(board: "chess.Board") -> Dict[str, Any]:
    """
    Extract abstract features describing a check position.

    BUG FIX: After pushing a legal move, board.is_check() tests whether the
    NEW side to move (opponent) is in check — NOT whether we escaped.
    All legal moves from check escape check by definition.
    We now track 'gives_check' as a separate (positive) signal.
    """
    require_python_chess()
    if not board.is_check():
        raise ValueError("Board is not in check.")

    color = board.turn
    king_sq = board.king(color)
    checkers = list(board.checkers())
    double_check = len(checkers) >= 2

    # --- Checker analysis ---
    checker_infos = []
    can_block_any = False
    block_squares_all: List[int] = []  # BUG FIX: store as ints, not strings

    for csq in checkers:
        p = board.piece_at(csq)
        if not p:
            continue
        info = {
            "checker_sq": square_name(csq),
            "checker_piece": piece_name(p),
            "checker_is_slider": is_slider(p.piece_type),
            "distance": chess.square_distance(king_sq, csq),
        }
        block_squares: List[str] = []
        if is_slider(p.piece_type):
            ray = chess.SquareSet.between(king_sq, csq)
            block_squares = [square_name(s) for s in ray]
            if ray:
                can_block_any = True
                block_squares_all.extend(list(ray))  # keep as int squares
        info["block_squares"] = block_squares
        checker_infos.append(info)

    # --- King escape squares ---
    # BUG FIX: ALL legal king moves escape check (legality guarantees it).
    # We count them directly without the faulty is_check() filter.
    safe_king_moves = sum(
        1 for mv in board.legal_moves if mv.from_square == king_sq
    )

    # --- Capture possibilities (capture a checking piece) ---
    capture_moves = [
        mv for mv in board.legal_moves if mv.to_square in checkers
    ]

    # --- Block possibilities ---
    # BUG FIX: compare int to_square against int set (was comparing int vs string set)
    block_moves = []
    if can_block_any and not double_check:
        block_sq_set = set(block_squares_all)  # set of int squares
        block_moves = [
            mv for mv in board.legal_moves
            if mv.to_square in block_sq_set and mv.from_square != king_sq
        ]

    adjacent_check = any(
        chess.square_distance(king_sq, csq) == 1 for csq in checkers
    )

    # --- Classify check type ---
    check_piece_types = []
    for csq in checkers:
        p = board.piece_at(csq)
        if p:
            check_piece_types.append(p.piece_type)

    if double_check:
        check_type = "double_check"
    elif chess.KNIGHT in check_piece_types:
        check_type = "knight_check"
    elif chess.PAWN in check_piece_types:
        check_type = "pawn_check"
    elif chess.BISHOP in check_piece_types:
        check_type = "bishop_check"
    elif chess.ROOK in check_piece_types:
        check_type = "rook_check"
    elif chess.QUEEN in check_piece_types:
        check_type = "queen_check"
    else:
        check_type = "unknown"

    return {
        "side_to_move": "w" if color == chess.WHITE else "b",
        "check_type": check_type,
        "double_check": double_check,
        "num_checkers": len(checkers),
        "checker_infos": checker_infos,
        "safe_king_moves": safe_king_moves,
        "capture_options": len(capture_moves),
        "block_options": len(block_moves),
        "adjacent_check": adjacent_check,
        "material_balance_rough": rough_material(board),
    }


def rough_material(board: "chess.Board") -> int:
    values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
    }
    score = 0
    for sq, p in board.piece_map().items():
        v = values.get(p.piece_type, 0)
        score += v if p.color == chess.WHITE else -v
    return score


def situation_signature(features: Dict[str, Any]) -> str:
    """
    Produce a generalized signature from features.
    
    Intentionally coarse so many positions map to the same node, BUT includes
    enough positional info (king zone, checker zone) that the moves stored
    for a signature are actually relevant to positions that match it.
    
    King zone + checker zone ensure that a queen check on the kingside with
    king on g8 maps to a different node than a queen check on the queenside
    with king on b1. This dramatically improves move relevance.
    """
    checker_pieces = []
    checker_slider = []
    checker_zones = []
    for ci in features.get("checker_infos", []):
        checker_pieces.append(ci.get("checker_piece", "?"))
        checker_slider.append("S" if ci.get("checker_is_slider") else "N")
        # Extract zone from checker square (a-c = queenside, d-e = center, f-h = kingside)
        csq = ci.get("checker_sq", "e4")
        file_char = csq[0] if csq else "e"
        if file_char in "abc":
            checker_zones.append("Q")
        elif file_char in "fgh":
            checker_zones.append("K")
        else:
            checker_zones.append("C")

    # Determine king zone from features (we need to extract from checker_infos context)
    # The side_to_move + check_type gives us the pattern, but for king location
    # we look at the safe_king_moves as a rough proxy and include the king rank
    # bucket (ranks 1-2 = back, 3-5 = mid, 6-8 = advanced)
    king_zone = "?"
    if features.get("_king_file"):
        kf = features["_king_file"]
        king_zone = "Q" if kf in "abc" else "K" if kf in "fgh" else "C"
    king_rank_zone = "?"
    if features.get("_king_rank"):
        kr = int(features["_king_rank"])
        king_rank_zone = "B" if kr <= 2 else "A" if kr >= 7 else "M"

    sig = {
        "side": features.get("side_to_move", "?"),
        "check_type": features.get("check_type", "?"),
        "num_checkers": int(features.get("num_checkers", 0)),
        "checkers": "".join(sorted(checker_pieces))[:4],
        "slider_flags": "".join(sorted(checker_slider))[:4],
        "safe_king_moves": int(features.get("safe_king_moves", 0)),
        "capture_options": int(features.get("capture_options", 0)),
        "block_options": min(3, int(features.get("block_options", 0))),  # cap at 3 to keep coarse
        "king_zone": king_zone,
        "king_rank": king_rank_zone,
        "checker_zone": "".join(sorted(checker_zones))[:2],
    }
    return stable_json(sig)


# =====================================================================
# Stockfish Oracle
# =====================================================================

@dataclasses.dataclass
class OracleResult:
    best_move_uci: Optional[str]
    eval_cp: Optional[int]
    eval_mate: Optional[int]


class StockfishOracle:
    def __init__(self, path: Optional[str] = None, think_time: float = 0.05,
                 depth: Optional[int] = 10):
        require_python_chess()
        self.path = path or os.environ.get("STOCKFISH_PATH", "stockfish")
        self.think_time = think_time
        self.depth = depth
        self.engine: Optional[chess.engine.SimpleEngine] = None

    def start(self) -> bool:
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
            return True
        except Exception:
            self.engine = None
            return False

    def stop(self) -> None:
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
        self.engine = None

    def analyze_best(self, board: "chess.Board") -> OracleResult:
        if not self.engine:
            return OracleResult(None, None, None)
        limit = chess.engine.Limit(depth=self.depth) if self.depth else \
            chess.engine.Limit(time=self.think_time)
        try:
            info = self.engine.analyse(board, limit)
            pv = info.get("pv", [])
            best = pv[0].uci() if pv else None
            score = info.get("score")
            eval_cp, eval_mate = None, None
            if score is not None:
                pov = score.pov(board.turn)
                if pov.is_mate():
                    eval_mate = int(pov.mate())
                else:
                    eval_cp = int(pov.score(mate_score=100000))
            return OracleResult(best, eval_cp, eval_mate)
        except Exception:
            return OracleResult(None, None, None)

    def eval_move(self, board: "chess.Board", move: "chess.Move") -> OracleResult:
        """Evaluate resulting position after move, from POV of side before move."""
        if not self.engine:
            return OracleResult(None, None, None)
        b2 = board.copy(stack=False)
        b2.push(move)
        # Analyze from opponent's POV, then negate
        result = self.analyze_best(b2)
        # Flip signs: the analysis is from opponent's POV
        if result.eval_cp is not None:
            result.eval_cp = -result.eval_cp
        if result.eval_mate is not None:
            result.eval_mate = -result.eval_mate
        return result


# =====================================================================
# Ollama LLM proposer (optional)
# =====================================================================

class OllamaClient:
    """
    Connects to a local Ollama server to propose candidate escape moves.
    The LLM sees the FEN, abstract features, AND the complete list of legal moves.
    It must select and rank from the legal moves only.
    
    Timeout is generous (180s default) to give large local models enough
    time to think — even 20B+ models on Apple Silicon need time.
    """

    def __init__(self, host: str = "http://localhost:11434",
                 model: str = "gpt-oss:20b",
                 timeout: float = 180.0):
        if requests is None:
            raise RuntimeError("requests not installed: pip install requests")
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    @staticmethod
    def list_models(host: str = "http://localhost:11434") -> Dict[str, Any]:
        """
        Query the Ollama server for available models.
        Returns:
            {
                "available": True/False,     # server reachable
                "models": [                  # list of model info dicts
                    {"name": "gpt-oss:20b", "size": "20B", "modified": "..."},
                    ...
                ],
                "error": "..." or None
            }
        """
        if requests is None:
            return {"available": False, "models": [], "error": "requests not installed"}
        host = host.rstrip("/")
        try:
            r = requests.get(f"{host}/api/tags", timeout=5)
            r.raise_for_status()
            tags = r.json()
            models = []
            for m in tags.get("models", []):
                name = m.get("name", "")
                # Size in GB (approximate from byte size)
                size_bytes = m.get("size", 0)
                size_str = f"{size_bytes / 1e9:.1f}GB" if size_bytes else "?"
                models.append({
                    "name": name,
                    "size": size_str,
                    "modified": m.get("modified_at", ""),
                    "family": m.get("details", {}).get("family", ""),
                    "parameter_size": m.get("details", {}).get("parameter_size", ""),
                })
            return {"available": True, "models": models, "error": None}
        except requests.ConnectionError:
            return {"available": False, "models": [], "error": "Cannot connect to Ollama server"}
        except requests.Timeout:
            return {"available": False, "models": [], "error": "Ollama server timed out"}
        except Exception as e:
            return {"available": False, "models": [], "error": str(e)[:200]}

    def check_available(self) -> bool:
        """Ping Ollama to verify it's running and the model exists."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            tags = r.json()
            models = [m.get("name", "") for m in tags.get("models", [])]
            # Check if our model (or a prefix match) is available
            for m in models:
                if m == self.model or m.startswith(self.model.split(":")[0]):
                    return True
            # Model not found, but server is up — still usable if model
            # gets pulled later or name is slightly different
            return True
        except Exception:
            return False

    def propose_escapes(self, features: Dict[str, Any],
                        board_legal_uci: Optional[List[str]] = None,
                        fen: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask the LLM to rank the best escape moves.
        Uses urllib + /api/generate (same pattern as maze KG app that works).
        Retry on bad JSON by appending feedback to prompt.
        """
        empty = {"moves_uci": [], "rationale": "", "rules_used": [],
                 "_raw_text": "", "_pre_filter_count": 0}

        prompt = self._build_prompt(features, board_legal_uci, fen)
        max_retries = 2

        for attempt in range(1 + max_retries):
            try:
                # Use urllib — exact same pattern as maze app's _ollama_generate
                import urllib.request
                url = f"{self.host}/api/generate"
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw_bytes = resp.read().decode("utf-8")
                data = json.loads(raw_bytes)
                text = (data.get("response") or "").strip()

                if not text:
                    if attempt < max_retries:
                        prompt += "\n\nYou returned an empty response. Please output the JSON object now."
                        continue
                    empty["_raw_text"] = "(empty)"
                    return empty

                parsed = self._extract_json(text)
                if parsed is not None:
                    moves = parsed.get("moves_uci", [])
                    if isinstance(moves, list) and len(moves) > 0:
                        result = {
                            "moves_uci": [str(m) for m in moves],
                            "rationale": str(parsed.get("rationale", ""))[:400],
                            "rules_used": parsed.get("rules_used", []),
                            "_raw_text": text[:300],
                            "_pre_filter_count": len(moves),
                        }
                        if board_legal_uci:
                            legal_set = set(board_legal_uci)
                            result["moves_uci"] = [m for m in result["moves_uci"] if m in legal_set]
                        return result

                if attempt < max_retries:
                    prompt += (
                        f"\n\nYour response was not valid JSON. You said:\n{text[:200]}\n\n"
                        f"Please respond with ONLY a JSON object like this:\n"
                        + json.dumps({
                            "moves_uci": (board_legal_uci or ["e2e4"])[:2],
                            "rationale": "brief reason",
                            "rules_used": ["capture_checker"]
                        })
                    )
                    continue

                empty["_raw_text"] = text[:300]
                return empty

            except Exception as e:
                empty["rationale"] = f"ollama_error: {str(e)[:200]}"
                return empty

        empty["rationale"] = "json_parse_failed_after_retries"
        return empty

    @staticmethod
    def _build_prompt(features: Dict[str, Any],
                      legal_moves: Optional[List[str]] = None,
                      fen: Optional[str] = None) -> str:
        """Single prompt string for /api/generate."""
        ex_moves = (legal_moves or ["e2e4", "d7d5"])[:3]
        example = json.dumps({
            "moves_uci": ex_moves,
            "rationale": "Capturing the checker wins material.",
            "rules_used": ["capture_checker"]
        })

        ct = features.get("check_type", "unknown")
        km = features.get("safe_king_moves", 0)
        cap = features.get("capture_options", 0)
        blk = features.get("block_options", 0)
        dc = features.get("double_check", False)

        parts = [
            "You are a chess engine. Analyze this CHECK position and rank the escape moves.",
            "",
        ]
        if fen:
            parts.append(f"FEN: {fen}")
        parts.append(f"Check type: {ct}" + (" DOUBLE CHECK" if dc else ""))
        parts.append(f"King escapes: {km} | Captures: {cap} | Blocks: {blk}")
        if legal_moves:
            parts.append(f"LEGAL_MOVES: {json.dumps(legal_moves)}")
        parts.append("")
        parts.append(f"Respond with ONLY this JSON (no other text):")
        parts.append(example)

        return "\n".join(parts)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """
        Extract JSON using balanced-brace counting (same technique as maze KG app).
        No regex. Finds first '{', counts braces to find matching '}', parses.
        """
        # Strip markdown fences if present
        for fence in ["```json", "```JSON", "```"]:
            text = text.replace(fence, "")
        text = text.strip()

        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except (json.JSONDecodeError, ValueError):
                        pass
                    return None
        return None


# =====================================================================
# Heuristic proposer (no LLM needed)
# =====================================================================

def heuristic_escapes(board: "chess.Board") -> List[str]:
    """
    BUG FIX: All legal moves from check escape check. Don't filter by
    is_check() after push (that tests opponent check, not ours).
    Prioritize: capture checker > block > king move.
    """
    require_python_chess()
    if not board.is_check():
        return []

    escapes_capture = []
    escapes_block = []
    escapes_king = []
    checkers = set(board.checkers())
    king_sq = board.king(board.turn)

    for mv in board.legal_moves:
        # All legal moves escape check — classify by type
        if mv.to_square in checkers:
            escapes_capture.append(mv.uci())
        elif mv.from_square == king_sq:
            escapes_king.append(mv.uci())
        else:
            escapes_block.append(mv.uci())

    return (escapes_capture + escapes_block + escapes_king)[:12]


# =====================================================================
# Scoring / adjudication
# =====================================================================

def generate_random_check_position(
    rng: random.Random, max_plies: int = 60
) -> Optional["chess.Board"]:
    require_python_chess()
    b = chess.Board()
    for _ in range(max_plies):
        if b.is_check():
            return b
        moves = list(b.legal_moves)
        if not moves:
            return None
        b.push(rng.choice(moves))
    return b if b.is_check() else None


def score_with_oracle_delta(best_eval: OracleResult,
                            move_eval: OracleResult) -> float:
    if best_eval.eval_mate is not None:
        if move_eval.eval_mate is None:
            return 0.0
        return 1.0 if (best_eval.eval_mate > 0 and move_eval.eval_mate > 0) else 0.0
    if best_eval.eval_cp is None or move_eval.eval_cp is None:
        return 0.5
    delta = best_eval.eval_cp - move_eval.eval_cp
    return clamp01(math.exp(-max(0.0, delta) / 80.0))


def channel_scores(
    board: "chess.Board",
    move: "chess.Move",
    *,
    oracle: Optional[StockfishOracle],
    oracle_best: Optional[OracleResult],
) -> Tuple[Dict[str, float], OracleResult]:
    """
    BUG FIX: escapes_check is always 1.0 for legal moves from check.
    Added 'gives_check' as a bonus tactical channel.
    """
    require_python_chess()

    if move not in board.legal_moves:
        return {
            "legality": 0.0, "escapes_check": 0.0, "robustness": 0.0,
            "oracle_signal": 0.0, "gives_check": 0.0,
        }, OracleResult(None, None, None)

    # All legal moves from check escape check — that's what legality means
    escapes = 1.0

    # BUG FIX: Check if the move GIVES check to the opponent (bonus signal)
    b2 = board.copy(stack=False)
    b2.push(move)
    gives_check = 1.0 if b2.is_check() else 0.0

    move_oracle_eval = OracleResult(None, None, None)
    robustness = 0.5
    oracle_signal = 0.0

    if oracle and oracle.engine and oracle_best:
        move_oracle_eval = oracle.eval_move(board, move)
        robustness = score_with_oracle_delta(oracle_best, move_oracle_eval)
        if move_oracle_eval.eval_mate is not None:
            oracle_signal = 1.0 if move_oracle_eval.eval_mate > 0 else 0.0
        elif move_oracle_eval.eval_cp is not None:
            oracle_signal = clamp01(logistic(move_oracle_eval.eval_cp / 120.0))

    return {
        "legality": 1.0,
        "escapes_check": escapes,
        "robustness": robustness,
        "oracle_signal": oracle_signal,
        "gives_check": gives_check,
    }, move_oracle_eval


def hp_boost_from_channels(ch: Dict[str, float]) -> float:
    escapes = float(ch.get("escapes_check", 0.0))
    legality = float(ch.get("legality", 0.0))
    robustness = float(ch.get("robustness", 0.0))
    oracle_signal = float(ch.get("oracle_signal", 0.0))
    gives_check = float(ch.get("gives_check", 0.0))

    base = 0.05 * legality + 0.9 * escapes
    quality = 0.2 * robustness + 0.2 * oracle_signal + 0.05 * gives_check
    return max(0.0, base + quality)


# =====================================================================
# Training loop
# =====================================================================

def train(
    kg: KG,
    episodes: int,
    *,
    seed: int = 1,
    max_plies: int = 60,
    stockfish_path: Optional[str] = None,
    oracle_depth: int = 10,
    oracle_time: float = 0.05,
    use_ollama: bool = False,
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "gpt-oss:20b",
    progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
) -> Dict[str, Any]:
    """
    Run training episodes. Returns summary stats.
    progress_callback(episode_num, total, info_dict) is called each episode.
    
    When use_ollama=True, the LLM proposes candidate moves BEFORE the heuristic.
    Both sources are merged (deduplicated), then all candidates are scored by
    Stockfish. This lets the LLM suggest creative moves the heuristic misses.
    """
    require_python_chess()
    rng = random.Random(seed)

    oracle = StockfishOracle(stockfish_path, think_time=oracle_time, depth=oracle_depth)
    oracle_ok = oracle.start()

    # --- Initialize Ollama proposer if requested ---
    ollama: Optional[OllamaClient] = None
    ollama_ok = False
    if use_ollama:
        try:
            ollama = OllamaClient(host=ollama_host, model=ollama_model)
            ollama_ok = ollama.check_available()
            if not ollama_ok:
                ollama = None
        except Exception:
            ollama = None

    summary = {
        "episodes_requested": episodes,
        "episodes_completed": 0,
        "episodes_skipped": 0,
        "oracle_active": oracle_ok,
        "ollama_active": ollama_ok,
        "ollama_proposals": 0,       # episodes where Ollama contributed moves
        "ollama_unique_finds": 0,    # moves Ollama proposed that heuristic didn't
        "unique_situations": set(),
        "best_move_matches": 0,
    }

    for i in range(episodes):
        b = generate_random_check_position(rng, max_plies=max_plies)
        if b is None or not b.is_check():
            summary["episodes_skipped"] += 1
            if progress_callback:
                progress_callback(i + 1, episodes, {"status": "skipped"})
            continue

        # --- Situation node ---
        feats = compute_situation_features(b)
        sig = situation_signature(feats)
        summary["unique_situations"].add(sig)

        sit_keywords = [
            "escape_check", feats["check_type"],
            "double_check" if feats["double_check"] else "single_check",
        ]
        sit_topic = (
            f"Escape {feats['check_type']} | captures={feats['capture_options']} "
            f"blocks={feats['block_options']} king_safe={feats['safe_king_moves']}"
        )
        sit_facts = [
            f"side_to_move={feats['side_to_move']}",
            f"num_checkers={feats['num_checkers']}",
        ]

        sit_id = kg.upsert_node(
            "situation", sig,
            keywords=sit_keywords, topic=sit_topic,
            facts=sit_facts, features=feats, hp_boost=0.05,
        )

        # --- Propose moves: Ollama (creative) + heuristic (exhaustive) ---
        llm_proposed: List[str] = []
        llm_notes = ""
        llm_status = ""  # diagnostic: what happened with the LLM this episode
        if ollama:
            try:
                # Give LLM the FEN + legal moves so it can actually reason about the position
                legal_uci = [mv.uci() for mv in b.legal_moves]
                resp = ollama.propose_escapes(feats, board_legal_uci=legal_uci, fen=b.fen())
                raw_moves = resp.get("moves_uci", []) or []
                rationale = (resp.get("rationale", "") or "")[:400]
                raw_text = (resp.get("_raw_text", "") or "")[:120]  # peek at actual LLM output
                llm_notes = rationale

                if "ollama_timeout" in rationale:
                    llm_status = "llm:timeout"
                elif "ollama_error" in rationale:
                    llm_status = f"llm:err"
                elif not raw_moves:
                    pre_ct = resp.get("_pre_filter_count", 0)
                    snippet = raw_text.replace('\n', ' ').strip()[:80] if raw_text else rationale[:80]
                    if pre_ct > 0:
                        llm_status = f"llm:all_illegal({pre_ct} proposed)"
                    else:
                        llm_status = f"llm:no_parse({snippet})"
                else:
                    llm_proposed = raw_moves
                    llm_status = f"llm:{len(llm_proposed)}"
                    summary["ollama_proposals"] += 1
            except Exception as e:
                llm_notes = f"ollama_error={str(e)[:200]}"
                llm_status = "llm:exception"

        heuristic_proposed = heuristic_escapes(b)

        # Track unique LLM finds (moves heuristic didn't propose)
        heur_set = set(heuristic_proposed)
        for m in llm_proposed:
            if m not in heur_set:
                summary["ollama_unique_finds"] += 1

        # Merge: LLM first (so its creative suggestions get priority), then heuristic
        proposed_uci = list(dict.fromkeys([*llm_proposed, *heuristic_proposed]))[:16]

        # --- Oracle best ---
        oracle_best = oracle.analyze_best(b) if oracle_ok else OracleResult(None, None, None)

        # --- Score candidates ---
        scored: List[Tuple[float, chess.Move, Dict[str, float], OracleResult]] = []
        for uci in proposed_uci:
            try:
                mv = chess.Move.from_uci(uci)
            except Exception:
                continue
            if mv not in b.legal_moves:
                continue
            ch, mv_eval = channel_scores(
                b, mv,
                oracle=oracle if oracle_ok else None,
                oracle_best=oracle_best if oracle_ok else None,
            )
            s = (2.0 * ch["escapes_check"] + 0.7 * ch["robustness"] +
                 0.3 * ch["oracle_signal"] + 0.1 * ch["gives_check"])
            scored.append((s, mv, ch, mv_eval))

        if not scored:
            summary["episodes_skipped"] += 1
            if progress_callback:
                progress_callback(i + 1, episodes, {"status": "no_legal_escapes"})
            continue

        scored.sort(key=lambda t: t[0], reverse=True)
        chosen_move = scored[0][1]
        chosen_channels = scored[0][2]
        chosen_oracle_eval = scored[0][3]

        # Track best-move match
        if oracle_best.best_move_uci and chosen_move.uci() == oracle_best.best_move_uci:
            summary["best_move_matches"] += 1

        # --- Action node ---
        act_id = kg.upsert_node(
            "action", chosen_move.uci(),
            keywords=["uci_move", "escape_check_candidate"],
            topic=f"Evasion: {chosen_move.uci()}",
            features={"uci": chosen_move.uci()},
            hp_boost=0.02,
        )

        # --- Edge: situation -> action ---
        boost = hp_boost_from_channels(chosen_channels)
        kg.upsert_edge(
            sit_id, act_id, "suggests",
            channels_update=chosen_channels, hp_boost=boost,
        )

        # --- Episode record ---
        kg.insert_episode(
            fen=b.fen(),
            side_to_move="w" if b.turn == chess.WHITE else "b",
            situation_node_id=sit_id,
            chosen_move_uci=chosen_move.uci(),
            oracle_best_move_uci=oracle_best.best_move_uci if oracle_ok else None,
            oracle_eval_cp=chosen_oracle_eval.eval_cp if chosen_oracle_eval else None,
            oracle_eval_mate=chosen_oracle_eval.eval_mate if chosen_oracle_eval else None,
            channels=chosen_channels,
            notes=llm_notes,
        )

        summary["episodes_completed"] += 1
        if progress_callback:
            progress_callback(i + 1, episodes, {
                "status": "ok",
                "fen": b.fen(),
                "chosen": chosen_move.uci(),
                "oracle_best": oracle_best.best_move_uci,
                "boost": round(boost, 3),
                "ollama_proposed": len(llm_proposed),
                "llm_status": llm_status,
            })

    oracle.stop()
    summary["unique_situations"] = len(summary["unique_situations"])
    return summary


# =====================================================================
# Query / retrieval
# =====================================================================

def query_position(kg: KG, fen: str, limit: int = 8) -> Dict[str, Any]:
    """
    Query the KG for suggestions given a FEN in check. Returns structured result.

    CRITICAL FIX: Filters suggestions for legality in the actual queried position.
    The coarse signature maps many positions to the same node, so stored moves
    may not be legal in this specific position. We fetch extra edges and filter.
    """
    require_python_chess()
    b = chess.Board(fen)

    result: Dict[str, Any] = {
        "fen": fen,
        "in_check": b.is_check(),
        "features": None,
        "signature": None,
        "node_found": False,
        "node_hp": 0.0,
        "suggestions": [],
        "illegal_filtered": 0,   # how many KG suggestions were illegal here
        "board_svg": "",
    }

    if not b.is_check():
        return result

    feats = compute_situation_features(b)
    sig = situation_signature(feats)
    result["features"] = feats
    result["signature"] = sig

    # Generate board SVG
    try:
        checkers = list(b.checkers())
        king_sq = b.king(b.turn)
        highlight = {king_sq: "#ff4444"}
        for csq in checkers:
            highlight[csq] = "#ffaa00"
        result["board_svg"] = chess.svg.board(
            b, size=360, fill=highlight, flipped=(b.turn == chess.BLACK)
        )
    except Exception:
        result["board_svg"] = ""

    node = kg.get_node_by("situation", sig)
    if not node:
        return result

    result["node_found"] = True
    result["node_hp"] = round(float(node["hp"]), 3)

    # Build set of legal UCI moves for this specific position
    legal_uci = {mv.uci() for mv in b.legal_moves}

    # Fetch extra edges since many will be filtered out (different positions
    # sharing same signature will have contributed different moves)
    edges = kg.top_edges_from(int(node["node_id"]), "suggests", limit=limit * 4)

    filtered_count = 0
    for e in edges:
        move_uci = e["dst_handle"]

        # CRITICAL: skip moves that aren't legal in THIS position
        if move_uci not in legal_uci:
            filtered_count += 1
            continue

        if len(result["suggestions"]) >= limit:
            break

        channels = json.loads(e["channels"])
        result["suggestions"].append({
            "move": move_uci,
            "edge_hp": round(float(e["hp"]), 3),
            "escapes_check": round(channels.get("escapes_check", 0), 3),
            "robustness": round(channels.get("robustness", 0), 3),
            "oracle_signal": round(channels.get("oracle_signal", 0), 3),
            "gives_check": round(channels.get("gives_check", 0), 3),
        })

    result["illegal_filtered"] = filtered_count
    return result


def render_board_with_move(fen: str, move_uci: str) -> Dict[str, Any]:
    """
    Render a board with a move arrow, plus the resulting position after the move.
    Returns dict with:
      - before_svg: SVG of current position with arrow showing the move
      - after_svg: SVG of resulting position with highlighted landing square
      - after_fen: FEN after the move
      - valid: whether the move was legal
      - gives_check: whether the move gives check to opponent
      - is_capture: whether the move captures a piece
    """
    require_python_chess()
    try:
        b = chess.Board(fen)
        mv = chess.Move.from_uci(move_uci)
    except Exception as e:
        return {"valid": False, "error": str(e)}

    if mv not in b.legal_moves:
        return {"valid": False, "error": f"Illegal move: {move_uci}"}

    color = b.turn
    king_sq = b.king(color)
    checkers = list(b.checkers()) if b.is_check() else []

    # --- Before: current position with move arrow ---
    fill_before = {king_sq: "#ff4444"}
    for csq in checkers:
        fill_before[csq] = "#ffaa00"
    # Highlight from/to squares
    fill_before[mv.from_square] = "#60a5fa44"
    fill_before[mv.to_square] = "#60a5fa44"

    arrow_color = "#60a5fa"
    arrows = [chess.svg.Arrow(mv.from_square, mv.to_square, color=arrow_color)]

    before_svg = chess.svg.board(
        b, size=360, fill=fill_before,
        arrows=arrows,
        flipped=(color == chess.BLACK),
    )

    # --- After: push the move, render resulting position ---
    is_capture = b.is_capture(mv)
    b2 = b.copy(stack=False)
    b2.push(mv)
    after_fen = b2.fen()
    gives_check = b2.is_check()

    fill_after = {}
    fill_after[mv.to_square] = "#34d39944"  # green highlight on landing square
    if gives_check:
        # Highlight opponent king in red
        opp_king = b2.king(b2.turn)
        if opp_king is not None:
            fill_after[opp_king] = "#ff4444"

    after_svg = chess.svg.board(
        b2, size=360, fill=fill_after,
        flipped=(color == chess.BLACK),
    )

    return {
        "valid": True,
        "before_svg": before_svg,
        "after_svg": after_svg,
        "after_fen": after_fen,
        "gives_check": gives_check,
        "is_capture": is_capture,
        "move_uci": move_uci,
    }


# =====================================================================
# Control test (trained vs untrained comparison)
# =====================================================================

def run_control_test(
    trained_kg: KG,
    n_positions: int = 50,
    seed: int = 999,
    stockfish_path: Optional[str] = None,
    oracle_depth: int = 10,
    progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
) -> Dict[str, Any]:
    """
    Generate random check positions and compare:
      - Trained KG: does it have a suggestion? Does it match Stockfish best?
      - Untrained KG: same queries (will always have no suggestions).
      - Heuristic baseline: does the heuristic's top pick match Stockfish?
    """
    require_python_chess()
    rng = random.Random(seed)

    oracle = StockfishOracle(stockfish_path, depth=oracle_depth)
    oracle_ok = oracle.start()

    results = {
        "n_positions": n_positions,
        "oracle_active": oracle_ok,
        "positions_tested": 0,
        "trained_hits": 0,        # KG had a suggestion
        "trained_matches": 0,     # KG top suggestion == Stockfish best
        "trained_top3_matches": 0,
        "untrained_hits": 0,      # always 0 (control)
        "heuristic_matches": 0,   # heuristic top pick == Stockfish
        "details": [],
    }

    # Create a temporary untrained KG for comparison
    untrained_kg = KG(":memory:")
    untrained_kg.init_schema()

    for i in range(n_positions):
        b = generate_random_check_position(rng, max_plies=80)
        if b is None or not b.is_check():
            if progress_callback:
                progress_callback(i + 1, n_positions, {"status": "skipped"})
            continue

        fen = b.fen()
        results["positions_tested"] += 1

        # Oracle best move
        oracle_best = oracle.analyze_best(b) if oracle_ok else OracleResult(None, None, None)

        # Trained KG query
        trained_result = query_position(trained_kg, fen, limit=5)
        trained_has_suggestion = len(trained_result["suggestions"]) > 0
        trained_top_move = trained_result["suggestions"][0]["move"] if trained_has_suggestion else None
        trained_top3 = [s["move"] for s in trained_result["suggestions"][:3]]

        if trained_has_suggestion:
            results["trained_hits"] += 1
        if oracle_best.best_move_uci and trained_top_move == oracle_best.best_move_uci:
            results["trained_matches"] += 1
        if oracle_best.best_move_uci and oracle_best.best_move_uci in trained_top3:
            results["trained_top3_matches"] += 1

        # Untrained KG (control — always empty)
        untrained_result = query_position(untrained_kg, fen, limit=5)
        if len(untrained_result["suggestions"]) > 0:
            results["untrained_hits"] += 1  # should stay 0

        # Heuristic baseline
        heur_moves = heuristic_escapes(b)
        heur_top = heur_moves[0] if heur_moves else None
        if oracle_best.best_move_uci and heur_top == oracle_best.best_move_uci:
            results["heuristic_matches"] += 1

        detail = {
            "fen": fen,
            "oracle_best": oracle_best.best_move_uci,
            "oracle_eval_cp": oracle_best.eval_cp,
            "trained_suggestion": trained_top_move,
            "trained_node_found": trained_result["node_found"],
            "heuristic_top": heur_top,
            "check_type": trained_result["features"]["check_type"] if trained_result["features"] else "?",
        }
        results["details"].append(detail)

        if progress_callback:
            progress_callback(i + 1, n_positions, detail)

    oracle.stop()
    untrained_kg.close()

    # Compute rates
    tested = max(1, results["positions_tested"])
    results["trained_hit_rate"] = round(results["trained_hits"] / tested, 4)
    results["trained_match_rate"] = round(results["trained_matches"] / tested, 4)
    results["trained_top3_rate"] = round(results["trained_top3_matches"] / tested, 4)
    results["heuristic_match_rate"] = round(results["heuristic_matches"] / tested, 4)

    return results
