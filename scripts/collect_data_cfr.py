"""Deep CFR data collection script.

This script loads a strategy network trained via ``scripts.train_cfr`` and
runs self-play games to generate replay data compatible with ``spectate.py``.

- Model load path: <project_root>/models/strategy_net.pt
- Output data path: <project_root>/data/game_data_for_spectate.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

import torch
from train_cfr import (
    ACTION_SIZE,
    NUM_PLAYERS,
    STATE_SIZE,
    StrategyNetwork,
    encode_state,
)

from one_o_one.game import Action, State, action_mask, reset, step

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "strategy_net.pt"
OUTPUT_DATA_PATH = PROJECT_ROOT / "data" / "game_data_for_spectate.json"

NUM_GAMES = 1


class LogEntry(TypedDict):
    """One turn of game data for replay."""

    turn: int
    player: int
    total_before: int
    action: str
    played_card: str
    total_after: int
    reward: float
    done: bool
    direction: int
    penalty: int
    lp_before_json: list[int]
    hands_before_json: list[list[str]]


class CFRPolicyAgent:
    """Wraps a strategy network to provide action selection."""

    def __init__(self, model_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StrategyNetwork(STATE_SIZE, ACTION_SIZE).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def select_action(self, state: State) -> Action:
        with torch.no_grad():
            state_vec = torch.tensor(encode_state(state), device=self.device)
            logits = self.model(state_vec)
            mask = torch.tensor(
                action_mask(state), dtype=torch.bool, device=self.device
            )
            logits[~mask] = -1e9
            return Action(int(logits.argmax().item()))


def collect_game_data(agent: CFRPolicyAgent, num_games: int) -> list[LogEntry]:
    """Run simulated games and gather replay logs."""
    all_data: list[LogEntry] = []
    for i in range(num_games):
        logger.info("Simulating game %d/%d...", i + 1, num_games)
        s = reset(NUM_PLAYERS)
        done = False
        turn = 0
        while not done:
            action = agent.select_action(s)
            next_s, reward, done, _ = step(s, action)
            cur_idx = s.public.turn
            if action in (Action.PLAY_HAND_0, Action.PLAY_HAND_1):
                card = s.players[cur_idx].hand[action.value]
                played = card.rank.name if card is not None else "NONE"
            else:
                played = "DECK"
            entry: LogEntry = {
                "turn": turn,
                "player": cur_idx,
                "total_before": s.public.total,
                "action": action.name,
                "played_card": played,
                "total_after": next_s.public.total,
                "reward": reward,
                "done": done,
                "direction": next_s.public.direction,
                "penalty": next_s.public.penalty_level,
                "lp_before_json": [p.lp for p in s.players],
                "hands_before_json": [
                    [c.rank.name if c is not None else "None" for c in p.hand]
                    for p in s.players
                ],
            }
            all_data.append(entry)
            s = next_s
            turn += 1
        logger.info("Game %d finished after %d turns.", i + 1, turn)
    return all_data


def main() -> None:
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        agent = CFRPolicyAgent(MODEL_PATH)
    except FileNotFoundError:
        logger.error("Model file not found: %s", MODEL_PATH)
        return
    data = collect_game_data(agent, NUM_GAMES)
    with open(OUTPUT_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info("Saved replay data for %d game(s) to %s", NUM_GAMES, OUTPUT_DATA_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
