"""
学習済みAIエージェント同士を対戦させ、その棋譜（ゲームログ）を収集するスクリプト。

このスクリプトは、ローカルに保存された学習済みモデル（D3QN）を
4体のエージェントにロードし、シミュレーションを実行します。
生成されたデータは、観戦用アニメーション（spectate.py）の入力として使用できます。

- モデル読込元: <プロジェクトルート>/models/101_d3qn.pth
- データ保存先: <プロジェクトルート>/data/game_data_for_spectate.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from one_o_one.game import Action, Card, State, reset, step

logger = logging.getLogger(__name__)

# --- 定数と設定 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "101_d3qn.pth"
OUTPUT_DATA_PATH = PROJECT_ROOT / "data" / "game_data_for_spectate.json"

NUM_GAMES = 1  # 生成するゲーム数
STATE_SIZE = 35
ACTION_SIZE = 4

# --- AIモデルと状態ベクトル化関数 (train_ai.pyから再利用) ---


def _to_one_hot(num: int, max_val: int) -> list[float]:
    """指定された値をOne-Hotエンコーディングする。"""
    return [1.0 if i == num else 0.0 for i in range(max_val)]


def _rank_value(card: Card | None) -> int:
    return int(card.rank) if card is not None else 0


def get_vector(s: State) -> np.ndarray:
    """ゲーム状態をAIの入力となる固定長のベクトルに変換する。"""
    me = s.players[s.public.turn]
    hand_vec = _to_one_hot(_rank_value(me.hand[0]), 16) + _to_one_hot(
        _rank_value(me.hand[1]), 16
    )
    total_vec = [s.public.total / 101.0]
    dir_vec = [1.0 if s.public.direction == 1 else 0.0]
    penalty_vec = [(s.public.penalty_level - 1) / 5.0]
    state_vector = np.array(
        hand_vec + total_vec + dir_vec + penalty_vec, dtype=np.float32
    )
    return np.pad(
        state_vector,
        (0, STATE_SIZE - len(state_vector)),
        "constant",
        constant_values=0,
    )


class DuelingDQN(nn.Module):  # type: ignore[misc]
    """Dueling Networkアーキテクチャを持つDQNモデル。"""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.feature_layer = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU())
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_size)
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.feature_layer(x)
        advantages: torch.Tensor = self.advantage_stream(features)
        values: torch.Tensor = self.value_stream(features)
        qvals: torch.Tensor = values + (
            advantages - advantages.mean(dim=1, keepdim=True)
        )
        return qvals


class AIAgent:
    """学習済みモデルを使用して最適な行動を選択するエージェント。"""

    def __init__(self, model_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQN(STATE_SIZE, ACTION_SIZE).to(self.device)

        if not model_path.exists():
            raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_path}")

        logger.info("Loading model from: %s", model_path)
        self.policy_net.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.policy_net.eval()

    def select_action(self, state: np.ndarray) -> Action:
        """現在の状態で最も評価値が高い行動を選択する。"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values: torch.Tensor = self.policy_net(state_tensor)
            return Action(q_values.argmax().item())


# --- データ収集メイン処理 ---


def collect_game_data(agent: AIAgent, num_games: int) -> list[dict[str, object]]:
    """指定された数のゲームシミュレーションを行い、ログを収集する。"""
    all_games_data = []
    for i in range(num_games):
        logger.info("Simulating game %d/%d...", i + 1, num_games)
        s = reset(4)
        game_log = []
        done = False
        turn = 0

        while not done:
            state_vec = get_vector(s)
            action = agent.select_action(state_vec)
            next_s, reward, done, _ = step(s, action)

            # 観戦用に詳細なログを記録
            cur_idx = s.public.turn
            if action in (Action.PLAY_HAND_0, Action.PLAY_HAND_1):
                card = s.players[cur_idx].hand[action.value]
                played = card.rank.name if card is not None else "NONE"
            else:
                played = "DECK"
            log_entry = {
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
            game_log.append(log_entry)
            s = next_s
            turn += 1

        all_games_data.extend(game_log)
        logger.info("Game %d finished after %d turns.", i + 1, turn)

    return all_games_data


def main() -> None:
    """メイン関数：モデルをロードし、データ収集を実行して保存する。"""
    # 出力ディレクトリの作成
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        agent = AIAgent(MODEL_PATH)
        game_data = collect_game_data(agent, NUM_GAMES)

        # 収集したデータをJSONファイルとして保存
        with open(OUTPUT_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(game_data, f, indent=4)

        logger.info("Successfully collected and saved data for %d game(s).", NUM_GAMES)
        logger.info("Data saved to: %s", OUTPUT_DATA_PATH)

    except FileNotFoundError as e:
        logger.error("Error: %s", e)
        logger.error(
            "Please run the training script (train_ai.py) first to generate the model "
            "file."
        )
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
