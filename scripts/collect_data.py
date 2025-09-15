"""
学習済みAIエージェント同士を対戦させ、その棋譜（ゲームログ）を収集するスクリプト。

このスクリプトは、ローカルに保存されたTransformerベースの学習済みモデルを
4体のエージェントにロードし、シミュレーションを実行します。
生成されたデータは、観戦用アニメーション（spectate.py）の入力として使用できます。

- モデル読込元: <プロジェクトルート>/models/101_transformer.pth
- データ保存先: <プロジェクトルート>/data/game_data_for_spectate.json
"""

# mypy: disallow-subclassing-any=False

import json
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn

from one_o_one.game import Action, Card, State, action_mask, reset, step

logger = logging.getLogger(__name__)

# --- 定数と設定 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "101_transformer.pth"
OUTPUT_DATA_PATH = PROJECT_ROOT / "data" / "game_data_for_spectate.json"

NUM_GAMES = 1  # 生成するゲーム数
NUM_PLAYERS = 4
HISTORY_LENGTH = 8
ACTION_SIZE = 3
STATE_SIZE = 35 + (NUM_PLAYERS + ACTION_SIZE) * HISTORY_LENGTH

# --- AIモデルと状態ベクトル化関数 (train_ai.pyから再利用) ---


class LogEntry(TypedDict):
    """観戦用ログの1ターン分のデータ構造。"""

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


def _to_one_hot(num: int, max_val: int) -> list[float]:
    """指定された値をOne-Hotエンコーディングする。"""
    return [1.0 if i == num else 0.0 for i in range(max_val)]


def _rank_value(card: Card | None) -> int:
    return int(card.rank) if card is not None else 0


def encode_state(s: State) -> np.ndarray:
    """単一のゲーム状態をベクトルに変換する。"""
    me = s.players[s.public.turn]
    hand_vec = _to_one_hot(_rank_value(me.hand[0]), 16) + _to_one_hot(
        _rank_value(me.hand[1]), 16
    )
    total_vec = [s.public.total / 101.0]
    dir_vec = [1.0 if s.public.direction == 1 else 0.0]
    penalty_vec = [(s.public.penalty_level - 1) / 5.0]
    history_vec: list[float] = []
    recent = s.public.history[-HISTORY_LENGTH:]
    for player_idx, act in recent:
        history_vec.extend(_to_one_hot(player_idx, NUM_PLAYERS))
        history_vec.extend(_to_one_hot(act, ACTION_SIZE))
    missing = HISTORY_LENGTH - len(recent)
    history_vec.extend([0.0] * (missing * (NUM_PLAYERS + ACTION_SIZE)))

    state_vector = np.array(
        hand_vec + total_vec + dir_vec + penalty_vec + history_vec,
        dtype=np.float32,
    )
    return state_vector


def get_vector(
    states: list[State], actions: list[int], rewards: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ゲーム開始からの状態・行動・報酬のシーケンスを返す。"""
    state_vecs = [encode_state(st) for st in states]
    action_seq = actions + [0]
    reward_seq = rewards + [0.0]
    return (
        np.array(state_vecs, dtype=np.float32),
        np.array(action_seq, dtype=np.int64),
        np.array(reward_seq, dtype=np.float32),
    )


class DecisionTransformer(nn.Module):
    """状態・行動・報酬の系列から次の行動を予測するTransformerモデル。"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.state_embed = nn.Linear(state_size, hidden_size)
        self.action_embed = nn.Embedding(action_size, hidden_size)
        self.reward_embed = nn.Linear(1, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.predict_head = nn.Linear(hidden_size, action_size)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """各時刻の状態・行動・報酬から次の行動のロジットを計算する。"""
        state_tokens = self.state_embed(states)
        action_tokens = self.action_embed(actions)
        reward_tokens = self.reward_embed(rewards.unsqueeze(-1))
        tokens = state_tokens + action_tokens + reward_tokens
        tokens = tokens.unsqueeze(1)  # (T, 1, hidden)
        out = self.encoder(tokens)
        logits: torch.Tensor = self.predict_head(out.squeeze(1))
        return logits


class AIAgent:
    """学習済みTransformerモデルを使用して行動を選択するエージェント。"""

    def __init__(self, model_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DecisionTransformer(STATE_SIZE, ACTION_SIZE).to(self.device)

        if not model_path.exists():
            raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_path}")

        logger.info("Loading model from: %s", model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def select_action(
        self,
        state: State,
        state_seq: np.ndarray,
        action_seq: np.ndarray,
        reward_seq: np.ndarray,
    ) -> Action:
        """現在の状態で最も評価値が高い行動を選択する。"""
        with torch.no_grad():
            states_t = torch.FloatTensor(state_seq).to(self.device)
            actions_t = torch.LongTensor(action_seq).to(self.device)
            rewards_t = torch.FloatTensor(reward_seq).to(self.device)
            logits: torch.Tensor = self.model(states_t, actions_t, rewards_t)[-1]
            mask = torch.tensor(
                action_mask(state), dtype=torch.bool, device=self.device
            )
            logits[~mask] = -1e9
            return Action(int(logits.argmax().item()))


# --- データ収集メイン処理 ---


def collect_game_data(agent: AIAgent, num_games: int) -> list[LogEntry]:
    """指定された数のゲームシミュレーションを行い、ログを収集する。"""
    all_games_data: list[LogEntry] = []
    for i in range(num_games):
        logger.info("Simulating game %d/%d...", i + 1, num_games)
        s = reset(NUM_PLAYERS)
        game_log = []
        done = False
        turn = 0
        state_history: list[State] = []
        action_history: list[int] = []
        reward_history: list[float] = []

        while not done:
            state_history.append(s)
            state_seq, action_seq, reward_seq = get_vector(
                state_history, action_history, reward_history
            )
            action = agent.select_action(s, state_seq, action_seq, reward_seq)
            next_s, reward, done, _ = step(s, action)
            action_history.append(action.value)
            reward_history.append(reward)

            # 観戦用に詳細なログを記録
            cur_idx = s.public.turn
            if action in (Action.PLAY_HAND_0, Action.PLAY_HAND_1):
                card = s.players[cur_idx].hand[action.value]
                played = card.rank.name if card is not None else "NONE"
            else:
                played = "DECK"
            log_entry: LogEntry = {
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
