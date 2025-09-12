"""
カードゲーム「101」のAIエージェントを強化学習で訓練するスクリプト。

Dueling Double Deep Q-Network (D3QN) を使用し、
自己対戦を通じてゲームの最適戦略を学習します。

学習データは以下のディレクトリに保存されます:
- エピソードデータ: <プロジェクトルート>/data/episodes/
- モデルの重み: <プロジェクトルート>/models/
"""

import random
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from one_o_one.game import (
    Action,
    State,
    step,
)

# 将来の非推奨に関する警告を非表示にする
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# --- 定数と設定 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPISODE_DATA_DIR = PROJECT_ROOT / "data" / "episodes"
MODEL_SAVE_DIR = PROJECT_ROOT / "models"

# 訓練パラメータ
NUM_EPISODES = 10000  # 訓練エピソード数
REPLAY_BUFFER_SIZE = 10000  # リプレイバッファのサイズ
BATCH_SIZE = 128
GAMMA = 0.99  # 割引率
EPS_START = 1.0  # ε-greedy法の開始ε
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10  # ターゲットネットワークの更新頻度
LEARNING_RATE = 0.001
STATE_SIZE = 35  # 状態ベクトルのサイズ
ACTION_SIZE = 4  # アクションの数


# --- 状態のベクトル化 ---


def _to_one_hot(num: int, max_val: int) -> list[float]:
    """指定された値をOne-Hotエンコーディングする。"""
    return [1.0 if i == num else 0.0 for i in range(max_val)]


def get_vector(s: State) -> np.ndarray:
    """ゲーム状態をAIの入力となる固定長のベクトルに変換する。"""
    me = s.players[s.public.current_player]
    hand_vec = _to_one_hot(me.hand[0].value, 14) + _to_one_hot(me.hand[1].value, 14)
    total_vec = [s.public.total / 101.0]
    dir_vec = [1.0 if s.public.direction == 1 else 0.0]
    penalty_vec = [(s.public.penalty - 1) / 5.0]

    # 状態ベクトルを結合
    state_vector = np.array(
        hand_vec + total_vec + dir_vec + penalty_vec, dtype=np.float32
    )
    # 期待されるサイズになるようにパディング
    return np.pad(
        state_vector,
        (0, STATE_SIZE - len(state_vector)),
        "constant",
        constant_values=0,
    )


# --- AIモデル (Dueling DQN) ---


class DuelingDQN(nn.Module):
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
        features = self.feature_layer(x)
        advantages = self.advantage_stream(features)
        values = self.value_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


# --- リプレイバッファ ---


class ReplayBuffer:
    """経験を保存し、ランダムサンプリングするためのバッファ。"""

    def __init__(self, capacity: int):
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.buffer, batch_size), strict=False
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# --- AIエージェント ---


class DQNAgent:
    """DQNアルゴリズムを実装したエージェント。"""

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.epsilon = EPS_START

    def select_action(self, state: np.ndarray) -> Action:
        """ε-greedy法に基づいて行動を選択する。"""
        if random.random() < self.epsilon:
            return Action(random.randint(0, self.action_size - 1))

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return Action(q_values.argmax().item())

    def learn(self):
        """リプレイバッファからの経験を用いてネットワークを更新する。"""
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(-1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.ByteTensor(dones).unsqueeze(-1).to(self.device)

        # Double DQN
        with torch.no_grad():
            best_actions = self.policy_net(next_states_t).argmax(1).unsqueeze(-1)
            next_q_values = self.target_net(next_states_t).gather(1, best_actions)

        target_q_values = rewards_t + (GAMMA * next_q_values * (1 - dones_t))

        current_q_values = self.policy_net(states_t).gather(1, actions_t)

        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        """εを減衰させる。"""
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)


# --- 自己対戦と学習ループ ---


def main():
    """学習のメインループを実行する。"""
    # 保存用ディレクトリを作成
    EPISODE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    model_path = MODEL_SAVE_DIR / "101_d3qn.pth"
    if model_path.exists():
        print(f"Loading model from {model_path}")
        agent.policy_net.load_state_dict(torch.load(model_path))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

    all_episode_data = []

    for episode in range(NUM_EPISODES):
        s: State = State.initial()
        episode_reward = 0
        episode_data: list[dict] = []
        done = False

        while not done:
            current_player = s.public.current_player
            state_vec = get_vector(s)
            action = agent.select_action(state_vec)

            next_s, reward, done, _ = step(s, action)
            next_state_vec = get_vector(next_s)

            agent.memory.push(state_vec, action.value, reward, next_state_vec, done)
            agent.learn()

            episode_reward += reward
            episode_data.append({"state": state_vec, "action": action.value})
            s = next_s

        all_episode_data.extend(episode_data)
        agent.update_epsilon()

        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if episode % 100 == 0:
            print(
                f"Episode {episode}, Total Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}"
            )
            # モデルを保存
            torch.save(agent.policy_net.state_dict(), model_path)

            # エピソードデータをParquet形式で保存
            df = pd.DataFrame(all_episode_data)
            df.to_parquet(EPISODE_DATA_DIR / f"episode_data_{episode}.parquet")
            all_episode_data = []

    print("Training finished.")
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Final model saved to {model_path}")


if __name__ == "__main__":
    main()
