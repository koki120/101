"""
AIモデルが存在しない場合に、初期状態のモデルファイルを生成するスクリプト。

このスクリプトを実行すると、'models/101_d3qn.pth'が存在しない場合、
ランダムな重みを持つ新しいモデルが作成されます。
これにより、`train_ai.py`での学習を開始する前や、
`collect_data.py`を学習済みモデルなしで実行したい場合に便利です。
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

# --- 定数と設定 (train_ai.py, collect_data.pyと共通) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SAVE_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_SAVE_DIR / "101_d3qn.pth"
STATE_SIZE = 35
ACTION_SIZE = 3

logger = logging.getLogger(__name__)


# --- AIモデル定義 (train_ai.pyから流用) ---
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


def initialize_model() -> None:
    """モデルが存在しない場合に初期モデルを生成する。"""
    try:
        # モデル保存ディレクトリが存在しない場合は作成
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        if MODEL_PATH.exists():
            logger.info("Model already exists at '%s'. No action taken.", MODEL_PATH)
            return

        logger.info("Model not found. Initializing a new model...")

        # デバイスを設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 新しいモデルをインスタンス化
        initial_model = DuelingDQN(STATE_SIZE, ACTION_SIZE).to(device)
        initial_model.eval()  # 推論モードに設定

        # モデルの初期状態を保存
        torch.save(initial_model.state_dict(), MODEL_PATH)

        logger.info(
            "Successfully created and saved an initial model to '%s'", MODEL_PATH
        )

    except Exception as e:
        logger.error("An unexpected error occurred during model initialization: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_model()
