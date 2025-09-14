# 101

## 1\. 🤖 AIの訓練 (`train_ai.py`)

まず、AIエージェントを学習させ、モデルファイルを作成します。この処理は完了までに時間がかかることがあります。

```bash
uv run python scripts/train_ai.py
```

実行が始まると、ターミナルに以下のような進捗状況が表示されます。

```
Using device: cpu
Episode 0, Total Reward: -1.0, Epsilon: 0.9950
Episode 100, Total Reward: -1.0, Epsilon: 0.6017
...
```

学習が進むと、プロジェクト内に`models/101_d3qn.pth`というモデルファイルと、`data/episodes/`ディレクトリに学習過程のデータが保存されます。

-----

## 2\. 📊 観戦用データの収集 (`collect_data.py`)

次に、ステップ1で学習させたモデルを使ってAI同士の対戦をシミュレートし、観戦用のゲームログを生成します。

**（注意：このコマンドは、ステップ1が完了し、`models/101_d3qn.pth`が作成された後に実行してください。）**

```bash
uv run python scripts/collect_data.py
```

成功すると、以下のメッセージが表示され、`data/game_data_for_spectate.json`ファイルが作成されます。

```
Loading model from: <...>/101-ai-project/models/101_d3qn.pth
Simulating game 1/1...
Game 1 finished after 58 turns.

Successfully collected and saved data for 1 game(s).
Data saved to: <...>/101-ai-project/data/game_data_for_spectate.json
```

-----

## 3\. 🎬 アニメーションの観戦 (`spectate.py`)

最後に、ステップ2で生成したゲームログを読み込み、ブラウザで観戦用アニメーションを再生します。

**（注意：このコマンドは、ステップ2が完了し、`data/game_data_for_spectate.json`が作成された後に実行してください。）**

```bash
uv run python scripts/spectate.py
```

実行すると、ターミナルに完了メッセージが表示され、自動的にデフォルトのWebブラウザが起動して`game_replay.html`が開き、対戦アニメーションが始まります。

```
✅ Replay HTML file has been generated: game_replay.html
```

-----

## テストの実行

作成したテストコード（`tests/test_game.py`）を実行するには、`uv run pytest`コマンドを使用します。

```bash
uv run pytest
```

すべてのテストがパスすれば、ゲームのコアロジックが正しく実装されていることを確認できます。

```
============================= test session starts ==============================
...
tests/test_game.py ........                                              [100%]

============================== 8 passed in 0.01s ===============================
```