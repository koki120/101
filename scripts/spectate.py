"""
収集したゲームデータをもとに、対戦の様子をアニメーションで観戦するスクリプト。

'collect_data.py'で生成されたJSONファイルを読み込み、
ゲームの進行を視覚化するスタンドアロンのHTMLファイルを生成します。
実行後、自動的にデフォルトのWebブラウザでHTMLファイルが開きます。

- データ読込元: <プロジェクトルート>/data/game_data_for_spectate.json
- HTML出力先: <プロジェクトルート>/game_replay.html
"""

import json
import logging

# import webbrowser
from pathlib import Path

# --- 定数と設定 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "game_data_for_spectate.json"
OUTPUT_HTML_PATH = PROJECT_ROOT / "game_replay.html"


logger = logging.getLogger(__name__)


# --- HTML, CSS, JavaScript テンプレート ---
# ノートブックのコードをそのままPythonのf-stringに埋め込みます
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>101 Game Replay</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #333;
            color: #fff;
        }}
        .game-container {{
            display: grid;
            grid-template-areas:
                ". p1 ."
                "p0 board p2"
                ". p3 .";
            grid-template-columns: 200px 300px 200px;
            grid-template-rows: 180px 300px 180px;
            gap: 20px;
        }}
        .player-area {{
            border: 2px solid #555;
            border-radius: 10px;
            padding: 10px;
            background-color: #444;
            transition: all 0.3s ease;
            text-align: center;
        }}
        .player-area.active {{
            border-color: #ffc107;
            box-shadow: 0 0 15px #ffc107;
        }}
        .board {{
            grid-area: board;
            border: 2px solid #777;
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background-color: #2c2c2c;
            position: relative;
        }}
        #p0 {{ grid-area: p0; }} #p1 {{ grid-area: p1; }}
        #p2 {{ grid-area: p2; }} #p3 {{ grid-area: p3; }}
        .total-display {{ font-size: 4em; font-weight: bold; }}
        .penalty-display {{ font-size: 1.2em; margin-top: 10px; color: #ff6b6b; }}
        .card {{
            width: 60px; height: 90px;
            border: 1px solid #ccc; border-radius: 5px;
            background-color: #fff; color: #333;
            display: inline-flex; justify-content: center; align-items: center;
            font-size: 1.5em; font-weight: bold;
            margin: 5px;
        }}
        .hand {{ margin-top: 10px; }}
        .player-info {{ margin-bottom: 10px; }}
        .lp-bar-container {{
            width: 100%; background-color: #555;
            border-radius: 5px; overflow: hidden; height: 10px;
        }}
        .lp-bar {{
            height: 100%; background-color: #4caf50;
            transition: width 0.5s ease-in-out;
        }}
        #event-display {{
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            font-size: 3em;
            font-weight: bold;
            color: #ffeb3b;
            background-color: rgba(0,0,0,0.7);
            padding: 20px;
            border-radius: 15px;
            opacity: 0;
            transition: opacity 0.5s;
            pointer-events: none; /* マウスイベントを無視 */
        }}
    </style>
</head>
<body>
    <div class="game-container">
        <div id="board" class="board">
            <div id="total-display" class="total-display">0</div>
            <div id="penalty-display" class="penalty-display">Penalty: x1</div>
            <div id="event-display"></div>
        </div>
        <div id="p0" class="player-area"></div>
        <div id="p1" class="player-area"></div>
        <div id="p2" class="player-area"></div>
        <div id="p3" class="player-area"></div>
    </div>

    <script>
        const gameData = {game_data_json};

        const NUM_PLAYERS = 4;
        const totalDisplay = document.getElementById('total-display');
        const penaltyDisplay = document.getElementById('penalty-display');
        const eventDisplay = document.getElementById('event-display');
        // プレイヤー領域の取得
        // Array.from にオブジェクトを渡すと空配列となるため、
        // プレイヤー数に応じて DOM 要素を確実に取得する
        const playerAreas = Array.from(
            {{ length: NUM_PLAYERS }},
            (_, i) => document.getElementById(`p${{i}}`)
        );

        let currentTurn = 0;

        function showEvent(message) {{
            eventDisplay.textContent = message;
            eventDisplay.style.opacity = 1;
            setTimeout(() => {{ eventDisplay.style.opacity = 0; }}, 1500);
        }}

        function updateUI(data) {{
            totalDisplay.textContent = data.total_after;
            penaltyDisplay.textContent = `Penalty: x${{data.penalty}}`;

            playerAreas.forEach((area, i) => {{
                area.classList.remove('active');
                const lp = data.lp_before_json[i];
                const hand = data.hands_before_json[i];
                area.innerHTML = `
                    <div class="player-info">
                        <strong>Player ${{i}}</strong>
                        <div>LP: ${{lp}}</div>
                    </div>
                    <div class="lp-bar-container">
                        <div class="lp-bar" style="width: ${{lp * 10}}%"></div>
                    </div>
                    <div class="hand">
                        <div class="card">${{hand[0]}}</div>
                        <div class="card">${{hand[1]}}</div>
                    </div>
                `;
            }});

            const activePlayer = playerAreas[data.player];
            if (activePlayer) {{
                activePlayer.classList.add('active');
            }}

            if (data.total_before > 101) {{ showEvent("BURST!"); }}
            if (data.total_before === 101) {{ showEvent("RESET!"); }}
        }}

        function gameLoop() {{
            if (currentTurn >= gameData.length) {{
                showEvent("GAME SET");
                clearInterval(gameInterval);
                return;
            }}
            updateUI(gameData[currentTurn]);
            currentTurn++;
        }}

        function initializeBoard() {{
            if (!gameData || gameData.length === 0) return;
            const initialData = gameData[0];
            const lps = initialData.lp_before_json || [10,10,10,10];
            const hands = initialData.hands_before_json || [];
            for (let i = 0; i < NUM_PLAYERS; i++) {{
                const hand = hands[i] || ['?', '?'];
                playerAreas[i].innerHTML = `
                    <div class="player-info">
                        <strong>Player ${{i}}</strong>
                        <div>LP: ${{lps[i] || 10}}</div>
                    </div>
                    <div class="lp-bar-container">
                        <div class="lp-bar" style="width: ${{lps[i] * 10}}%"></div>
                    </div>
                    <div class="hand">
                        <div class="card">
                            ${{hand[0] !== undefined ? hand[0] : '?'}}
                        </div>
                        <div class="card">
                            ${{hand[1] !== undefined ? hand[1] : '?'}}
                        </div>
                    </div>
                `;
            }}
        }}

        initializeBoard();
        const gameInterval = setInterval(gameLoop, 800); // 0.8秒ごとに進行
    </script>
</body>
</html>
"""


def create_and_show_replay() -> None:
    """ゲームデータからHTMLリプレイファイルを生成し、ブラウザで表示する。"""
    try:
        # データを読み込む
        with open(DATA_PATH, encoding="utf-8") as f:
            game_data = json.load(f)

        # データをJSON文字列に変換してHTMLテンプレートに埋め込む
        game_data_json_string = json.dumps(game_data, ensure_ascii=False)
        final_html = HTML_TEMPLATE.format(game_data_json=game_data_json_string)

        # HTMLファイルとして保存
        with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
            f.write(final_html)

        logger.info(
            "Replay HTML file has been generated: %s",
            OUTPUT_HTML_PATH.name,
        )

        # ブラウザで開く
        # webbrowser.open_new_tab(OUTPUT_HTML_PATH.as_uri())

    except FileNotFoundError:
        logger.error("Error: Data file not found at '%s'", DATA_PATH)
        logger.error("Please run 'collect_data.py' first to generate the game data.")
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_and_show_replay()
