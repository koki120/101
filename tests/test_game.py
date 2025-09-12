"""
カードゲーム「101」のコアロジックのテストスイート。

このファイルには、ゲームのルールが正しく実装されていることを検証するための
ユニットテストが含まれています。Pytestフレームワークを使用して実行されます。
"""

from one_o_one.game import (
    Action,
    PlayerState,
    PublicState,
    Rank,
    State,
    step,
)


def _mk_state_for_test(
    total: int,
    my_hand: tuple[Rank, ...],
    deck: tuple[Rank, ...] = (Rank.R1, Rank.R2, Rank.R3),
    current_player: int = 0,
    num_players: int = 4,
    direction: int = 1,
    penalty: int = 1,
) -> State:
    """テストケースを簡単に作成するための状態生成関数。"""
    return State(
        public=PublicState(
            total=total,
            deck=deck,
            current_player=current_player,
            direction=direction,
            penalty=penalty,
            turn=0,
        ),
        players=[
            PlayerState(
                hand=(my_hand if i == current_player else (Rank.R4, Rank.R5)),
                lp=10,
            )
            for i in range(num_players)
        ],
    )


# --- ゲームルールのテストケース ---


def test_pass_card_does_not_change_total():
    """8（パス）のカードは、場の合計値を変更しないことをテストする。"""
    initial_state = _mk_state_for_test(total=50, my_hand=(Rank.R8, Rank.R2))
    next_state, _, _, _ = step(initial_state, Action.PLAY_HAND_0)
    assert next_state.public.total == 50, "8のカードで合計値が変動してはならない"


def test_reverse_card_reverses_direction():
    """9（リバース）のカードは、進行方向を反転させることをテストする。"""
    # 正方向 -> 逆方向
    state_forward = _mk_state_for_test(
        total=50, my_hand=(Rank.R9, Rank.R2), direction=+1
    )
    state_after_reverse, _, _, _ = step(state_forward, Action.PLAY_HAND_0)
    assert state_after_reverse.public.direction == -1, "進行方向が+1から-1に変わるべき"
    assert state_after_reverse.public.total == 50, (
        "9のカードで合計値が変動してはならない"
    )

    # 逆方向 -> 正方向
    state_backward = _mk_state_for_test(
        total=50, my_hand=(Rank.R9, Rank.R2), direction=-1
    )
    state_after_reverse_2, _, _, _ = step(state_backward, Action.PLAY_HAND_0)
    assert state_after_reverse_2.public.direction == +1, (
        "進行方向が-1から+1に変わるべき"
    )


def test_ten_card_subtracts_10_from_total():
    """10のカードは、場の合計値から10を減算することをテストする。"""
    initial_state = _mk_state_for_test(total=50, my_hand=(Rank.R10, Rank.R2))
    next_state, _, _, _ = step(initial_state, Action.PLAY_HAND_0)
    assert next_state.public.total == 40, "合計値が50から40に減るべき"


def test_king_card_sets_total_to_99():
    """K（キング）のカードは、場の合計値を99に設定することをテストする。"""
    # 合計が99より低い場合
    state_below_99 = _mk_state_for_test(total=50, my_hand=(Rank.RK, Rank.R2))
    state_after_king_1, _, _, _ = step(state_below_99, Action.PLAY_HAND_0)
    assert state_after_king_1.public.total == 99, "合計値が50から99になるべき"

    # 合計が99より高い場合（例：10の効果後など）
    state_above_99 = _mk_state_for_test(total=100, my_hand=(Rank.RK, Rank.R2))
    state_after_king_2, _, _, _ = step(state_above_99, Action.PLAY_HAND_0)
    assert state_after_king_2.public.total == 99, "合計値が100から99になるべき"


def test_bust_reduces_lp_and_resets_board():
    """合計値が101を超えた場合（バースト）、LPが減少し、場がリセットされることをテストする。"""
    initial_state = _mk_state_for_test(total=100, my_hand=(Rank.R2, Rank.R3), penalty=3)
    next_state, _, done, _ = step(initial_state, Action.PLAY_HAND_0)

    assert done is True, "バーストしたためラウンドが終了するべき"
    assert next_state.players[0].lp == 10 - 3, (
        "現在のペナルティレベル(3)分のダメージを受けるべき"
    )
    assert next_state.public.total == 0, "バースト後は合計値が0にリセットされるべき"
    assert next_state.public.penalty == 1, "バースト後はペナルティが1に戻るべき"


def test_exact_101_resets_total_and_increases_penalty():
    """合計値がちょうど101になった場合、合計値が0にリセットされ、ペナルティレベルが上昇することをテストする。"""
    initial_state = _mk_state_for_test(total=100, my_hand=(Rank.R1, Rank.R3), penalty=3)
    next_state, _, done, _ = step(initial_state, Action.PLAY_HAND_0)

    assert done is False, "101リセットではラウンドは終了しない"
    assert next_state.public.total == 0, "101リセット後は合計値が0にリセットされるべき"
    assert next_state.public.penalty == 3 + 1, (
        "101リセット後はペナルティが1増加するべき"
    )
    assert next_state.players[0].lp == 10, "101リセットではLPは減少しない"


def test_player_draws_card_after_playing():
    """プレイヤーはカードをプレイした後、山札から1枚カードを引くことをテストする。"""
    initial_state = _mk_state_for_test(
        total=50, my_hand=(Rank.R2, Rank.R3), deck=(Rank.R10, Rank.RJ, Rank.RQ)
    )
    next_state, _, _, _ = step(initial_state, Action.PLAY_HAND_0)  # R2をプレイ

    new_hand = next_state.players[0].hand
    assert Rank.R2 not in new_hand, "プレイしたカード(R2)が手札からなくなるべき"
    assert Rank.R3 in new_hand, "プレイしていないカード(R3)は手札に残るべき"
    assert Rank.R10 in new_hand, "山札の一番上のカード(R10)を引いているべき"
    assert len(next_state.public.deck) == 2, "山札が1枚減少しているべき"


def test_turn_progression():
    """手番が正しく次のプレイヤーに移ることをテストする。"""
    # 順方向 (+1)
    state_forward = _mk_state_for_test(
        total=50, my_hand=(Rank.R2, Rank.R3), current_player=0, direction=+1
    )
    next_state_f, _, _, _ = step(state_forward, Action.PLAY_HAND_0)
    assert next_state_f.public.current_player == 1, "手番がプレイヤー0から1に移るべき"

    # 順方向 (+1) でプレイヤーがループするケース
    state_loop = _mk_state_for_test(
        total=50,
        my_hand=(Rank.R2, Rank.R3),
        num_players=4,
        current_player=3,
        direction=+1,
    )
    next_state_l, _, _, _ = step(state_loop, Action.PLAY_HAND_0)
    assert next_state_l.public.current_player == 0, "手番がプレイヤー3から0に移るべき"

    # 逆方向 (-1)
    state_backward = _mk_state_for_test(
        total=50, my_hand=(Rank.R2, Rank.R3), current_player=1, direction=-1
    )
    next_state_b, _, _, _ = step(state_backward, Action.PLAY_HAND_0)
    assert next_state_b.public.current_player == 0, "手番がプレイヤー1から0に移るべき"
