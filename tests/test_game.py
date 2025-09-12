from one_o_one.game import (
    Action,
    Card,
    PlayerState,
    PublicState,
    Rank,
    State,
    step,
)


def _mk_state_for_test(
    total: int,
    my_hand: tuple[Rank, Rank],
    deck: tuple[Rank, ...] = (Rank.R2, Rank.R3, Rank.R4),
    turn: int = 0,
    num_players: int = 4,
    direction: int = 1,
    penalty_level: int = 1,
) -> State:
    players = []
    for i in range(num_players):
        if i == turn:
            hand = (Card(my_hand[0]), Card(my_hand[1]))
        else:
            hand = (Card(Rank.R4), Card(Rank.R5))
        players.append(PlayerState(lp=10, hand=hand))
    public = PublicState(
        turn=turn,
        direction=direction,
        total=total,
        penalty_level=penalty_level,
        deck=tuple(Card(r) for r in deck),
        discard=tuple(),
        last_player=None,
    )
    alive = tuple(True for _ in range(num_players))
    return State(players=tuple(players), public=public, alive=alive)


def test_pass_card_does_not_change_total() -> None:
    initial_state = _mk_state_for_test(total=50, my_hand=(Rank.R8, Rank.R2))
    next_state, _, _, _ = step(initial_state, Action.PLAY_HAND_0)
    assert next_state.public.total == 50


def test_reverse_card_reverses_direction() -> None:
    state_forward = _mk_state_for_test(
        total=50, my_hand=(Rank.R9, Rank.R2), direction=1
    )
    state_after, _, _, _ = step(state_forward, Action.PLAY_HAND_0)
    assert state_after.public.direction == -1


def test_ten_card_moves_total_toward_101() -> None:
    initial_state = _mk_state_for_test(total=50, my_hand=(Rank.R10, Rank.R2))
    next_state, _, _, _ = step(initial_state, Action.PLAY_HAND_0)
    assert next_state.public.total == 60


def test_king_card_adds_30_to_total() -> None:
    initial_state = _mk_state_for_test(total=50, my_hand=(Rank.K, Rank.R2))
    next_state, _, _, _ = step(initial_state, Action.PLAY_HAND_0)
    assert next_state.public.total == 80


def test_bust_reduces_lp_and_resets_board() -> None:
    initial_state = _mk_state_for_test(
        total=100, my_hand=(Rank.R2, Rank.R3), penalty_level=3
    )
    next_state, _, done, _ = step(initial_state, Action.PLAY_HAND_0)
    assert done is False
    assert next_state.players[0].lp == 7
    assert next_state.public.total == 0
    assert next_state.public.penalty_level == 1


def test_exact_101_resets_total_and_increases_penalty() -> None:
    initial_state = _mk_state_for_test(
        total=100, my_hand=(Rank.A, Rank.R3), penalty_level=3
    )
    next_state, _, done, _ = step(initial_state, Action.PLAY_HAND_0)
    assert done is False
    assert next_state.public.total == 2
    assert next_state.public.penalty_level == 4
    assert next_state.players[0].lp == 10


def test_player_draws_card_after_playing() -> None:
    initial_state = _mk_state_for_test(
        total=50,
        my_hand=(Rank.R2, Rank.R3),
        deck=(Rank.R10, Rank.J, Rank.Q),
    )
    next_state, _, _, _ = step(initial_state, Action.PLAY_HAND_0)
    new_hand = next_state.players[0].hand
    ranks = [c.rank for c in new_hand if c is not None]
    assert Rank.R2 not in ranks
    assert Rank.R3 in ranks
    assert Rank.R10 in ranks
    assert len(next_state.public.deck) == 2


def test_turn_progression() -> None:
    state_forward = _mk_state_for_test(
        total=50, my_hand=(Rank.R2, Rank.R3), turn=0, direction=1
    )
    next_state_f, _, _, _ = step(state_forward, Action.PLAY_HAND_0)
    assert next_state_f.public.turn == 1

    state_loop = _mk_state_for_test(
        total=50,
        my_hand=(Rank.R2, Rank.R3),
        num_players=4,
        turn=3,
        direction=1,
    )
    next_state_l, _, _, _ = step(state_loop, Action.PLAY_HAND_0)
    assert next_state_l.public.turn == 0

    state_backward = _mk_state_for_test(
        total=50, my_hand=(Rank.R2, Rank.R3), turn=1, direction=-1
    )
    next_state_b, _, _, _ = step(state_backward, Action.PLAY_HAND_0)
    assert next_state_b.public.turn == 0
