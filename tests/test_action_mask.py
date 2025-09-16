from one_o_one.game import (
    Action,
    Card,
    PlayerState,
    PublicState,
    Rank,
    State,
    action_mask,
    step,
)


def _mk_state(
    *,
    hand: tuple[Rank | None, Rank | None],
    deck: tuple[Rank, ...],
    turn: int = 0,
    num_players: int = 2,
    total: int = 0,
) -> State:
    """Construct a minimal ``State`` for mask testing."""

    players = []
    for i in range(num_players):
        if i == turn:
            h = (
                Card(hand[0]) if hand[0] is not None else None,
                Card(hand[1]) if hand[1] is not None else None,
            )
        else:
            h = (Card(Rank.R4), Card(Rank.R5))
        players.append(PlayerState(lp=10, hand=h))

    public = PublicState(
        turn=turn,
        direction=1,
        total=total,
        penalty_level=1,
        deck=tuple(Card(r) for r in deck),
        discard=tuple(),
        last_player=None,
    )

    alive = tuple(True for _ in range(num_players))
    return State(players=tuple(players), public=public, alive=alive)


def test_mask_allows_only_legal_actions() -> None:
    state = _mk_state(hand=(Rank.R2, Rank.R3), deck=(Rank.R4,))
    mask = action_mask(state)
    assert len(mask) == len(Action)
    assert mask[Action.PLAY_HAND_0] == 1
    assert mask[Action.PLAY_HAND_1] == 1
    assert mask[Action.PLAY_DECK] == 1
    assert mask[Action.PLAY_TEN_PLUS] == 0
    assert mask[Action.PLAY_TEN_MINUS] == 0
    assert mask[Action.PLAY_ACE_ONE] == 0
    assert mask[Action.PLAY_ACE_ELEVEN] == 0

    state = _mk_state(hand=(Rank.R2, None), deck=(Rank.R4,))
    mask = action_mask(state)
    assert mask[Action.PLAY_HAND_0] == 1
    assert mask[Action.PLAY_HAND_1] == 0
    assert mask[Action.PLAY_DECK] == 1

    state = _mk_state(hand=(Rank.R2, None), deck=tuple())
    mask = action_mask(state)
    assert mask[Action.PLAY_DECK] == 0


def test_mask_restricts_to_choice_actions_during_pending() -> None:
    state = _mk_state(hand=(Rank.R10, Rank.R3), deck=(Rank.R4,), total=50)
    choice_state, _, _, _ = step(state, Action.PLAY_HAND_0)
    mask = action_mask(choice_state)
    assert mask[Action.PLAY_TEN_PLUS] == 1
    assert mask[Action.PLAY_TEN_MINUS] == 1
    assert mask[Action.PLAY_HAND_0] == 0
    assert mask[Action.PLAY_DECK] == 0


def test_minus_choice_blocked_when_total_low() -> None:
    state = _mk_state(hand=(Rank.R10, Rank.R3), deck=(Rank.R4,), total=9)
    choice_state, _, _, _ = step(state, Action.PLAY_HAND_0)
    mask = action_mask(choice_state)
    assert mask[Action.PLAY_TEN_PLUS] == 1
    assert mask[Action.PLAY_TEN_MINUS] == 0
