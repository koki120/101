from one_o_one.game import State, PublicState, PlayerState, Card, Rank, Action, step


def test_penalty_level_preserved_on_deck_exhaustion():
    players = (
        PlayerState(lp=10, hand=(Card(Rank.R2), Card(Rank.R3))),
        PlayerState(lp=10, hand=(Card(Rank.R4), Card(Rank.R5))),
    )
    public = PublicState(
        turn=0,
        direction=1,
        total=50,
        penalty_level=3,
        deck=tuple(),
        discard=tuple(),
        last_player=None,
    )
    state = State(players=players, public=public, alive=(True, True))

    next_state, _, done, info = step(state, Action.PLAY_HAND_0)

    assert info["event"] == "deck_exhausted"
    assert done is False
    assert next_state.public.penalty_level == 3
