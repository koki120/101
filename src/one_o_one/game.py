from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import NamedTuple

# ==== Card and action definitions ====


class Rank(IntEnum):
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7
    R8 = 8
    R9 = 9
    R10 = 10
    J = 11
    Q = 12
    K = 13
    A = 14
    JK = 15


class Card(NamedTuple):
    rank: Rank


class Action(IntEnum):
    PLAY_HAND_0 = 0
    PLAY_HAND_1 = 1
    PLAY_DECK = 2


@dataclass(frozen=True)
class PlayerState:
    lp: int
    hand: tuple[Card | None, Card | None]


@dataclass(frozen=True)
class PublicState:
    turn: int
    direction: int
    total: int
    penalty_level: int
    deck: tuple[Card, ...]
    discard: tuple[Card, ...]
    last_player: int | None


@dataclass(frozen=True)
class State:
    players: tuple[PlayerState, ...]
    public: PublicState
    alive: tuple[bool, ...]


# ==== Deck generation and shuffling ====


def _standard_deck() -> tuple[Card, ...]:
    ranks = [
        Rank.R2,
        Rank.R3,
        Rank.R4,
        Rank.R5,
        Rank.R6,
        Rank.R7,
        Rank.R8,
        Rank.R9,
        Rank.R10,
        Rank.J,
        Rank.Q,
        Rank.K,
        Rank.A,
    ]
    deck = [Card(r) for r in ranks for _ in range(4)]
    deck.extend([Card(Rank.JK), Card(Rank.JK)])
    return tuple(deck)


def _shuffle(deck: tuple[Card, ...], seed: int | None) -> tuple[Card, ...]:
    rng = random.Random(seed)
    d = list(deck)
    rng.shuffle(d)
    return tuple(d)


def _draw_top(deck: tuple[Card, ...]) -> tuple[Card | None, tuple[Card, ...]]:
    if not deck:
        return None, deck
    return deck[0], deck[1:]


def _next_turn_idx(turn: int, direction: int, alive: tuple[bool, ...]) -> int:
    n = len(alive)
    i = turn
    while True:
        i = (i + direction) % n
        if alive[i]:
            return i


def _first_alive(alive: tuple[bool, ...]) -> int:
    for i, a in enumerate(alive):
        if a:
            return i
    raise ValueError("No players are alive")


# ==== Initialization and legal actions ====

START_LP_DEFAULT = 10


def reset(
    num_players: int, start_lp: int = START_LP_DEFAULT, seed: int | None = None
) -> State:
    if num_players < 2:
        raise ValueError("The game requires at least two players")
    deck = _shuffle(_standard_deck(), seed)
    players: list[PlayerState] = []
    for _ in range(num_players):
        c1, deck = _draw_top(deck)
        c2, deck = _draw_top(deck)
        players.append(PlayerState(lp=start_lp, hand=(c1, c2)))
    public = PublicState(
        turn=0,
        direction=+1,
        total=0,
        penalty_level=1,
        deck=deck,
        discard=tuple(),
        last_player=None,
    )
    alive = tuple(True for _ in range(num_players))
    return State(players=tuple(players), public=public, alive=alive)


def legal_actions(state: State) -> tuple[Action, ...]:
    p = state.public
    me = state.players[p.turn]
    actions = []
    if me.hand[0] is not None:
        actions.append(Action.PLAY_HAND_0)
    if me.hand[1] is not None:
        actions.append(Action.PLAY_HAND_1)
    if p.deck:
        actions.append(Action.PLAY_DECK)
    return tuple(actions)


# ==== Card effect logic ====


class EffectResult(NamedTuple):
    total: int
    direction: int
    reset_triggered: bool
    counter_triggered: bool
    used_card: Card


def _prefer_plus(total: int) -> bool:
    plus = total + 10
    minus = total - 10

    def score(t: int) -> tuple[int, int]:
        return (0 if t <= 101 else 1, abs(101 - t))

    return score(plus) <= score(minus)


def _apply_card_effect(total: int, direction: int, card: Card) -> EffectResult:
    r = card.rank
    if r == Rank.R8:
        return EffectResult(total, direction, False, False, card)
    if r == Rank.R9:
        return EffectResult(total, -direction, False, False, card)
    if Rank.R2 <= r <= Rank.R7:
        new_total = total + int(r)
        return EffectResult(new_total, direction, new_total == 101, False, card)
    if r == Rank.R10:
        plus_total = total + 10
        minus_total = total - 10
        if total <= 9:
            new_total = plus_total
        else:
            new_total = plus_total if _prefer_plus(total) else minus_total
        return EffectResult(new_total, direction, new_total == 101, False, card)
    if r == Rank.J:
        new_total = total + 10
        return EffectResult(new_total, direction, new_total == 101, False, card)
    if r == Rank.Q:
        new_total = total + 20
        return EffectResult(new_total, direction, new_total == 101, False, card)
    if r == Rank.K:
        new_total = total + 30
        return EffectResult(new_total, direction, new_total == 101, False, card)
    if r == Rank.A:
        candidates = [total + 1, total + 11]
        for t in candidates:
            if t == 101:
                return EffectResult(t, direction, True, False, card)
        safe = [t for t in candidates if t <= 101]
        if safe:
            new_total = max(safe)
        else:
            new_total = max(candidates)
        return EffectResult(new_total, direction, new_total == 101, False, card)
    if r == Rank.JK:
        if total == 100:
            return EffectResult(total, direction, False, True, card)
        new_total = total + 50
        return EffectResult(new_total, direction, new_total == 101, False, card)
    raise ValueError("Unknown card rank")


class RewardScheme(NamedTuple):
    on_burst_loss: float = -1.0
    on_burst_win: float = +1.0
    on_counter_self: float = +0.2
    on_counter_prev: float = -0.2
    step_penalty: float = 0.0


# ==== Main step function ====


def step(
    state: State, action: Action, reward_scheme: RewardScheme = RewardScheme()
) -> tuple[State, float, bool, dict[str, int | str | int | None]]:
    s = state
    p = s.public
    me_idx = p.turn
    me = s.players[me_idx]
    used_card: Card | None = None
    new_deck = p.deck
    new_hand = me.hand
    if action == Action.PLAY_HAND_0:
        used_card = me.hand[0]
        if used_card is None:
            return s, 0.0, False, {"invalid": True}
    elif action == Action.PLAY_HAND_1:
        used_card = me.hand[1]
        if used_card is None:
            return s, 0.0, False, {"invalid": True}
    elif action == Action.PLAY_DECK:
        used_card, new_deck = _draw_top(p.deck)
        if used_card is None:
            return s, 0.0, False, {"invalid": True}
    else:
        return s, 0.0, False, {"invalid": True}
    eff = _apply_card_effect(p.total, p.direction, used_card)
    reward: float = reward_scheme.step_penalty
    info: dict[str, int | str | int | None] = {"used_card": int(used_card.rank)}
    # Joker counter effect
    if eff.counter_triggered:
        new_players_list = list(s.players)
        new_players_list[me_idx] = replace(
            new_players_list[me_idx], lp=new_players_list[me_idx].lp + p.penalty_level
        )
        reward += reward_scheme.on_counter_self
        prev_idx = p.last_player
        if prev_idx is not None and s.alive[prev_idx]:
            new_players_list[prev_idx] = replace(
                new_players_list[prev_idx],
                lp=new_players_list[prev_idx].lp - p.penalty_level,
            )
            reward += reward_scheme.on_counter_prev
        next_state = _start_next_round(
            players=tuple(new_players_list),
            alive=s.alive,
            carry_penalty_level=False,
            prev_penalty_level=p.penalty_level,
        )
        next_state = _apply_elimination(next_state)
        done, winner = _check_game_end(next_state)
        info["event"] = "counter"
        info["winner"] = winner
        return next_state, reward, done, info
    # Reset effect
    if eff.reset_triggered:
        top_card, after_deck = _draw_top(new_deck)
        if top_card is None:
            return _round_draw_and_continue(s, reward, info)
        next_turn = _next_turn_idx(me_idx, eff.direction, s.alive)
        if action != Action.PLAY_DECK:
            drawn, remaining_deck = _draw_top(after_deck)
            if drawn is None:
                return _round_draw_and_continue(s, reward, info)
            new_hand = _replace_hand(
                me.hand, 0 if action == Action.PLAY_HAND_0 else 1, drawn
            )
            after_deck = remaining_deck
        new_players = list(s.players)
        new_players[me_idx] = replace(me, hand=new_hand)
        new_public = PublicState(
            turn=next_turn,
            direction=eff.direction,
            total=_start_total_from_card(top_card),
            penalty_level=p.penalty_level + 1,
            deck=after_deck,
            discard=p.discard + (used_card,),
            last_player=me_idx,
        )
        next_state = replace(s, players=tuple(new_players), public=new_public)
        info["event"] = "reset"
        return next_state, reward, False, info
    # Normal play
    new_total = eff.total
    burst = new_total > 101
    if action != Action.PLAY_DECK:
        drawn, new_deck2 = _draw_top(new_deck)
        if drawn is None:
            return _round_draw_and_continue(s, reward, info)
        new_deck = new_deck2
        new_hand = _replace_hand(
            me.hand, 0 if action == Action.PLAY_HAND_0 else 1, drawn
        )
    next_turn = _next_turn_idx(me_idx, eff.direction, s.alive)
    new_players = list(s.players)
    new_players[me_idx] = replace(me, hand=new_hand)
    new_public = PublicState(
        turn=next_turn,
        direction=eff.direction,
        total=new_total,
        penalty_level=p.penalty_level,
        deck=new_deck,
        discard=p.discard + (used_card,),
        last_player=me_idx,
    )
    next_state = replace(s, players=tuple(new_players), public=new_public)
    if burst:
        damaged_players = list(next_state.players)
        damaged_players[me_idx] = replace(
            damaged_players[me_idx], lp=damaged_players[me_idx].lp - p.penalty_level
        )
        reward += reward_scheme.on_burst_loss
        next_state = replace(next_state, players=tuple(damaged_players))
        next_state = _start_next_round(
            players=next_state.players,
            alive=next_state.alive,
            carry_penalty_level=False,
            prev_penalty_level=p.penalty_level,
        )
        next_state = _apply_elimination(next_state)
        done, winner = _check_game_end(next_state)
        if done:
            reward += reward_scheme.on_burst_win
        info["event"] = "burst"
        info["winner"] = winner
        return next_state, reward, done, info
    if not new_public.deck:
        return _round_draw_and_continue(next_state, reward, info)
    return next_state, reward, False, info


def _start_total_from_card(card: Card) -> int:
    r = card.rank
    if Rank.R2 <= r <= Rank.R7:
        return int(r)
    if r == Rank.R8 or r == Rank.R9:
        return 0
    if r == Rank.R10:
        return 10
    if r == Rank.J:
        return 10
    if r == Rank.Q:
        return 20
    if r == Rank.K:
        return 30
    if r == Rank.A:
        return 11
    if r == Rank.JK:
        return 50
    raise ValueError


def _round_draw_and_continue(
    s: State, reward: float, info: dict[str, int | str | int | None]
) -> tuple[State, float, bool, dict[str, int | str | int | None]]:


    next_state = _start_next_round(
        players=s.players,
        alive=s.alive,
        carry_penalty_level=True,
        prev_penalty_level=s.public.penalty_level,
    )
    done, winner = _check_game_end(next_state)
    info["event"] = "deck_exhausted"
    info["winner"] = winner
    return next_state, reward, done, info


def _start_next_round(
    players: tuple[PlayerState, ...],
    alive: tuple[bool, ...],
    carry_penalty_level: bool,
    prev_penalty_level: int = 1,
) -> State:
    """Start a fresh round.

    Args:
        players: Current player states.
        alive: Tuple indicating which players are still in the game.
        carry_penalty_level: If ``True`` the previous penalty level is kept,
            otherwise it is reset to 1.
        prev_penalty_level: The penalty level from the previous round.

    Returns:
        A new ``State`` representing the beginning of the round.
    """

    deck = _shuffle(_standard_deck(), seed=None)
    new_players_list: list[PlayerState] = []
    for i, pl in enumerate(players):
        if not alive[i]:
            new_players_list.append(pl)
            continue
        c1, deck = _draw_top(deck)
        c2, deck = _draw_top(deck)
        new_players_list.append(replace(pl, hand=(c1, c2)))

    penalty_level = prev_penalty_level if carry_penalty_level else 1

    public = PublicState(
        turn=_first_alive(alive),
        direction=+1,
        total=0,
        penalty_level=penalty_level,
        deck=deck,
        discard=tuple(),
        last_player=None,
    )
    return State(players=tuple(new_players_list), public=public, alive=alive)


def _replace_hand(
    hand: tuple[Card | None, Card | None], idx: int, new_card: Card
) -> tuple[Card | None, Card | None]:
    return (new_card, hand[1]) if idx == 0 else (hand[0], new_card)


def _apply_elimination(state: State) -> State:
    new_alive = list(state.alive)
    for i, pl in enumerate(state.players):
        if new_alive[i] and pl.lp <= 0:
            new_alive[i] = False
    return replace(state, alive=tuple(new_alive))


def _check_game_end(state: State) -> tuple[bool, int | None]:
    alive_idxs = [i for i, a in enumerate(state.alive) if a]
    if len(alive_idxs) <= 1:
        return True, (alive_idxs[0] if alive_idxs else None)
    return False, None


# ==== Observation and encoding ====


@dataclass(frozen=True)
class Observation:
    player_index: int
    my_lp: int
    my_hand: tuple[int | None, int | None]
    total: int
    penalty_level: int
    direction: int
    deck_count: int
    alive_count: int
    last_player: int  # -1 if none


def observe(state: State, player_idx: int) -> Observation:
    me = state.players[player_idx]
    return Observation(
        player_index=player_idx,
        my_lp=me.lp,
        my_hand=(
            int(me.hand[0].rank) if me.hand[0] is not None else None,
            int(me.hand[1].rank) if me.hand[1] is not None else None,
        ),
        total=state.public.total,
        penalty_level=state.public.penalty_level,
        direction=state.public.direction,
        deck_count=len(state.public.deck),
        alive_count=sum(1 for a in state.alive if a),
        last_player=-1
        if state.public.last_player is None
        else state.public.last_player,
    )


def encode(obs: Observation) -> tuple[int, ...]:
    h0 = obs.my_hand[0] if obs.my_hand[0] is not None else -1
    h1 = obs.my_hand[1] if obs.my_hand[1] is not None else -1
    return (
        obs.player_index,
        obs.my_lp,
        h0,
        h1,
        obs.total,
        obs.penalty_level,
        obs.direction,
        obs.deck_count,
        obs.alive_count,
        obs.last_player,
    )
