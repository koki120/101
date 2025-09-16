from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import IntEnum
from functools import reduce
from typing import NamedTuple, TypeAlias

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
    PLAY_TEN_PLUS = 3
    PLAY_TEN_MINUS = 4
    PLAY_ACE_ONE = 5
    PLAY_ACE_ELEVEN = 6


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
    history: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    pending_choice: PendingChoice | None = None


@dataclass(frozen=True)
class State:
    players: tuple[PlayerState, ...]
    public: PublicState
    alive: tuple[bool, ...]


# ==== Deck generation and shuffling ====


def _standard_deck() -> tuple[Card, ...]:
    ranks = (
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
    )
    return tuple(Card(rank) for rank in ranks for _ in range(4)) + (
        Card(Rank.JK),
        Card(Rank.JK),
    )


def _shuffle(deck: tuple[Card, ...], seed: int | None) -> tuple[Card, ...]:
    rng = random.Random(seed)
    return tuple(rng.sample(deck, k=len(deck)))


def _draw_top(deck: tuple[Card, ...]) -> tuple[Card | None, tuple[Card, ...]]:
    if not deck:
        return None, deck
    return deck[0], deck[1:]


def _draw_hand(
    deck: tuple[Card, ...],
) -> tuple[tuple[Card | None, Card | None], tuple[Card, ...]]:
    first, remaining = _draw_top(deck)
    second, remaining_after = _draw_top(remaining)
    return (first, second), remaining_after


def _next_turn_idx(turn: int, direction: int, alive: tuple[bool, ...]) -> int:
    n = len(alive)
    indices = ((turn + direction * step) % n for step in range(1, n + 1))
    try:
        return next(idx for idx in indices if alive[idx])
    except StopIteration as exc:
        raise ValueError("No players are alive") from exc


def _first_alive(alive: tuple[bool, ...]) -> int:
    try:
        return next(i for i, alive_flag in enumerate(alive) if alive_flag)
    except StopIteration as exc:
        raise ValueError("No players are alive") from exc


# ==== Initialization and legal actions ====

START_LP_DEFAULT = 10


def _create_initial_players(
    num_players: int, start_lp: int, deck: tuple[Card, ...]
) -> tuple[tuple[PlayerState, ...], tuple[Card, ...]]:
    def deal(
        acc: tuple[tuple[PlayerState, ...], tuple[Card, ...]], _: int
    ) -> tuple[tuple[PlayerState, ...], tuple[Card, ...]]:
        players_acc, current_deck = acc
        hand, remaining_deck = _draw_hand(current_deck)
        player = PlayerState(lp=start_lp, hand=hand)
        return players_acc + (player,), remaining_deck

    empty_players: tuple[PlayerState, ...] = tuple()
    initial: tuple[tuple[PlayerState, ...], tuple[Card, ...]] = (
        empty_players,
        deck,
    )
    return reduce(deal, range(num_players), initial)


def reset(
    num_players: int, start_lp: int = START_LP_DEFAULT, seed: int | None = None
) -> State:
    if num_players < 2:
        raise ValueError("The game requires at least two players")
    deck = _shuffle(_standard_deck(), seed)
    players, deck = _create_initial_players(num_players, start_lp, deck)
    public = PublicState(
        turn=0,
        direction=+1,
        total=0,
        penalty_level=1,
        deck=deck,
        discard=tuple(),
        last_player=None,
        history=tuple(),
    )
    alive = tuple(True for _ in range(num_players))
    return State(players=players, public=public, alive=alive)


def legal_actions(state: State) -> tuple[Action, ...]:
    p = state.public
    pending = p.pending_choice
    if pending is not None:
        rank = pending.resolution.used_card.rank
        if rank == Rank.R10:
            plus = (Action.PLAY_TEN_PLUS,)
            minus = (Action.PLAY_TEN_MINUS,) if p.total > 9 else tuple()
            return plus + minus
        if rank == Rank.A:
            return (Action.PLAY_ACE_ONE, Action.PLAY_ACE_ELEVEN)
        return tuple()
    me = state.players[p.turn]
    hand_actions = tuple(
        action
        for action, card in (
            (Action.PLAY_HAND_0, me.hand[0]),
            (Action.PLAY_HAND_1, me.hand[1]),
        )
        if card is not None
    )
    deck_action = (Action.PLAY_DECK,) if p.deck else tuple()
    return hand_actions + deck_action


def action_mask(state: State) -> tuple[int, ...]:
    """Return a binary mask of legal actions.

    The tuple has an entry for every action in the :class:`Action` enum.  A
    value of ``1`` indicates that the action is currently legal for the player
    whose turn it is, while ``0`` marks an illegal action.  This utility is
    particularly useful when integrating the environment with reinforcement
    learning agents that expect a fixed-size action space with masking support.
    """

    legal = frozenset(legal_actions(state))
    return tuple(int(action in legal) for action in Action)


# ==== Card effect logic ====


class EffectResult(NamedTuple):
    total: int
    direction: int
    reset_triggered: bool
    counter_triggered: bool
    used_card: Card


@dataclass(frozen=True)
class ResolvedAction:
    used_card: Card
    deck_after_play: tuple[Card, ...]
    replacement_index: int | None


@dataclass(frozen=True)
class PendingChoice:
    player_index: int
    resolution: ResolvedAction


InfoValue: TypeAlias = int | str | bool | None
InfoDict: TypeAlias = dict[str, InfoValue]


def _apply_card_effect(total: int, direction: int, card: Card) -> EffectResult:
    r = card.rank
    if r == Rank.R8:
        return EffectResult(total, direction, False, False, card)
    if r == Rank.R9:
        return EffectResult(total, -direction, False, False, card)
    if Rank.R2 <= r <= Rank.R7:
        new_total = total + int(r)
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
    if r in (Rank.R10, Rank.A):
        raise ValueError("Choice-based card requires explicit resolution")
    if r == Rank.JK:
        if total == 100:
            return EffectResult(total, direction, False, True, card)
        new_total = total + 50
        return EffectResult(new_total, direction, new_total == 101, False, card)
    raise ValueError("Unknown card rank")


def _apply_choice_effect(
    total: int, direction: int, card: Card, action: Action
) -> EffectResult | None:
    r = card.rank
    if r == Rank.R10:
        if action == Action.PLAY_TEN_PLUS:
            new_total = total + 10
            return EffectResult(new_total, direction, new_total == 101, False, card)
        if action == Action.PLAY_TEN_MINUS and total > 9:
            new_total = total - 10
            return EffectResult(new_total, direction, new_total == 101, False, card)
        return None
    if r == Rank.A:
        if action == Action.PLAY_ACE_ONE:
            new_total = total + 1
            return EffectResult(new_total, direction, new_total == 101, False, card)
        if action == Action.PLAY_ACE_ELEVEN:
            new_total = total + 11
            return EffectResult(new_total, direction, new_total == 101, False, card)
        return None
    return None


class RewardScheme(NamedTuple):
    on_burst_loss: float = -1.0
    on_burst_win: float = +1.0
    on_counter_self: float = +0.2
    on_counter_prev: float = -0.2
    step_penalty: float = 0.0


# ==== Main step function ====


def _resolve_action(state: State, action: Action) -> ResolvedAction | None:
    public = state.public
    me = state.players[public.turn]
    if action == Action.PLAY_HAND_0:
        card = me.hand[0]
        return ResolvedAction(card, public.deck, 0) if card is not None else None
    if action == Action.PLAY_HAND_1:
        card = me.hand[1]
        return ResolvedAction(card, public.deck, 1) if card is not None else None
    if action == Action.PLAY_DECK:
        card, remaining_deck = _draw_top(public.deck)
        return ResolvedAction(card, remaining_deck, None) if card is not None else None
    return None


def _start_pending_choice_state(
    state: State, player_index: int, action: Action, resolution: ResolvedAction
) -> tuple[State, float, bool, InfoDict]:
    pending = PendingChoice(player_index=player_index, resolution=resolution)
    history = state.public.history + ((player_index, int(action)),)
    updated_public = replace(
        state.public,
        deck=resolution.deck_after_play,
        history=history,
        pending_choice=pending,
    )
    next_state = replace(state, public=updated_public)
    info: InfoDict = {
        "used_card": int(resolution.used_card.rank),
        "needs_choice": True,
    }
    return next_state, 0.0, False, info


def _resolve_pending_choice(
    state: State, action: Action, reward_scheme: RewardScheme
) -> tuple[State, float, bool, InfoDict]:
    pending = state.public.pending_choice
    if pending is None or pending.player_index != state.public.turn:
        return state, 0.0, False, {"invalid": True}
    card = pending.resolution.used_card
    effect = _apply_choice_effect(
        state.public.total, state.public.direction, card, action
    )
    if effect is None:
        return state, 0.0, False, {"invalid": True}
    base_info: InfoDict = {"used_card": int(card.rank)}
    history = state.public.history + ((pending.player_index, int(action)),)
    reward = reward_scheme.step_penalty
    cleared_public = replace(state.public, pending_choice=None, history=history)
    cleared_state = replace(state, public=cleared_public)
    me_idx = pending.player_index
    me = cleared_state.players[me_idx]
    return (
        _handle_counter(cleared_state, me_idx, reward, base_info, reward_scheme)
        if effect.counter_triggered
        else (
            _handle_reset(
                cleared_state,
                me_idx,
                me,
                effect,
                reward,
                base_info,
                pending.resolution,
            )
            if effect.reset_triggered
            else _handle_standard_play(
                cleared_state,
                me_idx,
                me,
                effect,
                reward,
                base_info,
                reward_scheme,
                pending.resolution,
                history,
            )
        )
    )


def _update_player(
    players: tuple[PlayerState, ...],
    target_idx: int,
    update_fn: Callable[[PlayerState], PlayerState],
) -> tuple[PlayerState, ...]:
    return tuple(
        update_fn(player) if idx == target_idx else player
        for idx, player in enumerate(players)
    )


def _maybe_replace_hand(
    hand: tuple[Card | None, Card | None],
    replacement_index: int | None,
    deck: tuple[Card, ...],
) -> tuple[tuple[Card | None, Card | None], tuple[Card, ...]] | None:
    if replacement_index is None:
        return hand, deck
    drawn, remaining = _draw_top(deck)
    if drawn is None:
        return None
    return _replace_hand(hand, replacement_index, drawn), remaining


def _handle_counter(
    state: State,
    me_idx: int,
    reward: float,
    base_info: InfoDict,
    reward_scheme: RewardScheme,
) -> tuple[State, float, bool, InfoDict]:
    penalty = state.public.penalty_level
    incremented_players = _update_player(
        state.players, me_idx, lambda pl: replace(pl, lp=pl.lp + penalty)
    )
    prev_idx = state.public.last_player
    updated_players = (
        _update_player(
            incremented_players,
            prev_idx,
            lambda pl: replace(pl, lp=pl.lp - penalty),
        )
        if prev_idx is not None and state.alive[prev_idx]
        else incremented_players
    )
    reward_delta = reward_scheme.on_counter_self + (
        reward_scheme.on_counter_prev
        if prev_idx is not None and state.alive[prev_idx]
        else 0.0
    )
    next_state, done, winner = _start_round_after_event(
        players=updated_players,
        alive=state.alive,
        carry_penalty_level=False,
        prev_penalty_level=state.public.penalty_level,
    )
    return (
        next_state,
        reward + reward_delta,
        done,
        base_info | {"event": "counter", "winner": winner},
    )


def _handle_reset(
    state: State,
    me_idx: int,
    me: PlayerState,
    effect: EffectResult,
    reward: float,
    base_info: InfoDict,
    resolution: ResolvedAction,
) -> tuple[State, float, bool, InfoDict]:
    top_card, after_deck = _draw_top(resolution.deck_after_play)
    if top_card is None:
        return _round_draw_and_continue(state, reward, base_info)
    replacement = _maybe_replace_hand(me.hand, resolution.replacement_index, after_deck)
    if replacement is None:
        return _round_draw_and_continue(state, reward, base_info)
    new_hand, remaining_deck = replacement
    updated_players = _update_player(
        state.players, me_idx, lambda pl: replace(pl, hand=new_hand)
    )
    new_public = PublicState(
        turn=_next_turn_idx(me_idx, effect.direction, state.alive),
        direction=effect.direction,
        total=_start_total_from_card(top_card),
        penalty_level=state.public.penalty_level + 1,
        deck=remaining_deck,
        discard=state.public.discard + (resolution.used_card,),
        last_player=me_idx,
        history=tuple(),
    )
    next_state = replace(state, players=updated_players, public=new_public)
    return next_state, reward, False, base_info | {"event": "reset"}


def _handle_standard_play(
    state: State,
    me_idx: int,
    me: PlayerState,
    effect: EffectResult,
    reward: float,
    base_info: InfoDict,
    reward_scheme: RewardScheme,
    resolution: ResolvedAction,
    history: tuple[tuple[int, int], ...],
) -> tuple[State, float, bool, InfoDict]:
    replacement = _maybe_replace_hand(
        me.hand, resolution.replacement_index, resolution.deck_after_play
    )
    if replacement is None:
        return _round_draw_and_continue(state, reward, base_info)
    new_hand, deck_after = replacement
    updated_players = _update_player(
        state.players, me_idx, lambda pl: replace(pl, hand=new_hand)
    )
    new_public = PublicState(
        turn=_next_turn_idx(me_idx, effect.direction, state.alive),
        direction=effect.direction,
        total=effect.total,
        penalty_level=state.public.penalty_level,
        deck=deck_after,
        discard=state.public.discard + (resolution.used_card,),
        last_player=me_idx,
        history=history,
    )
    next_state = replace(state, players=updated_players, public=new_public)
    if effect.total > 101:
        penalty = state.public.penalty_level
        damaged_players = _update_player(
            next_state.players,
            me_idx,
            lambda pl: replace(pl, lp=pl.lp - penalty),
        )
        damaged_state = replace(next_state, players=damaged_players)
        post_round_state, done, winner = _start_round_after_event(
            players=damaged_state.players,
            alive=damaged_state.alive,
            carry_penalty_level=False,
            prev_penalty_level=state.public.penalty_level,
        )
        burst_reward = reward + reward_scheme.on_burst_loss
        final_reward = burst_reward + (reward_scheme.on_burst_win if done else 0.0)
        return (
            post_round_state,
            final_reward,
            done,
            base_info | {"event": "burst", "winner": winner},
        )
    if not new_public.deck:
        return _round_draw_and_continue(next_state, reward, base_info)
    return next_state, reward, False, base_info


def step(
    state: State, action: Action, reward_scheme: RewardScheme | None = None
) -> tuple[State, float, bool, InfoDict]:
    reward_scheme = reward_scheme or RewardScheme()
    pending = state.public.pending_choice
    if pending is not None:
        return _resolve_pending_choice(state, action, reward_scheme)
    resolution = _resolve_action(state, action)
    if resolution is None:
        return state, 0.0, False, {"invalid": True}
    me_idx = state.public.turn
    me = state.players[me_idx]
    card = resolution.used_card
    if card.rank in (Rank.R10, Rank.A):
        return _start_pending_choice_state(state, me_idx, action, resolution)
    effect = _apply_card_effect(state.public.total, state.public.direction, card)
    base_info: InfoDict = {"used_card": int(card.rank)}
    history = state.public.history + ((me_idx, int(action)),)
    reward = reward_scheme.step_penalty
    return (
        _handle_counter(state, me_idx, reward, base_info, reward_scheme)
        if effect.counter_triggered
        else (
            _handle_reset(state, me_idx, me, effect, reward, base_info, resolution)
            if effect.reset_triggered
            else _handle_standard_play(
                state,
                me_idx,
                me,
                effect,
                reward,
                base_info,
                reward_scheme,
                resolution,
                history,
            )
        )
    )


def _start_total_from_card(card: Card) -> int:
    r = card.rank
    if Rank.R2 <= r <= Rank.R10:
        return int(r)
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
    s: State, reward: float, base_info: InfoDict
) -> tuple[State, float, bool, InfoDict]:
    next_state, done, winner = _start_round_after_event(
        players=s.players,
        alive=s.alive,
        carry_penalty_level=True,
        prev_penalty_level=s.public.penalty_level,
    )
    return (
        next_state,
        reward,
        done,
        base_info | {"event": "deck_exhausted", "winner": winner},
    )


def _deal_round_players(
    players: tuple[PlayerState, ...],
    alive: tuple[bool, ...],
    deck: tuple[Card, ...],
) -> tuple[tuple[PlayerState, ...], tuple[Card, ...]]:
    def deal(
        acc: tuple[tuple[PlayerState, ...], tuple[Card, ...]],
        payload: tuple[int, PlayerState],
    ) -> tuple[tuple[PlayerState, ...], tuple[Card, ...]]:
        players_acc, current_deck = acc
        idx, player = payload
        if not alive[idx]:
            return players_acc + (player,), current_deck
        hand, remaining_deck = _draw_hand(current_deck)
        return players_acc + (replace(player, hand=hand),), remaining_deck

    empty_players: tuple[PlayerState, ...] = tuple()
    initial: tuple[tuple[PlayerState, ...], tuple[Card, ...]] = (
        empty_players,
        deck,
    )
    return reduce(deal, enumerate(players), initial)


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
    new_players, deck = _deal_round_players(players, alive, deck)
    penalty_level = prev_penalty_level if carry_penalty_level else 1

    public = PublicState(
        turn=_first_alive(alive),
        direction=+1,
        total=0,
        penalty_level=penalty_level,
        deck=deck,
        discard=tuple(),
        last_player=None,
        history=tuple(),
    )
    return State(players=new_players, public=public, alive=alive)


def _replace_hand(
    hand: tuple[Card | None, Card | None], idx: int, new_card: Card
) -> tuple[Card | None, Card | None]:
    return (new_card, hand[1]) if idx == 0 else (hand[0], new_card)


def _apply_elimination(state: State) -> State:
    return replace(
        state,
        alive=tuple(
            False if alive_flag and player.lp <= 0 else alive_flag
            for alive_flag, player in zip(state.alive, state.players, strict=True)
        ),
    )


def _check_game_end(state: State) -> tuple[bool, int | None]:
    alive_idxs = tuple(i for i, flag in enumerate(state.alive) if flag)
    return (
        (True, alive_idxs[0])
        if len(alive_idxs) == 1
        else (True, None)
        if not alive_idxs
        else (False, None)
    )


def _start_round_after_event(
    players: tuple[PlayerState, ...],
    alive: tuple[bool, ...],
    carry_penalty_level: bool,
    prev_penalty_level: int,
) -> tuple[State, bool, int | None]:
    next_state = _start_next_round(
        players=players,
        alive=alive,
        carry_penalty_level=carry_penalty_level,
        prev_penalty_level=prev_penalty_level,
    )
    cleaned_state = _apply_elimination(next_state)
    done, winner = _check_game_end(cleaned_state)
    return cleaned_state, done, winner


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
