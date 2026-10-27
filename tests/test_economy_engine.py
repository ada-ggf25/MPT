"""The market without a GPU: the auction, revenue-share settlement, per-agent rent, availability,
and the population ledger (EoM's economy). Run: python -m pytest tests (needs only numpy)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from economy_engine import ChainTurnEngine, PopulationLedger, settle_revenue_share  # noqa: E402

LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)


class FixedBids:
    """Callbacks with fixed bids per agent; records who was asked to bid and the emitted rows."""

    def __init__(self, bids):
        self.bids = bids
        self.asked = set()
        self.rows = []

    def bid(self, i, mask):
        self.asked.add(i)
        return np.full(mask.shape, self.bids[i])

    def act(self, i, mask):
        pass

    def emit(self, i, kind, turn, mask, credit, rent):
        self.rows.append((i, kind, turn, mask.copy(), np.asarray(credit).copy(), rent))


def play(engine, cb, rewards, available=None, bs=1):
    engine.begin_episode()
    active = np.ones(bs, dtype=bool)
    for t, r in enumerate(rewards):
        engine.run_turn(active, t, bs, train_bid_rows=True, train_act_rows=True, cb=cb,
                        available=available)
        engine.note_step_rewards(np.full(bs, r))
    return engine.finalize(np.full(bs, sum(rewards)), bs, True, cb)


def test_revenue_share_conserves_reward():
    profit, total, leak = settle_revenue_share([0.0, 0.4, 0.6], [0.0, 0.0, 1.0], seg_scale=False)
    assert sum(profit) == pytest.approx(total - leak)
    # turn 2 keeps 1 - 0.6 of its income, passes 0.6 back to turn 1, which keeps 0.6 * (1 - 0.4)
    assert profit == pytest.approx([0.24, 0.36, 0.4])


def test_highest_bid_wins_and_pays_its_own_rent():
    eng = ChainTurnEngine(3, bid_levels=LEVELS, reward_rent=[0.05, 0.05, 0.5], reward_scale=1.0,
                          seg_scale=False, rng=np.random.default_rng(0))
    cb = FixedBids([0.2, 0.4, 0.8])
    play(eng, cb, [1.0])
    held = [r for r in cb.rows if r[1] in ("act",)]
    assert len(held) == 1 and held[0][0] == 2 and held[0][5] == 0.5


def test_scalar_and_uniform_list_rent_are_identical():
    out = []
    for rent in (0.05, [0.05, 0.05, 0.05]):
        eng = ChainTurnEngine(3, bid_levels=LEVELS, reward_rent=rent, reward_scale=1.0,
                              seg_scale=True, rng=np.random.default_rng(0))
        cb = FixedBids([0.2, 0.6, 0.4])
        play(eng, cb, [0.0, 1.0, 0.0, 2.0], bs=2)
        out.append([(i, k, t, c.tolist(), r) for i, k, t, _, c, r in cb.rows])
    assert out[0] == out[1]


def test_unavailable_agent_is_not_asked_and_cannot_win():
    eng = ChainTurnEngine(3, bid_levels=LEVELS, reward_rent=0.05, reward_scale=1.0,
                          seg_scale=False, rng=np.random.default_rng(0))
    cb = FixedBids([0.2, 0.4, 0.8])
    play(eng, cb, [1.0], available=np.array([True, True, False]))
    assert cb.asked == {0, 1}
    assert all(i != 2 for i, *_ in cb.rows)
    assert eng.chosen[0][0] == 1


def test_no_available_agent_is_an_error():
    eng = ChainTurnEngine(2, bid_levels=LEVELS, reward_rent=0.05, reward_scale=1.0,
                          seg_scale=False)
    with pytest.raises(ValueError):
        play(eng, FixedBids([0.2, 0.4]), [1.0], available=np.array([False, False]))


def test_ledger_books_winner_profit_minus_rent_and_bankrupts():
    eng = ChainTurnEngine(2, bid_levels=LEVELS, reward_rent=[0.05, 0.3], reward_scale=1.0,
                          seg_scale=False, rng=np.random.default_rng(0))
    led = PopulationLedger(2, wealth_init=0.2, bankruptcy=True, threshold=0.0)
    cb = FixedBids([0.2, 0.8])  # agent 1 always wins, earns nothing, pays 0.3 per turn
    bundle = play(eng, cb, [0.0, 0.0])
    earn = led.settle(eng.chosen, bundle["profit"], eng.rents, eng.term_at)
    assert earn == pytest.approx([0.0, -0.6])
    assert led.available.tolist() == [True, False] and led.bankrupt.tolist() == [False, True]
    led.set_available(1, True)  # readmitted as a newcomer with the initial wealth
    assert led.wealth[1] == pytest.approx(0.2) and not led.bankrupt[1]


def test_ledger_defaults_never_change_who_bids():
    led = PopulationLedger(3)
    led.settle([np.array([0])], np.array([[-5.0]]), np.full(3, 0.05), np.array([0]))
    assert led.everyone_available
