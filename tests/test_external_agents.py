"""External agents (a simulated person, an API model) inside an MPT society: they bid, win, act,
are settled and booked, and never produce training rows. Run: python -m pytest tests

Without torch / verl / transformers installed (e.g. on a laptop) the few names the orchestra
imports from them are stubbed; with them installed the real modules are used."""

import os
import sys
import types

import numpy as np

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)


def _stub_heavy_deps():
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass
    tr = types.ModuleType("transformers")
    tr.PreTrainedTokenizer = object
    sys.modules.setdefault("transformers", tr)
    verl = types.ModuleType("verl")
    verl.DataProto = object
    proto = types.ModuleType("verl.protocol")
    for f in ("pad_dataproto_to_divisor", "unpad_dataproto", "extract_dataproto_via_active_mask",
              "restore_dataproto_via_active_mask"):
        setattr(proto, f, lambda *a, **k: None)
    verl.protocol = proto
    sys.modules.setdefault("verl", verl)
    sys.modules.setdefault("verl.protocol", proto)
    # package __init__s that would pull in torch: load their submodules without running them
    for name in ("agent_system.environments", "agent_system.environments.env_package",
                 "agent_system.environments.env_package.minecraft"):
        pkg = types.ModuleType(name)
        pkg.__path__ = [os.path.join(SRC, *name.split("."))]
        sys.modules.setdefault(name, pkg)


_stub_heavy_deps()

from agent_system.agent.agents.external import OracleHumanAgent, make_external  # noqa: E402
from agent_system.agent.orchestra.chain_base import ChainOrchestraBase  # noqa: E402
from agent_system.agent.registry import AgentRegistry  # noqa: E402

NS = types.SimpleNamespace


class _Batch:
    def __init__(self, **cols):
        self.non_tensor_batch = {k: np.array(v, dtype=object) for k, v in cols.items()}


class _FakeDesk:
    BID = 0.0

    def __init__(self, wg_id, tokenizer, processor, config):
        pass

    def reset(self):
        pass

    def bid_call(self, gen_batch, env_obs, team_context, actor_rollout_wg, agent_active_mask, step):
        bs = len(env_obs["text"])
        return _Batch(raw_response=["<bid>x"] * bs), np.full(bs, self.BID)

    def call(self, gen_batch, env_obs, team_context, actor_rollout_wg, agent_active_mask, step):
        bs = len(env_obs["text"])
        emitted = ["<action>pick up the bowl</action>"] * bs
        return _Batch(raw_response=emitted, instruction=["pick up the bowl"] * bs,
                      first_word=["pick"] * bs, whitelist_ok=[True] * bs), emitted


@AgentRegistry.register("Test Desk (passes)")
class _PassDesk(_FakeDesk):
    BID = 0.0


@AgentRegistry.register("Test Desk (bids 0.6)")
class _BidDesk(_FakeDesk):
    BID = 0.6


class _Orch(ChainOrchestraBase):
    DOMAIN_BLOCK = "test_chain"
    AGENT_MODULE = __name__


def _orch(desks, block):
    config = NS(agent=NS(orchestra=NS(test_chain=block)))
    wg = {d: f"wg{i}" for i, d in enumerate(desks)}
    return _Orch(agent_ids=desks, model_ids=[None] * len(desks), agents_to_wg_mapping=wg,
                 tokenizers={w: None for w in wg.values()},
                 processors={w: None for w in wg.values()}, config=config)


def _obs(hints):
    return {"text": [f"TASK {i}" for i in range(len(hints))], "image": None,
            "anchor": [f"TASK {i}" for i in range(len(hints))],
            "privileged": [{"oracle_next": h} for h in hints]}


def _episode(orch, hints, reward):
    orch.reset()
    orch.begin_episode(is_train=True)
    wgs = {f"wg{i}": object() for i in range(orch.n_trainable)}
    actions, rows = orch.run(NS(meta_info={}), _obs(hints), wgs, np.ones(len(hints), bool), step=0)
    orch.note_step_rewards(np.full(len(hints), reward))
    final = orch.finalize_episode(_obs(hints), np.full(len(hints), reward), is_train=True)
    return actions, rows, final


HUMAN = {"name": "Human (oracle)", "kind": "oracle_human", "rent": 0.5, "bid": 0.2}


def test_human_wins_when_every_desk_passes_and_is_never_trained():
    orch = _orch(["Test Desk (passes)"], NS(external_agents=[HUMAN]))
    assert orch.desk_ids == ["Test Desk (passes)", "Human (oracle)"] and orch.n_trainable == 1
    assert orch.engine.rents.tolist() == [0.05, 0.5]
    actions, rows, final = _episode(orch, ["open the top drawer", "open the top drawer"], 1.0)
    assert actions == ["<action>open the top drawer</action>"] * 2
    # only the desk's losing bid rows exist; nothing for the human, at the turn or at settlement
    assert {r["agent_id"] for r in rows + final} == {"Test Desk (passes)"}
    # the human was paid like any winner: revenue share keeps (1 - bid) of the income (0.8 of R;
    # at turn 0 there is no earlier actor to receive the bid share), minus its own rent
    assert np.isclose(orch.ledger.wealth[1], 1.0 + (1.0 - 0.2) * 1.0 - 0.5)


def test_desk_that_outbids_the_human_acts_and_is_trained():
    orch = _orch(["Test Desk (bids 0.6)"], NS(external_agents=[HUMAN]))
    actions, rows, final = _episode(orch, ["open the top drawer"], 1.0)
    assert actions == ["<action>pick up the bowl</action>"]
    credit = [r["batch"].non_tensor_batch["econ_credit_internal"][0] for r in final]
    # bid row + act row, both at profit - rent = (1 - 0.6) * R - 0.05
    assert len(final) == 2 and np.allclose(credit, (1.0 - 0.6) * 1.0 - 0.05)


def test_human_is_not_asked_when_unavailable():
    block = NS(external_agents=[HUMAN], economy=NS(unavailable=["Human (oracle)"]))
    orch = _orch(["Test Desk (passes)", "Test Desk (bids 0.6)"], block)
    actions, _, _ = _episode(orch, ["open the top drawer"], 1.0)
    assert actions == ["<action>pick up the bowl</action>"]


def test_oracle_human_flaws():
    h = make_external({**HUMAN, "p_correct": 0.0, "wrong_instruction": "wave"})
    assert isinstance(h, OracleHumanAgent)
    obs = _obs(["open the top drawer", None])
    assert h.bid(obs, np.ones(2, bool), 0).tolist() == [0.2, 0.0]   # no hint -> pass
    assert h.act(obs, np.array([True, False]), 0)[0] == "<action>wave</action>"
    tired = make_external({**HUMAN, "fatigue": 0.0})                # wrong after the first call
    first = tired.act(obs, np.array([True, False]), 0)[0]
    second = tired.act(obs, np.array([True, False]), 0)[0]
    assert first == "<action>open the top drawer</action>" and "nearest object" in second
