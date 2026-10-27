from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import numpy as np

from agent_system.agent.orchestra.chain_base import ChainOrchestraBase

# The env server (eom_robotics.stage2) renders, per turn:
#     TASK: put both the alphabet soup and the tomato sauce in the basket
#     Decision 3/12 · goal conditions met: 1/2
#     Last: executed 'pick up the tomato sauce'
_GOALS_RE = re.compile(r"(?:goal conditions met|sub-goals done):\s*(\d+)\s*/\s*(\d+)")
_DEC_RE = re.compile(r"Decision\s+(\d+)\s*/\s*(\d+)")


def parse_progress(text: str) -> Dict[str, Any]:
    text = text if isinstance(text, str) else ""
    m_g, m_d = _GOALS_RE.search(text), _DEC_RE.search(text)
    return {"goals_met": int(m_g.group(1)) if m_g else None,
            "goals_total": int(m_g.group(2)) if m_g else None,
            "decision": int(m_d.group(1)) if m_d else None}


class LiberoChainOrchestra(ChainOrchestraBase):
    """MPT auction over LIBERO env turns (Grasp / Place / Articulate desks; a frozen pi0.5
    executes the winner's instruction on the env server). The training logic is
    ChainOrchestraBase; this class adds the progress parsing and the trace."""

    DOMAIN_BLOCK = "libero_chain"
    AGENT_MODULE = "agent_system.agent.agents.libero"
    ALT_COUNTER_ENV = "LIBERO_ALT_COUNTER"
    TRACE_ENV = "LIBERO_CHAIN_TRACE"
    LOG_TAG = "libero-chain"
    SEG_SCALE_DEFAULT = True

    def _init_domain(self):
        self._trace_text_max = int(os.environ.get("LIBERO_CHAIN_TRACE_TEXT_MAX", "800"))

    def _turn_context(self, env_obs, bs):
        raw = env_obs.get('anchor')
        if raw is None or len(raw) != bs:
            raw = env_obs['text']
        return [parse_progress(raw[i]) for i in range(bs)]

    def _after_turn(self, t, step, bs, active, res, bid_texts, acts, ctx):
        if self._trace_path:
            recs: List[dict] = []
            for s in np.nonzero(active)[0]:
                w = int(res.winner[s])
                recs.append({
                    "kind": "turn", "pid": self._pid, "episode": self._ep_seq, "sample": int(s),
                    "train": self._is_train, "env_step": int(step), "turn": int(t),
                    "bids": {self.desk_ids[i]: round(float(res.raw_bids[i][s]), 4)
                             for i in range(self.n) if res.bidders[i][s]},
                    "bid_texts": {self.desk_ids[i]: str(bid_texts[i][s])[:self._trace_text_max]
                                  for i in range(self.n) if bid_texts.get(i) is not None},
                    "bid_forced": bool(res.bid_forced[s]),
                    "winner": self.desk_ids[w], "price_paid": round(float(res.price[s]), 4),
                    "instruction": acts["instr"][s], "first_word": acts["verb"][s],
                    "whitelist_ok": bool(acts["ok"][s]), "raw_response": acts["raw"][s],
                    "progress_before": ctx[s] if ctx is not None else None,
                })
            self._trace(recs)
        nwin = int(active.sum())
        by_desk = {self.desk_ids[i]: int((res.chosen == i).sum()) for i in range(self.n)}
        passes = {self.desk_ids[i]: int((active & res.bidders[i] & (res.raw_bids[i] == 0.0)).sum())
                  for i in range(self.n)}
        print(f"[{self.LOG_TAG}] turn={t} active={nwin} wins={by_desk} passes={passes} "
              f"bid_fallbacks={self._bid_fallbacks} "
              f"beta[mean]={float(res.price[active].mean()) if nwin else 0.0:.2f} "
              f"whitelist_ok={int(sum(acts['ok']))}/{nwin}", flush=True)

    def _after_finalize(self, bundle, bs, episode_rewards, is_train):
        self._episode_summary_trace(bs, episode_rewards)
