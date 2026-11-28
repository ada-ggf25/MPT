from __future__ import annotations

from typing import Any, List

import numpy as np
from transformers import PreTrainedTokenizer

from agent_system.agent.agents.libero import ladder_prompts as lp
from agent_system.agent.agents.libero.libero_desk_agents import _LiberoDeskBase
from agent_system.agent.registry import AgentRegistry
from agent_system.environments.env_package.minecraft.envs import extract_instruction

# The LADDER phase agents: the LIBERO desks' bidding and acting, with LADDER's prompts (ladder_prompts.py, shared with the warm-up
# and the plug-ins) and a regular-expression parser in place of LIBERO's sentence menus: the Picker may write
# "pick up the <object>", the Placer "put the <object> on <destination>".


class _LadderDeskBase(_LiberoDeskBase):
    AGENT_NAME = "Desk"

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        from agent_system.agent.agents.base import BaseAgent
        cfg = getattr(getattr(getattr(config, "agent", None), "orchestra", None), "libero_chain", None)

        def _get(k, d):
            return getattr(cfg, k, d) if cfg is not None else d

        self._bid_max = float(_get("bid_max", 1.0))
        self._bid_max_tokens = int(_get("bid_max_tokens", lp.BID_TOKENS))
        self._bid_image = bool(_get("bid_sees_image", True))
        try:
            self._n_desks = len(config.agent.agent_ids)
        except Exception:
            self._n_desks = len(lp.AGENTS)
        self.SPEC = {**lp.AGENTS, "Single Desk (ladder)": lp.SINGLE}[self.AGENT_NAME]
        self.KEY, self.DESK = self.SPEC["key"], self.SPEC["desk"]
        self.VERBS = (self.KEY,)
        self.MENU = (self.SPEC["verbs"],)
        self._exact = True
        assert bool(_get("bid_pass", True)), "MPT uses bid_pass=True (level 0 = pass)"
        self._bid_prompt = (lp.IMAGE_HEADER if self._bid_image else "") + lp.fill(lp.BID_BODY, self.SPEC, self._n_desks)
        BaseAgent.__init__(self, self.AGENT_NAME, lp.IMAGE_HEADER + lp.fill(lp.ACT_BODY, self.SPEC, self._n_desks),
                           wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)

    def enforce(self, text_responses: List[str], mask=None):
        """The parser: a rejected row carries no <action> tag, so the env server answers INVALID with reward 0 and a consumed
        turn (the bid is still paid)."""
        n = len(text_responses)
        mask = np.ones(n, dtype=bool) if mask is None else mask
        emitted, instrs, verbs, oks = [], [], [], []
        for i in range(n):
            if not mask[i]:
                emitted.append(""); instrs.append(None); verbs.append(""); oks.append(False)
                continue
            instr = extract_instruction(text_responses[i] if isinstance(text_responses[i], str) else "")
            ok = bool(instr) and lp.valid(self.KEY, instr)
            instr = lp.normalize(instr) if ok else instr
            instrs.append(instr)
            verbs.append((instr or "").split()[0] if instr else "")
            oks.append(ok)
            emitted.append(f"<action>{instr}</action>" if ok else
                           f"[whitelist-reject] the {self.DESK} agent may only write: {self.SPEC['verbs']}; got {instr!r}")
        return emitted, instrs, verbs, oks


@AgentRegistry.register("Picker Desk (ladder)")
class PickerDeskAgent(_LadderDeskBase):
    AGENT_NAME = "Picker Desk (ladder)"


@AgentRegistry.register("Placer Desk (ladder)")
class LadderPlacerDeskAgent(_LadderDeskBase):
    AGENT_NAME = "Placer Desk (ladder)"


@AgentRegistry.register("Single Desk (ladder)")
class SingleDeskAgent(_LadderDeskBase):
    """The monolithic baseline: one model that may write either kind of instruction (a society of one: allow_single, no competing bids)."""
    AGENT_NAME = "Single Desk (ladder)"
