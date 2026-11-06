from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import PreTrainedTokenizer

from verl import DataProto
from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.agents.libero.command_prompts import ACT_BODY, AGENTS, BID_BODY, IMAGE_HEADER, fill
from agent_system.agent.agents.minecraft.minecraft_chain_agent import first_verb
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.utils import parse_open_bid
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.environments.env_package.minecraft.envs import extract_instruction

# The LIBERO agents follow the navigation desks one for one: each owns a slice of the action space,
# here movement commands for a shared controller on the env server, and a parser enforces it. The
# prompts and slices live in command_prompts.py, because the warm-up must train on the same text.


class _LiberoDeskBase(BaseAgent):
    AGENT_NAME = "Desk"

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        cfg = getattr(getattr(getattr(config, "agent", None), "orchestra", None), "libero_chain", None)

        def _get(k, d):
            return getattr(cfg, k, d) if cfg is not None else d

        self._bid_max = float(_get("bid_max", 1.0))
        self._bid_max_tokens = int(_get("bid_max_tokens", 16))
        self._bid_image = bool(_get("bid_sees_image", True))
        try:
            self._n_desks = len(config.agent.agent_ids)
        except Exception:
            self._n_desks = 3

        self.SPEC = AGENTS[self.AGENT_NAME]
        self.DESK, self.VERBS = self.SPEC["desk"], self.SPEC["first_words"]

        assert bool(_get("bid_pass", True)), "MPT uses bid_pass=True (level 0 = pass)"
        self._bid_prompt = (IMAGE_HEADER if self._bid_image else "") + fill(BID_BODY, self.SPEC, self._n_desks)
        super().__init__(self.AGENT_NAME, IMAGE_HEADER + fill(ACT_BODY, self.SPEC, self._n_desks), wg_id=wg_id,
                         tokenizer=tokenizer, processor=processor, config=config)

    def _build_obs(self, template: str, env_obs, team_context, step, with_image: bool):
        obs = {k: v for k, v in env_obs.items() if k not in ('text', 'image')}
        obs['image'] = env_obs.get('image') if with_image else None
        obs['text'] = [template.format(env_prompt=t, team_context=team_context[i], step=step)
                       for i, t in enumerate(env_obs['text'])]
        return obs

    def build_prompt(self, env_obs, team_context, step):
        return self._build_obs(self.prompt, env_obs, team_context, step, with_image=True)

    def bid_call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str],
                 actor_rollout_wg, agent_active_mask, step: int) -> Tuple[DataProto, np.ndarray]:
        obs = self._build_obs(self._bid_prompt, env_obs, team_context, step, with_image=self._bid_image)
        batch = preprocess_batch(gen_batch=gen_batch, obs=obs, config=self.config,
                                 tokenizer=self.tokenizer, processor=self.processor)
        meta = {**gen_batch.meta_info,
                'sampling_overrides': {'max_new_tokens': self._bid_max_tokens, 'stop': ['</bid>']}}
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, meta, kind="bid")
        bids, has_bid, _ = parse_open_bid(text_responses, self._bid_max)
        n = len(text_responses)
        batch.non_tensor_batch['bid'] = bids.astype(np.float32)
        batch.non_tensor_batch['raw_response'] = np.array(list(text_responses), dtype=object)
        batch.non_tensor_batch['is_action_valid'] = has_bid
        batch.non_tensor_batch['env_step'] = np.array([step] * n, dtype=object)
        batch.non_tensor_batch['call_kind'] = np.array(['bid'] * n, dtype=object)
        return batch, bids.astype(np.float32)

    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str],
             actor_rollout_wg, agent_active_mask, step: int) -> Tuple[DataProto, List[str]]:
        obs = self.build_prompt(env_obs, team_context, step)
        batch = preprocess_batch(gen_batch=gen_batch, obs=obs, config=self.config,
                                 tokenizer=self.tokenizer, processor=self.processor)
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask,
                                                        gen_batch.meta_info, kind="act")
        emitted, instrs, verbs, oks = self.enforce(text_responses, agent_active_mask)
        n = len(text_responses)
        batch.non_tensor_batch['raw_response'] = np.array(list(text_responses), dtype=object)
        batch.non_tensor_batch['instruction'] = np.array(instrs, dtype=object)
        batch.non_tensor_batch['first_word'] = np.array(verbs, dtype=object)
        batch.non_tensor_batch['whitelist_ok'] = np.array(oks, dtype=bool)
        batch.non_tensor_batch['is_action_valid'] = np.array(oks, dtype=bool)
        batch.non_tensor_batch['env_step'] = np.array([step] * n, dtype=object)
        batch.non_tensor_batch['call_kind'] = np.array(['act'] * n, dtype=object)
        return batch, emitted

    def enforce(self, text_responses: List[str], mask=None):
        """Whitelist gate, as in the Minecraft desks: a rejected row carries no <action> tag, so the
        env server answers INVALID with reward 0 and a consumed turn (the bid is still paid)."""
        n = len(text_responses)
        if mask is None:
            mask = np.ones(n, dtype=bool)
        emitted, instrs, verbs, oks = [], [], [], []
        for i in range(n):
            if not mask[i]:
                emitted.append(""); instrs.append(None); verbs.append(""); oks.append(False)
                continue
            instr = extract_instruction(text_responses[i] if isinstance(text_responses[i], str) else "")
            fw = first_verb(instr)
            ok = bool(instr) and fw in self.VERBS
            instrs.append(instr)
            verbs.append(fw)
            oks.append(ok)
            if ok:
                emitted.append(f"<action>{instr}</action>")
            else:
                emitted.append(f"[whitelist-reject] the {self.DESK} agent may only write commands "
                               f"starting with {'|'.join(self.VERBS)}; got {fw!r}")
        return emitted, instrs, verbs, oks


@AgentRegistry.register("Mover Desk (libero)")
class MoverDeskAgent(_LiberoDeskBase):
    AGENT_NAME = "Mover Desk (libero)"


@AgentRegistry.register("Hand Desk (libero)")
class HandDeskAgent(_LiberoDeskBase):
    AGENT_NAME = "Hand Desk (libero)"


@AgentRegistry.register("Stopper Desk (libero)")
class StopperDeskAgent(_LiberoDeskBase):
    AGENT_NAME = "Stopper Desk (libero)"


DESK_VERBS = {name: spec["first_words"] for name, spec in AGENTS.items()}

__all__ = ["MoverDeskAgent", "HandDeskAgent", "StopperDeskAgent", "DESK_VERBS"]
