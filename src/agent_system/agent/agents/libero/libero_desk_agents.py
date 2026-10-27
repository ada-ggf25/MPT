from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import PreTrainedTokenizer

from verl import DataProto
from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.agents.minecraft.minecraft_chain_agent import first_verb
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.utils import parse_open_bid
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.environments.env_package.minecraft.envs import extract_instruction

# The LIBERO desks follow the Minecraft desks (minecraft_chain_agent.py) one for one: a short bid
# probe on the five levels, then the winner's instruction, whose FIRST WORD must be on the desk's
# whitelist (its slice of the action space). The instruction is executed by a frozen low-level
# policy (pi0.5) on the env server. No format-SFT checkpoints exist yet for these prompts.

_IMAGE_HEADER = "# Camera view\n<image>\n\n"

BID_BODY = (
    "{env_prompt}\n\n"
    "# You are the {desk} desk\n"
    "You are one of {n_desks} competing desks that share control of this single robot arm.\n"
    "Exactly ONE desk acts each turn, decided by a sealed-bid auction that you are bidding in now.\n"
    "If you win you must issue an instruction whose FIRST WORD is one of: {verbs}.\n"
    "You cannot issue any other kind of instruction, so a turn you win that your verbs cannot\n"
    "advance is a turn wasted for the whole team, and you still pay the rent.\n"
    "\n"
    "# Bidding (revenue share, five levels)\n"
    "Your bid is the share of your income this turn that you promise to pass back to the desk\n"
    "that acted before you (you are buying the state it produced). Your income is what the\n"
    "environment pays your action plus the share the NEXT winner passes back to you, so value\n"
    "flows backward along the whole chain. The HIGHEST bid wins the turn; only the winner pays\n"
    "a small rent.\n"
    "First look at the camera view and the status lines and decide whether YOUR verbs can advance\n"
    "the task from THIS exact scene. Then bid:\n"
    "  0    = pass. My verbs cannot help right now; another desk must act first.\n"
    "  0.2  = my verbs barely help here.\n"
    "  0.4  = my verbs are useful here.\n"
    "  0.6  = my verbs are important here.\n"
    "  0.8  = I am the bottleneck desk: the task cannot move without my verbs right now.\n"
    "Your bid MUST be exactly one of: 0, 0.2, 0.4, 0.6, 0.8 (no other numbers).\n"
    "Answer with ONE short sentence of reasoning about the scene, then the number, for example:\n"
    "<think>one sentence about what the scene needs next and what my verbs can do</think>\n"
    "<bid>0.4</bid>\n"
)

ACT_BODY = (
    "{env_prompt}\n\n"
    "# You are the {desk} desk and you WON this turn's auction\n"
    "You issue the single instruction for this turn. A low-level robot policy executes it for a\n"
    "few seconds; it understands short, concrete manipulation commands about the objects in view.\n"
    "\n"
    "HARD CONSTRAINT, enforced by a parser and not negotiable: the FIRST WORD of your instruction\n"
    "must be one of: {verbs}\n"
    "Anything else is REJECTED before it reaches the robot: the turn is wasted, nothing happens,\n"
    "and you still pay your promised share. {desk_hint}\n"
    "\n"
    "Keep it short and concrete. Example: <action>{example}</action>\n"
    "Reply with your reasoning, then the instruction wrapped EXACTLY like this:\n"
    "<action>your short instruction here</action>\n"
)


class _LiberoDeskBase(BaseAgent):
    AGENT_NAME = "Desk"
    DESK = "Desk"
    VERBS: Tuple[str, ...] = ()
    EXAMPLE = "pick up the black bowl"
    HINT = ""

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        cfg = getattr(getattr(getattr(config, "agent", None), "orchestra", None), "libero_chain", None)

        def _get(k, d):
            return getattr(cfg, k, d) if cfg is not None else d

        self._bid_max = float(_get("bid_max", 1.0))
        self._bid_max_tokens = int(_get("bid_max_tokens", 96))
        self._bid_image = bool(_get("bid_sees_image", True))
        try:
            self._n_desks = len(config.agent.agent_ids)
        except Exception:
            self._n_desks = 3

        def _fill(body: str) -> str:
            # .replace, not .format: only {env_prompt} survives into the per-turn formatting
            for k, v in (("{desk}", self.DESK), ("{verbs}", " | ".join(self.VERBS)),
                         ("{example}", self.EXAMPLE), ("{desk_hint}", self.HINT),
                         ("{n_desks}", str(self._n_desks))):
                body = body.replace(k, v)
            return body

        assert bool(_get("bid_pass", True)), "MPT uses bid_pass=True (level 0 = pass)"
        self._bid_prompt = (_IMAGE_HEADER if self._bid_image else "") + _fill(BID_BODY)
        super().__init__(self.AGENT_NAME, _IMAGE_HEADER + _fill(ACT_BODY), wg_id=wg_id,
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
                emitted.append(f"[whitelist-reject] the {self.DESK} desk may only emit instructions "
                               f"whose first word is one of {'|'.join(self.VERBS)}; got {fw!r}")
        return emitted, instrs, verbs, oks


class _GraspDesk(_LiberoDeskBase):
    DESK = "GRASP"
    VERBS = ("pick", "grasp", "lift", "take")
    EXAMPLE = "pick up the black bowl"
    HINT = "You are the grasping desk: getting an object into the gripper."


class _PlaceDesk(_LiberoDeskBase):
    DESK = "PLACE"
    VERBS = ("put", "place", "stack", "drop")
    EXAMPLE = "put the bowl on the plate"
    HINT = "You are the placing desk: putting a held object where the task needs it."


class _ArticulateDesk(_LiberoDeskBase):
    DESK = "ARTICULATE"
    VERBS = ("open", "close", "turn", "push")
    EXAMPLE = "open the top drawer of the cabinet"
    HINT = "You are the articulation desk: drawers, doors, knobs, switches and pushing objects."


@AgentRegistry.register("Grasp Desk (libero)")
class GraspDeskAgent(_GraspDesk):
    AGENT_NAME = "Grasp Desk (libero)"


@AgentRegistry.register("Place Desk (libero)")
class PlaceDeskAgent(_PlaceDesk):
    AGENT_NAME = "Place Desk (libero)"


@AgentRegistry.register("Articulate Desk (libero)")
class ArticulateDeskAgent(_ArticulateDesk):
    AGENT_NAME = "Articulate Desk (libero)"


DESK_VERBS = {
    "Grasp Desk (libero)": _GraspDesk.VERBS,
    "Place Desk (libero)": _PlaceDesk.VERBS,
    "Articulate Desk (libero)": _ArticulateDesk.VERBS,
}

__all__ = ["GraspDeskAgent", "PlaceDeskAgent", "ArticulateDeskAgent", "DESK_VERBS"]
