"""External agents: auction participants WITHOUT trainable weights.

A person (simulated or real) or an API model (e.g. a frontier VLM) can join an MPT society. It
bids in the same sealed-bid auction, can win the turn and act, is settled by the same revenue
share, pays its own rent, and is booked in the population ledger -- but it has no worker group,
so the trainer never sees its rows. The trainable agents learn around it through the market:
passing on a turn they cannot advance lets the external agent act, and buying the state it
produces costs them their bid.

External agents are declared in the orchestra block, NOT in agent.agent_ids (which the trainer
turns into models):

    agent.orchestra.<block>.external_agents:
      - {name: "Human (oracle)", kind: oracle_human, rent: 0.5, bid: 0.2, p_correct: 0.9}
      - {name: "Frontier (API)", kind: api_vlm, rent: 0.3, model: gpt-6-astra,
         base_url: https://api.openai.com/v1, api_key_env: OPENAI_API_KEY}

Interface: bid(env_obs, mask, step) -> float[bs] and act(env_obs, mask, step) -> List[str] (the
emitted env actions, `<action>...</action>`), both answering only where mask is True.
"""
from __future__ import annotations

import base64
import io
import os
import re
from typing import Any, Dict, List

import numpy as np

EXTERNAL_AGENTS: Dict[str, type] = {}


def register_external(kind: str):
    def deco(cls):
        if kind in EXTERNAL_AGENTS:
            raise ValueError(f"external agent kind {kind!r} already registered")
        EXTERNAL_AGENTS[kind] = cls
        return cls
    return deco


def make_external(spec: Any) -> "ExternalAgent":
    spec = dict(spec)
    kind = spec.get("kind")
    if kind not in EXTERNAL_AGENTS:
        raise KeyError(f"unknown external agent kind {kind!r}; registered: {list(EXTERNAL_AGENTS)}")
    return EXTERNAL_AGENTS[kind](spec)


class ExternalAgent:
    trainable = False

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.name = str(spec["name"])

    def reset(self):
        pass

    def bid(self, env_obs: Dict[str, Any], mask: np.ndarray, step: int) -> np.ndarray:
        raise NotImplementedError

    def act(self, env_obs: Dict[str, Any], mask: np.ndarray, step: int) -> List[str]:
        raise NotImplementedError


@register_external("oracle_human")
class OracleHumanAgent(ExternalAgent):
    """A simulated person who knows the next step: the env server's privileged
    `env_obs['privileged'][s]['oracle_next']` (the next correct sub-instruction).

    Flaws, all optional, so the economy has something to audit:
      p_correct      probability the advice is right (else `wrong_instruction` is given);
      unreliable_when / p_correct_there   a substring of the status text where the person is
                     worse (e.g. a task they don't know), with its own probability;
      fatigue        multiplies the probability by fatigue**calls (calls this episode).
    Bidding: `bid` (a level, default the 0.2 floor) whenever a hint exists, otherwise pass (0).
    A real person would answer the same two questions through a UI; this class is the stand-in.
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.level = float(spec.get("bid", 0.2))
        self.p_correct = float(spec.get("p_correct", 1.0))
        self.unreliable_when = spec.get("unreliable_when")
        self.p_correct_there = float(spec.get("p_correct_there", self.p_correct))
        self.fatigue = float(spec.get("fatigue", 1.0))
        self.wrong = str(spec.get("wrong_instruction", "pick up the nearest object"))
        self.rng = np.random.default_rng(int(spec.get("seed", 0)))
        self.calls = 0

    def reset(self):
        self.calls = 0

    @staticmethod
    def _hint(env_obs, s):
        priv = env_obs.get("privileged")
        if priv is None or s >= len(priv) or not priv[s]:
            return None
        return priv[s].get("oracle_next")

    def bid(self, env_obs, mask, step):
        out = np.zeros(len(mask), dtype=np.float64)
        for s in np.nonzero(mask)[0]:
            out[s] = self.level if self._hint(env_obs, s) else 0.0
        return out

    def act(self, env_obs, mask, step):
        texts = env_obs.get("anchor") or env_obs.get("text") or []
        out = [""] * len(mask)
        for s in np.nonzero(mask)[0]:
            hint = self._hint(env_obs, s)
            p = self.p_correct
            if self.unreliable_when and s < len(texts) and self.unreliable_when in str(texts[s]):
                p = self.p_correct_there
            p *= self.fatigue ** self.calls
            self.calls += 1
            instr = hint if (hint and self.rng.random() < p) else self.wrong
            out[s] = f"<action>{instr}</action>"
        return out


_BID_RE = re.compile(r"<bid>\s*([0-9]*\.?[0-9]+)")


@register_external("api_vlm")
class ApiVlmAgent(ExternalAgent):
    """A VLM behind an OpenAI-compatible chat-completions endpoint (a frontier model, or a large
    open model served with vLLM / sglang). Same two-step protocol as the desks: a short bid probe
    ending in <bid>x</bid>, then, if it wins, one <action>...</action>. Bids snap in the engine.

    Spec: model, base_url (default https://api.openai.com/v1), api_key_env (default
    OPENAI_API_KEY), bid_prompt / act_prompt (strings with {status}; defaults below), verbs
    (informational, shown in the prompt), max_tokens, timeout. Untested against a live API.
    """

    BID_PROMPT = (
        "{status}\n\nYou are the FRONTIER desk: one of several agents that share control of a robot "
        "arm. Each turn they bid in a sealed-bid auction; the highest bid wins and gives the next "
        "instruction, which a low-level policy executes. Your bid is the share of your income you "
        "pass back to the previous winner. You are expensive: bid only where you add value.\n"
        "Bid exactly one of 0 (pass), 0.2, 0.4, 0.6, 0.8. Answer with one sentence of reasoning "
        "then <bid>x</bid>.")
    ACT_PROMPT = (
        "{status}\n\nYou are the FRONTIER desk and you won this turn. Give ONE short, concrete "
        "manipulation instruction for the low-level policy, wrapped as <action>...</action>.")

    def __init__(self, spec):
        super().__init__(spec)
        self.model = str(spec["model"])
        self.base_url = str(spec.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = os.environ.get(str(spec.get("api_key_env", "OPENAI_API_KEY")), "")
        self.bid_prompt = str(spec.get("bid_prompt", self.BID_PROMPT))
        self.act_prompt = str(spec.get("act_prompt", self.ACT_PROMPT))
        self.max_tokens = int(spec.get("max_tokens", 256))
        self.timeout = float(spec.get("timeout", 120))

    @staticmethod
    def _image_url(img):
        if img is None:
            return None
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _ask(self, prompt: str, image) -> str:
        import requests

        content = [{"type": "text", "text": prompt}]
        url = self._image_url(image)
        if url:
            content.append({"type": "image_url", "image_url": {"url": url}})
        r = requests.post(f"{self.base_url}/chat/completions", timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          json={"model": self.model, "max_tokens": self.max_tokens,
                                "messages": [{"role": "user", "content": content}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    def _per_sample(self, env_obs, mask, template):
        texts = env_obs.get("anchor") or env_obs.get("text") or []
        images = env_obs.get("image") or [None] * len(mask)
        out = {}
        for s in np.nonzero(mask)[0]:
            try:
                out[s] = self._ask(template.format(status=texts[s]), images[s])
            except Exception as e:  # an API failure is a pass / an invalid action, never a crash
                print(f"[external:{self.name}] API call failed: {e}", flush=True)
                out[s] = ""
        return out

    def bid(self, env_obs, mask, step):
        out = np.zeros(len(mask), dtype=np.float64)
        for s, text in self._per_sample(env_obs, mask, self.bid_prompt).items():
            m = _BID_RE.search(text)
            out[s] = float(np.clip(float(m.group(1)), 0.0, 1.0)) if m else 0.0
        return out

    def act(self, env_obs, mask, step):
        out = [""] * len(mask)
        for s, text in self._per_sample(env_obs, mask, self.act_prompt).items():
            out[s] = text
        return out
