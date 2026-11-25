"""What the LADDER phase agents (Picker, Placer) read and may write. Plain text, standard library only, because the warm-up and the
LADDER plug-ins (eom-robotics, ladder_spike/phases.py) load this file BY PATH and must train and act on exactly what the trainer serves.

The agents split the action space BY PHASE of a pick-and-place (Vault Design-2026-11-25-Agent-Roles-on-LADDER): the Picker writes
``pick up the <object>`` and the Placer ``put the <object> on <destination>``; a parser rejects anything else, as MPT's slices do.
A frozen low-level policy (LADDER's Bridge policy) carries out the winner's instruction for the next ~20 control steps.

Replies follow MPT's crafting prompts (paper App. B.1.4): a bid is ONE short sentence of reasoning, then the level; an action is
reasoning, then the instruction in ``<action>`` tags (the env server and MPT's projection read ``<action>``).
"""
import re

IMAGE_HEADER = "# Camera: the workspace as the robot sees it now\n<image>\n\n"
SYSTEM = "You are a helpful and harmless assistant."
BID_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)
OBJECTS = ("carrot", "spoon", "eggplant")
DESTINATIONS = ("the plate", "the blue towel")
CADENCE = 20  # control steps between decisions

BID_BODY = (
    "{env_prompt}\n\n"
    "# You are the {desk} agent\n"
    "You are one of {n_desks} specialists ({team}) that share control of one robot arm. A trained low-level policy carries out ONE "
    "short instruction at a time, such as \"pick up the carrot\" or \"put the carrot on the plate\", and works on it until the next "
    "decision (about " + str(CADENCE) + " control steps). Exactly ONE agent writes each decision's instruction, decided by a sealed-bid "
    "auction you are bidding in now. If you win you may ONLY write: {verbs}. {desk_hint}\n"
    "# Bidding (revenue share, five levels)\n"
    "Your bid is the share of your income this turn you pass back to the agent that acted before you "
    "(you are buying the state it produced). The HIGHEST bid wins; only the winner pays a small rent.\n"
    "First read the gripper line and the picture and decide whether YOUR instructions can advance the task from THIS exact state. "
    "Then bid:\n"
    "  0    = pass. My instruction cannot help now; another agent must act first.\n"
    "  0.2  = my instruction barely helps here.\n"
    "  0.4  = my instruction is useful here.\n"
    "  0.6  = my instruction is important here.\n"
    "  0.8  = I am the bottleneck: the task cannot progress without me right now.\n"
    "Answer with ONE short sentence of reasoning that cites the picture or the gripper, then the number, for example:\n"
    "<think>one sentence about what the gripper holds and what my instructions can do</think>\n"
    "<bid>0.4</bid>\n"
)

ACT_BODY = (
    "{env_prompt}\n\n"
    "# You are the {desk} agent and you WON this turn's auction\n"
    "You write the single instruction for this decision. A trained robot policy carries it out. "
    "HARD CONSTRAINT, enforced by a parser: you may ONLY write: {verbs}. Anything else is REJECTED, "
    "the turn is wasted, and you still pay. {desk_hint}\n"
    "Reply with <think>your reasoning</think> then the instruction wrapped EXACTLY like this:\n"
    "<action>{example}</action>\n"
)

# MPT's response caps for the crafting setting (Table 3): an action 320 tokens, a bid probe 96.
ACT_TOKENS, BID_TOKENS = 320, 96

PICK = "pick up the <object>, where <object> is carrot, spoon or eggplant"
PLACE = ("put the <object> on <destination>, where <object> is carrot, spoon or eggplant and <destination> is the plate or "
         "the blue towel")
AGENTS = {
    "Picker Desk (ladder)": {"key": "picker", "desk": "PICKER", "verbs": PICK, "example": "pick up the carrot",
                             "hint": "You send the arm to an object and have it grasp it. You cannot say where to put it."},
    "Placer Desk (ladder)": {"key": "placer", "desk": "PLACER", "verbs": PLACE, "example": "put the carrot on the plate",
                             "hint": "You have the arm carry what it holds to its destination and let go. You cannot ask it to "
                                     "pick something up."},
}
# One model holding both slices: the monolithic baseline. Not part of the society.
SINGLE = {"key": "single", "desk": "ONLY", "verbs": f"{PICK}; or {PLACE}", "example": "pick up the carrot",
          "hint": "You do everything: pick up an object while the gripper is empty, put it down where the task says once it holds it."}

_PICK = re.compile(r"^pick up the (carrot|spoon|eggplant)$")
_PUT = re.compile(r"^put the (carrot|spoon|eggplant) on (the plate|the blue towel)$")


def team_name() -> str:
    return "/".join(a["desk"].capitalize() for a in AGENTS.values())


def fill(body: str, agent: dict, n_desks: int) -> str:
    """Fix an agent's text. .replace, not .format: {env_prompt} survives for each turn."""
    for key, value in (("{desk}", agent["desk"]), ("{n_desks}", str(n_desks)), ("{team}", team_name()),
                       ("{verbs}", agent["verbs"]), ("{desk_hint}", agent["hint"]), ("{example}", agent["example"])):
        body = body.replace(key, value)
    return body


def normalize(instruction: str) -> str:
    """An instruction as the parser compares it: lower case, single spaces, no final full stop."""
    return " ".join((instruction or "").strip().lower().rstrip(".").split())


def owner_key(instruction: str):
    """Which agent's slice an instruction belongs to ('picker', 'placer'), or None."""
    text = normalize(instruction)
    return "picker" if _PICK.match(text) else "placer" if _PUT.match(text) else None


def valid(key: str, instruction: str) -> bool:
    """The parser rule: the agent ``key`` may write this instruction (the single model may write either)."""
    who = owner_key(instruction)
    return who is not None if key == "single" else who == key
