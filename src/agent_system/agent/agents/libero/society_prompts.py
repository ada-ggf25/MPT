"""What the LIBERO agents read and may write. Plain text with no imports, because the warm-up
(eom-robotics, stage2/warm_agents.py) must train on exactly what the trainer serves.

Agents partition the action space, as in MPT's crafting and navigation: each writes one short
instruction for a shared pretrained executor (pi0.5), and a parser rejects anything outside its slice.
"""

IMAGE_HEADER = "# Cameras: the scene on the left, the wrist on the right\n<image>\n\n"
SYSTEM = "You are a helpful and harmless assistant."
BID_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)

BID_BODY = (
    "{env_prompt}\n\n"
    "# You are the {desk} agent\n"
    "You are one of {n_desks} specialists ({team}) that share control of one robot arm. A trained "
    "policy carries out the winner's instruction. Exactly ONE agent acts each turn, decided by a "
    "sealed-bid auction you are bidding in now. If you win you may ONLY write: {verbs}. {desk_hint}\n"
    "# Bidding (revenue share, five levels)\n"
    "Your bid is the share of your income this turn you pass back to the agent that acted before you. "
    "The HIGHEST bid wins; only the winner pays a small rent.\n"
    "  0    = pass. My instruction cannot help now; another agent must act first.\n"
    "  0.2  = my instruction barely helps here.\n"
    "  0.4  = my instruction is useful here.\n"
    "  0.6  = my instruction is important here.\n"
    "  0.8  = I am the bottleneck: the task cannot progress without me right now.\n"
    "Reply with exactly one of 0, 0.2, 0.4, 0.6, 0.8 in this form:\n"
    "<bid>0.4</bid>\n"
)

ACT_BODY = (
    "{env_prompt}\n\n"
    "# You are the {desk} agent and you WON this turn's auction\n"
    "You write the single instruction for this turn. A trained robot policy carries it out. "
    "HARD CONSTRAINT, enforced by a parser: you may ONLY write: {verbs}. Anything else is REJECTED, "
    "the turn is wasted, and you still pay. {desk_hint}\n"
    "Reply with one instruction in this form:\n"
    "<action>{example}</action>\n"
)

PLACE = ("put the bowl on the plate | put the cream cheese in the bowl | "
         "put the wine bottle on top of the cabinet | put the wine bottle on the rack")
OPERATE = ("turn on the stove | open the middle drawer of the cabinet | "
           "push the plate to the front of the stove")
STOP = "done"

AGENTS = {
    "Placer Desk (libero)": {"key": "placer", "desk": "PLACER", "first_words": ("put",), "verbs": PLACE,
                             "example": "put the bowl on the plate",
                             "hint": "You move objects to where the task needs them."},
    "Operator Desk (libero)": {"key": "operator", "desk": "OPERATOR", "first_words": ("open", "turn", "push"),
                               "verbs": OPERATE, "example": "turn on the stove",
                               "hint": "You work the stove and the drawer, and push things."},
    "Stopper Desk (libero)": {"key": "stopper", "desk": "STOPPER", "first_words": ("done",),
                              "verbs": STOP, "example": "done",
                              "hint": "You end the episode, only once every goal holds."},
}
# One model holding every slice: the monolithic baseline. Not part of the society.
SINGLE = {"key": "single", "desk": "ONLY", "first_words": ("put", "open", "turn", "push", "done"),
          "verbs": f"{PLACE} | {OPERATE} | {STOP}", "example": "put the bowl on the plate",
          "hint": "You do everything."}


def team_name() -> str:
    return "/".join(a["desk"].capitalize() for a in AGENTS.values())


def fill(body: str, agent: dict, n_desks: int) -> str:
    """Fix an agent's text. .replace, not .format: {env_prompt} survives for each turn."""
    for key, value in (("{desk}", agent["desk"]), ("{n_desks}", str(n_desks)), ("{team}", team_name()),
                       ("{verbs}", agent["verbs"]), ("{desk_hint}", agent["hint"]),
                       ("{example}", agent["example"])):
        body = body.replace(key, value)
    return body


def normalize(instruction: str) -> str:
    """An instruction as the menus compare it: lower case, single spaces, no final full stop."""
    return " ".join(instruction.strip().lower().rstrip(".").split())


def sentences(agent: dict) -> tuple:
    """An agent's menu: the exact sentences it may write."""
    return tuple(normalize(s) for s in agent["verbs"].split("|"))


def agent_for(instruction: str, agents: dict = AGENTS) -> str | None:
    """Which agent's slice an instruction belongs to."""
    words = instruction.strip().lower().split()
    return next((name for name, a in agents.items() if words and words[0] in a["first_words"]), None)
