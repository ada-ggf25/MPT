"""What the LIBERO agents read and may write. Plain text with no imports, because the warm-up
(eom-robotics, stage2/warm_agents.py) must train on exactly what the trainer serves.

Agents partition the action space, as in MPT's navigation (Turner / Mover / Stopper): each writes one
movement command for a shared controller, and a parser rejects anything outside its slice.
"""

IMAGE_HEADER = "# Cameras: the scene on the left, the wrist on the right\n<image>\n\n"
SYSTEM = "You are a helpful and harmless assistant."
BID_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)

BID_BODY = (
    "{env_prompt}\n\n"
    "# You are the {desk} agent\n"
    "You are one of {n_desks} specialists ({team}) that share control of one robot arm whose gripper "
    "points straight down. Exactly ONE agent acts each turn, decided by a sealed-bid auction you are "
    "bidding in now. If you win you may ONLY write: {verbs}. {desk_hint}\n"
    "# Bidding (revenue share, five levels)\n"
    "Your bid is the share of your income this turn you pass back to the agent that acted before you. "
    "The HIGHEST bid wins; only the winner pays a small rent.\n"
    "  0    = pass. My command cannot help now; another agent must act first.\n"
    "  0.2  = my command barely helps here.\n"
    "  0.4  = my command is useful here.\n"
    "  0.6  = my command is important here.\n"
    "  0.8  = I am the bottleneck: the task cannot progress without my command right now.\n"
    "Reply with exactly one of 0, 0.2, 0.4, 0.6, 0.8 in this form:\n"
    "<bid>0.4</bid>\n"
)

ACT_BODY = (
    "{env_prompt}\n\n"
    "# You are the {desk} agent and you WON this turn's auction\n"
    "You write the single command for this turn, for a robot arm whose gripper points straight down. "
    "HARD CONSTRAINT, enforced by a parser: you may ONLY write: {verbs}. Anything else is REJECTED, "
    "the turn is wasted, and you still pay. {desk_hint}\n"
    "Reply with one command in this form:\n"
    "<action>{example}</action>\n"
)

MOVE = "move x=<cm> y=<cm> z=<cm> (each from -10 to +10)"
HAND = "gripper open | gripper close | rotate yaw=<degrees> (from -45 to +45)"
STOP = "done"

AGENTS = {
    "Mover Desk (libero)": {"key": "mover", "desk": "MOVER", "first_words": ("move",), "verbs": MOVE,
                            "example": "move x=+5 y=-2 z=+0",
                            "hint": "You place the gripper: reach, descend, lift and carry."},
    "Hand Desk (libero)": {"key": "hand", "desk": "HAND", "first_words": ("gripper", "rotate"),
                           "verbs": HAND, "example": "gripper close",
                           "hint": "You work the gripper: grasp, release and turn."},
    "Stopper Desk (libero)": {"key": "stopper", "desk": "STOPPER", "first_words": ("done",),
                              "verbs": STOP, "example": "done",
                              "hint": "You end the episode, only once every goal holds."},
}
# One model holding every slice: the monolithic baseline. Not part of the society.
SINGLE = {"key": "single", "desk": "ONLY", "first_words": ("move", "gripper", "rotate", "done"),
          "verbs": f"{MOVE} | {HAND} | {STOP}", "example": "move x=+5 y=-2 z=+0",
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


def agent_for(command: str, agents: dict = AGENTS) -> str | None:
    """Which agent's slice a command belongs to."""
    words = command.strip().lower().split()
    return next((name for name, a in agents.items() if words and words[0] in a["first_words"]), None)
