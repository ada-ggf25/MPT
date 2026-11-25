# Importing this module registers the three LIBERO agents (Placer / Operator / Stopper).
from .libero_desk_agents import PlacerDeskAgent, OperatorDeskAgent, StopperDeskAgent, DESK_VERBS

__all__ = ["PlacerDeskAgent", "OperatorDeskAgent", "StopperDeskAgent", "DESK_VERBS"]

# The LADDER phase agents (Picker / Placer) register in the same package, so the LIBERO orchestra and env manager serve them.
from . import ladder_desk_agents  # noqa: E402,F401
