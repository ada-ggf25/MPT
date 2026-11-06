# Importing this module registers the three LIBERO agents (Placer / Operator / Stopper).
from .libero_desk_agents import PlacerDeskAgent, OperatorDeskAgent, StopperDeskAgent, DESK_VERBS

__all__ = ["PlacerDeskAgent", "OperatorDeskAgent", "StopperDeskAgent", "DESK_VERBS"]
