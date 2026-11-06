# Importing this module registers the three LIBERO agents (Mover / Hand / Stopper).
from .libero_desk_agents import MoverDeskAgent, HandDeskAgent, StopperDeskAgent, DESK_VERBS

__all__ = ["MoverDeskAgent", "HandDeskAgent", "StopperDeskAgent", "DESK_VERBS"]
