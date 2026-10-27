# Importing this module registers the three LIBERO desks (Grasp / Place / Articulate).
from .libero_desk_agents import GraspDeskAgent, PlaceDeskAgent, ArticulateDeskAgent, DESK_VERBS

__all__ = ["GraspDeskAgent", "PlaceDeskAgent", "ArticulateDeskAgent", "DESK_VERBS"]
