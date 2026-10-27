from .envs import build_libero_envs
# Same validity rule as Minecraft: an action is valid iff it carries a non-empty <action>...</action>.
from agent_system.environments.env_package.minecraft.projection import minecraft_projection as libero_projection
