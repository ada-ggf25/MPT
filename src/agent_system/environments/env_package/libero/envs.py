# LIBERO environment for the MPT trainer -- CLIENT side.
#
# The simulator (LIBERO / MuJoCo) and the low-level executor (pi0.5, served by openpi) live in a
# separate env server: eom-robotics' Stage 2 server (`python -m eom_robotics.stage2.server`), which
# speaks the same JSON/HTTP contract as MPT's Minecraft server (/health, /reset, /step, /snapshot,
# /restore, /step_slot). So the client IS the Minecraft client, pointed at LIBERO_ENV_URL; only the
# health check and the counterfactual binding read that server's own keys.
#
#   MPT VLM agent --(<action>text instruction</action>)--> [server] pi0.5 -> up to K control steps
#                   <--(agent-view frame + status text + goal-condition reward)---------
import os
from typing import List

import requests

from agent_system.environments.env_package.minecraft.envs import MinecraftMultiProcessEnv

LIBERO_ENV_URL = os.environ.get("LIBERO_ENV_URL", "http://127.0.0.1:31700")


class LiberoMultiProcessEnv(MinecraftMultiProcessEnv):
    """MPT env contract, backed by the remote LIBERO slot pool."""

    def __init__(self, seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True,
                 url: str = None):
        super().__init__(seed=seed, env_num=env_num, group_n=group_n, is_train=is_train,
                         url=url or LIBERO_ENV_URL)

    def _check(self):
        total = 0
        for u in self.urls:
            try:
                h = requests.get(f"{u}/health", timeout=30).json()
            except Exception as e:
                raise RuntimeError(
                    f"LIBERO env server unreachable at {u} ({e}). Start it with "
                    f"`python -m eom_robotics.stage2.server` and set LIBERO_ENV_URL.") from e
            n = int(h.get("slots", 0))
            self.slots.append(n)
            self._health.append(h)
            total += n
            print(f"[libero] server OK at {u}: {h}", flush=True)
        if total < self.batch_size:
            raise RuntimeError(
                f"{len(self.urls)} server(s) offer {total} slots but this run needs "
                f"batch_size={self.batch_size}; add servers, raise --slots, or lower the batch")

    def cf_bind(self, timeout: float = None) -> List[int]:
        if timeout is not None:
            self._cf_timeout = float(timeout)
        if len(self.urls) != 1:
            raise RuntimeError(f"cf collector supports exactly ONE LIBERO server, got {self.urls}")
        total, bs = int(self.slots[0]), int(self.batch_size)
        if total < bs + 2:
            raise RuntimeError(
                f"cf collector: server at {self.urls[0]} offers {total} slots but the run needs "
                f"batch_size+2 = {bs + 2} (two spare slots for branch rollouts); relaunch it with "
                f"--slots {bs + 2}.")
        if str((self._health[0] if self._health else {}).get("snapshot")) != "1":
            raise RuntimeError(f"cf collector: server at {self.urls[0]} does not report "
                               "snapshot=1 (snapshot / restore / step_slot endpoints)")
        return [bs, bs + 1]


def build_libero_envs(seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True):
    return LiberoMultiProcessEnv(seed=seed, env_num=env_num, group_n=group_n, is_train=is_train)
