from typing import Iterable, List

import numpy as np


class PopulationLedger:
    """EoM's population economy across episodes, which MPT's fixed society does not have.

    Each agent carries a wealth from one episode to the next: on every turn it WON it earns the
    settled profit minus its rent (exactly the credit its rows train on), averaged over the
    batch's parallel episodes. With `bankruptcy` on, an agent whose wealth drops below
    `threshold` becomes unavailable: it is no longer asked to bid, so it can no longer win.
    `available` is also the hook for agents leaving and joining a deployed society (flexibility
    tests) -- see ChainTurnEngine.run_turn(available=...).

    With the defaults (no bankruptcy, everyone available) the ledger only keeps accounts and
    never changes who may bid.
    """

    def __init__(self, n: int, *, wealth_init: float = 1.0, bankruptcy: bool = False,
                 threshold: float = 0.0, unavailable: Iterable[int] = ()):
        self.n = int(n)
        self.wealth_init = float(wealth_init)
        self.bankruptcy = bool(bankruptcy)
        self.threshold = float(threshold)
        self.wealth = np.full(self.n, self.wealth_init, dtype=np.float64)
        self.available = np.ones(self.n, dtype=bool)
        self.bankrupt = np.zeros(self.n, dtype=bool)
        for i in unavailable:
            self.available[int(i)] = False

    @property
    def everyone_available(self) -> bool:
        return bool(self.available.all())

    def set_available(self, i: int, flag: bool, reset_wealth: bool = True) -> None:
        """An agent leaves (flag=False) or (re)joins the society. A joining agent starts with
        the initial wealth, like an EoM newcomer, and is no longer counted as bankrupt."""
        self.available[i] = bool(flag)
        if flag:
            self.bankrupt[i] = False
            if reset_wealth:
                self.wealth[i] = self.wealth_init

    def settle(self, chosen: List[np.ndarray], profit: np.ndarray, rents: np.ndarray,
               term_at: np.ndarray) -> np.ndarray:
        """Book one batch of episodes. `chosen[t][s]` is the winner of turn t in sample s (-1 if
        inactive), `profit` (T, bs) the engine's settled profit, `rents` (n,) the per-agent rent,
        `term_at[s]` the sample's last turn (-1 = never active). Returns the per-agent earnings
        added to the wealth (mean over the batch's episodes)."""
        T = len(chosen)
        earn = np.zeros(self.n, dtype=np.float64)
        n_eps = 0
        for s in range(profit.shape[1] if T else 0):
            last = int(term_at[s])
            if last < 0:
                continue
            n_eps += 1
            for t in range(min(last + 1, T)):
                w = int(chosen[t][s])
                if w >= 0:
                    earn[w] += float(profit[t, s]) - float(rents[w])
        if n_eps:
            earn /= n_eps
        self.wealth += earn
        if self.bankruptcy:
            newly = self.available & (self.wealth < self.threshold)
            self.available[newly] = False
            self.bankrupt |= newly
        return earn
