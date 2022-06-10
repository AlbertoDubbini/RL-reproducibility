import gym
import numpy as np

from ..mcts.mcts import MCTS
import stoch_mcts.mcts.selector as sel

class Agent:
    def __init__(self, mcts: MCTS):
        self.mcts = mcts

    def act(self, env: gym.Env, state: np.ndarray, n_rollouts: int) -> int:
        root = self.mcts(env, state, n_rollouts)
        vals = root.avg_vals()
        # print("agent: vals = ", root.avg_vals())
        if n_rollouts == 0:
            output = sel.wikipedia_policy(root)
        else:
            output = [np.random.choice(np.flatnonzero(vals == vals.max())), vals]
            # output = np.random.choice(np.flatnonzero(vals == vals.max()))
        # print("agent: reward = ", root.avg_vals())
        # print("agent: np.flatnonzero(vals == vals.max()) = ", np.flatnonzero(vals == vals.max()))
            # print("agent: output = ", output)
        return output
