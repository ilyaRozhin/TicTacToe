import TicTacToeEnv
from TicTacToeEnv import TicTacToe
import numpy as np

if __name__ == "__main__":
    env = TicTacToe(start_figure=0)
    episode_actions = np.array([3, 4, 6, 7, 1, 2, 8, 5, 9])
    observations = []
    for action in episode_actions:
        obs, _, _, _ = env.step(action)
        observations.append(obs.copy())
    TicTacToeEnv.render_game(observations, 0.8)

