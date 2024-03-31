import numpy as np
import time
import os


class TicTacToe:
    def __init__(self, start_figure):
        self.observation_space = np.zeros(shape=(3, 3), dtype=int)
        self.action_space = np.zeros(shape=(9, 1), dtype=int)
        self.start_figure = start_figure
        self.counter = 0

    def step(self, action):
        cords = ((action-1)//3, (action-1) % 3)
        self.observation_space[cords] = 1 if (self.counter % 2 + self.start_figure) == 1 else -1
        self.counter += 1
        if self.action_space[action-1] == 0:
            self.action_space[action-1] = 1
            is_done = self.game_is_done()
            reward = 1 if is_done else 0
            if not is_done and self.counter == 9:
                is_done = True
            return self.observation_space, self.action_space, reward, is_done
        return self.observation_space, self.action_space, -10, False

    def game_is_done(self):
        main_sums = np.concatenate([self.observation_space.sum(axis=0), self.observation_space.sum(axis=1),
                                   self.diagonal_check()], axis=0)
        if 3 in main_sums:
            return True
        elif -3 in main_sums:
            return True
        else:
            return False

    def diagonal_check(self):
        main_diagonal = 0
        sub_diagonal = 0
        for i in range(3):
            main_diagonal += self.observation_space[i][i]
            sub_diagonal += self.observation_space[2-i][i]
        return np.array([main_diagonal, sub_diagonal])

    def reset(self):
        self.observation_space = np.zeros(shape=(3, 3), dtype=float)
        self.action_space = np.zeros(shape=(9, 1), dtype=float)
        self.counter = 0


def print_observe(observation):
    for i in range(3):
        for j in range(3):
            if observation[i][j] == 1:
                print_figure = "|X|"
            elif observation[i][j] == -1:
                print_figure = "|O|"
            else:
                print_figure = "|_|"
            print(print_figure, end=" ")
        print("\n")
    print("\n")


def render_game(observations, render_time):
    for obs in observations:
        os.system("cls" if os.name == "nt" else "clear")
        print_observe(obs)
        time.sleep(render_time)
