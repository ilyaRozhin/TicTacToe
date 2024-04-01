import numpy as np
import time
import os
import random


class TicTacToe:
    def __init__(self, number_start_figure):
        self.state = np.zeros(shape=9, dtype=str)
        self.action_space = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.number_start_figure = number_start_figure
        self.counter = 0

    def step(self, action):
        if action in self.action_space:
            number_next_figure = self.counter % 2 + self.number_start_figure
            self.state[action-1] = "X" if number_next_figure == 1 else "O"
            self.action_space.remove(action)
            is_done = self.game_is_done()
            reward = 1 if is_done else 0
            self.counter += 1
            if self.counter == 9:
                is_done = True
            return self.state, reward, is_done
        return self.state, -10, True

    def game_is_done(self):
        observation_matrix = convert_state_to_digits(self.state)
        observation_matrix = observation_matrix.reshape((3, 3))
        main_sums = np.concatenate([observation_matrix.sum(axis=0), observation_matrix.sum(axis=1),
                                   diagonal_check(observation_matrix)], axis=0)
        if 3 in main_sums:
            return True
        elif -3 in main_sums:
            return True
        return False

    def reset(self, number_figure=0):
        self.state = np.zeros(shape=9, dtype=str)
        self.action_space = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.counter = 0
        self.number_start_figure = number_figure
        return self.state.copy()

    def action_sample(self):
        return random.choice(self.action_space)


def convert_to_nums(symbol):
    if symbol == "X":
        return 1
    elif symbol == "O":
        return -1
    return 0


def convert_state_to_digits(state):
    digit_state = np.array(list(map(convert_to_nums, state)))
    return digit_state


def diagonal_check(state):
    main_diagonal = 0
    sub_diagonal = 0
    for i in range(3):
        main_diagonal += state[i][i]
        sub_diagonal += state[2-i][i]
    return np.array([main_diagonal, sub_diagonal])


def print_current_state(state):
    for i in range(3):
        for j in range(3):
            if state[3*i + j]:
                print(state[3 * i + j], end=" ")
            else:
                print("_", end=" ")

        print("\n")
    print("\n")


def render_game(observations, render_time=0.5):
    for state in observations:
        os.system("cls" if os.name == "nt" else "clear")
        print_current_state(state)
        time.sleep(render_time)
