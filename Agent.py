import collections
import random
import numpy as np
from TicTacToeEnv import TicTacToe

GAMMA = 0.9
ALPHA = 0.05


class Agent:
    def __init__(self, name):
        self.name = name
        self.values = collections.defaultdict(float)
        self.actions = []

    def reset(self):
        self.actions = []

    def backward(self, observations, reward, index):
        size_observe = len(observations)
        revers_obs = observations.copy()
        revers_obs.reverse()
        revers_act = self.actions.copy()
        revers_act.reverse()
        new_state = None
        new_action = None
        for i in range(index, size_observe, 2):
            if i == index:
                old_state = "".join(list(map(convert_none, revers_obs[i])))
                if i == 0:
                    old_action = None
                else:
                    old_action = revers_act[int((i-1)/2)]
                self.value_update(old_state, old_action, reward, new_state, new_action)
            else:
                old_state = "".join(list(map(convert_none, revers_obs[i])))
                old_action = revers_act[int((i-1)/2)]
                self.value_update(old_state, old_action, 0, new_state, new_action)
            new_state = old_state
            new_action = old_action

    def best_action(self, state, epsilon, acts):
        best_value, best_action = None, None
        actions = list(set(range(1, 10)) - set(acts))
        new_state = "".join(list(map(convert_none, state)))
        if random.random() <= epsilon:
            action = random.choice(actions)
            best_action = action
            self.actions.append(best_action)
            return self.values[(new_state, best_action)], best_action
        for action in actions:
            action_value = self.values[(new_state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        self.actions.append(best_action)
        return self.values[(new_state, best_action)], best_action

    def value_update(self, old_state, old_action, reward, new_state, new_action):
        if new_state is None and new_action is None:
            self.values[(old_state, old_action)] += (1-ALPHA) * reward
        else:
            new_val = self.values[(new_state, new_action)]
            self.values[(old_state, old_action)] += (1-ALPHA) * reward + GAMMA * ALPHA * new_val

def convert_none(symbol):
    if symbol == "":
        return "_"
    return symbol
