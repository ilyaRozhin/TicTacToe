import collections
import random
import numpy as np
from TicTacToeEnv import TicTacToe

GAMMA = 0.9
ALPHA = 0.01


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
                old_action = None
                self.value_update(old_state, old_action, reward, new_state, new_action)
            else:
                old_state = "".join(list(map(convert_none, revers_obs[i])))
                old_action = revers_act[int((i-2)/2)]
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
            self.values[(old_state, old_action)] = (1-ALPHA) * self.values[(old_state, old_action)] + ALPHA * reward
        else:
            new_val = self.values[(new_state, new_action)]
            self.values[(old_state, old_action)] = (1-ALPHA) * self.values[(old_state, old_action)] + ALPHA * (reward + GAMMA * new_val)

def convert_none(symbol):
    if symbol == "":
        return "_"
    return symbol


def union(agents):
    first_agent = agents[0]
    second_agent = agents[1]
    for key, value in first_agent.values.items():
        second_agent.values[key] += value
    first_agent.values = second_agent.values.copy()
