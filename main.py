import collections
import numpy as np
from statistics import mean
import Agent
import TicTacToeEnv
from random import choice
import random
from TicTacToeEnv import TicTacToe
from torch.utils.tensorboard import SummaryWriter

SUB_EPISODES = 5
TEST_EPISODES = 10
LEARN_EPISODES = 1000
AGENT_LEARN_EPISODES = 400 #100
RANDOM_LEARN_EPISODES = 400 #50
EPSILON = 1


def play_grade_episode(environment, agent, beta_g, epsilon_g):
    action_counter = 0
    observations = [environment.reset(random.choice([0, 1]))]
    action = environment.action_sample()
    actions = []
    while True:
        actions.append(action)
        state, reward_g, is_done = environment.step(action)
        action_counter += 1
        observations.append(state.copy())
        if is_done:
            if action_counter % 2 == 1:
                #if random.random() <= beta_g:
                #   agent.backward(observations, -1 if reward_g <= 0 else 1, int(not random_choose)) #0
                #agent.reset()
                result = ""
                if reward_g == 0:
                    result = "agent_win"
                elif reward_g == 1:
                    result = "random_win"
            else:
                result = ""
                if reward_g == 0:
                    result = "agent_win"
                elif reward_g == 1:
                    result = "agent_win"
            return result
        else:
            if action_counter % 2 == 0:
                action = environment.action_sample()
            else:
                _, action = agent.best_action(environment.state, epsilon_g, actions)


def play_random_episode(environment, agent, epsilon_r):
    observations = []
    #rewards = []
    actions = []
    #goes_first = choice([0, 1])
    start_figure = choice([0, 1])
    observations.append(environment.reset(number_figure=start_figure))
    #agent_sequence = [first_agent, second_agent] if goes_first else [second_agent, first_agent]
    agent_sequence_rewards = [0.0, 0.0]
    #current_agent = agent_sequence[0]
    current_reward = 0
    agent_rewards = [0.0, 0.0]
    agent_actions = [[], []]
    current_agent = 0
    actions = []
    while True:
        is_done = False
        for i in range(2):
            value, action = agent.best_action(environment.state, epsilon_r, actions)
            agent_rewards[i] += value
            agent_actions[i].append(action)
            actions.append(action)
            new_state, reward_a, is_done = environment.step(action)
            observations.append(new_state.copy())
            current_agent = i
            current_reward = reward_a
            if is_done:
                break
        if is_done:
            for i in range(2):
                if current_reward == 0:
                    agent_sequence_rewards[i] = -1 #-1
                else:
                    agent_sequence_rewards[i] = 1 if current_agent == i else -1 #-1
            for i in range(2):
                agent.reset()
                agent.actions = agent_actions[i]
                agent.backward(observations, agent_sequence_rewards[i], i)
            agent.reset()
            #Agent.union(agent_sequence)
            break
    if current_reward == 0:
        return "draw", 0 #return "second_agent" if current_agent == 0 else "first_agent", agent_sequence_rewards[current_agent] #
    else:
        return "first_agent" if current_agent == 0 else "second_agent", agent_sequence_rewards[current_agent]


if __name__ == "__main__":
    """Agent Q-learning"""
    env = TicTacToe(number_start_figure=0)
    agent = Agent.Agent("Peter")
    writer = SummaryWriter(comment="TicTacToe Q-learning")
    agent_1_wins = 0
    agent_2_wins = 0
    agent_counter = 0
    start_agent = random.choice([0, 1])
    for i in range(LEARN_EPISODES):
        epsilon = 0.15 if EPSILON - i / LEARN_EPISODES < 0.15 else EPSILON - i / LEARN_EPISODES
        beta = 0  # 0.1 if 1 - i/LEARN_EPISODES < 0.1 else 1 - i/LEARN_EPISODES
        agent_1_wins = 0
        agent_2_wins = 0
        draw = 0
        for j in range(AGENT_LEARN_EPISODES):
            epsilon = 0.15 if epsilon - j / AGENT_LEARN_EPISODES < 0.15 else epsilon - j / AGENT_LEARN_EPISODES
            name, reward = play_random_episode(env, agent, epsilon)
            if name == "first_agent":
                if reward > 0:
                    agent_1_wins += 1
                else:
                    agent_2_wins += 1
            elif name == "second_agent":
                if reward > 0:
                    agent_2_wins += 1
                else:
                    agent_1_wins += 1
            else:
                draw += 1
            env.reset()
        freq_win_agent_1 = agent_1_wins / AGENT_LEARN_EPISODES
        freq_win_agent_2 = agent_2_wins / AGENT_LEARN_EPISODES
        freq_draw_learning = draw / AGENT_LEARN_EPISODES
        writer.add_scalar("Agent1LearningRate", freq_win_agent_1, i)
        writer.add_scalar("Agent2LearningRate", freq_win_agent_2, i)
        writer.add_scalar("DrawLearningRate", freq_draw_learning, i)
        agent_win = 0
        draw = 0
        id_x = 0
        for j in range(RANDOM_LEARN_EPISODES):
            epsilon = 0.15 if epsilon - j / RANDOM_LEARN_EPISODES < 0.15 else epsilon - j / RANDOM_LEARN_EPISODES
            result = play_grade_episode(env, agent, beta, epsilon)
            if result == "agent_win":  # 1-id_x/(TEST_EPISODES*SUB_EPISODES)
                agent_win += 1
            elif result == "draw":
                draw += 1
            env.reset()
            id_x += 1
            #Agent.union(agents)
        agent_counter += 1
        freq_win_random = (agent_win / RANDOM_LEARN_EPISODES)
        freq_draw_random = (draw / RANDOM_LEARN_EPISODES)
        writer.add_scalar("RandomLearningRate", freq_win_random, i)
        writer.add_scalar("RandomLearningRateDraws", freq_draw_random, i)
    writer.close()



