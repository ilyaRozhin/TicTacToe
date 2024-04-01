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
LEARN_EPISODES = 50000
EPSILON = 1


def play_grade_episode(environment, agent, epsilon=0.1):
    action_counter = 1
    random_choose = random.random() > 0.5
    action = None
    observations = [environment.reset(random.choice([0, 1]))]
    if random_choose:
        action = environment.action_sample()
    else:
        _, action = agent.best_action(environment.state, epsilon, agent.actions)
    while True:
        state, reward, is_done = environment.step(action)
        action_counter += 1
        observations.append(state.copy())
        if is_done:

            if (int(random_choose) + action_counter) % 2 == 0:
                agent.backward(observations, reward, int(not random_choose))
                agent.reset()
                return "random_win" if reward < 0 else "agent_win"
            else:
                agent.backward(observations, 1 if reward < 0 else -1, int(not random_choose))
                agent.reset()
                return "agent_win" if reward < 0 else "random_win"
        else:
            if (int(random_choose) + action_counter) % 2 == 0:
                action = environment.action_sample()
            else:
                _, action = agent.best_action(environment.state, 0, agent.actions)


def play_random_episode(environment, first_agent, second_agent, epsilon):
    observations = []
    rewards = []
    actions = []
    goes_first = choice([0, 1])
    start_figure = choice([0, 1])
    observations.append(environment.reset(number_figure=start_figure))
    agent_sequence = [first_agent, second_agent] if goes_first else [second_agent, first_agent]
    agent_sequence_rewards = {"first_agent": 0, "second_agent": 0}
    current_agent = agent_sequence[0]
    current_reward = 0
    agent_rewards = {"first_agent": 0, "second_agent": 0}
    while True:
        is_done = False
        for agent in agent_sequence:
            value, action = agent.best_action(environment.state, epsilon, actions)
            agent_rewards[agent.name] += value
            actions.append(action)
            new_state, reward, is_done = environment.step(action)
            observations.append(new_state.copy())
            current_agent = agent
            current_reward = reward
            if is_done:
                break
        if is_done:
            for key in agent_sequence_rewards.keys():
                if current_reward == 0:
                    agent_sequence_rewards[key] = -1
                elif current_reward == 1:
                    agent_sequence_rewards[key] = 1 if current_agent.name == key else -1
                else:
                    agent_sequence_rewards[key] = -10 if current_agent.name == key else 0
            for agent in agent_sequence:
                counter_agent = 1 if agent.name == current_agent.name else 0
                agent.backward(observations, agent_sequence_rewards[agent.name], counter_agent)
                agent.reset()
            break
    return agent_rewards["first_agent"], agent_rewards["second_agent"]


if __name__ == "__main__":

    """Agent Q-learning"""
    env = TicTacToe(number_start_figure=0)
    test_env = TicTacToe(number_start_figure=1)
    agent_1 = Agent.Agent("first_agent")
    agent_2 = Agent.Agent("second_agent")
    agents = [agent_1, agent_2]
    writer = SummaryWriter(comment="TicTacToe Q-learning")
    for i in range(LEARN_EPISODES):
        epsilon = 0.15 if EPSILON - i/LEARN_EPISODES < 0.15 else EPSILON - i/LEARN_EPISODES
        reward_agent1, reward_agent2 = play_random_episode(env, agent_1, agent_2, epsilon)
        writer.add_scalar("reward_agent1", reward_agent1, i)
        writer.add_scalar("reward_agent2", reward_agent2, i)
        if i % 100 == 0:
            freqs = []
            for _ in range(SUB_EPISODES):
                agent_win = 0
                id_x = 0
                for _ in range(TEST_EPISODES):
                    if play_grade_episode(test_env, random.choice(agents), epsilon) == "agent_win": #1-id_x/(TEST_EPISODES*SUB_EPISODES)
                        agent_win += 1
                    test_env.reset()
                    id_x += 1
                freq_win = (agent_win / TEST_EPISODES)
                freqs.append(freq_win)
            writer.add_scalar("Agent_win/TEST_EPISODES", mean(freqs), i//100)
    writer.close()



