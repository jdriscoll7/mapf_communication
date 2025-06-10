import time

import primalite.algorithms as algorithms
from primalite.controller.a2c_controller import A2CController
import primalite.maps as maps
import primalite.experiments as experiments
from primalite.constants import *
from os.path import join
from torch import Tensor, zeros
import sys
from copy import deepcopy, copy
import timeit
import torch.nn.functional as F
import random

params = {}
completion_rates_final = []
filename = ""
map_size = "10"
density = "0.2"


def int_joint_action_to_vector(action, group):

    joint_action = zeros(size=(len(action),), dtype=int)
    for i in range(5 ** len(group)):
        for j, agent_id in enumerate(group):
            if agent_id in group:
                joint_action[agent_id] = (i // (5 ** j)) % 5

    return joint_action


class LookaheadController(A2CController):

    def __init__(self, params, env, communication_pattern=None) -> None:
        super(LookaheadController, self).__init__(params)
        self.env = env

        # If communication pattern is not provided, assume fully connected. Create default here.
        if communication_pattern is None:
            self.communication_pattern = [tuple([i for i in range(self.nr_agents)])]
        else:
            self.communication_pattern = communication_pattern

        self.policy_evals = 0

    @classmethod
    def construct_from_controller(cls, controller, env, communication_pattern=None):

        controller.__class__ = LookaheadController
        controller.env = env

        # If communication pattern is not provided, assume fully connected. Create default here.
        if communication_pattern is None:
            controller.communication_pattern = [tuple([i for i in range(controller.nr_agents)])]
        else:
            controller.communication_pattern = communication_pattern

        controller.policy_evals = 0

        return controller

    def value_function(self, observation):
        joint_observation = observation.view(1, self.nr_agents, -1)
        action_logits = self.policy_network(joint_observation)
        probs = F.softmax(action_logits, dim=-1).detach()

        q_eval = self.critic_network.q_net(observation)
        #
        # q_eval = q_eval * q_eval.std() - q_eval.mean()
        # q_eval = (q_eval - q_eval.mean())/q_eval.std()

        return torch.sum(q_eval * probs, dim=1).clone()
        # return torch.max(q_eval* probs, dim=1)[0].clone()
        # return torch.sum(action_logits * probs, dim=1).clone()
        # return torch.max(action_logits, dim=1)[0].clone()


    def joint_future_observations(self):

        # Store original environment to restore after trying each joint action.
        env = copy(self.env)
        previous_env = copy(env)

        # Output observations - list of observation-sized tensors.
        observations = {}
        valid_observations = {}
        for group in self.communication_pattern:
            # observations[group] = [None for _ in range(5**len(group))]
            # valid_observations[group] = [None for _ in range(5**len(group))]
            observations[group] = [None for _ in range(5**len(group))]
            valid_observations[group] = [None for _ in range(5**len(group))]

        # Store joint action in action tensor - fill out based on base 5 representation of i.

        # Iterate over all possible joint actions, making observations and restoring environment each time.
        # int_joint_action_to_vector
        for group in self.communication_pattern:
            joint_action = zeros(size=(self.nr_agents,), dtype=int)
            for i in range(5**len(group)):
                for j, agent_id in enumerate(group):
                    if agent_id in group:
                        joint_action[agent_id] = (i // (5**j)) % 5

                # Apply joint action to environment and make observation.
                # start = time.time()

                observations[group][i], valid_observations[group][i] = env.soft_step(joint_action)

                # Restore environment.
                # env = copy(previous_env)
                # env.reset()

        return observations, valid_observations

    def joint_policy(self, observation, value_network=False):

        self.policy_evals += 1

        if random.random() < 0.75:
        # if self.policy_evals % 5 != 0:
            return super(LookaheadController, self).joint_policy(observation)

        # TESTING TO CHECK Q-NET - REMOVE LATER.
        test_joint_action = zeros(size=(self.nr_agents,), dtype=int)
        for group in self.communication_pattern:
            for j, agent_id in enumerate(group):
                test_joint_action[agent_id] = torch.argmax(self.critic_network.q_net(observation)[group, :])

        # Obtain all possible future observations.
        future_observations, valid_observations = self.joint_future_observations()

        # Initialize dictionary that maps groups to lookahead value mappings.
        group_lookahead_dict = {}
        for group in self.communication_pattern:
            group_lookahead_dict[group] = {}

        # Iterate over communication pattern - this is a list of tuples denoting fully connected communication groups.

        observation_q_evaluation = self.critic_network.q_net(observation)

        for group in self.communication_pattern:
            for i, f_observation in enumerate(future_observations[group]):
                if valid_observations[group][i] or i==0:
                    if value_network:
                        group_lookahead_dict[group][i] = torch.sum(self.mean(self.critic_network(f_observation).view(-1)[group], dim=1))
                    else:
                        #
                        # print(self.critic_network.q_net(f_observation)[group, :])
                        #
                        # group_lookahead_dict[group][i] = torch.sum(self.critic_network.q_net(f_observation)[group, :], dim=0)[0]
                        # group_lookahead_dict[group][i] = torch.sum(torch.max(self.critic_network.q_net(f_observation)[group, :], dim=1)[0])

                        # print(torch.sum(self.critic_network.q_net(f_observation)[group, :], dim=0)[0])
                        # print("\n\n")

                        # print(self.critic_network.q_net(f_observation))
                        # print(self.critic_network.q_net(f_observation)[group, :][:, 0])
                        # group_lookahead_dict[group][i] = torch.max(torch.sum(self.critic_network.q_net(f_observation)[group, :], dim=0), dim=0)[0]

                        # group_lookahead_dict[group][i] = torch.sum(self.critic_network.q_net(f_observation)[group, 0])
                        # group_lookahead_dict[group][i] = torch.sum(torch.max(self.critic_network.q_net(f_observation)[group, :], dim=1)[0])

                        # print("Future observation " + str(i) + ":")
                        # # print(self.critic_network.q_net(f_observation)[group, :])
                        # print(torch.max(self.critic_network.q_net(f_observation)[group, :], dim=1)[0])
                        # print("\n\n\n")

                        # Current closest.
                        local_joint_policy = super(LookaheadController, self).joint_policy(f_observation)
                        group_lookahead_dict[group][i] = torch.sum(self.critic_network.q_net(f_observation)[group, local_joint_policy[list(group)]], dim=0).item()
                        # group_lookahead_dict[group][i] = torch.sum(self.critic_network.q_net(f_observation)[group, :], dim=0).item()


                        # group_lookahead_dict[group][i] = (torch.sum(self.value_function(f_observation)[list(group)])).item()
                        # group_lookahead_dict[group][i] = torch.sum(self.value_function(f_observation)[list(group)])
                        # print(i)
                        # print(self.critic_network.q_net(f_observation)[group, local_joint_policy[group]])
                        # print("\n\n")

                        # group_lookahead_dict[group][i] = torch.sum(self.critic_network.q_net(f_observation)[group, :], dim=0)[0]

                        # print("Action decided by maximizing Q-table: \t\t" + str(test_joint_action[0]))
                        # print("Action decided by learned policy: \t\t\t" + str(super(LookaheadController, self).joint_policy(f_observation)[0]))
                        # print("Critic network row-sum evaluation: \t\t\t" + str(torch.sum(self.critic_network.q_net(f_observation), dim=0)))
                        # print("Critic network future evaluation: \n" + str(self.critic_network.q_net(f_observation)))
                        # print("Critic network present evaluation: \n" + str(self.critic_network.q_net(observation)))
                        # print("\n")


                        # group_lookahead_dict[group][i] = torch.sum(self.critic_network.q_net(f_observation)[group, 0])
                        # group_lookahead_dict[group][i] = self.critic_network.q_net(f_observation)
                        # group_lookahead_dict[group][i] = self.critic_network.q_net(f_observation)[group, :]

        # Find best joint action, assigning global joint policy by communication groups.
        joint_action = zeros(size=(self.nr_agents, ), dtype=int)
        # [print(group_lookahead_dict[g]) for g in self.communication_pattern]
        for group in self.communication_pattern:

            # Find joint action with largest joint Q-function value.
            max_joint_value = max(group_lookahead_dict[group].values())
            # print([i for i, j in group_lookahead_dict[group].items() if j == max_joint_value])
            action_key = [i for i, j in group_lookahead_dict[group].items() if j == max_joint_value][-1]

            # print("Max action: \t\t\t\t" + str(action_key))
            # print("Max action future value: \t\t" + str(group_lookahead_dict[group][action_key]))

            # Convert action key back to action and write to joint action.

            for j, agent_id in enumerate(group):
                if agent_id in group:
                    joint_action[agent_id] = (action_key // (5**j)) % 5

        # print(super(LookaheadController, self).joint_policy(observation))
        # print(joint_action)
        # print("\n\n")
        # return super(LookaheadController, self).joint_policy(observation)


        # # print("Action decided by maximizing Q-table: \t\t" + str(test_joint_action))
        # print("Action decided by learned policy: \t\t\t" + str(super(LookaheadController, self).joint_policy(observation)))
        # print("Action decided by cooperative policy: \t\t" + str(joint_action))
        # # print("Critic network row-sum evaluation: \t\t\t" + str(torch.sum(self.critic_network.q_net(observation), dim=0)))
        # print("Critic network evaluation: \n" + str(self.critic_network.q_net(observation)))
        # print("\n")

        return joint_action.clone()



    # def joint_policy(self, observation, value_network=False):
    #
    #     # Initialize dictionary that maps groups to lookahead value mappings.
    #     group_lookahead_dict = {}
    #     for group in self.communication_pattern:
    #         group_lookahead_dict[group] = {}
    #
    #     # Iterate over communication pattern - this is a list of tuples denoting fully connected communication groups.
    #
    #     for group in self.communication_pattern:
    #         if value_network:
    #             group_lookahead_dict[group][i] = torch.sum(self.critic_network(observation).view(-1)[group])
    #         else:
    #             # print(self.critic_network.q_net(f_observation))
    #             # print(self.critic_network.q_net(f_observation)[group, :][:, 0])
    #             # group_lookahead_dict[group][i] = torch.sum(torch.max(self.critic_network.q_net(f_observation)[group, :], dim=1)[0])
    #             group_lookahead_dict[group] = torch.max(torch.sum(self.critic_network.q_net(observation)[group, :], dim=0), dim=0)[0]
    #             # group_lookahead_dict[group][i] = torch.sum(self.critic_network.q_net(f_observation)[group, 0])
    #             # group_lookahead_dict[group][i] = self.critic_network.q_net(f_observation)
    #             # group_lookahead_dict[group][i] = self.critic_network.q_net(f_observation)[group, :]
    #
    #     # Find best joint action, assigning global joint policy by communication groups.
    #     joint_action = zeros(size=(self.nr_agents, ), dtype=int)
    #     for group in self.communication_pattern:
    #
    #         # Find joint action with largest joint Q-function value.
    #         action_key = max(group_lookahead_dict[group], key=group_lookahead_dict[group].get)
    #
    #         # Convert action key back to action and write to joint action.
    #
    #         for j, agent_id in enumerate(group):
    #             if agent_id in group:
    #                 joint_action[agent_id] = (action_key // (5**j)) % 5
    #
    #     # print(super(LookaheadController, self).joint_policy(observation))
    #     # print(joint_action)
    #     # print("\n\n")
    #
    #     # return super(LookaheadController, self).joint_policy(observation)
    #     print(joint_action)
    #     return joint_action


# for nr_agents in [4, 8, 16, 32, 64, 128, 256]:
for nr_agents in [4]:
    completion_rates = []
    for map_id in range(100):
        start = time.time()
        # Environment setup.
        params[ENV_NR_AGENTS] = nr_agents
        params[MAP_NAME] = f"primal-{params[ENV_NR_AGENTS]}_agents_{map_size}_size_{density}_density_id_{map_id}_environment"
        params[EPISODES_PER_EPOCH] = 32
        params[ALGORITHM_NAME] = ALGORITHM_PPO_QMIX
        params[HIDDEN_LAYER_DIM] = 64
        params[NUMBER_OF_EPOCHS] = 1000
        params[EPISODES_PER_EPOCH] = 1
        params[EPOCH_LOG_INTERVAL] = 50
        params[ENV_TIME_LIMIT] = 256
        params[ENV_INIT_GOAL_RADIUS] = 10
        env = maps.make_test_map(params)

        # Lookahead controller setup - create by casting A2C controller to lookahead controller type.
        communication_groups = [(i, ) for i in range(nr_agents)]

        controller = LookaheadController.construct_from_controller(algorithms.make(params), env=env, communication_pattern=[(0, 1, 2, 3)])
        # controller = LookaheadController.construct_from_controller(algorithms.make(params), env=env, communication_pattern=communication_groups)
        # controller.joint_future_observations()

        # Evaluation and logging.
        controller.load_model_weights(join("output", filename))
        results = experiments.run_episodes(params[EPISODES_PER_EPOCH], [env], controller, params, training_mode=False, render_mode=True)
        end = time.time()
        # print(end - start)
        completion_rates.append(results[COMPLETION_RATE])
        print(f"Run {nr_agents} agents in {map_id+1}/{100}: completion={numpy.mean(completion_rates)}\t\t\t\t", end='\r', flush=True)

    completion_rates_final.append(numpy.mean(completion_rates))
print("Completion rate:", completion_rates_final)
