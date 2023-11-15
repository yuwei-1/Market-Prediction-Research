import gymnasium as gym
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.feed_forward_network import FeedForward
from models.reinforcement_learning_agents import Agent

from IPython import display
is_ipython = 'inline' in matplotlib.get_backend()

class DQNAgent(Agent):

    def __init__(self, 
                double_dqn=False, 
                target_net_update='hard',
                tau=0.005,
                hard_update_interval=500,
                mem_length=10000, 
                environment='CartPole-v1', 
                gradient_clipping=None, 
                **net_kwargs):
        
        self.mem_length = mem_length
        self.double_dqn = double_dqn
        self.tau = tau
        self.hard_update_interval = hard_update_interval
        self.target_net_update = self.update_guard(target_net_update)
        self.gradient_clipping = gradient_clipping
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env, self.n_obs, self.n_actions = self.init_env(environment)
        self.replay_memory = self.init_agent_memory()
        self.net = self.init_agent(self.n_obs, self.n_actions, **net_kwargs)
        if self.double_dqn:
            self.target_net = self.init_agent(self.n_obs, self.n_actions, **net_kwargs)
            self.target_net.load_state_dict(self.net.state_dict())
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        
        
    def init_env(self, environment):
        try:
            env = gym.make(environment, render_mode="human")
            n_actions = env.action_space.n
            n_obs = env.observation_space.shape[0]
            return env, n_obs, n_actions

        except Exception as error:
            print("not a valid environment")
            raise error

    def init_agent(self, input_size, output_size, q_func_approximator=FeedForward, **net_kwargs):
        return q_func_approximator(input_size, output_size, **net_kwargs).to(self.device)

    def train(self, episodes=2000, discount=0.99, batch_size=128, train_threshold=500, learning_rate=0.0001, optimizer=optim.AdamW, loss=nn.MSELoss()):
        
        self.discount = discount
        self.batch_size = batch_size
        self.train_threshold = train_threshold
        self.loss_fn = loss
        self.optimizer = optimizer(self.net.parameters(), lr=learning_rate, amsgrad=True)
        self.steps_done = 0
        self.episode_durations = []
        
        for ep in range(episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state).to(self.device)
            for t in count():
                # agent chooses an action
                action = self.get_action(state)
                # observe new obs from environment
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                # logic for different scenarios
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(next_state).to(self.device)
                # save experience to replay memory
                reward = torch.tensor([reward]).to(self.device)
                self.retain_experience(state, action, next_state, reward)
                # change current state
                state = next_state
                # learn from some past experiences
                self.step_action_value_function()
                # terminate the episode under stopping conditions
                if terminated or truncated:
                    self.episode_durations.append(t+1)
                    self.plot_durations()
                    break
                
                if self.double_dqn:
                    if self.target_net_update == "soft":
                        self.perform_soft_target_update()
                    elif self.target_net_update == "hard":
                        self.perform_hard_target_update()

                self.steps_done += 1

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def perform_soft_target_update(self):
        net_state_dict = self.net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in net_state_dict.keys():
            target_net_state_dict[key] = self.tau*net_state_dict[key] + (1-self.tau)*target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def perform_hard_target_update(self):
        if self.steps_done % self.hard_update_interval == 0 and self.steps_done > 0:
            net_state_dict = self.net.state_dict()
            target_net_state_dict = self.target_net.state_dict()
            for key in net_state_dict.keys():
                target_net_state_dict[key] = net_state_dict[key]
            self.target_net.load_state_dict(target_net_state_dict)

    def step_action_value_function(self):

        if self.get_memory_len() < self.train_threshold:
            return
        
        # sample past memories
        batch = self.transition(*self.recall_experience(self.batch_size))

        states = torch.stack(batch.state)
        rewards = torch.stack(batch.reward).squeeze()
        actions = torch.stack(batch.action)

        # y = r for terminal steps, and y = r + gamma*max_a Q_hat(s', a)
        non_terminal_mask = torch.tensor(tuple(map(lambda ns:ns is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_terminal_states = torch.stack([ns for ns in batch.next_state if ns is not None]).to(self.device)

        current_estimate = torch.gather(self.net(states), 1, actions)
        target = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            if self.double_dqn:
                target[non_terminal_mask] = (self.discount * torch.max(self.target_net(non_terminal_states), dim=1)[0])
            else:
                target[non_terminal_mask] = (self.discount * torch.max(self.net(non_terminal_states), dim=1)[0])
        
        loss = self.loss_fn(current_estimate.squeeze(), target + rewards)
        #print(current_estimate.squeeze().shape, (target + rewards).shape)
        self.optimizer.zero_grad()
        loss.backward()

        if self.gradient_clipping > 0:
            torch.nn.utils.clip_grad_value_(self.net.parameters(), self.gradient_clipping)
        self.optimizer.step()

    
    def get_action(self, state):
        # TODO: how to refactor this so can use other behaviour policies?

        # Epsilon Greedy
        def epsilon_greedy(start, end, n_steps, it):
            return max(start - (start - end) * (it / n_steps), end)

        ep_threshold = epsilon_greedy(1.0, .05, 1000, self.steps_done)
        rand = random.uniform(0,1)
        if rand <= ep_threshold:
            # choose random action
            return torch.tensor([self.env.action_space.sample()], device=self.device, dtype=torch.long)
        else:
            # greedy: choose action with largest value
            with torch.no_grad():
                return torch.argmax(self.net(state)).unsqueeze(0)


    def init_agent_memory(self):
        return deque([], maxlen=self.mem_length)

    def retain_experience(self, *args):
        self.replay_memory.append(self.transition(*args))

    def recall_experience(self, sample_size):
        return zip(*random.sample(self.replay_memory, sample_size))
    
    def get_memory_len(self):
        return len(self.replay_memory)
    
    @staticmethod
    def update_guard(update_type):
        assert isinstance(update_type, str), "update type must be string"
        update_type = update_type.lower()
        allowed = {'soft', 'hard'}
        assert update_type in allowed, "target updates must be either hard or soft"
        return update_type