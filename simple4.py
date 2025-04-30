import os
import datetime
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from PIL import Image
import gym
import gym_super_mario_bros
from gym.spaces import Box
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
import time
#used this to test internally

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=5): # 5 frames
        super().__init__(env)
        self._skip = skip
    def step(self, action):
        total_reward = 0.0
        done = False

        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation
        
class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84 or w != 84:
            print("Either the height or width is broken! ")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if "online" in model:
            return self.online(input)
        elif "target" in model:
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

class DQNAgent:
    def __init__(self, n_states, n_actions, save_dir='.', checkpoint=None):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_states     = n_states
        self.n_actions    = n_actions
        self.save_dir      = save_dir
        self.device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.qnet          = QNet(self.n_states, self.n_actions).float().to(self.device)
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.999995
        self.step_count    = 0
        self.update_every  = 50_000

        self.memory    = TensorDictReplayBuffer(storage=LazyMemmapStorage(100_000,device=torch.device("cpu")))
        self.batch_size = 32
        self.gamma = 0.99

        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=0.00025)
        self.loss_fn   = torch.nn.SmoothL1Loss()

        if checkpoint:
            self.load(checkpoint)

        self.burnin     = 1000
        self.learn_every = 1
        self.sync_every  = 50_000

    def act(self, state, deterministic=False):
        if not deterministic and random.random() < self.epsilon:
            action_idx = np.random.randint(self.n_actions)
        else:
            state_t    = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state_t    = torch.tensor(state_t, device=self.device).unsqueeze(0)
            action_vals= self.qnet(state_t, model="online")
            action_idx = torch.argmax(action_vals, axis=1).item()

        self.epsilon    = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        if isinstance(state, tuple):
            state      = state[0].__array__()
        else:
            state      = state.__array__()

        if isinstance(next_state, tuple):
            next_state = next_state[0].__array__()
        else:
            next_state = next_state.__array__()

        td = TensorDict({
            "state":      torch.tensor(state),
            "next_state": torch.tensor(next_state),
            "action":     torch.tensor([action]),
            "reward":     torch.tensor([reward]),
            "done":       torch.tensor([done])
        }, batch_size=[])
        self.memory.add(td)

    def recall(self):
        batch       = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(k) for k in ("state","next_state","action","reward","done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        return self.qnet(state, model="online")[
            np.arange(0, self.batch_size), action
        ]

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_q = self.qnet(next_state, model="online")
        best_action  = torch.argmax(next_state_q, axis=1)
        next_q       = self.qnet(next_state, model="target")[
                         np.arange(0, self.batch_size), best_action
                       ]
        return (reward + (1 - done.float()) * self.gamma * next_q).float()

    def update_q_online(self, td_est, td_tgt):
        loss = self.loss_fn(td_est, td_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_q_target(self):
        self.qnet.target.load_state_dict(self.qnet.online.state_dict())

    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.step_count//self.update_every)}.chkpt"
        torch.save({
            "model":             self.qnet.state_dict(),
            "exploration_rate":  self.epsilon
        }, save_path)

    def load(self, load_path):
        if not load_path.exists():
            print("no")
        
        ckp = torch.load(load_path, map_location=('cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path}")
        self.qnet.load_state_dict(state_dict)
        self.epsilon = exploration_rate

    def learn(self):
        if self.step_count % self.sync_every == 0:
            self.sync_q_target()
        if self.step_count % self.update_every == 0:
            self.save()

        if self.step_count < self.burnin or (self.step_count % self.learn_every):
            return None, None

        s, ns, a, r, d = self.recall()
        td_est = self.td_estimate(s, a)
        td_tgt = self.td_target(r, ns, d)
        loss   = self.update_q_online(td_est, td_tgt)
        return (td_est.mean().item(), loss)

class LogProgress:
    def __init__(self, save_dir: Path, ma_window=100):
        self.save_path = save_dir / "learning_overview.jpg"
        self.ma_window = ma_window
        self.rewards = []
        self.lengths = []
        # (optional) if you want to store them
        self.epsilons = []
        self.steps    = []

    # add epsilon and step as optional kwargs
    def record(self, episode, reward, length, *, epsilon=None, step=None):
        self.rewards.append(reward)
        self.lengths.append(length)

        if epsilon is not None:
            self.epsilons.append(epsilon)
        if step is not None:
            self.steps.append(step)

        window = self.ma_window
        ma_rewards = (
            np.convolve(self.rewards, np.ones(window)/window, mode="valid")
            if len(self.rewards) >= window
            else np.array(self.rewards)
        )
        ma_lengths = (
            np.convolve(self.lengths, np.ones(window)/window, mode="valid")
            if len(self.lengths) >= window
            else np.array(self.lengths)
        )

        plt.clf()
        plt.plot(ma_rewards, label="Reward MA")
        plt.plot(ma_lengths, label="Length MA")
        plt.xlabel("Episode")
        plt.ylabel(f"{window}-Episode Moving Avg")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

agent = DQNAgent(
    n_states = (env.observation_space.shape[0],
                env.observation_space.shape[1],
                env.observation_space.shape[2]),
    n_actions = env.action_space.n,
    save_dir = Path('demo'),
    checkpoint = Path('new_checkpoint2') / 'mario_net_123.chkpt'
)

agent.epsilon = 0.0

num_episodes = 100
total_rewards = []
for ep in range(1, num_episodes + 1):
    state = env.reset()
    done = False
    total = 0.0
    while not done:
        if ep == 1 or ep % 1 == 0: #only print the last one
            env.render()
        action = agent.act(state, deterministic=False) # True)
        state, reward, done, info = env.step(action)
        total += reward
        # time.sleep(0.02) #render delay
    print(f"Episode {ep}: total_reward = {total:.2f}")
    total_rewards.append(total)

env.close()

avg_reward = sum(total_rewards) / len(total_rewards)
print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")


# 1067.18 on train5
# 2278
# 3000 at 50 at 6k episodes ish
# 3030 chkpt 105

# 117: 3061
# 120: 2897
# 123: 3139
