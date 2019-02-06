import time
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


def visualize_random():
    env = gym.make('CartPole-v1')
    # env.seed(2)

    observation = env.reset()
    for _ in range(100):
      env.render()
      action = env.action_space.sample()
      observation, reward, done, info = env.step(action)

      time.sleep(0.1)

      if done:
          print('Fail!')
          break

    env.close()

visualize_random()

# %%

def gather_random_data(n_steps=100):

    data = []

    env = gym.make('CartPole-v1')
    observation = env.reset()
    for _ in range(n_steps):
      action = env.action_space.sample() # your agent here (this takes random actions)

      next_observation, reward, done, info = env.step(action)

      data.append((observation, action, reward, done, next_observation))

      observation = next_observation

      if done:
          observation = env.reset()

    env.close()

    return data

def get_dataset(data):

    data2 = list(zip(*data))
    data_new = [np.stack(d, axis=0) for d in data2]

    # let reward be negative square angle of pole
    data_new[2] = -np.abs(data_new[4][:, 2])

    data_new = [d.astype(np.float32) for d in data_new]

    data_new[2] -= data_new[3] # done is bad

    dataset = torch.utils.data.TensorDataset(*[torch.from_numpy(d) for d in data_new])

    return dataset

class ActorNetwork(nn.Module):
    def __init__(self, n_inputs=4, n_actions=2):
        super().__init__()
        self.linear_layer1 = nn.Linear(n_inputs, 16)
        self.linear_layer2 = nn.Linear(16, 8)
        self.linear_layer3_mean = nn.Linear(8, 1)

        self.linear_layer3_action = nn.Linear(8, n_actions)

    def forward(self, observation):
        x = self.linear_layer1(observation)
        x = F.softplus(x)
        x = self.linear_layer2(x)
        x = F.softplus(x)
        # duelling dqn
        # TODO: should remove mean from second term but network is already trained
        x = self.linear_layer3_mean(x) + self.linear_layer3_action(x)
        return x



def train(dataset, n_epochs=100, batch_size=10):

    sampler = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(dataset),
        batch_size,
        drop_last=True
    )

    for epoch in range(n_epochs):
        for batch_number, batch_index in enumerate(sampler, start=1):
            # if False:
            #     batch_index = list(sampler)[1]

            observation, action, reward, done, next_observation = dataset[batch_index]
            pred_q = network(observation)
            with torch.no_grad():
                next_pred_q = previous_network(next_observation)
            # NOTE: should keep old version of network and not get gradients from both

            target = reward.view(-1, 1) + gamma*next_pred_q.max(dim=1)[0]

            loss = torch.mean(
                (target - pred_q[torch.arange(batch_size), action.long()])**2
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        previous_network.load_state_dict(network.state_dict())

        print(f'epoch: {epoch}, loss: {loss}')

def visualize_game(network, n_steps=100, seed=None):
    env = gym.make('CartPole-v1')
    if seed is not None:
        env.seed(seed)

    observation = env.reset()
    for _ in range(n_steps):
      env.render()
      # action = env.action_space.sample() # your agent here (this takes random actions)

      with torch.no_grad():
          action_value = network(torch.from_numpy(observation).float())
      action = action_value.argmax()

      print(f'action_value: {action_value}, action: {action}')
      observation, reward, done, info = env.step(action.item())
      print(f'angle: {observation[2].item()}, reward: {-np.abs(observation[2].item())}')

      time.sleep(0.1)


      # print(f'reward: {reward}, obs: {observation}')
      if done:
          print('Fail!')
          break

    env.close()

def gather_new_data(network, n_steps=100, eps=0.05):
    data = []

    env = gym.make('CartPole-v1')
    observation = env.reset()
    for _ in range(n_steps):
      with torch.no_grad():
          action_value = network(torch.from_numpy(observation).float())
      action = action_value.argmax().item()

      if np.random.rand() <= eps:
          action = int(np.random.rand() >= 0.5)

      next_observation, reward, done, info = env.step(action)

      data.append((observation, action, reward, done, next_observation))

      observation = next_observation

      if done:
          observation = env.reset()

    env.close()

    return data

# %%

previous_network = ActorNetwork()
network = ActorNetwork()
previous_network.load_state_dict(network.state_dict())

optimizer = optim.Adam(network.parameters(), lr=1e-3)

gamma = 0.90

# %%


data = gather_random_data(n_steps=1000)
dataset = get_dataset(data)
train(dataset, batch_size=100)

# %%

new_data = gather_new_data(network, n_steps=1000, eps=0.1)
data += new_data
dataset = get_dataset(data)
train(dataset, n_epochs=10, batch_size=10)

# %%

visualize_game(network, n_steps=1000)

# %%

# torch.save(network.state_dict(), 'test3-model-v3.pth')

# %%

network.load_state_dict(torch.load('dqn-model-v1.pth'))
visualize_game(network, n_steps=200, seed=1)

# %%

network.load_state_dict(torch.load('dqn-model-v2.pth'))
visualize_game(network, n_steps=200, seed=0)

# %%

network.load_state_dict(torch.load('dqn-model-v3.pth'))
visualize_game(network, n_steps=1000)#, seed=0)
