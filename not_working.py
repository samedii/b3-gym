import time
import gym


data = []

env = gym.make('CartPole-v1')
observation = env.reset()
for _ in range(10000):
  # env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)

  # print(action)
  next_observation, reward, done, info = env.step(action)

  data.append((observation, action, reward, done, next_observation))

  observation = next_observation

  # time.sleep(0.1)


  # print(f'reward: {reward}, obs: {observation}')
  if done:
      observation = env.reset()

env.close()

print(len(data))
# %%

data

# realign data
data2 = list(zip(*data))
data2


# %%

import numpy as np

data_new = [np.stack(d, axis=0) for d in data2]
data_new

# %%

data_new[2] = -data_new[4][:, 2]**2 # let reward be angle of pole
data_new = [d.astype(np.float32) for d in data_new]

data_new[2]

# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


class ActorNetwork(nn.Module):
    def __init__(self, n_inputs=4, n_actions=2):
        super().__init__()
        self.linear_layer1 = nn.Linear(n_inputs, 16)
        self.linear_layer2 = nn.Linear(16, 8)
        self.linear_layer3 = nn.Linear(8, n_actions)

    def forward(self, observation):
        x = self.linear_layer1(observation)
        x = F.softplus(x)
        x = self.linear_layer2(x)
        x = F.softplus(x)
        x = self.linear_layer3(x)
        return x


network = ActorNetwork()



optimizer = optim.Adam(network.parameters(), lr=1e-3)

# %%

gamma = 0.90
# pred_q = r + 0.9*next_pred_q

dataset = torch.utils.data.TensorDataset(*[torch.from_numpy(d) for d in data_new])

batch_size = 100
sampler = torch.utils.data.BatchSampler(
    torch.utils.data.RandomSampler(dataset),
    batch_size,
    drop_last=True
)



for epoch in range(100):
    for batch_number, batch_index in enumerate(sampler, start=1):
        if False:
            batch_index = list(sampler)[0]
        observation, action, reward, done, next_observation = dataset[batch_index]
        pred_q = network(observation)
        # next_pred_q = network(next_observation).detach()
        # NOTE: should keep old version of network and not get gradients from both

        # observation
        # network(observation)

        #reward.shape
        #next_pred_q.shape
        #next_pred_q.max(dim=1)
        # target = reward.view(-1, 1) #+ gamma*next_pred_q.max(dim=1)[0]
        #
        # pred_q[torch.arange(batch_size), action.long()]


        loss = torch.mean(
            (target - pred_q[torch.arange(batch_size), action.long()])**2
        )

        loss.backward()

        optimizer.step()


    print(f'epoch: {epoch}, loss: {loss}')

# %%

env = gym.make('CartPole-v1')

observation = env.reset()
for _ in range(100):
  env.render()
  # action = env.action_space.sample() # your agent here (this takes random actions)

  with torch.no_grad():
      action_value = network(torch.from_numpy(observation).float())
  action = action_value.argmax()

  print(f'action_value: {action_value}, action: {action}')
  observation, reward, done, info = env.step(action.item())
  print(f'angle: {observation[2].item()}, reward: {-observation[2].item()**2}')

  time.sleep(0.1)


  # print(f'reward: {reward}, obs: {observation}')
  if done:
      break

env.close()
