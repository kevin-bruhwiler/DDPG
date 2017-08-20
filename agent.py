import model
import memory
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import numpy as np
import noise

class Agent():
	def __init__(self, input_size, hidden_size, num_actions, mem_size, batch_size, cuda=False):
		self.input_size = input_size
		self.batch_size = batch_size
		self.cuda = cuda
		self.actor = model.Actor(input_size, hidden_size, num_actions)
		self.critic = model.Critic(input_size, hidden_size, num_actions)
		if cuda:
			self.actor.cuda()
			self.critic.cuda()
		self.memory = memory.Memory(mem_size)
		self.noise = noise.Noise()
		self.loss = nn.MSELoss()
		self.actor_optim = optim.Adam(self.actor.parameters(), lr=0.0001)
		self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.0001)
		
	#Don't use inside this class
	def getAction(self, x):
		y = self.actor.forward(self.toVariable(x))
		if self.cuda:
			action = y.cpu().data.numpy().astype(float)
			return self.noise.getNoise(action)+action
		action = y.data.numpy().astype(float)
		return self.noise.getNoise(action) + action
		
	def getReward(self, x, a):
		y = self.critic.forward(x, a)
		if self.cuda:
			return y.cpu().data.numpy().astype(float)
		return y.data.numpy().astype(float)
		
	def predictAction(self, x):
		return self.actor.forward(x)
		
	def predictReward(self, x, a):
		return self.critic.forward(x, a)
		
	def toVariable(self, x):
		if self.cuda:
			return Variable(torch.from_numpy(x).float()).cuda()
		else:
			return Variable(torch.from_numpy(x).float())
		
	def step(self, observation, action, reward, next_observation):
		self.memory.add((observation, action, reward, next_observation))
		obs, acts, rewards, new_obs = self.memory.sample(self.batch_size)
		
		reward_labels = []
		for i in range(len(new_obs)):
			if new_obs[i][0] != None:
				n_o = self.toVariable(np.asarray(new_obs[i])).unsqueeze(0)
				future_actions = self.predictAction(n_o)
				future_rewards = self.getReward(n_o, future_actions)
				reward_labels.append(rewards[i] + 0.9*future_rewards[0])
			else:
				reward_labels.append(rewards[i])
		
		reward_labels = np.asarray(reward_labels).astype(float)
		predicted_rewards = self.predictReward(self.toVariable(obs), self.toVariable(acts))
		reward_loss = self.loss(predicted_rewards, self.toVariable(reward_labels))
		self.critic_optim.zero_grad()
		reward_loss.backward()
		self.critic_optim.step()
		
		actions = self.predictAction(self.toVariable(obs))
		rewards = self.predictReward(self.toVariable(obs), actions)
		rewards = -torch.mean(rewards)
		self.actor_optim.zero_grad()
		rewards.backward()
		self.actor_optim.step()
