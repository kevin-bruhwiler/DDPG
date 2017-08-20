import torch
import torch.nn as nn

class Actor(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, output_size)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		
	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		x = self.tanh(x)
		return x
		
class Critic(nn.Module):
	def __init__(self, input_size, hidden_size, num_actions):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size+num_actions, hidden_size)
		self.fc3 = nn.Linear(hidden_size, 1)
		self.relu = nn.ReLU()
		
	def forward(self, x, a):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(torch.cat([x,a],1))
		x = self.relu(x)
		x = self.fc3(x)
		return x
