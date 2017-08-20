from agent import Agent
import itertools
import gym
from gym import wrappers

def run(env, agent, iters, render):
	for episode in range(iters):
		observation = env.reset()
		total_reward = 0
		for step in itertools.count():
			if render:
				env.render()
			action = agent.getAction(observation)
			new_observation, reward, done, info = env.step(2*action)
			total_reward += reward
			if done:
				agent.step(observation, action, reward, [None])
				print('Episode {2} finished in {0} steps with reward {1}'.format(step, reward, episode))
				break
			else:
				agent.step(observation, action, reward, new_observation)
				observation = new_observation

if __name__=='__main__':
	iterations = 10000
	mem_size = 1000000
	hidden_size = 400
	batch_size = 64
	render = True
	env = gym.make('Pendulum-v0')
	env = wrappers.Monitor(env, '/tmp/pendulum-experiment-1', force=True)
	agent = Agent(env.observation_space.shape[0], hidden_size,
				  env.action_space.shape[0], mem_size, batch_size, cuda=True)
	run(env, agent, iterations, render)
