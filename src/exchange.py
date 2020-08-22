import numpy as np
from typing import Sequence
from src.agent import Agent

class Exchange:
	"""
	Defines the basic structure of an exchange.
	"""
	def __init__(self, agents: Sequence[Agent]):
		"""Make an Exchange object.
		"""
		self.agents = agents
		self.t = 0 # tracks the time of the simulation
		pass

	def update_exchange_state(self, actions: np.array):
		"""Handle all of the actions that the agents take.
		This may involve updating an internal order book,
		prioritizing certain agents, etc.
		"""
		# Code that determines a new exchange state, based
		# on the agents' actions

		# Set the exchange state variable
		
		raise(NotImplementedError())

	def get_exchange_state(self):
		"""Encodes the state of the exchange. This 
		"""
		raise(NotImplementedError())

	def __get_reward(self, agent_index):
		"""Get the reward that an agent
		receives after taking their action.
		"""
		return 0

	def __get_done(self, agent_index):
		"""Get info on whether or not an
		agent is done. By default, there
		is no terminal state for agents 
		on this exchange.
		"""
		return False

	def __get_info(self, agent_index):
		"""Get information that is to be given to users.
		This does not need to be implemented.
		"""
		return None

	def simulate_step(self):
		"""Take a step, where the exchange gets an
		action from each agent, handles
		the actions (i.e. determines which trades occur,
		which participants are first place, etc.), then
		returns each agent's position.
		"""

		# Get the actions from each agent. In this simulation,
		# we assume that agents submit their actions at the same
		# time.
		initial_exchange_state = self.get_exchange_state()
		actions = np.array([agent.get_action(initial_exchange_state) for agent in self.agents])
		
		# Handle all of the agents' actions.
		self.update_exchange_state(actions)

		# Broadcast the action results
		new_exchange_state = self.get_exchange_state()
		for agent_index in range(len(self.agents)):
			self.agents[agent_index].action_results_update(
				new_exchange_state,
				self.__get_reward(agent_index),
				self.__get_done(agent_index),
				self.__get_info(agent_index))

		# Increment the internal timer
		self.t += 1
		return

	def simulate_steps(self, n):
		"""Simulate multiple steps of an exchange.
		"""
		for i in range(n):
			self.simulate_step()
		return

