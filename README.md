# agent-exchange

## What's this?
Simulation is a powerful tool. In complicated environments such as financial exchanges, empirical simulations provide a risk-free way to understand low-level interactions between market participants. In this project, we aim to make a clear interface for an exchange and its participants. Concretely, we create a standardized way to simulate the dynamics of an exchange and its participants. 

## Installation (for Mac/Linux)
1. Create a virtual environment: 
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
2. Install `agent-exchange`:
    ```
    pip install agent-exchange
    ```

## Usage
The following steps are the recommended workflow for using this project.
1. Implement an `Agent` class (or multiple!)
2. Instantiate multiple `Agent`s, either of the same class or of separate ones (all with the same interface)
3. Instantiate an `Exchange` object with all of the agent objects
4. Call `simulate_steps(n)`, where `n` is the number of steps you'd like to simulate

Afterwards, you can look at your `Agent`s to view their state at the end of the simulation. Moreover, you can define the `Agent` class such that it stores all of its previous state, which would allow you to view the agent's entire history.


## Concepts

### Inspiration
This project makes the most sense when viewed through the lens of reinforcement learning. In a reinforcement learning problem, there are agents that take actions in an environment and get rewards. The [`openai/gym`](https://github.com/openai/gym) project sought to standardize a reinforcement learning interface, however their interface does not allow for multiple agents to interact with each another. Here we aim to mirror `openai/gym`'s logic for multi-agent problems.

This multi-agent RL interface provides a straightforward tool for testing adversarial market strategies. Our examples are primarily be motivated by the scenario where multiple market participants (`agents`) are competing in a zero-sum game on an exchange.

### Exchanges
An exchange is a competitive environment. This name is drawn from the concept of financial exchanges, whereby participants compete to acquire as much money as possible.

### Agents
An agent is a participant in an exchange. There may be more than one agent participating in an exchange, and agents may compete with one another for scarce rewards.

### Project organization
All of the interface logic lives in the files `/agent_exchange/exchange.py` and `/agent_exchange/agent.py`.

## Learning
Examples are a straightforward way to learn mosts tools, and this tool is no exception. The `/examples/` directory contains some example implementations of the `agent-exchange` interface. The jupyter notebooks are meant to help provide inspiration for you to develop your own agent and exchange problems.

## Contributing
The interface should not change much over time, however it's always good to have more examples! If you are interested in modeling agent and exchange interactions using this tool, feel free to open a PR to add a simplified version of your use case into `/examples/`! Here are the guidelines for adding an example:
1. Create a subdirectory under `/examples/` for your example
2. Implement the Agent and Exchange interfaces in a `.py` file
3. Create a *running* jupyter notebook with all proper installations

