# agent-exchange

## Background
Simulation is a powerful tool. In complicated environments such as financial exchanges, empirical simulations provide a risk-free way to understand an idea. In this project, we aim to make a clear interface for an exchange and its participants. Concretely, we create a standardized way to simulate the dynamics of an exchange and its participants. The primary way to use this is to do the following:

1. Implement an `Agent` class (or multiple!)
2. Instantiate multiple `Agent`s, either of the same class or of separate ones (all with the same interface)
3. Instantiate an `Exchange` object with all of the agent objects
4. Call `simulate_steps(n)`, with `n` being the number of steps you'd like to simulate

Afterwards, you can look at your `Agent`s to view their state at the end of the simulation. Moreover, you can define the `Agent` class such that it stores all of its previous state, which would allow you to view the agent's entire history.