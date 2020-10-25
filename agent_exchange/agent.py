class Agent:
    def __init__(self):
        pass

    def get_action(self, exchange_state):
        """Given the state of an exchange, determine
        what action to take."""
        raise(NotImplementedError())

    def action_results_update(self, new_exchange_state, reward, done, info):
        """Get the results of the previous action. This allows the agent to
        update its state and/or its policy."""
        pass