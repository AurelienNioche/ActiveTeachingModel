class State:

    def get_possible_actions(self):
        """Returns an iterable of all actions which can be taken
        from this state"""

    def take_action(self, action):
        """Returns the state which results from taking action 'action'"""

    def is_terminal(self, ):
        """Returns whether this state is a terminal state"""

    def get_reward(self):
        """"Returns the reward for this state"""
