"""Monte Carlo state value prediction for the game of Blackjack."""

# Dependancies.
import numpy as np
import matplotlib.pyplot as plt

# Libraries.
import random
from constants import Constants

class Blackjack:
    """Blackjack environment."""

    def __init__(self):
        """Initialise the environment."""

        self.player = None
        self.dealer = None

        ACTIONS = Constants.ACTIONS
        self.state_space = {}
        # Initialise the state space.
        for player_sum in range(12, 21 + 1):
            for dealer_sum in range(1, 10 + 1):
                for usable_ace in range(0, 1 + 1):
                    key = (player_sum, dealer_sum, usable_ace)

                    self.state_space[key] = {
                        'state actions': [ACTIONS.HIT, ACTIONS.STICK],
                        'state reward': 0,
                        'state value': 0,
                    }

        # Deterministic policy. The player will hit on anything less than 20.
        self.policy = {}
        for state in self.state_space:
            player_sum, _, _ = state
            if player_sum >= 20:
                self.policy[state] = ACTIONS.STICK
            else:
                self.policy[state] = ACTIONS.HIT

        for state in self.state_space:
            print(f"state {state}\n{self.state_space[state]}")

        self.reset_episode()


    def reset_episode(self):
        """Reset the environment."""

        # Set the policy.

        # Simulate the player drawing cards.
        self.player = random

if __name__ == "__main__":
    beat_the_dealer = Blackjack()