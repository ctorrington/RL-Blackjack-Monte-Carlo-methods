"""Monte Carlo state value prediction for the game of Blackjack."""

# Dependancies.
import numpy as np
import matplotlib.pyplot as plt

# Libraries.
import random
import copy

from constants import Constants

class Blackjack:
    """Blackjack environment."""

    def __init__(self):
        """Initialise the environment."""

        self.player = None
        self.dealer = None

        self.ACTIONS = Constants.ACTIONS

        # Initialise the state space.
        self.state_space = {}
        for player_sum in range(12, 21 + 1):
            for dealer_sum in range(1, 10 + 1):
                for usable_ace in range(0, 1 + 1):
                    key = (player_sum, dealer_sum, usable_ace)

                    self.state_space[key] = {
                        'state actions': [self.ACTIONS.HIT, self.ACTIONS.STICK],
                        'state reward': 0,
                        'state value': 0,
                        'state returns': [],
                    }

        # Deterministic policy.
        # The player will hit on anything less than 20.
        self.policy = {}
        for state in self.state_space:
            player_sum, _, _ = state
            if player_sum >= 20:
                self.policy[state] = self.ACTIONS.STICK
            else:
                self.policy[state] = self.ACTIONS.HIT

        self.estimate_value_function()

    def play_hand(self) -> list:
        """Generate an episode under the policy."""

        # Using a dictionary for tracking states, actions & rewards in an episode.
        # The dictionaries will be indexable within the list for the time stamps.
        episode = []
        episode_time = {'state': tuple[int, int, int],
                        'action': int,
                        'reward': int}

        player_hand_value = random.randint(12, 21)
        dealer_hand_value = random.randint(1, 10)
        if player_hand_value > 11:
            player_has_ace = 1
        else:
            player_has_ace = random.randint(0, 1)

        # Game loop.
        # Whle the player is not bust.
        while player_hand_value <= 21:
            state = (player_hand_value, dealer_hand_value, player_has_ace)
            episode_time['state'] = state
            print(f"Game state {state}.")
            
            # Get the players next action.
            player_action =  self.policy[state]
            episode_time['action'] = player_action
            print(f"Agent action {player_action}.")

            episode_time['reward'] = 0
            episode.append(copy.copy(episode_time))

            # Resolve the players decision.
            if player_action == self.ACTIONS.HIT:
                print("\nAgent hits.")
                # Draw a card.
                player_card_value = random.randint(2, 11)
                print(f"Agent draws {player_card_value}.")
                # Check whether it can be an ace.
                if player_card_value == 11 and player_hand_value + player_card_value > 21:
                    player_card_value = 1
                elif player_card_value == 11 and player_hand_value + player_card_value <= 21:
                    player_has_ace = 1

                player_hand_value += player_card_value
                print(f"Agent hand value {player_hand_value}.")

                # Check whether the player busts.
                if player_hand_value > 21:
                    print(f"Agent busts.")
                    break

                # Continue playing until the player decides to stick.
                continue
            
            print("\nAgent sticks.")
            break

        # Play the dealers hand.
        # The dealer will hit everything below 17.
        while dealer_hand_value < 17 and player_hand_value <= 21:
            # Draw a card.
            dealer_card_value = random.randint(2, 11)
            print(f"dealer draws {dealer_card_value}.")
            # Check whether it can be an ace.
            if dealer_card_value == 11 and dealer_hand_value + dealer_card_value > 21:
                dealer_card_value = 1

            dealer_hand_value += dealer_card_value
            print(f"Dealer hand value {dealer_hand_value}.")

            # Check whether the dealer busts.
            if dealer_hand_value > 21:
                print("Dealer busts.")
                break

        # Determine the winner of the game.
        # Check if the dealer has won.
        if dealer_hand_value <= 21 and dealer_hand_value > player_hand_value \
            or player_hand_value > 21:
            print("Dealer wins.")
            # episode_time['reward'] = -1
            episode[-1]['reward'] = -1
            # episode.append(episode_time)
        # Check if the game is a draw.
        elif dealer_hand_value == player_hand_value:
            print("Dealer & Agent draw.")
            # episode_time['reward'] = 0
            # episode[-1]['reward'] = -1
            # episode.append(episode_time)
        # Else the player wins.
        else:
            print("Agent wins.")
            # episode_time['reward'] = 1
            episode[-1]['reward'] = 1
            # episode.append(episode_time)

        print(episode)
        return episode

    def estimate_value_function(self):
        """Estimate the value function under the policy with First-visit
        Monte Carlo Prediction."""

        maximum_number_of_episodes = 1

        # Loop for every episode.
        for episode_counter in range(maximum_number_of_episodes):

            # Play a hand of Blackjack under the policy.
            episode = self.play_hand()


if __name__ == "__main__":
    beat_the_dealer = Blackjack()