"""Monte Carlo state value prediction for the game of Blackjack."""

# Dependancies.
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

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
        self.player_values = range(12, 21 + 1)
        self.dealer_values = range(1, 10 + 1)
        self.usable_aces = range(0, 1 + 1)
        self.state_space = {}
        for player_sum in self.player_values:
            for dealer_sum in self.dealer_values:
                for usable_ace in self.usable_aces:
                    state = (player_sum, dealer_sum, usable_ace)
                    self.state_space[state] = {
                        'state actions': [self.ACTIONS.HIT, self.ACTIONS.STICK],
                        'state reward': 0,
                        'state value': 0,
                        'state returns': {
                            'entries': 0,
                        },
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
        self.plot_state_space(1)

    def plot_state_space(self, episodes: int):
        """Visualise the value function after Monte Carlo estimation."""

        player_values_list = list(self.player_values)
        dealer_values_list = list(self.dealer_values)
        state_value_function = np.zeros((len(player_values_list),
                                         len(dealer_values_list)))

        for i, player_value in enumerate(player_values_list):
            for j, dealer_value in enumerate(dealer_values_list):
                axis = (player_value, dealer_value, 1)
                state_value_function[i, j] = self.state_space[axis]['state value']

        X, Y = np.meshgrid(player_values_list, dealer_values_list)

        fig, axes = plt.subplots(1, 2, figsize=(6, 8),
                                 subplot_kw={'projection': '3d'})

        axes[1].plot_surface(X, Y, state_value_function, cmap='viridis')
        axes[1].set_xlabel('Player Hand Value')
        axes[1].set_ylabel('Dealer hand Value')
        axes[1].set_zlabel('Value Function')
        axes[1].set_title(f'Value Function after {episodes} episodes')

        plt.show()

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
            # print(f"Game state {state}.")

            # Get the players next action.
            player_action =  self.policy[state]
            episode_time['action'] = player_action
            # print(f"Agent action {player_action}.")

            episode_time['reward'] = 0
            episode.append(copy.copy(episode_time))

            # Resolve the players decision.
            if player_action == self.ACTIONS.HIT:
                # print("\nAgent hits.")
                # Draw a card.
                player_card_value = random.randint(2, 11)
                # print(f"Agent draws {player_card_value}.")
                # Check whether it can be an ace.
                if player_card_value == 11 and player_hand_value \
                                                + player_card_value > 21:
                    player_card_value = 1
                elif player_card_value == 11 and player_hand_value \
                                                + player_card_value <= 21:
                    player_has_ace = 1

                player_hand_value += player_card_value
                # print(f"Agent hand value {player_hand_value}.")

                # Check whether the player busts.
                if player_hand_value > 21:
                    # print(f"Agent busts.")
                    break

                # Continue playing until the player decides to stick.
                continue
            
            # print("\nAgent sticks.")
            break

        # Play the dealers hand.
        # The dealer will hit everything below 17.
        while dealer_hand_value < 17 and player_hand_value <= 21:
            # Draw a card.
            dealer_card_value = random.randint(2, 11)
            # print(f"dealer draws {dealer_card_value}.")
            # Check whether it can be an ace.
            if dealer_card_value == 11 and dealer_hand_value \
                                            + dealer_card_value > 21:
                dealer_card_value = 1

            dealer_hand_value += dealer_card_value
            # print(f"Dealer hand value {dealer_hand_value}.")

            # Check whether the dealer busts.
            if dealer_hand_value > 21:
                # print("Dealer busts.")
                break

        # Determine the winner of the game.
        # Check if the dealer has won.
        if dealer_hand_value <= 21 and dealer_hand_value > player_hand_value \
            or player_hand_value > 21:
            # print("Dealer wins.")
            episode[-1]['reward'] = -1
        # Check if the game is a draw.
        elif dealer_hand_value == player_hand_value:
            # print("Dealer & Agent draw.")
            pass
        # Else the player wins.
        else:
            # print("Agent wins.")
            episode[-1]['reward'] = 1

        # print(f"({player_hand_value}, {dealer_hand_value}, {player_has_ace})")
        # print(episode)
        return episode
    
    def check_state_visited_earlier(self, episode: list, index: int) -> bool:
        """Check if the state is visted earlier in the episode."""

        state_visited_earlier = False
        # Loop through the earlier states.
        for i in range(index):
            # Check if the state is visited earlier.
            if episode[i]['state'] == episode[index]['state']:
                state_visited_earlier = True
                break

        return state_visited_earlier
    
    def update_state_value(self, state: tuple[int, int, int],
                           expected_return: float) -> None:
        """Update the state value estimation."""

        # Update the state value estimation.
        self.state_space[state]['state returns']['entries'] += 1
        state_average_value = self.state_space[state]['state value']
        entries = self.state_space[state]['state returns']['entries']
        self.state_space[state]['state value'] += (expected_return \
                                                    - state_average_value) \
                                                    / entries
    
    def process_episode(self, episode: list,
                        gamma: float,
                        expected_return: float) -> None:
        """Process an episode to estimate the value function under the policy.
        This is done using first-visit Monte Carlo prediction."""

        for index, step in enumerate(episode):
            # Calculate the expected return.
            expected_return = gamma * expected_return + step['reward']

            # Check if the state is visited earlier in the episode.
            state_visited_earlier = self.check_state_visited_earlier(episode,
                                                                     index)

            # Only change the expected return for the state if it is not 
            # visited earlier in the episode. This is a first-visit MC Method.
            if not state_visited_earlier:
                # Update the average for the state value estimation.
                self.update_state_value(step['state'], expected_return)
                

    def estimate_value_function(self):
        """Estimate the value function under the policy with First-visit
        Monte Carlo Prediction."""

        maximum_number_of_episodes = 1000000
        gamma = 1

        # Loop for every episode.
        for episode_counter in range(maximum_number_of_episodes):
            # Play a hand of Blackjack under the policy.
            episode = self.play_hand()

            # Reverse the list.
            # I belive this is where the no bootstrapping thing comes in.
            episode = list(reversed(episode))
            expected_return = 0

            # Process the episode.
            self.process_episode(episode, gamma, expected_return)
            print(f"\rCompleted {episode_counter/maximum_number_of_episodes}",
                  end = "")
        # This is here for errors. It was hard to read them without a new line.
        print("")

if __name__ == "__main__":
    beat_the_dealer = Blackjack()
