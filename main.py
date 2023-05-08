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
        self.player_values = range(12, 21 + 1)
        self.dealer_values = range(1, 10 + 1)
        self.usable_aces = range(0, 1 + 1)
        self.plot0 = None
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
                            'average': 0,
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
        self.plot_state_space()

    def plot_state_space(self):
        """Visualise the value function after Monte Carlo estimation."""

        player_values_list = list(self.player_values)
        dealer_values_list = list(self.dealer_values)
        value_function_0 = np.zeros((len(player_values_list), len(dealer_values_list)))
        value_function_1 = np.zeros((len(player_values_list), len(dealer_values_list)))
        
        for i, player_value in enumerate(player_values_list):
            for j, dealer_value in enumerate(dealer_values_list):
                state_0 = (player_value, dealer_value, 1)
                state_1 = (player_value, dealer_value, 1)
                value_function_0[i, j] = self.plot0[state_0]['state value']
                value_function_1[i, j] = self.state_space[state_1]['state value']

        X, Y = np.meshgrid(player_values_list, dealer_values_list)

        fig, axes = plt.subplots(1, 2, figsize=(6, 8), subplot_kw={'projection': '3d'})
        fig.tight_layout()
        
        # Plot for usable_ace = 0
        axes[0].plot_surface(X, Y, value_function_0, cmap='viridis')
        axes[0].set_xlabel('Dealer Hand Value')
        axes[0].set_ylabel('Player Hand Value')
        axes[0].set_zlabel('Value Function')
        axes[0].set_title('Value Function after 10 episodes')

        # Plot for usable_ace = 1
        axes[1].plot_surface(X, Y, value_function_1, cmap='viridis')
        axes[1].set_xlabel('Dealer Hand Value')
        axes[1].set_ylabel('Player Hand Value')
        axes[1].set_zlabel('Value Function')
        axes[1].set_title('Value Function after 1 000 000 episodes')

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
                if player_card_value == 11 and player_hand_value + player_card_value > 21:
                    player_card_value = 1
                elif player_card_value == 11 and player_hand_value + player_card_value <= 21:
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
            if dealer_card_value == 11 and dealer_hand_value + dealer_card_value > 21:
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

    def estimate_value_function(self):
        """Estimate the value function under the policy with First-visit
        Monte Carlo Prediction."""

        maximum_number_of_episodes = 1000000
        gamma = 1

        # Loop for every episode.
        for episode_counter in range(maximum_number_of_episodes):
            # episode_list = []
            # Play a hand of Blackjack under the policy.
            episode = self.play_hand()
            # Reverse the list. I belive this is where the no bootstrapping comes in
            episode = list(reversed(episode))
            # episode_list.append(episode)
            expected_return = 0
            # print(episode)
            for index, step in enumerate(episode):
                # print(step)
                # print(f"next reward {step['reward']}")
                expected_return = gamma * expected_return + step['reward']
                # Check if the current state appears in the upcoming states 
                # (the beginning states), this list is reversed.
                state_upcoming = False
                # print("upcoming states:")
                for upcoming_state in range(index + 1, len(episode)):
                    # print(f"STAAATE: {episode[upcoming_state]}")
                    if step['state'] == episode[upcoming_state]['state']:
                        state_upcoming = True
                        break
                if not state_upcoming:

                    # Update the average for the state value estimation.
                    self.state_space[step['state']]['state returns']['entries'] += 1
                    state_average_value = self.state_space[step['state']]['state value']
                    entries = self.state_space[step['state']]['state returns']['entries']
                    self.state_space[step['state']]['state value'] += (expected_return - state_average_value) / entries

            if episode_counter == 10:
                self.plot0 = copy.deepcopy(self.state_space)
            print(f"\rCompleted {episode_counter/maximum_number_of_episodes}", end = "")
        print("")

if __name__ == "__main__":
    beat_the_dealer = Blackjack()