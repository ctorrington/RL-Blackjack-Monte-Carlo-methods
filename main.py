"""Monte Carlo state value prediction for the game of Blackjack."""

# Dependancies.
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

# Libraries.
import random
import copy
import time

from constants import Constants
from blackjack import generate_blackjack_episode
from off_policy_control import estimate_optimal_policy

class Blackjack:
    """Blackjack environment."""

    def __init__(self):
        """Initialise the environment."""

        self.player = None
        self.dealer = None

        self.ACTIONS = Constants.ACTIONS

        # Initialise the state space.
        self.player_values = range(12, 21 + 1)
        # TODO the dealer cant draw aces. idk why
        self.dealer_values = range(1, 10 + 1)
        self.usable_aces = range(0, 1 + 1)
        self.state_space = {}
        for player_sum in self.player_values:
            for dealer_sum in self.dealer_values:
                for usable_ace in self.usable_aces:
                    state = (player_sum, dealer_sum, usable_ace)

                    # State values.
                    self.state_space[state] = {
                        'state actions': {}, # State actions.
                        'policies': {}, # State policies.
                        'estimated return': 0, # Expected return for a state.
                        'state entries': 0, # Number of times a state is encountered.
                    }

                    # State actions.
                    for action in self.ACTIONS.as_tuple():
                        self.state_space[state]['state actions'][action] = {
                            # Add the action value and the cumulative weight
                            # for the action in the state.
                            'action value': 0, # Abitrary value.
                            'cumulative weights': 0, # Initialised as 0.
                        }

                    # State policy.
                    # Set the deterministic policy. Hit below 20.
                    player_sum, _, _ = state
                    if player_sum >= 20:
                        deterministic_action = self.ACTIONS.STICK
                    else:
                        deterministic_action = self.ACTIONS.HIT
                    # Set the behaviour policy.
                    self.state_space[state]['policies'] = {
                        'target': deterministic_action,
                        'behaviour': random.choice(self.ACTIONS.as_tuple()),
                    }
        self.data = []

        estimate_optimal_policy(self.state_space)
        self.estimate_value_function()
        self.animate_state_space()

    def exponential_frame_sequence(self, number_of_frames,
                                   number_of_values):
        """Create a sequence of frames for the animation."""

        frame_indices = np.linspace(0, number_of_frames - 1, num=number_of_frames) ** 5
        frame_indices = frame_indices.astype(int)
        frame_indices = np.clip(frame_indices, 0, number_of_values - 1)
        return frame_indices

    def animate_state_space(self):
        """Visualise the value function after Monte Carlo estimation."""

        # Plot the value function.

        # Initialise the plot.
        player_values_list = list(self.player_values)
        dealer_values_list = list(self.dealer_values)
        X, Y = np.meshgrid(player_values_list, dealer_values_list)
        z_data = np.zeros((len(player_values_list),
                           len(dealer_values_list)))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('Player Hand Value')
        ax.set_ylabel('Dealer hand Value')
        ax.set_zlabel('Value Function')
        print("Animating...")

        def update_plot(frame):
            """Update the plot."""

            print(frame)
            for i, player_value in enumerate(self.player_values):
                for j, dealer_value in enumerate(self.dealer_values):
                    # Usable ace.
                    state = (player_value, dealer_value, 1)
                    z_data[i, j] = self.data[frame][state]['estimated return']
                    ax.set_title(f'Value Function after {frame} episodes')

            if frame == len(self.data) - 1:
                for i, player_value in enumerate(self.player_values):
                    for j, dealer_value in enumerate(self.dealer_values):
                        # Usable ace.
                        state = (player_value, dealer_value, 1)
                        z_data[i, j] = self.data[-1][state]['estimated return']
                surface = ax.plot_surface(X, Y, z_data, cmap='viridis')
            else:
                surface = ax.plot_surface(X, Y, z_data, cmap='viridis')
                
    
        frames = self.exponential_frame_sequence(100, len(self.data))
        animation = FuncAnimation(fig, update_plot,
                                  frames=len(self.data),
                                  interval=1000,
                                  repeat=False)
        print("Completed animation.")
        plt.show()

        # Save the animation.
        # print("Saving animation...")
        # animation.save('blackjack_value_function.gif',
        #                writer='pillow',
        #                fps=5)
        # print("Animation saved.")
    
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
        self.state_space[state]['state entries'] += 1
        state_average_value = self.state_space[state]['estimated return']
        entries = self.state_space[state]['state entries']
        self.state_space[state]['estimated return'] += (expected_return \
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

    def track_data(self, episode_index: int,
                   start_time: float) -> None:
        """Track the data for the state value estimation."""

        track_data = copy.deepcopy(self.state_space)
        track_data['episode'] = episode_index
        track_data['computation duration'] = time.time() - start_time
        self.data.append(track_data)

    def estimate_value_function(self):
        """Estimate the value function under the policy with First-visit
        Monte Carlo Prediction."""

        print("Estimating the value function under the policy with First-visit"
              " Monte Carlo Prediction.")
              
        maximum_number_of_episodes = 100000
        gamma = 1
        track_data = True

        print(f"Computing {maximum_number_of_episodes} episodes "
              f"with gamma {gamma}.")

        # Loop for every episode.
        for episode_counter in range(maximum_number_of_episodes):
            # Play a hand of Blackjack under the policy.
            episode = generate_blackjack_episode(self.state_space, 'target')

            # Reverse the list.
            # I belive this is where the no bootstrapping thing comes in.
            episode = list(reversed(episode))
            expected_return = 0

            # Process the episode.
            start_time = time.time()
            self.process_episode(episode, gamma, expected_return)

            # Add the episode to the list of tracked episodes.
            if track_data and episode_counter >= maximum_number_of_episodes - 1:
                self.track_data(episode_counter, start_time)

            # Print the progress.
            print(f"\rCompleted {episode_counter/maximum_number_of_episodes}",
                  end = "")

        # This is here for errors. It was hard to read them without a new line.
        print("\rCompleted 1.0           \n")

if __name__ == "__main__":
    beat_the_dealer = Blackjack()
