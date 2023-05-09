"""Off-policy Monte Carlo Control Method for estimating optimal policy"""

# Dependencies
import numpy as np
from blackjack import generate_blackjack_episode
import copy

from constants import Constants

ACTIONS = Constants.ACTIONS



def estimate_optimal_policy(state_space):
    """Off-policy Monte Carlo Control Method for estimating optimal policy"""

    print("Estimating the optimal policy with Monte Carlo Control Methods.")

    maximum_number_of_episodes = 100000
    gamma = 1
    track_data = []

    print(f"Computing {maximum_number_of_episodes} episosdes "
          f"with gamma {gamma}.")

    # loop for each episode.
    for episode_counter in range(maximum_number_of_episodes):
        # Generate an episode under the behaviour policy.
        policy = 'behaviour'
        episode = generate_blackjack_episode(state_space, policy)

        # Initialise the expected return.
        expected_return = 0

        # Initialise the weight.
        weight = 1

        # Step through each element in the episode.
            # The episode is reversed because we need to step through the 
            # policy backwards.
        episode = list(reversed(episode))
        for index, step in enumerate(episode):
            # Update the expected return for the state.
            expected_return = gamma  * expected_return + step['reward']

            # Update the cumulative weight for the state.
            state_cumulative_weight = state_space[step['state']]['state actions'][step['action']]['cumulative weight']
            state_cumulative_weight = state_cumulative_weight + weight
            state_space[step['state']]['state actions'][step['action']]['cumulative weight'] = state_cumulative_weight

            # Update the action value for the state.
            action_value = state_space[step['state']]['state actions'][step['action']]['action value']
            weight_importance_sampling_ratio = weight / state_cumulative_weight
            action_value = action_value + weight_importance_sampling_ratio * (expected_return - action_value)
            state_space[step['state']]['state actions'][step['action']]['action value'] = action_value

            # Update the target policy for the state.
            action_values = []
            # Get the action values for the state.
            for action in state_space[step['state']]['state actions']:
                action_values.append(state_space[step['state']]['state actions'][action]['action value'])
            max_action_value_index = np.argmax(action_values)
            # Get the correct action from the index.
            max_action = ACTIONS.as_tuple()[max_action_value_index]
            # Set the target policy to choose the maximum action value.
            state_space[step['state']]['policies']['target'] = max_action

            # Check if the weights need to be adjusted for the state.
            # ie, behaviour action != target action            
            if step['action'] != state_space[step['state']]['policies']['target']:
                # Continue with the next loop.
                continue
            weight = weight * (1 / len(ACTIONS.as_tuple()))

        # Track the data for graphing.
        if episode_counter < 10:
            track_data.append(copy.deepcopy(state_space))

        # Print the progress.
        print(f"\rComplted {episode_counter/maximum_number_of_episodes}",
              end="")

    print("\rCompleted 1.0           \n")

    return track_data


            