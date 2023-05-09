"""Generate an episode of Blackjack."""

# Libraries
import random

from constants import Constants

ACTIONS = Constants.ACTIONS

def process_player_action(game_state):
    """Process the action taken by the player."""

    hand_value, _, has_ace = game_state['state']
    action = game_state['action']

    # Process the player choosing to hit.
    if action == ACTIONS.HIT:
        # Simulate drawing a card.
        card_value = random.randint(2, 11)
        # print(f"player draws {card_value}")

        # Check if they can avoid busting with an ace.
        if has_ace and hand_value + card_value > 21:
            # Change the aces value to a 1 if they player will bust.
            hand_value -= 10
            has_ace = 0

        # Check whether the card can be an ace.
        if card_value == 11 and hand_value + card_value > 21:
            card_value = 1
            has_ace = 0
        elif card_value == 11 and hand_value + card_value <= 21:
            has_ace = 1

        # Add the new card value to the hands value.
        hand_value += card_value
        # print(f"new player value {hand_value}")
    
    # Process the player choosing to stick
    elif action == ACTIONS.STICK:
        pass

    return (hand_value, _, has_ace)

def simulate_player_turn(state,
                         state_space,
                         policy):
    """Simulate the players turn."""

    player_hand_value, dealer_hand_value, player_has_ace = state
    player_bust = False
    # Initialise the episode list.
    episode = []

    # While the player can still play.
    while not player_bust:
        # A single step within the episode.
        # Rewards are 1 for winning,
        #             0 for drawing,
        #            -1 for losing,
        timestep = {
            'state': tuple[int, int, int],
            'action': ACTIONS,
            'reward': int,
        }

        # Record the state in the timestep.
        timestep['state'] = state

        # Get the players action for the current state.
        action = state_space[state]['policies'][policy]
        # print(f"chosen action {action} in state {state}.")
        # Record the action in the timestep.
        timestep['action'] = action

        # Record the reward in the timestep.
        # The reward is 0 by default, it will be updated
        # after simulating the dealers hand & determining
        # a winner
        timestep['reward'] = 0
        # Should not need a copy below, creating a new varible each step.
        episode.append(timestep)

        # Update values from processing the players turn.
        player_hand_value, \
        dealer_hand_value, \
        player_has_ace = process_player_action(timestep)
        state = (player_hand_value, dealer_hand_value, player_has_ace)

        # Check whether the players is bust.
        if player_hand_value > 21:
            player_bust = True

        # Exit loop if they player chooses to stick.
        if action == ACTIONS.STICK:
            break

    return episode, player_hand_value, player_has_ace

def simulate_dealers_turn(dealer_hand_value) -> int:
    """Simulate the dealers turn."""

    # print(f"dealer has {dealer_hand_value}")

    # While the dealer can still hit.
    while dealer_hand_value < 17:
        # Draw a card.
        card_value = random.randint(2, 10)
        # print(f"dealer draws card {card_value}")

        # Check whether it can be an ace.
        if card_value == 11 and dealer_hand_value + card_value > 21:
            card_value = 1

        dealer_hand_value += card_value
        # print(f"dealer hand value {dealer_hand_value}")

    return dealer_hand_value

def simulate_hand(state_space, policy):
    """Simulate a single hand of Blackjack."""

    # Simulate the player & dealer drawing cards.
    player_hand_value = random.randint(12, 21)
    dealer_hand_value = random.randint(1, 10)
    # This is a simplified version of Blackjack.
    # The player player can only hit or stick.
    # The player cannot bust with a hand value less
    # than 11, & so they will hit until they have a 
    # decision to be made.
    
    if player_hand_value > 20:
        player_has_ace = 1
    else:
        player_has_ace = random.randint(0, 1)
      
    state = (player_hand_value, dealer_hand_value, player_has_ace)
    state_space = state_space

    # Simulate the players turn.
    episode, player_hand_value, player_has_ace = simulate_player_turn(state,
                                   state_space, policy)
    
    # Check if the player has not bust:
    if player_hand_value <= 21:
        # Simulate the dealers turn.
        dealer_hand_value = simulate_dealers_turn(dealer_hand_value)

    # Determine a winner.
    # The dealer wins.
    if dealer_hand_value <= 21 and dealer_hand_value > player_hand_value \
        or player_hand_value > 21:
        # print("dealer wins")
        episode[-1]['reward'] = -1
    # The game is a draw.
    elif dealer_hand_value == player_hand_value:
        # print("its a draw")
        pass
    # The player wins.
    else:
        # print("player wins")
        episode[-1]['reward'] = 1

    # print(f"player hand value:{player_hand_value}, dealer hand value:{dealer_hand_value}, player has ace:{player_has_ace}")
    # print(episode)
    return episode
      

def generate_blackjack_episode(state_space, policy) -> list:
    """Generate an episode of Blackjack."""

    state_space = state_space

    return simulate_hand(state_space, policy)
