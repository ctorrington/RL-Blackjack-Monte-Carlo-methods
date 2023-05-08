# RL-Blackjack-MC-Prediction
Monte Carlo Value prediction for a simple deterministic policy.

# Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Examples](#examples)
5. [Contributing](#contributing)
6. [License](#license)

## Introduction
This is a simple implementation of Monte Carlo prediction for the value function of a deterministic policy in Blackjack. The policy is to stick if the player's hand value is 20 or 21, and to hit otherwise. The player is dealt cards from an infinite deck (i.e. cards are replaced after being drawn). The player plays against a fixed dealer policy: stick if hand value is 17 or more, and hit otherwise. The player receives a reward of +1 if they win, 0 if they draw, and -1 if they lose.
This is being used as a frist step into Monte Carlo prediction methods.

## Installation
This project was written in Python 3.11.3. To install, clone the repository and install the requirements:
```
pip install -r requirements.txt
```

## Usage
To run the program, use the following command:
```
python main.py
```

## Examples
Below is an example of the output of the program:

![Gif of the value function updating in a 3D space](https://github.com/ctorrington/RL-Blackjack-MC-Prediction/blob/main/images/blackjack_value_function.gif?raw=true)

![Example output](https://github.com/ctorrington/RL-Blackjack-MC-Prediction/blob/main/images/mesh%20plot%20for%20value%20function.png?raw=true)

The image on the left shows the value function for the policy after 10 games have been simulated.
The image on the right shows the value function for the policy after 1 000 000 games have been simulated. The value functions are estimated using First-Vist Monte Carlo prediction.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the terms of the [MIT](https://choosealicense.com/licenses/mit/) license.


<!-- This is a simplified version of Blackjack. The player can only hit or stick.
Because of this, the player does not have a decsion to make with a hand value
less than 12, because it is impossible to lose - they should just hit. -->
