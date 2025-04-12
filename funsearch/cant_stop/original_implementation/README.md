# Can't Stop Game Instructions

Welcome to the GitHub repository for the classic game, "Can't Stop". 

## The Program!
Here is the program that you need to understand and explain in our questionnaire. The strategy implemented within this code is known as Couto's strategy, named after its developer.
```
def get_action(self, state):
    actions = state.available_moves()
    if actions == ['y', 'n']:
        score = ((DifficultyScore * DifficultyScore) - sum(map((lambda _ :sum(neutrals)), neutrals) + DifficultyScore + sum(map((lambda x : (progress_value * (NumberAdvancedThisRound * PositionsPlayerHasSecuredInColumn))), neutrals))

        if WinAfterN(state):
            return 'n'
        elif AreThereAvailableColumnsToPlay(state):
            return 'y'
        else:
            if score >= 29:
                return 'n'
            else:
                return 'y'
    else:
        index = argmax(map((lambda x : sum(map((lambda x : (PositionsPlayerHasSecuredInColumn + move_value)), locallist))), actions))
    return actions[index]
```
You can find the implementation of this program by following this path:
`src/players/Couto_Player.py`



The next section will guide you through the process of running the game, explaining the different commands to interact with various settings.

## Getting Started and Running the Game!
In order to install all dependencies of the program, run the following command:

`python3 -m pip install -r requirements.txt`


Begin by navigating to the `src` folder, where you will find all the necessary files to run the game.

### Game Play Options

We offer different settings for diverse gameplay experiences:

#### Play against the Couto AI

To test your skills against the Couto AI player, use the following command in your terminal:

`python3 play_cant_stop.py self-c` 

#### Play against the Glenn AI

If you're up for a challenge against the Glenn AI player, run the following command:

`python3 play_cant_stop.py self-g` 

#### Watch Glenn vs Couto

Want to sit back and watch a match between Glenn and Couto? Simply execute the following command:

`python3 play_cant_stop.py gc` 

Please make sure that you are using Python 3 to run these scripts.

Enjoy the game!

----------
_The source codes are taken from [here](https://github.com/leandrocouto/sketch-learning/)._


