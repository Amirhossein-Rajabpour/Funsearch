from players.player import Player
import numpy as np
import copy
import random
import math

class Custom_Player(Player):

    def get_action(self, game):
        """Returns the chosen action (either a dice combination or a yes or no to end the round)."""
        available_moves = game.available_moves()
  
        # If there's only one move, return it
        if len(available_moves) == 1:
            return available_moves[0]
        
        outcomes = {}
        
        for move in available_moves:
            game_copy = copy.deepcopy(game)
            game_copy.play(move)
            
            # Calculate the immediate reward (number of finished columns)
            immediate_reward = len(game_copy.finished_columns)
            
            # Use a heuristic to estimate future rewards (e.g., number of finished columns if this move leads to optimal play in subsequent rounds)
            max_future_outcomes = -float('inf')
            for future_move in game_copy.available_moves():
                future_game_copy = copy.deepcopy(game_copy)
                future_game_copy.play(future_move)
                max_future_outcomes = max(max_future_outcomes, len(future_game_copy.finished_columns))
            
            # Expected value calculation considering potential future moves
            expected_value = immediate_reward + (0.9 * max_future_outcomes / len(available_moves))
            outcomes[move] = expected_value
        
        best_move = max(outcomes, key=lambda k: outcomes[k])
        return best_move
