"""Plays the game Can't Stop."""
import random
import math
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import copy

class Cell:
  def __init__(self) -> None:
    """A list of markers. Neutral markers are represented as 0. Markers from Player 1 are
    represented as 1, and so forth.
    """

    self.markers: List[int] = []


class Board:
  def __init__(self, column_range: List[int], offset: int, initial_height: int) -> None:
    """First two columns are unused. Used columns vary from range 2 to 12 (inclusive)."""

    self.column_range = column_range
    self.offset = offset
    self.initial_height = initial_height
    self.board: List[List[Cell]] = [[] for _ in range(self.column_range[1] + 1)]

    height = self.initial_height
    for x in range(self.column_range[0], self.column_range[1] + 1):
      for _ in range(height):
        self.board[x].append(Cell())
      if x < self.column_range[1] / 2 + 1:
        height += self.offset
      else:
        height -= self.offset

    def check_board_equality(self, board: 'Board') -> bool:
      """ Check if the boards self and board are equal."""

      height = self.initial_height
      for x in range(self.column_range[0], self.column_range[1] + 1):
        list_of_cells = self.board[x]
        list_of_cells_2 = board.board[x]
        for i in range(len(list_of_cells)):
          # Check if all Cell in [x] are equal
          if sorted(list_of_cells[i].markers) != sorted(list_of_cells_2[i].markers):
            return False
        if x < self.column_range[1] / 2 + 1:
          height += self.offset
        else:
          height -= self.offset
        return True

class Game:
  def __init__(self, n_players: int, dice_number: int, dice_value: int, column_range: List[int],
                offset: int, initial_height: int):
    """Class to represent a Can't Stop game state.
    - n_players is the number of players (only 2 is possible at the moment).
    - dice_number is the number of dice used in the Can't Stop game.
    - dice_value is the number of faces of a single die.
    - column_range is a list denoting the range of the board game columns.
    - offset is the height difference between columns.
    - initial_height is the height of the columns at the board border.
    - finished_columns is a list of tuples of 2 integers indicating which columns were won by a
      certain player. e.g.: (2,1) = column 2 won by player 1.
    - player_won_column is a list of 2-tuples indicating which columns were won by a certain
      player in the current round. e.g.: (2,1) = column 2 won by player 1 in the current round.
    - dice_action refers to which action the game is at the moment, if the player is choosing
      which combination or if the player is choosing if he wants to continue playing or not.
    - n_neutral_markers is the number of neutral markers currently in the board.
    - neutral_positions is a tuple of 2 integers storing where the neutral markers are stored
      in the board (column index, cell index).
    - current_roll refers to all dice_number dice roll.
    """

    self.n_players = n_players
    self.dice_number = dice_number
    self.dice_value = dice_value
    self.column_range = column_range
    self.offset = offset
    self.initial_height = initial_height
    self.board_game = Board(self.column_range, self.offset, self.initial_height)
    self.player_turn = 1
    self.finished_columns = []
    self.player_won_column = []
    self.dice_action = True
    self.current_roll = self.roll_dice()
    self.n_neutral_markers = 0
    self.neutral_positions = []
    self.actions_taken = []

  def check_game_equality(self, game: 'Game') -> bool:
    """Check if self and game are equal."""

    condition_1 = self.check_boardgame_equality(game) and self.player_turn == game.player_turn
    condition_2 = self.n_neutral_markers == game.n_neutral_markers
    condition_3 = sorted(self.neutral_positions) == sorted(game.neutral_positions)
    return condition_1 and condition_2 and condition_3

  def set_manual_board(self, manual_board: List[List[Cell]],
                        finished_columns: List[Tuple[int, int]],
                        player_won_column: List[Tuple[int, int]]) -> None:
    """Manually set self.board_game."""

    self.finished_columns = finished_columns
    self.player_won_column = player_won_column
    height = self.initial_height
    for x in range(self.column_range[0], self.column_range[1] + 1):
      list_of_cells = self.board_game.board[x]
      list_of_cells_2 = manual_board[x - 2]
      for i in range(len(list_of_cells)):
        list_of_cells[i].markers = list_of_cells_2[i]
      if x < self.column_range[1] / 2 + 1:
        height += self.offset
      else:
        height -= self.offset

  def check_boardgame_equality(self, game: 'Game') -> bool:
    """Check if self.board_game and game.board_game represent the same board."""

    condition_1 = self.board_game.check_board_equality(game.board_game)
    condition_2 = sorted(self.finished_columns) == sorted(game.finished_columns)
    condition_3 = sorted(self.player_won_column) == sorted(game.player_won_column)

    return condition_1 and condition_2 and condition_3

  def compute_advances_per_column(self) -> Dict[int, int]:
    """Together with self.similarity_boards, calculate how observationally similar two
    boards are. Explained in detail in the article "What can we Learn Even From the Weakest?
    Learning Sketches for Programmatic Strategies".
    """
    score = {}

    for i in range(2, 13):
      score[i] = 0

    for index_row in range(len(self.board_game.board)):
      advances_column = 0
      for index in range(len(self.board_game.board[index_row])):
        markers = self.board_game.board[index_row][index].markers
        if len(markers) == 2:
          advances_column = index + 1
          break
        if len(markers) == 1 and markers[0] == 1:
          advances_column = index + 1
          break
      score[index_row] = advances_column

    for complete_columns in self.finished_columns:
      if complete_columns[1] == 1:
        if complete_columns[0] <= 7:
          score[complete_columns[0]] = 3 + (complete_columns[0] - 2) * 2 + 1
        else:
          score[complete_columns[0]] = 3 + (12 - complete_columns[0]) * 2 + 1

    return score

  def similarity_boards(self, game: 'Game') -> float:
    """Together with self.compute_advances_per_column, calculate how observationally similar
    two boards are. Explained in detail in the article "What can we Learn Even From the
    Weakest? Learning Sketches for Programmatic Strategies".
    """
    score_oracle = self.compute_advances_per_column()
    score_learner = game.compute_advances_per_column()

    max_overlap = 0
    score_value_learner = 0

    for col, num in score_oracle.items():
      max_overlap += num
      score_value_learner += min(score_learner[col], num)

    return score_value_learner / max_overlap

  def play(self, chosen_play: Union[Tuple[int], str],
            ignore_busted: Optional[bool] = False) -> None:
    """Apply the chosen_play action to the current state of the game.
    - chosen_play can be either 'y' and 'n' that means the player wishes to respectively
    continue or stop playing the current turn. It can also be a tuple, e.g. (2,6), that means
    advancement in the columns 2 and 6.
    - ignore_busted is a parameter meant for debugging, it  ignores going bust and passing the
    turn.
    This method takes care of ignoring columns from chosen_play that are already completed or
    not achievable due to lack of neutral markers, the going bust functionality, passing turns,
    and editing the board accordingly.
    """
    # Prevent illegal moves.
    assert chosen_play in self.available_moves()
    if chosen_play == 'n':
      self.transform_neutral_markers()
      # Next action should be to choose a dice combination
      self.dice_action = True
      # Roll dice so the previous roll isn't stored in the game state across two turns.
      self.current_roll = self.roll_dice()
      return
    if chosen_play == 'y':
      # Next action should be to choose a dice combination
      self.dice_action = True
      # Roll dice so the previous roll isn't stored in the game state across two turns.
      self.current_roll = self.roll_dice()
      return
    if not ignore_busted:
      if self.is_player_busted(self.available_moves()):
        return
    for die_position in range(len(chosen_play)):
      current_position_zero = -1
      current_position_id = -1
      col = chosen_play[die_position]
      cell_list = self.board_game.board[col]
      for i in range(0, len(cell_list)):
        if 0 in cell_list[i].markers:
          current_position_zero = i
        if self.player_turn in cell_list[i].markers:
          current_position_id = i

      # If there's no zero and no player_id marker
      if current_position_zero == current_position_id == -1:
        cell_list[0].markers.append(0)
        self.n_neutral_markers += 1
        self.neutral_positions.append((col, 0))

      # If there's no zero but there is player id marker
      elif current_position_zero == -1:
        # First check if the player will win that column
        self.n_neutral_markers += 1
        if current_position_id == len(cell_list) - 1:
          if (col, self.player_turn) not in self.player_won_column:
            self.player_won_column.append((col, self.player_turn))
        else:
          cell_list[current_position_id + 1].markers.append(0)
          self.neutral_positions.append((col, current_position_id + 1))
      # If there's zero    
      else:
        # First check if the player will win that column
        if current_position_zero == len(cell_list) - 1:
          if (col, self.player_turn) not in self.player_won_column:
            self.player_won_column.append((col, self.player_turn))
            #added!
            cell_list[current_position_zero].markers.remove(0)
            self.neutral_positions.remove((col,current_position_zero))
        else:
          cell_list[current_position_zero].markers.remove(0)
          cell_list[current_position_zero + 1].markers.append(0)
          self.neutral_positions.remove((col, current_position_zero))
          self.neutral_positions.append((col, current_position_zero + 1))
    # added
    if self.n_neutral_markers ==4:
      self.n_neutral_markers -= 1

    # Next action should be [y,n]
    self.dice_action = False
    # Then a new dice roll is done (same is done if the player is busted)
    self.current_roll = self.roll_dice()

  def transform_neutral_markers(self) -> None:
    """Transform the neutral markers into player_id markers (1 or 2)."""
    for neutral in self.neutral_positions:
      markers = self.board_game.board[neutral[0]][neutral[1]].markers
      for i in range(len(markers)):
        if markers[i] == 0:
          markers[i] = self.player_turn
      # Remove the previous player_turn id in order to keep only the furthest one
      col_cell_list = self.board_game.board[neutral[0]]
      for i in range(neutral[1] - 1, -1, -1):
        if self.player_turn in col_cell_list[i].markers:
          col_cell_list[i].markers.remove(self.player_turn)
          break

    # Special case example: Player 1 is about to win, for ex. column 7 
    # but they rolled a (7,7) tuple. That would add two instances (1,7) 
    # in the finished columns list. Remove the duplicates 
    # from player_won_column.
    self.player_won_column = list(set(self.player_won_column))

    # Check if the player won some column and update it accordingly.
    for column_won in self.player_won_column:
      self.finished_columns.append((column_won[0], column_won[1]))

      for cell in self.board_game.board[column_won[0]]:
        cell.markers.clear()

    self.player_won_column.clear()

    if self.player_turn == self.n_players:
      self.player_turn = 1
    else:
      self.player_turn += 1

    self.n_neutral_markers = 0
    self.neutral_positions = []

  def erase_neutral_markers(self) -> None:
    """Remove the neutral markers because the player went bust."""

    for neutral in self.neutral_positions:
      markers = self.board_game.board[neutral[0]][neutral[1]].markers
      if 0 in markers:
        markers.remove(0)

    self.n_neutral_markers = 0
    self.neutral_positions = []

  def is_player_busted(self, all_moves: Union[List[Tuple[int]], List[str]]) -> bool:
    """Check if the player has no remaining plays.
    - all_moves is either a list of tuples of integers relating to the possible plays the
    player can make or the list ['y', 'n'] regarding  if the player wants to continue playing
    or not.
    """

    if all_moves == ['y', 'n']:
      return False
    if len(all_moves) == 0:
      self.erase_neutral_markers()
      self.player_won_column.clear()
      if self.player_turn == self.n_players:
        self.player_turn = 1
      else:
        self.player_turn += 1
      # A new dice roll is done (same is done if a play is completed)
      self.current_roll = self.roll_dice()
      return True
    if self.n_neutral_markers < 3:
      return False
    for move in all_moves:
      for i in range(len(move)):
        list_of_cells = self.board_game.board[move[i]]
        for cell in list_of_cells:
          if 0 in cell.markers:
            return False
    self.erase_neutral_markers()
    self.player_won_column.clear()
    if self.player_turn == self.n_players:
      self.player_turn = 1
    else:
      self.player_turn += 1
    # A new dice roll is done (same is done if a play is completed)
    self.current_roll = self.roll_dice()
    return True

  def roll_dice(self) -> Tuple[int]:
    """Return a tuple with integers representing the dice roll."""

    my_list = []
    for _ in range(self.dice_number):
      my_list.append(random.randrange(1, self.dice_value + 1))
    return tuple(my_list)

  def check_tuple_availability(self, tup) -> bool:
    """Check if there is a neutral marker in both tuples columns taking into account the number
    of neutral_markers currently on the board.
    """

    # Check if one of the columns from tup is already completed
    for tuple_finished in self.finished_columns:
      if tuple_finished[0] == tup[0] or tuple_finished[0] == tup[1]:
        return False
    for tuple_column in self.player_won_column:
      if tuple_column[0] == tup[0] or tuple_column[0] == tup[1]:
        return False

    # Variables to store if there is a neutral marker in tuples columns.
    is_first_value_valid = False
    is_second_value_valid = False

    for cell in self.board_game.board[tup[0]]:
      if 0 in cell.markers:
        is_first_value_valid = True

    for cell in self.board_game.board[tup[1]]:
      if 0 in cell.markers:
        is_second_value_valid = True

    if self.n_neutral_markers in [0, 1]:
      return True
    elif self.n_neutral_markers == 2:
      if is_first_value_valid and is_second_value_valid:
        return True
      elif not is_first_value_valid and is_second_value_valid:
        return True
      elif is_first_value_valid and not is_second_value_valid:
        return True
      elif tup[0] == tup[1]:
        return True
      else:
        return False
    else:
      if is_first_value_valid and is_second_value_valid:
        return True
      else:
        return False

  def check_value_availability(self, value: int) -> bool:
    """Check if the player is allowed to play in the column 'value'."""

    # First check if the column 'value' is already completed
    for tuple_finished in self.finished_columns:
      if tuple_finished[0] == value:
        return False
    for tuple_column in self.player_won_column:
      if tuple_column[0] == value:
        return False

    if self.n_neutral_markers < 3:
      return True
    list_of_cells = self.board_game.board[value]
    for cell in list_of_cells:
      if 0 in cell.markers:
        return True
    return False

  def available_moves(self) -> Union[List[str], List[Tuple[int]]]:
    """Calculate the available actions the player has at their disposal.
    It depends on self.dice_action if the available actions are either to continue or stop
    playing or calculate the available columns according to self.current_roll and columns that
    are not yet completed.
    """

    if not self.dice_action:
      return ['y', 'n']
    standard_combination = [(self.current_roll[0] + self.current_roll[1],
                              self.current_roll[2] + self.current_roll[3]),
                            (self.current_roll[0] + self.current_roll[2],
                              self.current_roll[1] + self.current_roll[3]),
                            (self.current_roll[0] + self.current_roll[3],
                              self.current_roll[1] + self.current_roll[2])]
    combination = []
    for comb in standard_combination:
      first_value_available = self.check_value_availability(comb[0])
      second_value_available = self.check_value_availability(comb[1])
      if self.check_tuple_availability(comb):
        combination.append(comb)
      elif first_value_available and second_value_available:
        combination.append((comb[0],))
        combination.append((comb[1],))
      if first_value_available and not second_value_available:
        combination.append((comb[0],))
      if second_value_available and not first_value_available:
        combination.append((comb[1],))

    # Remove duplicate actions (e.g.: dice = (2,6,6,6) -> actions = [(8,12), (8,12), (8,12)])
    # Remove redundant actions (Example: (8,12) and (12,8))
    return [t for t in {x[::-1] if x[0] > x[-1] else x for x in combination}]

  def is_finished(self) -> Tuple[int, bool]:
    """Check if the game is over given the current state of the game. Return which player won
    (0 if the game is not over).
    """
    won_columns_player_1 = 0
    won_columns_player_2 = 0
    for tuples in self.finished_columns:
      if tuples[1] == 1:
        won_columns_player_1 += 1
      else:
        won_columns_player_2 += 1
    # >= instead of == because the player can have 2 columns and win another 2 columns
    # in one turn.
    if won_columns_player_1 >= 3:
      return 1, True
    elif won_columns_player_2 >= 3:
      return 2, True
    else:
      return 0, False

class Player:
  def __init__(self, name='default'):
    self.player_name = name
      
  def get_name(self):
    return self.player_name
  
  def get_action(self, game, *args):
    """Return the action to be made by the player given the game state passed. Concrete classes
    must implement this method.
    """
    pass

class Couto_Player(Player):

  def __init__(self):
    # Incremental score for the player. If it reaches self.threshold, 
    # chooses the 'n' action, chooses 'y' otherwise.
    # Columns weights used for each type of action
    self.progress_value = [0, 0, 7, 7, 3, 2, 2, 1, 2, 2, 3, 7, 7]
    self.move_value = [0, 0, 7, 0, 2, 0, 4, 3, 4, 0, 2, 0, 7]
    # Difficulty score
    self.odds = 7
    self.evens = 1
    self.highs = 6
    self.lows = 5
    self.marker = 6
    self.threshold = 29

  def get_action(self, state):
    actions = state.available_moves()
    if actions == ['y', 'n']:
      # If the player stops playing now, they will win the game, therefore
      # stop playing
      if self.will_player_win_after_n(state):
        return 'n'
      # If there are available columns and neutral markers, continue playing
      if self.are_there_available_columns_to_play(state):
        return 'y'
      else:
        # Calculate score
        score = self.calculate_score(state)
        # Difficulty score
        score += (self.calculate_difficulty_score(state) * self.calculate_difficulty_score(state)) + self.calculate_difficulty_score(state)

        # If the current score surpasses the threshold, stop playing
        if score >= self.threshold:
          # Reset the score to zero for next player's round
          return 'n'
        else:
          return 'y'
    else:
      scores = np.zeros(len(actions))
      for i in range(len(scores)):
        # Iterate over all columns in action
        for column in actions[i]:
          scores[i] += self.positions_player_has_secured_column(state, column) + self.move_value[column]
      chosen_action = actions[np.argmax(scores)]
      return chosen_action

  def calculate_score(self, state):
    neutrals = [col[0] for col in state.neutral_positions]
    score = -sum(map((lambda _ :sum(neutrals)), neutrals))
    for col in neutrals:
      score += self.progress_value[col] * (self.number_cells_advanced_this_round_for_col(state, col) * self.positions_player_has_secured_column(state, col))
    return score

  def number_cells_advanced_this_round_for_col(self, state, column):
    """
    Return the number of positions advanced in this round for a given
    column by the player.
    """
    counter = 0
    previously_conquered = -1
    neutral_position = -1
    list_of_cells = state.board_game.board[column]

    for i in range(len(list_of_cells)):
      if state.player_turn in list_of_cells[i].markers:
        previously_conquered = i
      if 0 in list_of_cells[i].markers:
        neutral_position = i
    if previously_conquered == -1 and neutral_position != -1:
      counter += neutral_position + 1
      for won_column in state.player_won_column:
        if won_column[0] == column:
          counter += 1
    elif previously_conquered != -1 and neutral_position != -1:
      counter += neutral_position - previously_conquered
      for won_column in state.player_won_column:
        if won_column[0] == column:
          counter += 1
    elif previously_conquered != -1 and neutral_position == -1:
      for won_column in state.player_won_column:
        if won_column[0] == column:
          counter += len(list_of_cells) - previously_conquered
    return counter
  
  def positions_player_has_secured_column(self, state, column):
    if column not in list(range(2,13)):
      raise Exception('Out of range column passed to PlayerColumnAdvance()')
    counter = 0
    player = state.player_turn
    # First check if the column is already won
    for won_column in state.finished_columns:
      if won_column[0] == column and won_column[1] == player:
        return len(state.board_game.board[won_column[0]]) + 1
      elif won_column[0] == column and won_column[1] != player:
        return 0
    # If not, 'manually' count it while taking note of the neutral position
    previously_conquered = -1
    neutral_position = -1
    list_of_cells = state.board_game.board[column]

    for i in range(len(list_of_cells)):
      if player in list_of_cells[i].markers:
        previously_conquered = i
      if 0 in list_of_cells[i].markers:
        neutral_position = i
    if neutral_position != -1:
      counter += neutral_position + 1
      for won_column in state.player_won_column:
        if won_column[0] == column:
          counter += 1
    elif previously_conquered != -1 and neutral_position == -1:
      counter += previously_conquered + 1
      for won_column in state.player_won_column:
        if won_column[0] == column:
          counter += len(list_of_cells) - previously_conquered
    return counter

  def get_available_columns(self, state):
    """ Return a list of all available columns. """
    # List containing all columns, remove from it the columns that are
    # available given the current board
    available_columns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for neutral in state.neutral_positions:
      available_columns.remove(neutral[0])
    for finished in state.finished_columns:
      if finished[0] in available_columns:
        available_columns.remove(finished[0])

    return available_columns

  def will_player_win_after_n(self, state):
    """ 
    Return a boolean in regards to if the player will win the game or not 
    if they choose to stop playing the current round (i.e.: choose the 
    'n' action). 
    """
    clone_state = copy.deepcopy(state)
    clone_state.play('n')
    won_columns = 0
    for won_column in clone_state.finished_columns:
      if state.player_turn == won_column[1]:
        won_columns += 1
    #This means if the player stop playing now, they will win the game
    if won_columns == 3:
      return True
    else:
      return False

  def are_there_available_columns_to_play(self, state):
    """
    Return a booleanin regards to if there available columns for the player
    to choose. That is, if the does not yet have all three neutral markers
    used AND there are available columns that are not finished/won yet.
    """
    available_columns = self.get_available_columns(state)
    return state.n_neutral_markers != 3 and len(available_columns) > 0

  def calculate_difficulty_score(self, state):
    """
    Add an integer to the current score given the peculiarities of the
    neutral marker positions on the board.
    """
    difficulty_score = 0

    neutral = [n[0] for n in state.neutral_positions]
    # If all neutral markers are in odd columns
    if all([x % 2 != 0 for x in neutral]):
      difficulty_score += self.odds
    # If all neutral markers are in even columns
    if all([x % 2 == 0 for x in neutral]):
      difficulty_score += self.evens
    # If all neutral markers are is "low" columns
    if all([x <= 7 for x in neutral]):
      difficulty_score += self.lows
    # If all neutral markers are is "high" columns
    if all([x >= 7 for x in neutral]):
      difficulty_score += self.highs
    return difficulty_score

class RandomPlayer(Player):
  def get_action(self, game):
    actions = game.available_moves()
    return random.choice(actions)

class Rule_of_28_Player(Player):

  def __init__(self):
    # Incremental score for the player. If it reaches self.threshold, 
    # chooses the 'n' action, chooses 'y' otherwise.
    # Columns weights used for each type of action
    # self.progress_value = [0, 0, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6]
    # self.move_value = [0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
    # # Difficulty score
    # self.odds = 2
    # self.evens = -2
    # self.highs = 4
    # self.lows = 4
    # self.marker = 6
    # self.threshold = 28

    self.progress_value = [0, 0, 7, 7, 3, 2, 2, 1, 2, 2, 3, 7, 7]
    self.move_value = [0, 0, 7, 0, 2, 0, 4, 3, 4, 0, 2, 0, 7]
    
    # Difficulty score
    self.odds = 7
    self.evens = 1
    self.highs = 6
    self.lows = 5
    self.marker = 6
    self.threshold = 29

  def get_action(self, state):
    actions = state.available_moves()
    if actions == ['y', 'n']:
      # If the player stops playing now, they will win the game, therefore
      # stop playing
      if self.will_player_win_after_n(state):
        return 'n'
      # If there are available columns and neutral markers, continue playing
      elif self.are_there_available_columns_to_play(state):
        return 'y'
      else:
        # Calculate score
        score = self.calculate_score(state)
        # Difficulty score
        score += self.calculate_difficulty_score(state)
        # If the current score surpasses the threshold, stop playing
        if score >= self.threshold:
          # Reset the score to zero for next player's round
          return 'n'
        else:
          return 'y'
    else:
      scores = np.zeros(len(actions))
      for i in range(len(scores)):
        # Iterate over all columns in action
        for column in actions[i]:
          scores[i] += self.advance(actions[i]) * self.move_value[
            column] - self.marker * self.is_new_neutral(column, state)
      chosen_action = actions[np.argmax(scores)]
      return chosen_action

  def calculate_score(self, state):
    score = 0
    neutrals = [col[0] for col in state.neutral_positions]
    for col in neutrals:
      advance = self.number_cells_advanced_this_round_for_col(state, col)
      # +1 because whenever a neutral marker is used, the weight of that
      # column is summed
      score += (advance + 1) * self.progress_value[col]
    return score

  def number_cells_advanced_this_round_for_col(self, state, column):
    """
    Return the number of positions advanced in this round for a given
    column by the player.
    """
    counter = 0
    previously_conquered = -1
    neutral_position = -1
    list_of_cells = state.board_game.board[column]

    for i in range(len(list_of_cells)):
      if state.player_turn in list_of_cells[i].markers:
        previously_conquered = i
      if 0 in list_of_cells[i].markers:
        neutral_position = i
    if previously_conquered == -1 and neutral_position != -1:
      counter += neutral_position + 1
      for won_column in state.player_won_column:
        if won_column[0] == column:
          counter += 1
    elif previously_conquered != -1 and neutral_position != -1:
      counter += neutral_position - previously_conquered
      for won_column in state.player_won_column:
        if won_column[0] == column:
          counter += 1
    elif previously_conquered != -1 and neutral_position == -1:
      for won_column in state.player_won_column:
        if won_column[0] == column:
            counter += len(list_of_cells) - previously_conquered
    return counter

  def get_available_columns(self, state):
    """ Return a list of all available columns. """

    # List containing all columns, remove from it the columns that are
    # available given the current board
    available_columns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for neutral in state.neutral_positions:
      available_columns.remove(neutral[0])
    for finished in state.finished_columns:
      if finished[0] in available_columns:
        available_columns.remove(finished[0])

    return available_columns

  def will_player_win_after_n(self, state):
    """ 
    Return a boolean in regards to if the player will win the game or not 
    if they choose to stop playing the current round (i.e.: choose the 
    'n' action). 
    """
    clone_state = copy.deepcopy(state)
    clone_state.play('n')
    won_columns = 0
    for won_column in clone_state.finished_columns:
      if state.player_turn == won_column[1]:
        won_columns += 1
    # This means if the player stop playing now, they will win the game
    if won_columns == 3:
      return True
    else:
      return False

  def are_there_available_columns_to_play(self, state):
    """
    Return a booleanin regards to if there available columns for the player
    to choose. That is, if the does not yet have all three neutral markers
    used AND there are available columns that are not finished/won yet.
    """
    available_columns = self.get_available_columns(state)
    return state.n_neutral_markers != 3 and len(available_columns) > 0

  def calculate_difficulty_score(self, state):
    """
    Add an integer to the current score given the peculiarities of the
    neutral marker positions on the board.
    """
    difficulty_score = 0

    neutral = [n[0] for n in state.neutral_positions]
    # If all neutral markers are in odd columns
    if all([x % 2 != 0 for x in neutral]):
      difficulty_score += self.odds
    # If all neutral markers are in even columns
    if all([x % 2 == 0 for x in neutral]):
      difficulty_score += self.evens
    # If all neutral markers are is "low" columns
    if all([x <= 7 for x in neutral]):
      difficulty_score += self.lows
    # If all neutral markers are is "high" columns
    if all([x >= 7 for x in neutral]):
      difficulty_score += self.highs

    return difficulty_score

  def is_new_neutral(self, action, state):
    # Return a boolean representing if action will place a new neutral. """
    is_new_neutral = True
    for neutral in state.neutral_positions:
      if neutral[0] == action:
        is_new_neutral = False

    return is_new_neutral

  def advance(self, action):
    """ Return how many cells this action will advance for each column. """

    # Special case: doubled action (e.g. (6,6))
    if len(action) == 2 and action[0] == action[1]:
      return 2
    # All other cases will advance only one cell per column
    else:
      return 1

class Custom_Player(Player):
  def get_action(self, game):
    return get_custom_action(game)

def get_custom_action(game):
  return random.choice(game.available_moves()) # To be replaced.

def run_game(p2):
  """
  Run a game of Can't Stop and return 1 if the player won, zero otherwise.
  """
  game = Game(n_players = 2, dice_number = 4, dice_value = 6, column_range = [2,12], offset = 2, initial_height = 3)
  is_over = False
  who_won = None
    
  state_action_pairs = {}
  state_action_pairs[1] = []
  state_action_pairs[2] = []

  current_player = game.player_turn
  while not is_over:
    moves = game.available_moves()

    if game.is_player_busted(moves):
      if current_player == 1:
          current_player = 2
      else:
          current_player = 1
      continue
    else:
      if current_player != game.player_turn:
        exit()
      if game.player_turn == 1:
        moves = game.available_moves()
        chosen_play = get_action(game)
      else:
        chosen_play = p2.get_action(game)

      if chosen_play == 'n':               
        if current_player == 1:                        
          current_player = 2
        else:                
          current_player = 1

      game.play(chosen_play)

    who_won, is_over = game.is_finished()

    if is_over:
      return 1 if who_won == 1 else 0

@funsearch.run
def evaluate(player: int) -> float:
  """Returns the winrate of the action selection strategy over 4 strategies."""
  count = 0
  num_evals = 100

  if player == 0:
    p2 = RandomPlayer()
  if player == 1:
    p2 = Couto_Player()
  elif player == 2:
    p2 = Rule_of_28_Player()
  else:
    p2 = Custom_Player()

  for _ in range(num_evals):
    count += run_game(p2)

  return count / num_evals



def get_action(game: Game) -> Union[Tuple[int], str]:
  """Returns the chosen action (either a dice combination or a yes or no to end the round)."""
  actions = game.available_moves()
  if actions == ['y', 'n']:
    return function1(game)
  else:
    return function2(game)

@funsearch.evolve
def function1(game: Game)-> str:
  """Return 'y' or 'n' to continue or stop playing."""
  return random.choice(game.available_moves())

@funsearch.evolve
def function2(game: Game)-> Tuple[int]:
  """Return a tuple of integers representing the columns to advance."""
  return random.choice(game.available_moves())


