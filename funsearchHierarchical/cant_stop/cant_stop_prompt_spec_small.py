"""Specification for playing the game Can't Stop.

Classes:
  Cell:
    Variables:
      markers: List[int]
        A list of markers. Neutral markers are represented as 0. Markers from Player 1 are represented as 1, and so forth.
    Functions:
      __init__():
        Initializes the markers list to an empty list.
  Board:
    Variables:
      column_range: List[int]
        A list of two integers representing the range of columns in the board.
      offset: int
        The height difference between columns.
      initial_height: int
        The height of the columns at the board border.
      board: List[List[Cell]]
        A list of lists of Cell objects representing the board.
    Functions:
      __init__(column_range: List[int], offset: int, initial_height: int):
        Initializes the board with the given column range, offset, and initial height. The first two columns are unused. Used columns vary from range 2 to 12 (inclusive).
      check_board_equality(board: Board) -> bool:
        Check if the specified board is equal to the current board.
  Game:
    Variables:
      n_players: int
        The number of players. Only 2 players are supported.
      dice_number: int
        The number of dice used in the game.
      dice_value: int
        The number of faces of a single die.
      column_range: List[int]
        A list of two integers representing the range of columns in the board.
      offset: int
        The height difference between columns.
      initial_height: int
        The height of the columns at the board border.
      board_game: Board
        The board object representing the game board.
      player_turn: int
        Indicates which player's turn it is. Player 1 is represented as 1, and Player 2 is represented as 2.
      finished_columns: List[Tuple[int, int]]
        A list of tuples of two integers indicating which columns were won by a certain player.
      player_won_column: List[Tuple[int, int]]
        A list of 2-tuples indicating which columns were won by a certain player in the current round.
      dice_action: bool
        Indicates the current action of the game. True if the player is choosing a dice combination, False if the player is choosing to continue playing or not.
      current_roll: Tuple[int]
        A tuple representing the current dice roll.
      n_neutral_markers: int
        The number of neutral markers currently in the board.
      neutral_positions: List[Tuple[int, int]]
        A list of tuples of two integers storing where the neutral markers are stored in the board (column index, cell index).
      actions_taken: List
        A list of actions taken by the player.
    Functions:
      __init__(n_players: int, dice_number: int, dice_value: int, column_range: List[int], offset: int, initial_height: int):
        Initializes the game with the specified parameters.
      check_game_equality(game: Game) -> bool:
        Check if the specified game is equal to the current game.
      set_manual_board(manual_board: List[List[Cell]], finished_columns: List[Tuple[int, int]], player_won_column: List[Tuple[int, int]]) -> None:
        Manually set the board_game variable.
      check_boardgame_equality(game: Game) -> bool:
        Check if the board_game of the specified game is equal to the current game's board_game.
      compute_advances_per_column() -> Dict[int, int]:
        Calculate the advances per column in the board, which is used to determine the similarity between two boards.
      similarity_boards(game: Game) -> float:
        Calculate the similarity between two boards using the advances per column.
      play(chosen_play: Union[Tuple[int], str], ignore_busted: Optional[bool] = False) -> None:
        Apply the chosen_play action to the current state of the game. The chosen play can be a tuple of integers representing the columns to advance or 'y' and 'n' to continue or stop playing.
      transform_neutral_markers() -> None:
        Transform the neutral markers into player_id markers (1 or 2).
      erase_neutral_markers() -> None:
        Remove the neutral markers because the player went bust.
      is_player_busted(all_moves: Union[List[Tuple[int]], List[str]]) -> bool:
        Check if the player has no remaining plays.
      roll_dice() -> Tuple[int]:
        Return a tuple with integers representing the dice roll.
      check_tuple_availability(tup) -> bool:
        Check if there is a neutral marker in both tuples columns taking into account the number of neutral markers currently on the board.
      check_value_availability(value: int) -> bool:
        Check if the player is allowed to play in the column 'value'.
      available_moves() -> Union[List[str], List[Tuple[int]]]:
        Calculate the available actions the player has at their disposal.
      is_finished() -> Tuple[int, bool]:
        Check if the game is over given the current state of the game.
"""

def run_game() -> Any:
  """Run a game of Can't Stop."""
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
def evaluate(n: int) -> float:
  """Returns the winrate of the action selection strategy over 10 games."""
  count = 0
  for _ in range(10):
    count += run_game()
  return count / 10


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
