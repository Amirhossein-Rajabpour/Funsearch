from game import Game
from players.glenn_player import Glenn_Player
from players.random_player import RandomPlayer
import copy
import json
import pickle
from players.Couto_Player import Couto_Player
from players.Custom_Player import Custom_Player

def main(count, arg):
    game = Game(n_players = 2, dice_number = 4, dice_value = 6, column_range = [2,12],
            offset = 2, initial_height = 3)

    is_over = False
    who_won = None
    
    # p1 = Glenn_Player()
    # p2 = Couto_Player()
    p1 = None
    player1 = "You"
    if arg =='self-c':
        p2 = Couto_Player()
        player2 = "Couto"
    elif arg == 'self-g':
        p2 = Glenn_Player()
        player2 = "Glenn"
    elif arg == 'gc':
        p1 = Couto_Player()
        p2 = Glenn_Player()
        player1 = "Couto"
        player2 = "Glenn"
    elif arg =='self-custom':
        p2 = Custom_Player()
        player2 = "Custom"
    elif arg == 'custom-g':
        p1 = Custom_Player()
        p2 = Glenn_Player()
        player1 = "Custom"
        player2 = "Glenn"
      
    state_action_pairs = {}
    state_action_pairs[1] = []
    state_action_pairs[2] = []
    state_action_pairs_current_round = []

    current_player = game.player_turn
    while not is_over:
        moves = game.available_moves()

        if game.is_player_busted(moves):
            state_action_pairs_current_round = []
            
            if current_player == 1:
                print("Player 1 (%s) is busted!"%(player1))
                game.print_board()
                current_player = 2
            else:
                print("Player 2 (%s) is busted!"%(player2))
                game.print_board()
                current_player = 1
            continue
        else:
            if current_player != game.player_turn:
                exit()
            if game.player_turn == 1:
                # print(game.current_roll)
                moves = game.available_moves()
                print("<<<<Player: %s>>>>"%player1)
                print("Available actions: \t", moves)
                
                if moves[0] =='y' or moves[0]=='n':
                    if p1:
                        chosen_play = p1.get_action(game)
  
                    else:
                        print("Yes/NO")
                        chosen_play = p2.get_action(game)
                        print("Please choose yes/no! (y for yes and n for No)")
                        chosen_index = str(input())
                        while chosen_index != 'y' and chosen_index != 'n':
                            print("Please choose either y or n!")
                            chosen_index = str(input())
                        chosen_play = chosen_index
                
                    # assert game.n_neutral_markers <=3
                    
                else:
                    if p1:
                       chosen_play = p1.get_action(game) 

                    else:
                        print("Column decision")
                        print("Please choose the index of the action you want to choose (from 1 to %s)"%(len(moves)))
                        chosen_index = int(input())
                        while chosen_index > len(moves) or chosen_index<=0:
                            print("Please enter a value from 1 to ",len(moves))
                            chosen_index = int(input())
                        chosen_play = moves[chosen_index-1]
                        print("CHOSE", chosen_play)
                    
            else:
                print("<<<<Player: %s>>>>"%player2)
                print("Available actions: \t", moves)
                # moves = game.available_moves()
                # print("MOVES", moves)
                chosen_play = p2.get_action(game)
                # print(chosen_play)
                # print("\n\n")
            

            if chosen_play == 'n':               
                if current_player == 1:                        
                    current_player = 2
                else:                
                    current_player = 1

            game.play(chosen_play)
            game.print_board()
            

        who_won, is_over = game.is_finished()

        # print('who won = ', who_won)
        if is_over:
            if who_won==1:
                count +=1
            return count
import sys          
if __name__ == "__main__":
    count = 0
    arg = sys.argv[1]
    if arg== "gc" or arg== "self-c" or arg=="self-g" or arg=="self-custom" or arg=="custom-g":
        arg = str(arg)
    else:
        print("WRONGN ARGUMENT PASSED!")
        assert AssertionError
        exit()
    
    # print(arg)

    for i in range(1):
        count = main(count, arg)
    
    if arg == "gc":
        if count == 1:
            print("COUTO won!")
        elif count ==0:
            print("GLENN won!")
    elif arg == "self-c":
        if count == 1:
            print("YOU WON!")
        elif count == 0:
            print("COUTO won!")
    elif arg == "self-g":
        if count == 1:
            print("YOU WON!")
        elif count == 0:
            print("GLENN won!")
    elif arg == "self-custom":
        if count == 1:
            print("YOU WON!")
        elif count == 0:
            print("CUSTOM won!")
    elif arg == "custom-g":
        if count == 1:
            print("CUSTOM WON!")
        elif count == 0:
            print("GLENN won!")
