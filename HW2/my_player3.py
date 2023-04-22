import random
from copy import deepcopy
from read import readInput
from write import writeOutput
from host import GO

class MyPlayer():
    def __init__(self):
        self.type = 'minimax'

    # Record my piece type
    def set_piece_type(self, piece_type):
        self.piece_type = piece_type

    # Compare the current board with the previous board for KO condition (from host.py, slightly modified)
    def compare_board(self, board1, board2):
        for i in range(len(board1)):
            for j in range(len(board1[i])):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    # Copy the current board for potential testing (from host.py, slightly modified)
    def copy_board(self, board):
        return deepcopy(board)

    # Detect all the neighbors of a given piece (from host.py, slightly modified)
    def detect_neighbor(self, board, i, j):
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i - 1, j))
        if i < len(board) - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < len(board) - 1: neighbors.append((i, j + 1))
        return neighbors

    # Detect the neighbor allies of a given piece (from host.py, slightly modified)
    def detect_neighbor_ally(self, board, i, j):
        neighbors = self.detect_neighbor(board, i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    # Using DFS to search for all allies of a given piece (from host.py, slightly modified)
    def ally_dfs(self, board, i, j):
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(board, piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    # Find liberty of a given piece (from host.py, slightly modified)
    def find_liberty(self, board, i, j):
        ally_members = self.ally_dfs(board, i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(board, member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    # Find the died pieces that has no liberty in the board for a given piece type (from host.py, slightly modified)
    def find_died_pieces(self, board, piece_type):
        died_pieces = []
        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(board, i, j):
                        died_pieces.append((i,j))
        return died_pieces

    # Remove the dead pieces in the board (from host.py, slightly modified)
    def remove_died_pieces(self, board, piece_type):
        died_pieces = self.find_died_pieces(board, piece_type)
        if not died_pieces: return board
        new_board = self.remove_certain_pieces(board, died_pieces)
        return new_board

    # Remove the pieces of certain locations (from host.py, slightly modified)
    def remove_certain_pieces(self, board, positions):
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        return board

    # Check whether a placement is valid (from host.py, slightly modified)
    def valid_place_check(self, previous_board, board, piece_type, i, j):
        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            return False
        if not (j >= 0 and j < len(board)):
            return False
        
        # Check if the place already has a piece
        if board[i][j] != 0:
            return False

        # Copy the board for testing
        curr_board_copy = self.copy_board(board)

        # Check if the place has liberty
        curr_board_copy[i][j] = piece_type
        if self.find_liberty(curr_board_copy, i, j):
            return True
        
        # If not, remove the died pieces of opponent and check again
        dead_pieces = self.find_died_pieces(curr_board_copy, 3 - piece_type)
        curr_board_copy = self.remove_died_pieces(curr_board_copy, 3 - piece_type)
        if not self.find_liberty(curr_board_copy, i, j):
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if dead_pieces and self.compare_board(previous_board, curr_board_copy):
                return False

        return True

    # Find all valid actions for the current board
    def find_valid_moves(self, previous_board, board, piece_type):
        valid_moves = []
        for i in range(len(board)):
            for j in range(len(board[i])):
                if self.valid_place_check(previous_board, board, piece_type, i, j):
                    valid_moves.append((i,j))
        return valid_moves
    
    # Find the number of liberty for a piece
    def count_liberty(self, board, i, j):
        cnt = 0
        ally_members = self.ally_dfs(board, i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(board, member[0], member[1])
            for neighbor in neighbors:
                if board[neighbor[0]][neighbor[1]] == 0:
                    cnt += 1
        return cnt

    # Assign a score to the current board state for alpha-beta pruning
    def score(self, board, piece_type):
        player_cnt = opponent_cnt = 0
        player_lib = opponent_lib = 0
        for i in range(len(board)):
            for j in range(len(board[i])):
                # Count the number and liberty of my pieces
                if board[i][j] == self.piece_type:
                    player_cnt += 1
                    player_lib += self.count_liberty(board, i, j)
                # Count the number and liberty of my opponent's pieces
                elif board[i][j] == 3 - self.piece_type:
                    opponent_cnt += 1
                    opponent_lib += self.count_liberty(board, i, j)

        player = player_cnt + player_lib
        opponent = opponent_cnt + opponent_lib
        # If the next piece type is my piece type, then I'm the player
        if piece_type == self.piece_type:
            score = player - opponent
        # Or I'm the "opponent"
        else:
            score = opponent - player

        return score

    def minimax(self, previous_board, board, piece_type, alpha, beta, max_depth, score):
        if max_depth == 0:
            return score

        curr_board_copy = self.copy_board(board)
        valid_actions = self.find_valid_moves(previous_board, board, piece_type)
        best_score = score

        for action in valid_actions:
            # Apply the action and remove dead pieces
            next_board = self.copy_board(board)
            next_board[action[0]][action[1]] = piece_type
            next_board = self.remove_died_pieces(next_board, 3 - piece_type)

            # Evaluate the board
            next_board_score = self.score(next_board, 3 - piece_type)
            evaluation = self.minimax(curr_board_copy, next_board, 3 - piece_type, alpha, beta, max_depth - 1, next_board_score)
            curr_score = -1 * evaluation

            # Update the best score
            if curr_score > best_score:
               best_score = curr_score

            new_score = -1 * best_score
            # Play as minimizing player
            if piece_type == 3 - self.piece_type:
                player = new_score
                if player < alpha:
                    return best_score
                if best_score > beta:
                    beta = best_score

            # Play as maximizing player
            elif piece_type == self.piece_type:
                opponent = new_score
                if opponent < beta:
                    return best_score
                if best_score > alpha:
                    alpha = best_score
            
        return best_score

    # My minimax agent that pick the best action(s) from a list of valid actions
    def my_agent(self, previous_board, board, piece_type, alpha, beta):
        curr_board_copy = self.copy_board(board)
        valid_actions = self.find_valid_moves(previous_board, board, piece_type)
        if not valid_actions:
            return []

        # Adjust depth of tree based on number of valid actions
        if len(valid_actions) > 5:
            max_depth = 2
        else:
            max_depth = 3

        best_score = 0
        best_actions = []

        for action in valid_actions:
            next_board = self.copy_board(board)
            next_board[action[0]][action[1]] = piece_type
            next_board = self.remove_died_pieces(next_board, 3 - piece_type)

            next_board_score = self.score(next_board, 3 - piece_type)
            evaluation = self.minimax(curr_board_copy, next_board, 3 - piece_type, alpha, beta, max_depth, next_board_score)
            curr_score = -1 * evaluation

            if curr_score > best_score or not best_actions:
                best_score = curr_score
                alpha = best_score
                best_actions.append(action)
            elif curr_score == best_score:
                best_actions.append(action)

        return best_actions

if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)

    go = GO(N)
    go.set_board(piece_type, previous_board, board)

    player = MyPlayer()
    player.set_piece_type(piece_type)

    # Check the state of the board
    non_empty = 0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] != 0:
                non_empty += 1
                break

    # Play black & board is empty (first action in the game) -> directly place at (2, 2)
    if piece_type == 1 and non_empty == 0:
        action = (2, 2)
    # Play white & board has one non-empty (second action in the game) -> place at (2, 2) if empty
    elif piece_type == 2 and non_empty == 1:
        if board[2][2] == 0:
            action = (2, 2)
        else:
            action = (2, 3)
    # Remaining actions are derived from my agent
    else:
        alpha, beta = float('-inf'), float('inf')
        best_actions = player.my_agent(previous_board, board, piece_type, alpha, beta)
        # Randomly choose an action from all best actions returned by the agent
        if best_actions:
            action = random.choice(best_actions)
        # PASS if the agent cannot find an action
        else:
            action = 'PASS'

    writeOutput(action)