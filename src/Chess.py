import numpy as np


class Chess:
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.action_size = self.row_count * self.column_count
        self.promotion_size = 8 * 4
        self.__states_counter = {str(self.get_initial_state()): 1}
        self.__is_draw_due_to_repetitions = False
        self.__move_count_for_draw = 0
        self.__is_draw_due_to_50_move_rule = False
        self.__move_last_pawn = []
        self.__the_kings_was_moved = [False, False]
        self.__the_rooks1_was_moved = [False, False]
        self.__the_rooks2_was_moved = [False, False]

    def get_initial_state(self):
        initial_state = [
            [-5, -2, -3, -9, -10, -3, -2, -5],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [5, 2, 3, 9, 10, 3, 2, 5],
        ]
        return np.array(initial_state)

    def get_next_state(self, state, action, player):
        row_before, column_before = action[0][0], action[0][1]
        row_after, column_after = action[1][0], action[1][1]
        piece = state[row_before, column_before]
        piece_after = state[row_after, column_after] if row_after <= 7 else 100

        if piece == 1 and self.__move_last_pawn:
            # En passant
            row_before_last, column_before_last = self.__move_last_pawn[0][0], self.__move_last_pawn[0][1]
            row_after_last, column_after_last = self.__move_last_pawn[1][0], self.__move_last_pawn[1][1]
            if row_before_last == 6 and row_after_last == 4 and row_before == 3:
                if 7 - column_after_last == column_after:
                    state[row_after + 1, column_after] = 0

        if piece == 1 and row_after > 7:
            piece_if_promotion = {8: 2, 9: 3, 10: 5, 11: 9}[row_after]
            state[row_after, column_after] = piece_if_promotion
            state[row_before, column_before] = 0
        elif piece == 1:
            self.__move_last_pawn = [[row_before, column_before], [row_after, column_after]]
            state[row_after, column_after] = piece
            state[row_before, column_before] = 0
        elif piece == 10:
            self.__the_kings_was_moved[player == -1] = True
            self.__move_last_pawn = []
            self.__execute_the_king_move(state, row_before, column_before, row_after, column_after)
        elif piece == 5:
            if [row_before, column_before] == [7, 0]:  # don't sure about this
                self.__the_rooks1_was_moved[player == -1] = True
            elif [row_before, column_before] == [7, 7]:
                self.__the_rooks2_was_moved[player == -1] = True
            state[row_after, column_after] = piece
            state[row_before, column_before] = 0
            self.__move_last_pawn = []
        else:
            state[row_after, column_after] = piece
            state[row_before, column_before] = 0
            self.__move_last_pawn = []

        # Check state for draws
        key = str(state)
        self.__states_counter[key] = self.__states_counter.get(key, 0) + 1
        if self.__states_counter[key] >= 3:
            self.__is_draw_due_to_repetitions = True
        # Check move for draw
        self.__move_count_for_draw = 0 if piece == 1 or piece_after < 0 else self.__move_count_for_draw + 1
        if self.__move_count_for_draw >= 50:
            self.__is_draw_due_to_50_move_rule = True

        return state

    def __execute_the_king_move(self, state, row_before, column_before, row_after, column_after):
        if [row_before, column_before] == [7, 4] and [row_after, column_after] == [7, 6]:
            state[7, 4] = 0
            state[7, 5] = 5
            state[7, 6] = 10
            state[7, 7] = 0
        elif [row_before, column_before] == [7, 4] and [row_after, column_after] == [7, 2]:
            state[7, 0] = 0
            state[7, 1] = 0
            state[7, 2] = 10
            state[7, 3] = 5
            state[7, 4] = 0
        elif [row_before, column_before] == [7, 3] and [row_after, column_after] == [7, 1]:
            state[7, 0] = 0
            state[7, 1] = 10
            state[7, 2] = 5
            state[7, 3] = 0
        elif [row_before, column_before] == [7, 3] and [row_after, column_after] == [7, 5]:
            state[7, 3] = 0
            state[7, 4] = 5
            state[7, 5] = 10
            state[7, 6] = 0
            state[7, 7] = 0
        else:
            state[row_after, column_after] = 10
            state[row_before, column_before] = 0

    def __get_next_row_and_column(self, piece, row, column):  # Without pawns and castling
        next_row_and_column = []

        if piece == 2:
            next_row_and_column = [
                [row - 2, column - 1],
                [row - 2, column + 1],
                [row - 1, column - 2],
                [row + 1, column - 2],
                [row - 1, column + 2],
                [row + 1, column + 2],
                [row + 2, column - 1],
                [row + 2, column + 1]
            ]
        elif piece == 3:
            next_row_and_column = [
                [[row - num, column - num] for num in range(1, 8)],
                [[row + num, column + num] for num in range(1, 8)],
                [[row - num, column + num] for num in range(1, 8)],
                [[row + num, column - num] for num in range(1, 8)],
            ]
        elif piece == 5:
            next_row_and_column = [
                [[row, column - num] for num in range(1, 8)],
                [[row, column + num] for num in range(1, 8)],
                [[row - num, column] for num in range(1, 8)],
                [[row + num, column] for num in range(1, 8)],
            ]
        elif piece == 9:
            next_row_and_column = [
                [[row - num, column - num] for num in range(1, 8)],
                [[row + num, column + num] for num in range(1, 8)],
                [[row - num, column + num] for num in range(1, 8)],
                [[row + num, column - num] for num in range(1, 8)],
                [[row, column - num] for num in range(1, 8)],
                [[row, column + num] for num in range(1, 8)],
                [[row - num, column] for num in range(1, 8)],
                [[row + num, column] for num in range(1, 8)],
            ]
        elif piece == 10:
            next_row_and_column = [
                [row, column - 1],
                [row, column + 1],
                [row - 1, column],
                [row + 1, column],
                [row - 1, column + 1],
                [row + 1, column + 1],
                [row + 1, column - 1],
                [row - 1, column - 1]
            ]
        return next_row_and_column

    def __the_king_was_checked(self, state):
        king_coords = np.where(state == 10)
        row, column = king_coords

        array = [i for i in range(self.row_count)]  # Shows valid row or column
        # Check knight
        for direction in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
            dr, dc = direction
            if (row + dr in array) and (column + dc in array) and state[row + dr, column + dc] == -2:
                return True
        # Check pawns
        for direction in [(-1, 1), (-1, -1)]:
            dr, dc = direction
            if (row + dr in array) and (column + dc in array) and (state[row + dr, column + dc] == -1):
                return True
        # Check diagonals
        for direction in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            dr, dc = direction
            rowd = row + dr
            columnd = column + dc
            while (rowd in array) and (columnd in array):
                if state[rowd, columnd] == -3 or state[rowd, columnd] == -9:
                    return True
                if state[rowd, columnd] != 0:
                    break
                rowd += dr
                columnd += dc
        # Check horizontals and verticals
        for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            dr, dc = direction
            row += dr
            column += dc
            while (row in array) and (column in array):
                if state[row, column] == -5 or state[row, column] == -9:
                    return True
                if state[row, column] != 0:
                    break
                row += dr
                column += dc
        return False

    def __check_move_on_check(self, state, move):
        row_before, column_before = move[0][0], move[0][1]
        row_after, column_after = move[1][0], move[1][1]
        piece = state[row_before, column_before]
        possible_next_state = state.copy()

        if piece == 1 and self.__move_last_pawn:
            # En passant
            row_before_last, column_before_last = self.__move_last_pawn[0][0], self.__move_last_pawn[0][1]
            row_after_last, column_after_last = self.__move_last_pawn[1][0], self.__move_last_pawn[1][1]
            if row_before_last == 6 and row_after_last == 4 and row_before == 3:
                if 7 - column_after_last == column_after:
                    possible_next_state[row_after + 1, column_after] = 0

        if piece == 1 and row_after > 7:
            piece_if_promotion = {8: 2, 9: 3, 10: 5, 11: 9}[row_after]
            possible_next_state[row_after, column_after] = piece_if_promotion
            possible_next_state[row_before, column_before] = 0
        elif piece == 10:
            self.__execute_the_king_move(possible_next_state, row_before, column_before, row_after, column_after)
        else:
            possible_next_state[row_after, column_after] = piece
            possible_next_state[row_before, column_before] = 0

        king_1_was_checked = self.__the_king_was_checked(possible_next_state)
        if king_1_was_checked:
            return True
        return False

    def __fill_valid_pawns_moves(self, state, row, column, array, valid_moves):
        # Move forward
        if (row - 1 in array) and (state[row - 1, column] == 0):
            move = [[row, column], [row - 1, column]]
            if not self.__check_move_on_check(state, move):
                valid_moves.append(move)

            # Double pawn advance
            if row == 6 and state[row - 2, column] == 0:
                move = [[row, column], [row - 2, column]]
                if not self.__check_move_on_check(state, move):
                    valid_moves.append(move)

            # Promotion forward
            if row == 1:
                for figure in [8, 9, 10, 11]:  # 8 - knight, 9 - bishop, 10 - rook, 11 - queen
                    move = [[1, column], [figure, column]]
                    if not self.__check_move_on_check(state, move):
                        valid_moves.append(move)

        # Cutting down a piece
        for next_column in [column - 1, column + 1]:
            if (row - 1 in array) and (next_column in array) and (state[row - 1, next_column] < 0):
                move = [[row, column], [row - 1, next_column]]
                if not self.__check_move_on_check(state, move):
                    valid_moves.append(move)

                # Promotion cutting down
                if row == 1:
                    for figure in [8, 9, 10, 11]:  # 8 - knight, 9 - bishop, 10 - rook, 11 - queen
                        move = [[1, column], [figure, column]]
                        if not self.__check_move_on_check(state, move):
                            valid_moves.append(move)

        # En passant
        if self.__move_last_pawn:
            row_before_last, column_before_last = self.__move_last_pawn[0][0], self.__move_last_pawn[0][1]
            row_after_last, column_after_last = self.__move_last_pawn[1][0], self.__move_last_pawn[1][1]
            if row_before_last != 6 and row_after_last != 4 and row != 3:
                return
            for new_column in [column - 1, column + 1]:
                if 7 - column_after_last == new_column:
                    move = [[row, column], [row - 1, new_column]]
                    if not self.__check_move_on_check(state, move):
                        valid_moves.append(move)

    def __fill_valid_knight_and_king_moves(self, state, row, column, array, valid_moves):
        piece = state[row, column]
        next_row_and_column = self.__get_next_row_and_column(piece, row, column)
        for i in range(len(next_row_and_column)):
            # Checks the validity of a next row, column and cell
            if next_row_and_column[i][0] in array and next_row_and_column[i][1] in array and \
                    state[next_row_and_column[i][0], next_row_and_column[i][1]] < 1:
                move = [[row, column], [next_row_and_column[i][0], next_row_and_column[i][1]]]
                if not self.__check_move_on_check(state, move):
                    valid_moves.append(move)

        # Adding valid castling
        if piece != 10:
            return
        # White
        if row == 7 and column == 4:
            pseudo_player = 1
            if not self.__the_kings_was_moved[pseudo_player == -1]:
                # Big castling
                if not self.__the_rooks1_was_moved[pseudo_player == -1] and state[7, 1] == 0 and state[7, 2] == 0 and \
                        state[7, 3] == 0:
                    move = [[7, 4], [7, 2]]
                    if not self.__check_move_on_check(state, move):
                        valid_moves.append(move)

                # Small castling
                elif not self.__the_rooks2_was_moved[pseudo_player == -1] and state[7, 5] == 0 and state[7, 6] == 0:
                    move = [[7, 4], [7, 6]]
                    if not self.__check_move_on_check(state, move):
                        valid_moves.append(move)

        # Black
        elif row == 7 and column == 3:
            pseudo_player = -1
            if not self.__the_kings_was_moved[pseudo_player == -1]:
                # Small castling
                if not self.__the_rooks1_was_moved[pseudo_player == -1] and state[7, 1] == 0 and state[7, 2] == 0:
                    move = [[7, 3], [7, 1]]
                    if not self.__check_move_on_check(state, move):
                        valid_moves.append(move)

                # Big castling
                elif not self.__the_rooks2_was_moved[pseudo_player == -1] and state[7, 5] == 0 and state[7, 6] == 0 and\
                        state[7, 4] == 0:
                    move = [[7, 3], [7, 5]]
                    if not self.__check_move_on_check(state, move):
                        valid_moves.append(move)

    def __fill_valid_bishop_rook_queen_moves(self, state, row, column, array, valid_moves):
        piece = state[row, column]
        next_row_and_column = self.__get_next_row_and_column(piece, row, column)
        for i in range(len(next_row_and_column)):
            for j in range(len(next_row_and_column[i])):
                # Checks the validity of a next row and column
                if next_row_and_column[i][j][0] in array and next_row_and_column[i][j][1] in array:
                    # Checks the validity of a next cell
                    if state[next_row_and_column[i][j][0], next_row_and_column[i][j][1]] < 1:
                        move = [[row, column], [next_row_and_column[i][j][0], next_row_and_column[i][j][1]]]
                        if not self.__check_move_on_check(state, move):
                            valid_moves.append(move)

                    # Removing the ability to jump over pieces
                    if state[next_row_and_column[i][j][0], next_row_and_column[i][j][1]] != 0:
                        break
                # Skipping invalid row or column
                else:
                    break

    def get_valid_moves(self, state):
        valid_moves = []

        array = [i for i in range(self.row_count)]  # Shows valid row or column
        for row in range(self.row_count):
            for column in range(self.column_count):
                piece = state[row][column]
                if piece == 1:
                    self.__fill_valid_pawns_moves(state, row, column, array, valid_moves)

                elif piece == 2 or piece == 10:
                    self.__fill_valid_knight_and_king_moves(state, row, column, array, valid_moves)

                elif piece > 0:
                    self.__fill_valid_bishop_rook_queen_moves(state, row, column, array, valid_moves)

        return_arr = []
        for valid_move in valid_moves:
            return_arr.append([np.array(item, dtype=np.int8) for item in valid_move])
        return np.array(return_arr, dtype='object')

    def __check_draw(self, state, valid_moves_size, the_king_was_checked):
        # Stalemate
        if the_king_was_checked:
            return False
        if valid_moves_size == 0:
            return True
        # Repeats
        if self.__is_draw_due_to_repetitions:
            return True
        # The 50-move rule
        if self.__is_draw_due_to_50_move_rule:
            return True
        # Not enough pieces
        value_white, value_black = 0, 0
        for row in range(self.row_count):
            for column in range(self.column_count):
                piece = state[row, column]
                if piece > 0:
                    if piece == 1:
                        value_white += 100
                    else:
                        value_white += piece
                elif piece < 0:
                    if piece == -1:
                        value_black += 100
                    else:
                        value_black += -piece
        if value_white < 15 and value_black < 15:
            return True
        return False

    def get_value_and_terminated(self, state, action):
        if action is None:
            return 0, False
        row_after, column_after = action[1][0], action[1][1]
        player = 1 if state[row_after, column_after] > 0 else -1
        opponent_perspective = self.change_perspective(state, self.get_opponent(player))
        valid_moves_size = self.get_valid_moves(opponent_perspective).size
        the_king_was_checked = self.__the_king_was_checked(opponent_perspective)

        if the_king_was_checked and valid_moves_size == 0:
            return 1, True
        if self.__check_draw(state, valid_moves_size, the_king_was_checked):
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        rotated_state = state[::-1, ::-1]
        states = [state, rotated_state]
        return states[player == -1] * player

    def get_encoded_state(self, state):
        encoded_state = np.zeros((self.row_count, self.column_count, 13))
        pieces = [-5, -2, -3, -9, -10, -1, 0, 1, 2, 3, 5, 9, 10]
        for row in range(self.row_count):
            for column in range(self.column_count):
                piece = state[row, column]
                index = pieces.index(piece)
                encoded_state[row, column, index] = 1

        encoded_state = np.transpose(encoded_state, (2, 0, 1))
        return encoded_state.astype(np.float32)
