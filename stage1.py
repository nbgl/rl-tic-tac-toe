class GameFinished(Exception):
    """Thrown into the corourine when the game is over."""
    def __init__(self, winning_symbol):
        """winning_symbol = None for a draw"""
        self.winning_symbol = winning_symbol


class IllegalMove(Exception):
    """Thrown into the coroutine when the move is invalid."""


def board_to_strs(board):
    return [[' ' if cell is None else cell for cell in row] for row in board]


def print_board(board):
    board_strs = board_to_strs(board)
    print('  1   2   3')
    print(f'A {board_strs[0][0]} | {board_strs[0][1]} | {board_strs[0][2]}')
    print('    |   |  ')
    print('  -- --- --')
    print('    |   |  ')
    print(f'B {board_strs[1][0]} | {board_strs[1][1]} | {board_strs[1][2]}')
    print('    |   |  ')
    print(f'  -- --- --')
    print('    |   |  ')
    print(f'C {board_strs[2][0]} | {board_strs[2][1]} | {board_strs[2][2]}')


def cli_player(symbol):
    """The CLI player.

    It takes care of asking the user for moves, responding to invalid
    moves and printing game result messages.

    This neat solution is implemented using coroutines. To be fair, most
    students do not know coroutines. Similar functionality can be
    implemented with OOP or callbacks.
    """
    move = None  # For priming.
    while True:
        try:
            board = yield move
        except IllegalMove as e:
            print('Illegal move.')
        except GameFinished as e:
            if e.winning_symbol == symbol:
                # We win.
                print(f'{symbol} wins!')
            elif e.winning_symbol is None:
                print('Draw.')
            else:
                print(f'{e.winning_symbol} wins this one. Try again!')
            return
        else:
            print(f'You are {symbol}.')
            print('Current board:')
            print_board(board)
        while True:
            raw_move = input('Choose a move (e.g. B2): ')
            try:
                row, col = raw_move
                row_index = {'A': 0, 'B': 1, 'C': 2}[row]
                col_index = {'1': 0, '2': 1, '3': 2}[col]
            except (ValueError, KeyError):
                print('Invalid format.')
            else:
                # `move` gets yielded.
                move = row_index, col_index
                break


def first_legal_move(board):
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell is None:
                return i, j
    raise ValueError('no legal moves')


def simple_ai_player(symbol):
    """The simple player.

    Chooses the first legal move.
    """
    move = None  # For priming.
    while True:
        try:
            board = yield move
        except IllegalMove as e:
            raise RuntimeError('illegal move by AI player') from e
        except GameFinished as e:
            return
        try:
            move = first_legal_move(board)
        except ValueError as e:
            raise RuntimeError('AI player called with no legal moves') from e


def is_legal(move, board):
    row_i, col_i = move
    return board[row_i][col_i] is None


def find_winner_line(line):
    """Return the winner of the line if one exists.

    Otherwise return None.

    A line is a row, a column or a diagonal.
    """
    if len(line) != 3:
        raise ValueError('invalid line')
    symbols = set(line)
    if len(symbols) == 1:
        # All equal.
        return line[0]  # Could be None.
    return None


def find_winner(board):
    """Return the winner if one exists.

    Otherwise return None.
    """
    # We could also use itertools.chain for this.
    for row in board:
        winner = find_winner_line(row)
        if winner is not None:
            return winner
    for column in zip(*board):
        winner = find_winner_line(column)
        if winner is not None:
            return winner
    for diagonal in [(board[0][0], board[1][1], board[2][2]),
                     (board[0][2], board[1][1], board[2][0])]:
        winner = find_winner_line(diagonal)
        if winner is not None:
            return winner
    return None


def is_finished(board):
    # Game is finished when the board is full.
    return all(all(cell is not None for cell in row) for row in board)


def inform_game_result(agents, result):
    for agent in agents:
        try:
            agent.throw(GameFinished(result))
        except StopIteration:
            pass
        else:
            raise RuntimeError('StopIteration expected')


def place_move(board, move, symbol):
    """Place symbol in-place."""
    x, y = move
    board[x][y] = symbol


def game(agent1f, agent2f):
    """Play the game.

    `agent1f` is 'X' and `agent2f` is 'O'. `agent1f` hence begins the
    game.
    """
    board = [[None, None, None],
             [None, None, None],
             [None, None, None]]
    
    agent1 = agent1f('X')
    next(agent1)  # Prime.
    agent2 = agent2f('O')
    next(agent2)  # Prime.
    
    turn = False
    while not is_finished(board):
        agent_turn = [agent1, agent2][turn]
        symbol_turn = ['X', 'O'][turn]
        
        move = agent_turn.send(board)
        while not is_legal(move, board):
            move = agent_turn.throw(IllegalMove)

        place_move(board, move, symbol_turn)

        winner = find_winner(board)
        if winner is not None:
            inform_game_result([agent1, agent2], winner)
            return

        turn = not turn
    else:
        # No more legal moves but nobody won.
        inform_game_result([agent1, agent2], None)


def main():
    while True:
        human_symbol = input('Choose your symbol (X/O): ')
        if human_symbol in {'X', 'O'}:
            break
        else:
            print('Invalid symbol.')
    if human_symbol == 'X':
        agent1f = cli_player
        agent2f = simple_ai_player
    else:
        agent1f = simple_ai_player
        agent2f = cli_player
    game(agent1f, agent2f)


if __name__ == '__main__':
    main()
