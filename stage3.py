import random

ALPHA = 1  # Learning rate
GAMMA = .9  # Discount factor
EPSILON = .1  # How often we choose random action


# Global for persistance between episodes
Q = {}


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


def cli_player(symbol, training):
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


def find_legal_moves(board):
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell is None:
                yield i, j


def random_max(values, key):
    max_key = max(map(key, values))  # Might throw ValueError on empty
    return random.choice(tuple(filter(lambda v: key(v) == max_key, values)))


def is_epsilon_step():
    return random.random() < EPSILON


def get_board_order(board):
    return sum(1 if cell == 'X' else (2 if cell == 'O' else 0)
               for row in board for cell in row)


def compute_symmetry(board, move, symmetry):
    assert 0 <= symmetry < 8
    i, j = move
    if symmetry & 1:
        # Swap row order.
        board = board[::-1]
        i = 2 - i
    if symmetry & 2:
        # Swap column order.
        board = tuple(row[::-1] for row in board)
        j = 2 - j
    if symmetry & 4:
        # Transpose
        board = tuple(zip(*board))
        i, j = j, i
    return board, (i, j)


def normalise(board, move):
    return min((compute_symmetry(board, move, symmetry)
                for symmetry in range(8)),
               key=lambda bm: get_board_order(bm[0]))


def update(board, move, symbol):
    succ_board = place_move(board, move, symbol)
    try:
        best_Q = -max(Q.get(normalise(succ_board, action), 0)
                      for action in find_legal_moves(succ_board))
    except ValueError:
        # No legal moves.
        best_Q = 0
    Q_entry = normalise(board, move)
    Q[Q_entry] = (1 - ALPHA) * Q.get(Q_entry, 0) + ALPHA * GAMMA * best_Q


def rl_player(symbol, training):
    """The simple player.

    Chooses the first legal move.
    """
    move = None  # For priming.
    last_board = None
    while True:
        try:
            board = yield move
        except IllegalMove as e:
            raise RuntimeError('illegal move by AI player') from e
        except GameFinished as e:
            assert last_board is not None
            assert move is not None
            if e.winning_symbol == symbol:
                # We win.
                # Terminal state so no need to worry about learning
                # rates, etc...
                Q_entry = normalise(last_board, move)
                Q[Q_entry] = (1 - ALPHA) * Q.get(Q_entry, 0) + ALPHA
            else:
                update(last_board, move, symbol)
            return

        # Update step.
        if last_board is not None:
            assert move is not None
            update(last_board, move, symbol)

        legal_moves = tuple(find_legal_moves(board))
        if training and is_epsilon_step():
            try:
                move = random.choice(legal_moves)
            except IndexError as e:
                raise RuntimeError(
                    'AI player called with no legal moves') from e
        else:
            try:
                move = random_max(legal_moves,
                                  key=lambda m: Q.get(normalise(board, m), 0))
            except ValueError as e:
                raise RuntimeError(
                    'AI player called with no legal moves') from e
            
        last_board = board


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
    return tuple(tuple(symbol if j == y else cell
                       for j, cell in enumerate(row)) if i == x else row
                 for i, row in enumerate(board))


def game(agent1f, agent2f, training=False):
    """Play the game.

    `agent1f` is 'X' and `agent2f` is 'O'. `agent1f` hence begins the
    game.
    """
    board = ((None, None, None),
             (None, None, None),
             (None, None, None))
    
    agent1 = agent1f('X', training)
    next(agent1)  # Prime.
    agent2 = agent2f('O', training)
    next(agent2)  # Prime.
    
    turn = False
    while not is_finished(board):
        agent_turn = [agent1, agent2][turn]
        symbol_turn = ['X', 'O'][turn]
        
        move = agent_turn.send(board)
        while not is_legal(move, board):
            move = agent_turn.throw(IllegalMove)

        board = place_move(board, move, symbol_turn)

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
        while True:
            human_symbol = input('Choose your symbol (X/O): ')
            if human_symbol in {'X', 'O'}:
                break
            else:
                print('Invalid symbol.')
        if human_symbol == 'X':
            agent1f = cli_player
            agent2f = rl_player
        else:
            agent1f = rl_player
            agent2f = cli_player
        game(agent1f, agent2f)


def pretrain(episodes):
    for _ in range(episodes):
        game(rl_player, rl_player, training=True)


if __name__ == '__main__':
    pretrain(10000)
    main()
