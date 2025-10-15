import numpy as np
import subprocess

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x,y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
                            ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y if x != -1 else self.game.n ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a


class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]

class GTPOthelloPlayer():
    """
    Player that plays with Othello programs using the Go Text Protocol.
    """

    # The colours are reversed as the Othello programs seems to have the board setup with the opposite colours
    player_colors = {
        -1: "white",
         1: "black",
    }

    def __init__(self, game, gtpClient):
        """
        Input:
            game: the game instance
            gtpClient: list with the command line arguments to start the GTP client with.
                       The first argument should be the absolute path to the executable.
        """
        self.game = game
        self.gtpClient = gtpClient

    def startGame(self):
        """
        Should be called before the game starts in order to setup the board.
        """
        self._currentPlayer = 1 # Arena does not notify players about their colour so we need to keep track here
        self._process = subprocess.Popen(self.gtpClient, bufsize = 0, stdin = subprocess.PIPE, stdout = subprocess.PIPE)
        self._sendCommand("boardsize " + str(self.game.n))
        self._sendCommand("clear_board")

    def endGame(self):
        """
        Should be called after the game ends in order to clean-up the used resources.
        """
        if hasattr(self, "_process") and self._process is not None:
            self._sendCommand("quit")
            # Waits for the client to terminate gracefully for 10 seconds. If it does not - kills it.
            try:
                self._process.wait(10)
            except (subprocessTimeoutExpired):
                self._process.kill()
            self._process = None

    def notify(self, board, action):
        """
        Should be called after the opponent turn. This way we can update the GTP client with the opponent move.
        """
        color = GTPOthelloPlayer.player_colors[self._currentPlayer]
        move = self._convertActionToMove(action)
        self._sendCommand("play {} {}".format(color, move))
        self._switchPlayers()

    def play(self, board):
        color = GTPOthelloPlayer.player_colors[self._currentPlayer]
        move = self._sendCommand("genmove {}".format(color))
        action = self._convertMoveToAction(move)
        self._switchPlayers()
        return action

    def _switchPlayers(self):
        self._currentPlayer = -self._currentPlayer

    def _convertActionToMove(self, action):
        if action < self.game.n ** 2:
            row, col = int(action / self.game.n), int(action % self.game.n)
            return "{}{}".format(chr(ord("A") + col), row + 1)
        else:
            return "PASS"

    def _convertMoveToAction(self, move):
        if move != "PASS":
            col, row = ord(move[0]) - ord('A'), int(move[1:])
            return (row - 1) * self.game.n + col
        else:
            return self.game.n ** 2

    def _sendCommand(self, cmd):
        self._process.stdin.write(cmd.encode() + b"\n")

        response = ""
        while True:
            line = self._process.stdout.readline().decode()
            if line == "\n":
                if response:
                    break  # Empty line means end of the response is reached
                else:
                    continue  # Ignoring leading empty lines
            response += line

        # If the first character of the response is '=', then is success. '?' is error.
        if response.startswith("="):
            # Some clients return uppercase other lower case.
            # Normalizing to uppercase in order to simplify handling.
            return response[1:].strip().upper()
        else:
            raise Exception("Error calling GTP client: {}".format(response[1:].strip()))

    def __call__(self, game):
        return self.play(game)

import numpy as np
import time

class AlphaBetaOthelloPlayer:
    """
    Negamax + Alpha-Beta for Othello under AlphaZeroGeneral's Game API.
    Assumes the input board is canonical (player=+1 perspective), as Arena does.
    """
    def __init__(self, game, depth=3, time_limit=None):
        self.game = game
        self.depth = depth
        self.time_limit = time_limit  # seconds; if None, pure depth search
        self.n = getattr(game, 'n', game.getBoardSize()[0])

        self.W = np.array([
            [100, -20,  10,  5,  5, 10, -20, 100],
            [-20, -50,  -2, -2, -2, -2, -50, -20],
            [ 10,  -2,   2,  1,  1,  2,  -2,  10],
            [  5,  -2,   1,  0,  0,  1,  -2,   5],
            [  5,  -2,   1,  0,  0,  1,  -2,   5],
            [ 10,  -2,   2,  1,  1,  2,  -2,  10],
            [-20, -50,  -2, -2, -2, -2, -50, -20],
            [100, -20,  10,  5,  5, 10, -20, 100],
        ], dtype=np.float32)

    def play(self, board):
        start = time.time()
        best_act = self._best_legal(board)
        best_val = -float('inf')
        max_depth = 20 if self.time_limit else self.depth

        d = 1
        while d <= max_depth:
            val, act = self._negamax(board, d, -float('inf'), float('inf'), start)
            if act is not None:
                best_act, best_val = act, val
            d += 1
            if self.time_limit and (time.time() - start) >= self.time_limit:
                break
        return best_act

    # ---------- core search ----------

    def _negamax(self, board, depth, alpha, beta, start):
        if self.time_limit and (time.time() - start) >= self.time_limit:
            return self._eval(board), None

        ge = self.game.getGameEnded(board, 1)
        if ge != 0:
            if abs(ge) < 1e-6:
                return 0.0, None
            return (1.0 if ge > 0 else -1.0), None

        if depth == 0:
            return self._eval(board), None

        valids = self.game.getValidMoves(board, 1)
        actions = np.where(valids == 1)[0]
        if actions.size == 0:
            return self._eval(board), None

        scores = []
        for a in actions:
            flip_hint = self._flip_hint(board, a)
            corner_bonus = self._corner_bonus(a)
            scores.append((corner_bonus * 1000 + flip_hint, a))
        actions = [a for _, a in sorted(scores, reverse=True)]

        best_val, best_act = -float('inf'), actions[0]
        for a in actions:
            nb, np_player = self.game.getNextState(board, 1, a)
            nb_canon = self.game.getCanonicalForm(nb, np_player)
            v, _ = self._negamax(nb_canon, depth - 1, -beta, -alpha, start)
            v = -v
            if v > best_val:
                best_val, best_act = v, a
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best_val, best_act

    # ---------- heuristics ----------

    def _eval(self, board):
        b = np.array(board)
        n = self.n
        material = b[:n, :n].sum() / (n * n)

        my_moves = np.sum(self.game.getValidMoves(board, 1))
        opp_board = self.game.getCanonicalForm(board, -1)
        opp_moves = np.sum(self.game.getValidMoves(opp_board, 1))
        mobility = 0 if (my_moves + opp_moves) == 0 else (my_moves - opp_moves) / (my_moves + opp_moves)

        W = self.W[:n, :n]
        positional = float((W * b[:n, :n]).sum()) / (np.abs(W).sum())

        return 0.6 * positional + 0.3 * mobility + 0.1 * material

    def _best_legal(self, board):
        valids = self.game.getValidMoves(board, 1)
        actions = np.where(valids == 1)[0]
        if actions.size == 0:
            return 0
        actions_sorted = sorted(actions.tolist(), key=lambda a: self._corner_bonus(a), reverse=True)
        return actions_sorted[0]

    def _corner_bonus(self, a):
        n = self.n
        if a >= n * n:
            return 0
        r, c = divmod(a, n)
        return 1 if (r, c) in [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)] else 0

    def _flip_hint(self, board, a):
        if a >= self.n * self.n:
            return 0
        nb, np_player = self.game.getNextState(board, 1, a)
        b = np.array(self.game.getCanonicalForm(nb, np_player))
        return int((b[:self.n, :self.n] > 0).sum())
