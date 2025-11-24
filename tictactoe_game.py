"""
Tic Tac Toe Game with Multiple AI Agents
Interactive game and tournament system
"""

import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict
import random
import pickle
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns


# ==================== GAME LOGIC ====================

class Player(Enum):
    """Player identifiers."""
    EMPTY = 0
    X = 1
    O = 2
    
    def __str__(self):
        return {Player.EMPTY: ' ', Player.X: 'X', Player.O: 'O'}[self]
    
    def opponent(self):
        """Get opponent player."""
        if self == Player.X:
            return Player.O
        elif self == Player.O:
            return Player.X
        return Player.EMPTY


class GameResult(Enum):
    """Game result types."""
    X_WIN = 1
    O_WIN = 2
    DRAW = 0
    IN_PROGRESS = -1


class TicTacToe:
    """Tic Tac Toe game environment."""
    
    def __init__(self):
        """Initialize empty board."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = Player.X
        self.move_history = []
        
    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = Player.X
        self.move_history = []
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current board state."""
        return self.board.copy()
    
    def get_state_key(self) -> str:
        """Get hashable state representation."""
        return ''.join(map(str, self.board.flatten()))
    
    def get_available_actions(self) -> List[Tuple[int, int]]:
        """Get list of available moves."""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]
    
    def is_valid_action(self, action: Tuple[int, int]) -> bool:
        """Check if action is valid."""
        i, j = action
        return 0 <= i < 3 and 0 <= j < 3 and self.board[i, j] == 0
    
    def make_move(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, GameResult]:
        """
        Make a move on the board.
        
        Returns:
            state: New board state
            reward: Reward for the move
            done: Whether game is over
            result: Game result
        """
        if not self.is_valid_action(action):
            # Invalid move
            return self.get_state(), -10, True, GameResult.IN_PROGRESS
        
        i, j = action
        self.board[i, j] = self.current_player.value
        self.move_history.append((action, self.current_player))
        
        # Check game result
        result = self.check_winner()
        done = result != GameResult.IN_PROGRESS
        
        # Calculate reward from current player's perspective
        reward = self._calculate_reward(result)
        
        # Switch player
        if not done:
            self.current_player = self.current_player.opponent()
        
        return self.get_state(), reward, done, result
    
    def _calculate_reward(self, result: GameResult) -> float:
        """Calculate reward based on game result."""
        if result == GameResult.IN_PROGRESS:
            return 0
        elif result == GameResult.DRAW:
            return 0.5
        elif (result == GameResult.X_WIN and self.current_player == Player.X) or \
             (result == GameResult.O_WIN and self.current_player == Player.O):
            return 1.0
        else:
            return 0.0
    
    def check_winner(self) -> GameResult:
        """Check if there's a winner."""
        board = self.board
        
        # Check rows
        for i in range(3):
            if board[i, 0] != 0 and board[i, 0] == board[i, 1] == board[i, 2]:
                return GameResult.X_WIN if board[i, 0] == 1 else GameResult.O_WIN
        
        # Check columns
        for j in range(3):
            if board[0, j] != 0 and board[0, j] == board[1, j] == board[2, j]:
                return GameResult.X_WIN if board[0, j] == 1 else GameResult.O_WIN
        
        # Check diagonals
        if board[0, 0] != 0 and board[0, 0] == board[1, 1] == board[2, 2]:
            return GameResult.X_WIN if board[0, 0] == 1 else GameResult.O_WIN
        if board[0, 2] != 0 and board[0, 2] == board[1, 1] == board[2, 0]:
            return GameResult.X_WIN if board[0, 2] == 1 else GameResult.O_WIN
        
        # Check draw
        if len(self.get_available_actions()) == 0:
            return GameResult.DRAW
        
        return GameResult.IN_PROGRESS
    
    def get_winner_line(self) -> Optional[List[Tuple[int, int]]]:
        """Get winning line positions if game is won."""
        board = self.board
        
        # Check rows
        for i in range(3):
            if board[i, 0] != 0 and board[i, 0] == board[i, 1] == board[i, 2]:
                return [(i, 0), (i, 1), (i, 2)]
        
        # Check columns
        for j in range(3):
            if board[0, j] != 0 and board[0, j] == board[1, j] == board[2, j]:
                return [(0, j), (1, j), (2, j)]
        
        # Check diagonals
        if board[0, 0] != 0 and board[0, 0] == board[1, 1] == board[2, 2]:
            return [(0, 0), (1, 1), (2, 2)]
        if board[0, 2] != 0 and board[0, 2] == board[1, 1] == board[2, 0]:
            return [(0, 2), (1, 1), (2, 0)]
        
        return None
    
    def render(self):
        """Print board to console."""
        print("\n  0   1   2")
        for i in range(3):
            row_str = f"{i} "
            for j in range(3):
                cell = ' '
                if self.board[i, j] == 1:
                    cell = 'X'
                elif self.board[i, j] == 2:
                    cell = 'O'
                row_str += f" {cell} "
                if j < 2:
                    row_str += "|"
            print(row_str)
            if i < 2:
                print("  -----------")
        print()
    
    def clone(self):
        """Create a deep copy of the game."""
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        return new_game


# ==================== BASE AGENT CLASS ====================

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str, player: Player = Player.X):
        """
        Initialize agent.
        
        Args:
            name: Agent name
            player: Player symbol (X or O)
        """
        self.name = name
        self.player = player
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    @abstractmethod
    def choose_action(self, game: TicTacToe) -> Tuple[int, int]:
        """
        Choose an action given the current game state.
        
        Args:
            game: Current game state
            
        Returns:
            action: Tuple (row, col) representing the chosen move
        """
        pass
    
    def reset_stats(self):
        """Reset agent statistics."""
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    def update_stats(self, result: str):
        """
        Update agent statistics.
        
        Args:
            result: 'win', 'loss', or 'draw'
        """
        self.games_played += 1
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        elif result == 'draw':
            self.draws += 1
    
    def get_win_rate(self) -> float:
        """Get win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            'name': self.name,
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': self.get_win_rate()
        }
    
    def __str__(self):
        return f"{self.name} ({self.player})"


# ==================== AI AGENTS ====================

class RandomAgent(BaseAgent):
    """Agent that chooses random valid moves."""
    
    def __init__(self, player: Player = Player.X):
        super().__init__("Random", player)
    
    def choose_action(self, game: TicTacToe) -> Tuple[int, int]:
        """Choose a random valid action."""
        available = game.get_available_actions()
        return random.choice(available)


class ExhaustiveSearchAgent(BaseAgent):
    """Agent that evaluates all possible moves exhaustively."""
    
    def __init__(self, player: Player = Player.X):
        super().__init__("Exhaustive Search", player)
        # Transposition table keyed by state_key -> score
        self._tt: Dict[str, float] = {}
    
    def choose_action(self, game: TicTacToe) -> Tuple[int, int]:
        """Choose best action by exhaustive search."""
        # Reset cache per root call to avoid unbounded growth
        self._tt.clear()
        available = game.get_available_actions()
        best_action = None
        best_score = -float('inf')
        
        for action in available:
            # Simulate the move
            sim_game = game.clone()
            sim_game.make_move(action)
            score = self._evaluate_position(sim_game, depth=0)
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action else available[0]
    
    def _evaluate_position(self, game: TicTacToe, depth: int) -> float:
        """Evaluate position recursively."""
        # Memoization by state
        key = game.get_state_key()
        cached = self._tt.get(key)
        if cached is not None:
            return cached
        result = game.check_winner()
        
        # Terminal state
        if result == GameResult.X_WIN:
            val = (10 - depth) if self.player == Player.X else -(10 - depth)
            self._tt[key] = val
            return val
        elif result == GameResult.O_WIN:
            val = (10 - depth) if self.player == Player.O else -(10 - depth)
            self._tt[key] = val
            return val
        elif result == GameResult.DRAW:
            self._tt[key] = 0
            return 0
        
        # Recursively evaluate all moves
        available = game.get_available_actions()
        if game.current_player == self.player:
            # Maximizing player
            max_score = -float('inf')
            for action in available:
                sim_game = game.clone()
                sim_game.make_move(action)
                score = self._evaluate_position(sim_game, depth + 1)
                max_score = max(max_score, score)
            self._tt[key] = max_score
            return max_score
        else:
            # Minimizing player
            min_score = float('inf')
            for action in available:
                sim_game = game.clone()
                sim_game.make_move(action)
                score = self._evaluate_position(sim_game, depth + 1)
                min_score = min(min_score, score)
            self._tt[key] = min_score
            return min_score


class MinMaxAgent(BaseAgent):
    """Minimax agent with alpha-beta pruning."""
    
    def __init__(self, player: Player = Player.X):
        super().__init__("MinMax Alpha-Beta", player)
        # Transposition table: (state_key, maximizing) -> score
        self._tt: Dict[Tuple[str, bool], float] = {}
    
    def choose_action(self, game: TicTacToe) -> Tuple[int, int]:
        """Choose best action using minimax with alpha-beta pruning."""
        # Reset cache for this search
        self._tt.clear()
        available = game.get_available_actions()
        best_action = None
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        for action in available:
            sim_game = game.clone()
            sim_game.make_move(action)
            score = self._minimax(sim_game, depth=0, alpha=alpha, beta=beta, 
                                 maximizing=False)
            
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, score)
        
        return best_action if best_action else available[0]
    
    def _minimax(self, game: TicTacToe, depth: int, alpha: float, beta: float, 
                 maximizing: bool) -> float:
        """Minimax algorithm with alpha-beta pruning."""
        # Memoization
        key = (game.get_state_key(), maximizing)
        cached = self._tt.get(key)
        if cached is not None:
            return cached
        result = game.check_winner()
        
        # Terminal state
        if result == GameResult.X_WIN:
            val = (10 - depth) if self.player == Player.X else -(10 - depth)
            self._tt[key] = val
            return val
        elif result == GameResult.O_WIN:
            val = (10 - depth) if self.player == Player.O else -(10 - depth)
            self._tt[key] = val
            return val
        elif result == GameResult.DRAW:
            self._tt[key] = 0
            return 0
        
        available = game.get_available_actions()
        
        if maximizing:
            max_score = -float('inf')
            for action in available:
                sim_game = game.clone()
                sim_game.make_move(action)
                score = self._minimax(sim_game, depth + 1, alpha, beta, False)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # Beta cutoff
            self._tt[key] = max_score
            return max_score
        else:
            min_score = float('inf')
            for action in available:
                sim_game = game.clone()
                sim_game.make_move(action)
                score = self._minimax(sim_game, depth + 1, alpha, beta, True)
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break  # Alpha cutoff
            self._tt[key] = min_score
            return min_score


class MonteCarloAgent(BaseAgent):
    """Monte Carlo RL control (every-visit) with epsilon-greedy policy."""

    def __init__(self, player: Player = Player.X, epsilon: float = 1.0, alpha: float = 0.1, gamma: float = 1.0):
        super().__init__("Monte Carlo", player)
        self.q_table: Dict[Tuple[str, Tuple[int, int]], float] = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.is_training = False

    def _q(self, state_key: str, action: Tuple[int, int]) -> float:
        return self.q_table.get((state_key, action), 0.0)

    def choose_action(self, game: TicTacToe) -> Tuple[int, int]:
        available = game.get_available_actions()
        if not available:
            return (0, 0)
        if self.is_training and random.random() < self.epsilon:
            return random.choice(available)
        state_key = game.get_state_key()
        # Greedy
        best_a = max(available, key=lambda a: self._q(state_key, a))
        return best_a

    def train(self, episodes: int = 5000, epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: float = 0.995):
        self.is_training = True
        self.epsilon = epsilon_start
        opponent = RandomAgent(self.player.opponent())
        for ep in range(episodes):
            game = TicTacToe()
            # If we are O, opponent moves first
            if self.player == Player.O:
                a0 = opponent.choose_action(game)
                game.make_move(a0)
            trajectory: List[Tuple[str, Tuple[int, int]]] = []
            steps = 0
            while game.check_winner() == GameResult.IN_PROGRESS:
                s = game.get_state_key()
                a = self.choose_action(game)
                game.make_move(a)
                trajectory.append((s, a))
                steps += 1
                if game.check_winner() != GameResult.IN_PROGRESS:
                    break
                # Opponent move
                oa = opponent.choose_action(game)
                game.make_move(oa)
            # Episode return from outcome
            res = game.check_winner()
            if (res == GameResult.X_WIN and self.player == Player.X) or (res == GameResult.O_WIN and self.player == Player.O):
                G = 1.0
            elif res == GameResult.DRAW:
                G = 0.5
            else:
                G = 0.0
            # Every-visit MC update (use same terminal return for all own steps)
            for t, (s, a) in enumerate(trajectory):
                # Optionally discount by remaining steps; keep gamma=1.0 by default
                G_t = (self.gamma ** 0) * G
                q = self._q(s, a)
                self.q_table[(s, a)] = q + self.alpha * (G_t - q)
            # epsilon schedule
            self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)
        self.is_training = False

    def train_with_history(self, episodes: int = 2000, epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                            opp_agent: Optional[BaseAgent] = None, stop_event: Optional['threading.Event'] = None) -> Dict[str, List[float]]:
        history = {"rewards": [], "steps": [], "epsilons": [], "q_sizes": []}
        self.is_training = True
        self.epsilon = epsilon_start
        opponent = opp_agent if opp_agent is not None else RandomAgent(self.player.opponent())
        for ep in range(episodes):
            if stop_event is not None and stop_event.is_set():
                break
            game = TicTacToe()
            if self.player == Player.O:
                a0 = opponent.choose_action(game)
                game.make_move(a0)
            trajectory: List[Tuple[str, Tuple[int, int]]] = []
            steps = 0
            while game.check_winner() == GameResult.IN_PROGRESS:
                s = game.get_state_key()
                a = self.choose_action(game)
                game.make_move(a)
                trajectory.append((s, a))
                steps += 1
                if game.check_winner() != GameResult.IN_PROGRESS:
                    break
                oa = opponent.choose_action(game)
                game.make_move(oa)
            res = game.check_winner()
            if (res == GameResult.X_WIN and self.player == Player.X) or (res == GameResult.O_WIN and self.player == Player.O):
                G = 1.0
            elif res == GameResult.DRAW:
                G = 0.5
            else:
                G = 0.0
            for (s, a) in trajectory:
                q = self.q_table.get((s, a), 0.0)
                self.q_table[(s, a)] = q + self.alpha * (G - q)
            self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)
            history["rewards"].append(G)
            history["steps"].append(steps)
            history["epsilons"].append(self.epsilon)
            history["q_sizes"].append(len(self.q_table))
        self.is_training = False
        return history


class DynamicProgrammingAgent(BaseAgent):
    """Dynamic Programming planner with cached transitions and value iteration.
    Assumes opponent plays uniformly at random. Produces a standard q_table for visuals.
    """

    def __init__(self, player: Player = Player.X, gamma: float = 0.95, tol: float = 1e-4):
        super().__init__("Dynamic Programming", player)
        self.gamma = gamma
        self.tol = tol
        # Exposed Q for UI visuals: keys are (state_key:str, action:(r,c))
        self.q_table: Dict[Tuple[str, Tuple[int, int]], float] = {}

        # Internal caches over tuple states (length 9, values 0/1/2)
        self._built = False
        self._our_states: List[Tuple[int, ...]] = []
        self._actions: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
        # (s,a) -> list of (terminal:bool, reward:float, next_state:Optional[Tuple[int,...]])
        self._transitions: Dict[Tuple[Tuple[int, ...], Tuple[int, int]], List[Tuple[bool, float, Optional[Tuple[int, ...]]]]] = {}
        self._V: Dict[Tuple[int, ...], float] = {}

    # ---------- Fast board helpers (no TicTacToe objects) ----------
    @staticmethod
    def _winner(state: Tuple[int, ...]) -> Optional[int]:
        b = state
        lines = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
        for i,j,k in lines:
            if b[i] != 0 and b[i] == b[j] == b[k]:
                return b[i]
        return None

    @staticmethod
    def _draw(state: Tuple[int, ...]) -> bool:
        return 0 not in state and DynamicProgrammingAgent._winner(state) is None

    @staticmethod
    def _avail(state: Tuple[int, ...]) -> List[Tuple[int, int]]:
        return [(i//3, i%3) for i,v in enumerate(state) if v == 0]

    @staticmethod
    def _put(state: Tuple[int, ...], rc: Tuple[int, int], mark: int) -> Tuple[int, ...]:
        idx = rc[0]*3 + rc[1]
        lst = list(state); lst[idx] = mark
        return tuple(lst)

    @staticmethod
    def _turn(state: Tuple[int, ...]) -> int:
        x = 0; o = 0
        for v in state:
            if v == 1: x += 1
            elif v == 2: o += 1
        return 1 if x == o else 2

    def _reward_after_terminal(self, winner: Optional[int]) -> float:
        if winner is None:
            return 0.5
        return 1.0 if ((winner == 1 and self.player == Player.X) or (winner == 2 and self.player == Player.O)) else 0.0

    @staticmethod
    def _state_key(state: Tuple[int, ...]) -> str:
        return "".join(str(x) for x in state)

    def _build_graph(self):
        if self._built:
            return
        start: Tuple[int, ...] = (0,)*9
        our_mark = 1 if self.player == Player.X else 2
        opp_mark = 2 if our_mark == 1 else 1

        seen: set = set()

        def dfs(state: Tuple[int, ...]):
            if state in seen:
                return
            seen.add(state)
            w = self._winner(state)
            if w is not None or self._draw(state):
                return
            turn = self._turn(state)
            if turn == our_mark:
                # our decision state
                self._our_states.append(state)
                acts = self._avail(state)
                self._actions[state] = acts
                for a in acts:
                    s1 = self._put(state, a, our_mark)
                    w1 = self._winner(s1)
                    if w1 is not None or self._draw(s1):
                        r = self._reward_after_terminal(w1)
                        self._transitions[(state, a)] = [(True, r, None)]
                        continue
                    # opponent moves uniformly at random
                    outs: List[Tuple[bool, float, Optional[Tuple[int, ...]]]] = []
                    opp_acts = self._avail(s1)
                    for oa in opp_acts:
                        s2 = self._put(s1, oa, opp_mark)
                        w2 = self._winner(s2)
                        if w2 is not None or self._draw(s2):
                            outs.append((True, self._reward_after_terminal(w2), None))
                        else:
                            outs.append((False, 0.0, s2))
                            dfs(s2)
                    self._transitions[(state, a)] = outs
            else:
                # opponent turn: expand children
                acts = self._avail(state)
                mark = opp_mark if turn == opp_mark else our_mark
                for a in acts:
                    dfs(self._put(state, a, mark))

        dfs(start)

        # initialize V(s)
        for s in self._our_states:
            self._V[s] = self._V.get(s, 0.0)
        self._built = True

    def _greedy_eval_once(self, opponent: Optional['BaseAgent'] = None) -> Tuple[float, int]:
        """Play one evaluation game using our greedy Q policy vs provided opponent.
        Falls back to uniform-random opponent if none is provided.
        """
        our_player = self.player
        game = TicTacToe()
        # Ensure opponent player is opposite
        opp = opponent
        if opp is not None:
            opp.player = Player.O if our_player == Player.X else Player.X
        steps = 0
        import random as _rnd
        while game.check_winner() == GameResult.IN_PROGRESS:
            if game.current_player == our_player:
                # Greedy move by our Q
                key = game.get_state_key()
                avail = game.get_available_actions()
                if not avail:
                    break
                best = None; best_q = -1e9
                for a in avail:
                    q = self.q_table.get((key, a), 0.0)
                    if q > best_q:
                        best_q, best = q, a
                game.make_move(best if best is not None else _rnd.choice(avail))
            else:
                if opp is not None:
                    a = opp.choose_action(game)
                else:
                    a = _rnd.choice(game.get_available_actions())
                game.make_move(a)
            steps += 1
        res = game.check_winner()
        if res == GameResult.X_WIN:
            w = 1
        elif res == GameResult.O_WIN:
            w = 2
        elif res == GameResult.DRAW:
            w = None
        else:
            w = None
        return self._reward_after_terminal(w), steps

    def train_with_history(self, episodes: int = 200, opp_agent=None, stop_event=None,
                            tol: Optional[float] = None, record_every: int = 5, min_sweeps: int = 60) -> Dict[str, List[float]]:
        """Run value-iteration sweeps up to `episodes`.
        - Stops early at tolerance only after at least `min_sweeps` so plots have points.
        - Records every sweep during the first `min_sweeps`, then every `record_every` sweeps.
        """
        self._build_graph()
        tol = self.tol if tol is None else tol
        history = {"rewards": [], "steps": [], "epsilons": [], "q_sizes": []}

        for it in range(1, episodes + 1):
            if stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set():
                break
            delta = 0.0
            # Bellman optimality backup over our decision states
            V = self._V
            gamma = self.gamma
            actions = self._actions
            transitions = self._transitions
            for s in self._our_states:
                acts = actions[s]
                if not acts:
                    continue
                best = -1e9
                for a in acts:
                    outs = transitions[(s, a)]
                    if outs and outs[0][0]:
                        # all terminal
                        q = sum(o[1] for o in outs) / len(outs)
                    else:
                        q = 0.0
                        for term, r, nxt in outs:
                            q += (r if term else gamma * V.get(nxt, 0.0))
                        q /= max(1, len(outs))
                    if q > best:
                        best = q
                old = V.get(s, 0.0)
                V[s] = best if best != -1e9 else old
                delta = max(delta, abs(V[s] - old))

            # dense recording early on, then sparse
            if it <= min_sweeps or (record_every and it % record_every == 0) or it == episodes:
                # Rebuild q_table only when recording to avoid recomputation every sweep
                self.q_table.clear()
                Vrec = self._V
                gamma_rec = self.gamma
                for s in self._our_states:
                    key = self._state_key(s)
                    for a in self._actions[s]:
                        outs = self._transitions[(s, a)]
                        if outs and outs[0][0]:
                            q = sum(o[1] for o in outs) / len(outs)
                        else:
                            q = 0.0
                            for term, r, nxt in outs:
                                q += (r if term else gamma_rec * Vrec.get(nxt, 0.0))
                            q /= max(1, len(outs))
                        self.q_table[(key, a)] = q
                r, st = self._greedy_eval_once(opponent=opp_agent)
                history["rewards"].append(r)
                history["steps"].append(st)
                history["epsilons"].append(0.0)
                history["q_sizes"].append(len(self.q_table))
            else:
                # Keep history length equal to episodes: repeat last recorded values
                if history["rewards"]:
                    history["rewards"].append(history["rewards"][-1])
                    history["steps"].append(history["steps"][-1])
                    history["epsilons"].append(history["epsilons"][-1])
                    history["q_sizes"].append(history["q_sizes"][-1])
                else:
                    history["rewards"].append(0.0)
                    history["steps"].append(0)
                    history["epsilons"].append(0.0)
                    history["q_sizes"].append(0)


        return history

    def choose_action(self, game: TicTacToe) -> Tuple[int, int]:
        avail = game.get_available_actions()
        if not avail:
            return (0, 0)
        key = game.get_state_key()
        best = None; best_q = -1e9
        for a in avail:
            q = self.q_table.get((key, a), 0.0)
            if q > best_q:
                best_q, best = q, a
        return best if best is not None else avail[0]


class QLearningAgent(BaseAgent):
    """Q-Learning agent."""
    
    def __init__(self, player: Player = Player.X, epsilon: float = 0.1, 
                 alpha: float = 0.1, gamma: float = 0.95):
        super().__init__("Q-Learning", player)
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.is_training = False
    
    def choose_action(self, game: TicTacToe) -> Tuple[int, int]:
        """Choose action using epsilon-greedy policy."""
        available = game.get_available_actions()
        
        if self.is_training and random.random() < self.epsilon:
            return random.choice(available)
        
        # Greedy action
        state_key = game.get_state_key()
        best_action = None
        best_q = -float('inf')
        
        for action in available:
            q_value = self.q_table.get((state_key, action), 0.0)
            if q_value > best_q:
                best_q = q_value
                best_action = action
        
        return best_action if best_action else available[0]
    
    def update_q(self, state: str, action: Tuple[int, int], reward: float, 
                 next_state: str, next_actions: List[Tuple[int, int]], done: bool):
        """Update Q-value."""
        current_q = self.q_table.get((state, action), 0.0)
        
        if done:
            target = reward
        else:
            max_next_q = max([self.q_table.get((next_state, a), 0.0) 
                             for a in next_actions], default=0.0)
            target = reward + self.gamma * max_next_q
        
        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)
    
    def train(self, episodes: int = 10000):
        """Train the agent."""
        self.is_training = True
        opponent = RandomAgent(self.player.opponent())
        
        for episode in range(episodes):
            game = TicTacToe()
            if self.player == Player.O:
                # Opponent goes first
                action = opponent.choose_action(game)
                game.make_move(action)
            
            while True:
                state = game.get_state_key()
                available = game.get_available_actions()
                
                if len(available) == 0:
                    break
                
                # Agent's turn
                action = self.choose_action(game)
                _, reward, done, result = game.make_move(action)
                
                if done:
                    if (result == GameResult.X_WIN and self.player == Player.X) or \
                       (result == GameResult.O_WIN and self.player == Player.O):
                        reward = 1.0
                    elif result == GameResult.DRAW:
                        reward = 0.5
                    else:
                        reward = -1.0
                    
                    self.update_q(state, action, reward, game.get_state_key(), [], True)
                    break
                
                # Opponent's turn
                opp_action = opponent.choose_action(game)
                _, _, opp_done, opp_result = game.make_move(opp_action)
                
                next_state = game.get_state_key()
                next_actions = game.get_available_actions()
                
                if opp_done:
                    if (opp_result == GameResult.X_WIN and self.player == Player.O) or \
                       (opp_result == GameResult.O_WIN and self.player == Player.X):
                        reward = -1.0
                    else:
                        reward = 0.5
                    self.update_q(state, action, reward, next_state, next_actions, True)
                    break
                else:
                    self.update_q(state, action, 0.0, next_state, next_actions, False)
        
        self.is_training = False


class SARSAAgent(BaseAgent):
    """SARSA agent (on-policy TD learning)."""
    
    def __init__(self, player: Player = Player.X, epsilon: float = 0.1,
                 alpha: float = 0.1, gamma: float = 0.95):
        super().__init__("SARSA", player)
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.is_training = False
    
    def choose_action(self, game: TicTacToe) -> Tuple[int, int]:
        """Choose action using epsilon-greedy policy."""
        available = game.get_available_actions()
        
        if self.is_training and random.random() < self.epsilon:
            return random.choice(available)
        
        state_key = game.get_state_key()
        best_action = None
        best_q = -float('inf')
        
        for action in available:
            q_value = self.q_table.get((state_key, action), 0.0)
            if q_value > best_q:
                best_q = q_value
                best_action = action
        
        return best_action if best_action else available[0]
    
    def update_q(self, state: str, action: Tuple[int, int], reward: float,
                 next_state: str, next_action: Optional[Tuple[int, int]], done: bool):
        """Update Q-value using SARSA."""
        current_q = self.q_table.get((state, action), 0.0)
        
        if done or next_action is None:
            target = reward
        else:
            next_q = self.q_table.get((next_state, next_action), 0.0)
            target = reward + self.gamma * next_q
        
        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)
    
    def train(self, episodes: int = 10000):
        """Train the agent."""
        self.is_training = True
        opponent = RandomAgent(self.player.opponent())
        
        for episode in range(episodes):
            game = TicTacToe()
            if self.player == Player.O:
                action = opponent.choose_action(game)
                game.make_move(action)
            
            state = game.get_state_key()
            action = self.choose_action(game)
            
            while True:
                _, reward, done, result = game.make_move(action)
                
                if done:
                    if (result == GameResult.X_WIN and self.player == Player.X) or \
                       (result == GameResult.O_WIN and self.player == Player.O):
                        reward = 1.0
                    elif result == GameResult.DRAW:
                        reward = 0.5
                    else:
                        reward = -1.0
                    self.update_q(state, action, reward, "", None, True)
                    break
                
                # Opponent's turn
                opp_action = opponent.choose_action(game)
                _, _, opp_done, opp_result = game.make_move(opp_action)
                
                next_state = game.get_state_key()
                
                if opp_done:
                    if (opp_result == GameResult.X_WIN and self.player == Player.O) or \
                       (opp_result == GameResult.O_WIN and self.player == Player.X):
                        reward = -1.0
                    else:
                        reward = 0.5
                    self.update_q(state, action, reward, next_state, None, True)
                    break
                
                next_action = self.choose_action(game)
                self.update_q(state, action, 0.0, next_state, next_action, False)
                
                state = next_state
                action = next_action
        
        self.is_training = False


class ExpectedSARSAAgent(BaseAgent):
    """Expected SARSA agent."""
    
    def __init__(self, player: Player = Player.X, epsilon: float = 0.1,
                 alpha: float = 0.1, gamma: float = 0.95):
        super().__init__("Expected SARSA", player)
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.is_training = False
    
    def choose_action(self, game: TicTacToe) -> Tuple[int, int]:
        """Choose action using epsilon-greedy policy."""
        available = game.get_available_actions()
        
        if self.is_training and random.random() < self.epsilon:
            return random.choice(available)
        
        state_key = game.get_state_key()
        best_action = None
        best_q = -float('inf')
        
        for action in available:
            q_value = self.q_table.get((state_key, action), 0.0)
            if q_value > best_q:
                best_q = q_value
                best_action = action
        
        return best_action if best_action else available[0]
    
    def update_q(self, state: str, action: Tuple[int, int], reward: float,
                 next_state: str, next_actions: List[Tuple[int, int]], done: bool):
        """Update Q-value using Expected SARSA."""
        current_q = self.q_table.get((state, action), 0.0)
        
        if done:
            target = reward
        else:
            # Expected value under epsilon-greedy policy
            q_values = [self.q_table.get((next_state, a), 0.0) for a in next_actions]
            if q_values:
                max_q = max(q_values)
                expected_q = 0.0
                for q in q_values:
                    if q == max_q:
                        expected_q += (1 - self.epsilon + self.epsilon / len(q_values)) * q
                    else:
                        expected_q += (self.epsilon / len(q_values)) * q
                target = reward + self.gamma * expected_q
            else:
                target = reward
        
        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)
    
    def train(self, episodes: int = 10000):
        """Train the agent."""
        self.is_training = True
        opponent = RandomAgent(self.player.opponent())
        
        for episode in range(episodes):
            game = TicTacToe()
            if self.player == Player.O:
                action = opponent.choose_action(game)
                game.make_move(action)
            
            while True:
                state = game.get_state_key()
                available = game.get_available_actions()
                
                if len(available) == 0:
                    break
                
                action = self.choose_action(game)
                _, reward, done, result = game.make_move(action)
                
                if done:
                    if (result == GameResult.X_WIN and self.player == Player.X) or \
                       (result == GameResult.O_WIN and self.player == Player.O):
                        reward = 1.0
                    elif result == GameResult.DRAW:
                        reward = 0.5
                    else:
                        reward = -1.0
                    self.update_q(state, action, reward, "", [], True)
                    break
                
                opp_action = opponent.choose_action(game)
                _, _, opp_done, opp_result = game.make_move(opp_action)
                
                next_state = game.get_state_key()
                next_actions = game.get_available_actions()
                
                if opp_done:
                    if (opp_result == GameResult.X_WIN and self.player == Player.O) or \
                       (opp_result == GameResult.O_WIN and self.player == Player.X):
                        reward = -1.0
                    else:
                        reward = 0.5
                    self.update_q(state, action, reward, next_state, next_actions, True)
                    break
                else:
                    self.update_q(state, action, 0.0, next_state, next_actions, False)
        
        self.is_training = False


class DoubleQLearningAgent(BaseAgent):
    """Double Q-Learning agent."""
    
    def __init__(self, player: Player = Player.X, epsilon: float = 0.1,
                 alpha: float = 0.1, gamma: float = 0.95):
        super().__init__("Double Q-Learning", player)
        self.q_table1 = {}
        self.q_table2 = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.is_training = False
    
    def choose_action(self, game: TicTacToe) -> Tuple[int, int]:
        """Choose action using epsilon-greedy policy."""
        available = game.get_available_actions()
        
        if self.is_training and random.random() < self.epsilon:
            return random.choice(available)
        
        state_key = game.get_state_key()
        best_action = None
        best_q = -float('inf')
        
        for action in available:
            q1 = self.q_table1.get((state_key, action), 0.0)
            q2 = self.q_table2.get((state_key, action), 0.0)
            q_value = q1 + q2
            if q_value > best_q:
                best_q = q_value
                best_action = action
        
        return best_action if best_action else available[0]
    
    def update_q(self, state: str, action: Tuple[int, int], reward: float,
                 next_state: str, next_actions: List[Tuple[int, int]], done: bool):
        """Update Q-values using Double Q-Learning."""
        if random.random() < 0.5:
            # Update Q1
            current_q = self.q_table1.get((state, action), 0.0)
            
            if done:
                target = reward
            else:
                # Use Q1 to select action, Q2 to evaluate
                best_action = max(next_actions, 
                                 key=lambda a: self.q_table1.get((next_state, a), 0.0),
                                 default=None)
                if best_action:
                    next_q = self.q_table2.get((next_state, best_action), 0.0)
                    target = reward + self.gamma * next_q
                else:
                    target = reward
            
            self.q_table1[(state, action)] = current_q + self.alpha * (target - current_q)
        else:
            # Update Q2
            current_q = self.q_table2.get((state, action), 0.0)
            
            if done:
                target = reward
            else:
                # Use Q2 to select action, Q1 to evaluate
                best_action = max(next_actions,
                                 key=lambda a: self.q_table2.get((next_state, a), 0.0),
                                 default=None)
                if best_action:
                    next_q = self.q_table1.get((next_state, best_action), 0.0)
                    target = reward + self.gamma * next_q
                else:
                    target = reward
            
            self.q_table2[(state, action)] = current_q + self.alpha * (target - current_q)
    
    def train(self, episodes: int = 10000):
        """Train the agent."""
        self.is_training = True
        opponent = RandomAgent(self.player.opponent())
        
        for episode in range(episodes):
            game = TicTacToe()
            if self.player == Player.O:
                action = opponent.choose_action(game)
                game.make_move(action)
            
            while True:
                state = game.get_state_key()
                available = game.get_available_actions()
                
                if len(available) == 0:
                    break
                
                action = self.choose_action(game)
                _, reward, done, result = game.make_move(action)
                
                if done:
                    if (result == GameResult.X_WIN and self.player == Player.X) or \
                       (result == GameResult.O_WIN and self.player == Player.O):
                        reward = 1.0
                    elif result == GameResult.DRAW:
                        reward = 0.5
                    else:
                        reward = -1.0
                    self.update_q(state, action, reward, "", [], True)
                    break
                
                opp_action = opponent.choose_action(game)
                _, _, opp_done, opp_result = game.make_move(opp_action)
                
                next_state = game.get_state_key()
                next_actions = game.get_available_actions()
                
                if opp_done:
                    if (opp_result == GameResult.X_WIN and self.player == Player.O) or \
                       (opp_result == GameResult.O_WIN and self.player == Player.X):
                        reward = -1.0
                    else:
                        reward = 0.5
                    self.update_q(state, action, reward, next_state, next_actions, True)
                    break
                else:
                    self.update_q(state, action, 0.0, next_state, next_actions, False)
        
        self.is_training = False


# ==================== TOURNAMENT SYSTEM ====================

class Tournament:
    """Tournament manager for agent competitions."""
    
    def __init__(self, agents: List[BaseAgent]):
        """
        Initialize tournament.
        
        Args:
            agents: List of agents to compete
        """
        self.agents = agents
        self.results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0})
        self.matchups = {}
    
    def run_match(self, agent1: BaseAgent, agent2: BaseAgent, num_games: int = 10) -> Dict:
        """
        Run a match between two agents.
        
        Args:
            agent1: First agent (plays both X and O)
            agent2: Second agent (plays both X and O)
            num_games: Number of games per side
            
        Returns:
            Match statistics
        """
        stats = {
            'agent1_wins': 0,
            'agent2_wins': 0,
            'draws': 0,
            'games': num_games * 2
        }
        
        # Play num_games with agent1 as X
        for _ in range(num_games):
            result = self._play_game(agent1, agent2, Player.X, Player.O)
            if result == GameResult.X_WIN:
                stats['agent1_wins'] += 1
            elif result == GameResult.O_WIN:
                stats['agent2_wins'] += 1
            else:
                stats['draws'] += 1
        
        # Play num_games with agent2 as X
        for _ in range(num_games):
            result = self._play_game(agent2, agent1, Player.X, Player.O)
            if result == GameResult.X_WIN:
                stats['agent2_wins'] += 1
            elif result == GameResult.O_WIN:
                stats['agent1_wins'] += 1
            else:
                stats['draws'] += 1
        
        # Update overall results
        self.results[agent1.name]['wins'] += stats['agent1_wins']
        self.results[agent1.name]['losses'] += stats['agent2_wins']
        self.results[agent1.name]['draws'] += stats['draws']
        self.results[agent1.name]['games'] += stats['games']
        
        self.results[agent2.name]['wins'] += stats['agent2_wins']
        self.results[agent2.name]['losses'] += stats['agent1_wins']
        self.results[agent2.name]['draws'] += stats['draws']
        self.results[agent2.name]['games'] += stats['games']
        
        matchup_key = f"{agent1.name} vs {agent2.name}"
        self.matchups[matchup_key] = stats
        
        return stats
    
    def _play_game(self, agent_x: BaseAgent, agent_o: BaseAgent, 
                   player_x: Player, player_o: Player) -> GameResult:
        """Play a single game."""
        game = TicTacToe()
        
        # Temporarily set player assignments
        agent_x.player = player_x
        agent_o.player = player_o
        
        while game.check_winner() == GameResult.IN_PROGRESS:
            if game.current_player == Player.X:
                action = agent_x.choose_action(game)
            else:
                action = agent_o.choose_action(game)
            
            game.make_move(action)
        
        return game.check_winner()
    
    def run_round_robin(self, games_per_match: int = 10, verbose: bool = True):
        """
        Run round-robin tournament.
        
        Args:
            games_per_match: Number of games per match (both sides)
            verbose: Print progress
        """
        total_matches = len(self.agents) * (len(self.agents) - 1) // 2
        match_count = 0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"ROUND-ROBIN TOURNAMENT: {len(self.agents)} agents")
            print(f"{'='*60}\n")
        
        for i, agent1 in enumerate(self.agents):
            for agent2 in self.agents[i+1:]:
                match_count += 1
                if verbose:
                    print(f"Match {match_count}/{total_matches}: {agent1.name} vs {agent2.name}")
                
                stats = self.run_match(agent1, agent2, games_per_match)
                
                if verbose:
                    print(f"  {agent1.name}: {stats['agent1_wins']} wins")
                    print(f"  {agent2.name}: {stats['agent2_wins']} wins")
                    print(f"  Draws: {stats['draws']}")
                    print()
    
    def get_leaderboard(self) -> List[Tuple[str, Dict]]:
        """Get sorted leaderboard."""
        leaderboard = []
        for agent_name, stats in self.results.items():
            win_rate = stats['wins'] / stats['games'] if stats['games'] > 0 else 0
            stats['win_rate'] = win_rate
            leaderboard.append((agent_name, stats))
        
        leaderboard.sort(key=lambda x: (x[1]['wins'], x[1]['win_rate']), reverse=True)
        return leaderboard
    
    def print_leaderboard(self):
        """Print tournament leaderboard."""
        print(f"\n{'='*60}")
        print("LEADERBOARD")
        print(f"{'='*60}")
        print(f"{'Rank':<6} {'Agent':<25} {'Wins':<8} {'Losses':<8} {'Draws':<8} {'Win%':<8}")
        print(f"{'-'*60}")
        
        leaderboard = self.get_leaderboard()
        for rank, (agent_name, stats) in enumerate(leaderboard, 1):
            win_pct = stats['win_rate'] * 100
            print(f"{rank:<6} {agent_name:<25} {stats['wins']:<8} {stats['losses']:<8} "
                  f"{stats['draws']:<8} {win_pct:<8.1f}")
        
        print(f"{'='*60}\n")
    
    def visualize_results(self, save_path: str = "tournament_results.png"):
        """Create visualization of tournament results."""
        leaderboard = self.get_leaderboard()
        
        if not leaderboard:
            print("No results to visualize!")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Tournament Results', fontsize=16, fontweight='bold')
        
        # 1. Win Rate Comparison
        agents = [item[0] for item in leaderboard]
        win_rates = [item[1]['win_rate'] * 100 for item in leaderboard]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(agents)))
        bars = ax1.barh(agents, win_rates, color=colors)
        ax1.set_xlabel('Win Rate (%)', fontsize=12)
        ax1.set_title('Win Rate Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 100)
        
        for bar, rate in zip(bars, win_rates):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{rate:.1f}%', va='center', fontsize=10)
        
        # 2. Wins/Losses/Draws Distribution
        wins = [item[1]['wins'] for item in leaderboard]
        losses = [item[1]['losses'] for item in leaderboard]
        draws = [item[1]['draws'] for item in leaderboard]
        
        x = np.arange(len(agents))
        width = 0.25
        
        ax2.bar(x - width, wins, width, label='Wins', color='#2ecc71')
        ax2.bar(x, draws, width, label='Draws', color='#f39c12')
        ax2.bar(x + width, losses, width, label='Losses', color='#e74c3c')
        
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Win/Draw/Loss Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(agents, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Head-to-Head Heatmap
        n_agents = len(self.agents)
        h2h_matrix = np.zeros((n_agents, n_agents))
        agent_names = [agent.name for agent in self.agents]
        
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents):
                if i != j:
                    key1 = f"{agent1.name} vs {agent2.name}"
                    key2 = f"{agent2.name} vs {agent1.name}"
                    
                    if key1 in self.matchups:
                        stats = self.matchups[key1]
                        wins = stats['agent1_wins']
                        total = stats['games']
                        h2h_matrix[i, j] = (wins / total * 100) if total > 0 else 0
                    elif key2 in self.matchups:
                        stats = self.matchups[key2]
                        wins = stats['agent2_wins']
                        total = stats['games']
                        h2h_matrix[i, j] = (wins / total * 100) if total > 0 else 0
        
        sns.heatmap(h2h_matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                   xticklabels=agent_names, yticklabels=agent_names,
                   ax=ax3, cbar_kws={'label': 'Win Rate (%)'}, vmin=0, vmax=100)
        ax3.set_title('Head-to-Head Win Rates (%)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Opponent', fontsize=12)
        ax3.set_ylabel('Agent', fontsize=12)
        
        # 4. Total Games Played
        total_games = [item[1]['games'] for item in leaderboard]
        ax4.pie(total_games, labels=agents, autopct='%1.1f%%', startangle=90, colors=colors)
        ax4.set_title('Games Played Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tournament visualization saved to: {save_path}")
        plt.close()


# ==================== PYGAME GUI ====================

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Pygame not available. Install with: pip install pygame")


class TicTacToeGUI:
    """Interactive Pygame GUI for Tic Tac Toe."""
    
    def __init__(self, agent: Optional[BaseAgent] = None):
        """
        Initialize GUI.
        
        Args:
            agent: AI agent to play against (None for human vs human)
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("Pygame is required for GUI")
        
        pygame.init()
        
        # Constants
        self.WINDOW_SIZE = 600
        self.GRID_SIZE = 3
        self.CELL_SIZE = self.WINDOW_SIZE // self.GRID_SIZE
        self.LINE_WIDTH = 5
        self.MARK_WIDTH = 15
        
        # Colors
        self.BG_COLOR = (28, 170, 156)
        self.LINE_COLOR = (23, 145, 135)
        self.X_COLOR = (242, 235, 211)
        self.O_COLOR = (242, 85, 96)
        self.TEXT_COLOR = (255, 255, 255)
        self.BUTTON_COLOR = (52, 73, 94)
        self.BUTTON_HOVER_COLOR = (52, 152, 219)
        
        # Setup
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE + 100))
        pygame.display.set_caption("Tic Tac Toe")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 40)
        self.small_font = pygame.font.Font(None, 30)
        
        # Game state
        self.game = TicTacToe()
        self.agent = agent
        self.game_over = False
        self.human_player = Player.X if agent is None or agent.player == Player.O else Player.O
        
        # UI elements
        self.reset_button = pygame.Rect(200, self.WINDOW_SIZE + 20, 200, 60)
    
    def draw_grid(self):
        """Draw the game grid."""
        self.screen.fill(self.BG_COLOR)
        
        # Draw lines
        for i in range(1, self.GRID_SIZE):
            # Vertical lines
            pygame.draw.line(self.screen, self.LINE_COLOR,
                           (i * self.CELL_SIZE, 0),
                           (i * self.CELL_SIZE, self.WINDOW_SIZE),
                           self.LINE_WIDTH)
            # Horizontal lines
            pygame.draw.line(self.screen, self.LINE_COLOR,
                           (0, i * self.CELL_SIZE),
                           (self.WINDOW_SIZE, i * self.CELL_SIZE),
                           self.LINE_WIDTH)
    
    def draw_marks(self):
        """Draw X's and O's."""
        for row in range(3):
            for col in range(3):
                cell_value = self.game.board[row, col]
                
                if cell_value == 1:  # X
                    self.draw_x(row, col)
                elif cell_value == 2:  # O
                    self.draw_o(row, col)
    
    def draw_x(self, row: int, col: int):
        """Draw X mark."""
        offset = self.CELL_SIZE // 4
        start_x = col * self.CELL_SIZE + offset
        start_y = row * self.CELL_SIZE + offset
        end_x = (col + 1) * self.CELL_SIZE - offset
        end_y = (row + 1) * self.CELL_SIZE - offset
        
        pygame.draw.line(self.screen, self.X_COLOR,
                        (start_x, start_y), (end_x, end_y), self.MARK_WIDTH)
        pygame.draw.line(self.screen, self.X_COLOR,
                        (start_x, end_y), (end_x, start_y), self.MARK_WIDTH)
    
    def draw_o(self, row: int, col: int):
        """Draw O mark."""
        center_x = col * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = row * self.CELL_SIZE + self.CELL_SIZE // 2
        radius = self.CELL_SIZE // 3
        
        pygame.draw.circle(self.screen, self.O_COLOR,
                          (center_x, center_y), radius, self.MARK_WIDTH)
    
    def draw_winning_line(self):
        """Draw line through winning marks."""
        winner_line = self.game.get_winner_line()
        if winner_line:
            start_row, start_col = winner_line[0]
            end_row, end_col = winner_line[2]
            
            start_x = start_col * self.CELL_SIZE + self.CELL_SIZE // 2
            start_y = start_row * self.CELL_SIZE + self.CELL_SIZE // 2
            end_x = end_col * self.CELL_SIZE + self.CELL_SIZE // 2
            end_y = end_row * self.CELL_SIZE + self.CELL_SIZE // 2
            
            pygame.draw.line(self.screen, (255, 255, 255),
                           (start_x, start_y), (end_x, end_y), 10)
    
    def draw_ui(self):
        """Draw UI elements."""
        # Status text
        if not self.game_over:
            if self.game.current_player == self.human_player:
                text = "Your turn"
            else:
                text = "AI thinking..."
        else:
            result = self.game.check_winner()
            if result == GameResult.DRAW:
                text = "Draw!"
            elif (result == GameResult.X_WIN and self.human_player == Player.X) or \
                 (result == GameResult.O_WIN and self.human_player == Player.O):
                text = "You Win!"
            else:
                text = "AI Wins!"
        
        text_surface = self.font.render(text, True, self.TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(self.WINDOW_SIZE // 2, self.WINDOW_SIZE + 15))
        self.screen.blit(text_surface, text_rect)
        
        # Reset button
        mouse_pos = pygame.mouse.get_pos()
        button_color = self.BUTTON_HOVER_COLOR if self.reset_button.collidepoint(mouse_pos) else self.BUTTON_COLOR
        pygame.draw.rect(self.screen, button_color, self.reset_button, border_radius=10)
        
        button_text = self.small_font.render("New Game", True, self.TEXT_COLOR)
        button_rect = button_text.get_rect(center=self.reset_button.center)
        self.screen.blit(button_text, button_rect)
    
    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click."""
        x, y = pos
        
        # Check reset button
        if self.reset_button.collidepoint(pos):
            self.game.reset()
            self.game_over = False
            return
        
        # Check board click
        if y < self.WINDOW_SIZE and not self.game_over:
            if self.game.current_player == self.human_player:
                row = y // self.CELL_SIZE
                col = x // self.CELL_SIZE
                action = (row, col)
                
                if self.game.is_valid_action(action):
                    self.game.make_move(action)
                    
                    result = self.game.check_winner()
                    if result != GameResult.IN_PROGRESS:
                        self.game_over = True
    
    def ai_move(self):
        """Make AI move."""
        if self.agent and not self.game_over and self.game.current_player == self.agent.player:
            action = self.agent.choose_action(self.game)
            self.game.make_move(action)
            
            result = self.game.check_winner()
            if result != GameResult.IN_PROGRESS:
                self.game_over = True
    
    def run(self):
        """Run the GUI main loop."""
        running = True
        
        while running:
            self.clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
            
            # AI move
            self.ai_move()
            
            # Draw everything
            self.draw_grid()
            self.draw_marks()
            if self.game_over:
                self.draw_winning_line()
            self.draw_ui()
            
            pygame.display.flip()
        
        pygame.quit()


# ==================== HELPER FUNCTIONS ====================

def create_all_agents() -> List[BaseAgent]:
    """Create instances of all available agents."""
    return [
        RandomAgent(),
        ExhaustiveSearchAgent(),
        MinMaxAgent(),
        # MonteCarloAgent does not accept 'simulations' kwarg; keep signature (player, epsilon, alpha, gamma)
        MonteCarloAgent(),
        DynamicProgrammingAgent(),
    ]


def create_trained_rl_agents(episodes: int = 5000) -> List[BaseAgent]:
    """Create and train RL agents."""
    print("Training RL agents...")
    
    agents = []
    
    # Q-Learning
    print("  Training Q-Learning...")
    q_agent = QLearningAgent()
    q_agent.train(episodes)
    agents.append(q_agent)
    
    # SARSA
    print("  Training SARSA...")
    sarsa_agent = SARSAAgent()
    sarsa_agent.train(episodes)
    agents.append(sarsa_agent)
    
    # Expected SARSA
    print("  Training Expected SARSA...")
    exp_sarsa_agent = ExpectedSARSAAgent()
    exp_sarsa_agent.train(episodes)
    agents.append(exp_sarsa_agent)
    
    # Double Q-Learning
    print("  Training Double Q-Learning...")
    double_q_agent = DoubleQLearningAgent()
    double_q_agent.train(episodes)
    agents.append(double_q_agent)
    
    print("Training complete!\n")
    return agents


def quick_match(agent1: BaseAgent, agent2: BaseAgent, num_games: int = 10):
    """Quick match between two agents."""
    print(f"\n{'='*50}")
    print(f"MATCH: {agent1.name} vs {agent2.name}")
    print(f"{'='*50}\n")
    
    tournament = Tournament([agent1, agent2])
    stats = tournament.run_match(agent1, agent2, num_games)
    
    print(f"{agent1.name}: {stats['agent1_wins']} wins")
    print(f"{agent2.name}: {stats['agent2_wins']} wins")
    print(f"Draws: {stats['draws']}")
    print()


# ==================== BENCHMARK (optional manual run) ====================

if __name__ == "__main__":
    import time
    print("=== Benchmark: DP training and Tournament ===")
    # Build 9 agents
    agents: List[BaseAgent] = [
        RandomAgent(Player.X),
        ExhaustiveSearchAgent(Player.X),
        MinMaxAgent(Player.X),
        MonteCarloAgent(Player.X),
        DynamicProgrammingAgent(Player.X),
        QLearningAgent(Player.X),
        SARSAAgent(Player.X),
        ExpectedSARSAAgent(Player.X),
        DoubleQLearningAgent(Player.X),
    ]

    # DP training benchmark (episodes=200)
    dp = agents[4]
    t0 = time.perf_counter()
    if isinstance(dp, DynamicProgrammingAgent):
        dp.train_with_history(episodes=200)
    t1 = time.perf_counter()
    print(f"DP(200) elapsed: {t1 - t0:.2f}s")

    # Tournament ~9 agents, 100 per matchup including self-matches
    tour = Tournament(agents)
    n = len(agents)
    t2 = time.perf_counter()
    games_per_match = 100
    for i in range(n):
        for j in range(n):
            a1 = agents[i]
            if i == j:
                # self-match: clone same type for the O side
                a2 = type(a1)(Player.O) if not isinstance(a1, MonteCarloAgent) else MonteCarloAgent(Player.O)
            else:
                a2 = agents[j]
            tour.run_match(a1, a2, num_games=games_per_match)
    t3 = time.perf_counter()
    total = t3 - t2
    print(f"Tournament elapsed: {total:.2f}s for {n*n*games_per_match*2} games (both sides)")
    if total > 20:
        print("[warn] Tournament took >20s; consider increasing cache sizes or reducing games for your machine.")
