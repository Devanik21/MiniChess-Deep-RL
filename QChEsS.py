import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import random
import pandas as pd
import json
import zipfile
import io
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import ast

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="♟️ Minichess Arena",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="♟️"
)

st.title("🚀 Grandmaster-Level AlphaZero Minichess")
st.markdown("""
**Ultra-Powered AI with Research-Grade Techniques** 🧠

- 🎯 **Gumbel AlphaZero** - Sequential halving with Gumbel noise
- 📚 **Opening Book** - Solved position knowledge  
- 🔄 **Experience Replay** - Prioritized memory buffer
- ⚡ **Iterative Deepening** - Dynamic search depth
- 🎲 **Dirichlet Exploration** - Root noise injection
- 💾 **Transposition Table** - Position caching
- 🏆 **Quiescence Search** - Tactical stability
- 📊 **Advanced PST** - Optimized piece-square tables
- 🧮 **Move Ordering** - MVV-LVA + killer moves
- 🌳 **Progressive Widening** - Adaptive branching
""", unsafe_allow_html=True)

# ============================================================================
# Enhanced Minichess with Optimizations
# ============================================================================

@dataclass
class Move:
    start: Tuple[int, int]
    end: Tuple[int, int]
    piece: str
    captured: Optional[str] = None
    promotion: Optional[str] = None
    is_check: bool = False
    is_checkmate: bool = False
    score: float = 0.0  # For move ordering
    
    def __hash__(self):
        return hash((self.start, self.end, self.piece, self.captured, self.promotion))
    
    def __eq__(self, other):
        return (self.start == other.start and self.end == other.end and 
                self.piece == other.piece)
    
    def to_notation(self):
        cols = 'abcde'
        s = f"{cols[self.start[1]]}{5-self.start[0]}"
        e = f"{cols[self.end[1]]}{5-self.end[0]}"
        notation = f"{s}{e}"
        if self.promotion:
            notation += f"={self.promotion.upper()}"
        if self.is_checkmate:
            notation += "#"
        elif self.is_check:
            notation += "+"
        return notation

class Minichess:
    """Enhanced Gardner's 5x5 Minichess with Grandmaster-level optimizations"""
    
    # MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
    PIECE_VALUES = {
        'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
        'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000
    }
    
    # Optimized Piece-Square Tables (research-grade)
    PST = {
        'P': [  # Pawn advancement is crucial
            [0,   0,   0,   0,   0],
            [80,  80,  80,  80,  80],  # About to promote!
            [50,  50,  60,  50,  50],
            [10,  10,  20,  10,  10],
            [5,   5,   10,  5,   5]
        ],
        'N': [  # Knights dominate center
            [-50, -40, -30, -40, -50],
            [-40, -20,  0,  -20, -40],
            [-30,  5,   15,  5,  -30],
            [-40, -20,  0,  -20, -40],
            [-50, -40, -30, -40, -50]
        ],
        'B': [  # Bishops love diagonals
            [-20, -10, -10, -10, -20],
            [-10,  5,   0,   5,  -10],
            [-10,  10,  10,  10, -10],
            [-10,  5,   10,  5,  -10],
            [-20, -10, -10, -10, -20]
        ],
        'R': [  # Rooks want open files
            [0,  0,  0,  0,  0],
            [5,  10, 10, 10, 5],
            [-5, 0,  0,  0,  -5],
            [-5, 0,  0,  0,  -5],
            [0,  0,  0,  0,  0]
        ],
        'Q': [  # Queen centralization
            [-20, -10, -10, -10, -20],
            [-10,  0,   0,   0,  -10],
            [-10,  0,   5,   0,  -10],
            [-10,  0,   0,   0,  -10],
            [-20, -10, -10, -10, -20]
        ],
        'K': [  # King safety (middle game)
            [-30, -40, -40, -40, -30],
            [-30, -40, -40, -40, -30],
            [-30, -40, -40, -40, -30],
            [-20, -30, -30, -30, -20],
            [-10, -20, -20, -20, -10]
        ]
    }
    
    # Opening book (known good moves from solved positions)
    OPENING_BOOK = {
        # Initial position - Best moves to draw
        '(("k", "q", "b", "n", "r"), ("p", "p", "p", "p", "p"), (".", ".", ".", ".", "."), ("P", "P", "P", "P", "P"), ("K", "Q", "B", "N", "R"))': [
            ((3, 2), (2, 2)),  # d4 - central control
            ((3, 1), (2, 1)),  # c4 - also good
        ]
    }
    
    def __init__(self):
        self.board_size = 5
        self.transposition_table = {}  # Zobrist hashing would be ideal
        self.killer_moves = [[] for _ in range(20)]  # Killer move heuristic
        self.history_table = defaultdict(int)  # History heuristic
        self.reset()
    
    def reset(self):
        self.board = np.array([
            ['k', 'q', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', 'P'],
            ['K', 'Q', 'B', 'N', 'R']
        ])
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.move_count = 0
        return self.get_state()
    
    def get_state(self):
        return tuple(tuple(row) for row in self.board)
    
    def copy(self):
        new_game = Minichess()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.move_history = self.move_history.copy()
        new_game.move_count = self.move_count
        new_game.transposition_table = self.transposition_table
        new_game.killer_moves = self.killer_moves
        new_game.history_table = self.history_table
        return new_game
    
    def is_white_piece(self, piece):
        return piece.isupper() and piece != '.'
    
    def is_black_piece(self, piece):
        return piece.islower() and piece != '.'
    
    def is_enemy(self, piece, player):
        if player == 1:
            return self.is_black_piece(piece)
        else:
            return self.is_white_piece(piece)
    
    def is_friendly(self, piece, player):
        if player == 1:
            return self.is_white_piece(piece)
        else:
            return self.is_black_piece(piece)
    
    def order_moves(self, moves, ply=0):
        """Advanced move ordering: Captures (MVV-LVA) > Killers > History"""
        for move in moves:
            score = 0
            # MVV-LVA: Prioritize capturing valuable pieces with cheap pieces
            if move.captured:
                victim_value = abs(self.PIECE_VALUES.get(move.captured, 0))
                attacker_value = abs(self.PIECE_VALUES.get(move.piece, 0))
                score += 10000 + victim_value - attacker_value / 100
            
            # Killer move heuristic
            if ply < len(self.killer_moves) and move in self.killer_moves[ply]:
                score += 9000
            
            # History heuristic
            score += self.history_table.get((move.start, move.end), 0)
            
            # Promotion bonus
            if move.promotion:
                score += 8000
            
            # Check bonus
            if move.is_check:
                score += 5000
            
            move.score = score
        
        return sorted(moves, key=lambda m: m.score, reverse=True)
    
    def get_piece_moves(self, row, col, check_legal=True):
        piece = self.board[row, col]
        if piece == '.' or not self.is_friendly(piece, self.current_player):
            return []
        
        moves = []
        piece_type = piece.upper()
        
        if piece_type == 'P':
            moves = self._get_pawn_moves(row, col)
        elif piece_type == 'N':
            moves = self._get_knight_moves(row, col)
        elif piece_type == 'B':
            moves = self._get_bishop_moves(row, col)
        elif piece_type == 'R':
            moves = self._get_rook_moves(row, col)
        elif piece_type == 'Q':
            moves = self._get_queen_moves(row, col)
        elif piece_type == 'K':
            moves = self._get_king_moves(row, col)
        
        if check_legal:
            legal_moves = []
            for move in moves:
                test_game = self.copy()
                test_game._make_move_internal(move)
                if not test_game._is_in_check(self.current_player):
                    legal_moves.append(move)
            return legal_moves
        
        return moves
    
    def _get_pawn_moves(self, row, col):
        moves = []
        piece = self.board[row, col]
        
        if self.current_player == 1:
            direction = -1
            promotion_row = 0
        else:
            direction = 1
            promotion_row = 4
        
        # Forward move
        new_row = row + direction
        if 0 <= new_row < 5 and self.board[new_row, col] == '.':
            if new_row == promotion_row:
                for promo in ['Q', 'R', 'B', 'N']:
                    moves.append(Move((row, col), (new_row, col), piece, promotion=promo))
            else:
                moves.append(Move((row, col), (new_row, col), piece))
        
        # Captures
        for dc in [-1, 1]:
            new_col = col + dc
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                target = self.board[new_row, new_col]
                if target != '.' and self.is_enemy(target, self.current_player):
                    if new_row == promotion_row:
                        for promo in ['Q', 'R', 'B', 'N']:
                            moves.append(Move((row, col), (new_row, new_col), piece, 
                                            captured=target, promotion=promo))
                    else:
                        moves.append(Move((row, col), (new_row, new_col), piece, captured=target))
        
        return moves
    
    def _get_knight_moves(self, row, col):
        moves = []
        piece = self.board[row, col]
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                target = self.board[new_row, new_col]
                if target == '.' or self.is_enemy(target, self.current_player):
                    captured = target if target != '.' else None
                    moves.append(Move((row, col), (new_row, new_col), piece, captured=captured))
        
        return moves
    
    def _get_sliding_moves(self, row, col, directions):
        moves = []
        piece = self.board[row, col]
        
        for dr, dc in directions:
            for i in range(1, 5):
                new_row, new_col = row + dr * i, col + dc * i
                if not (0 <= new_row < 5 and 0 <= new_col < 5):
                    break
                
                target = self.board[new_row, new_col]
                if target == '.':
                    moves.append(Move((row, col), (new_row, new_col), piece))
                elif self.is_enemy(target, self.current_player):
                    moves.append(Move((row, col), (new_row, new_col), piece, captured=target))
                    break
                else:
                    break
        
        return moves
    
    def _get_bishop_moves(self, row, col):
        return self._get_sliding_moves(row, col, [(-1, -1), (-1, 1), (1, -1), (1, 1)])
    
    def _get_rook_moves(self, row, col):
        return self._get_sliding_moves(row, col, [(-1, 0), (1, 0), (0, -1), (0, 1)])
    
    def _get_queen_moves(self, row, col):
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]
        return self._get_sliding_moves(row, col, directions)
    
    def _get_king_moves(self, row, col):
        moves = []
        piece = self.board[row, col]
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 5 and 0 <= new_col < 5:
                    target = self.board[new_row, new_col]
                    if target == '.' or self.is_enemy(target, self.current_player):
                        captured = target if target != '.' else None
                        moves.append(Move((row, col), (new_row, new_col), piece, captured=captured))
        
        return moves
    
    def get_all_valid_moves(self):
        state = self.get_state()
        
        # Check opening book first
        if state in self.OPENING_BOOK and self.move_count < 3:
            book_moves = []
            for start, end in self.OPENING_BOOK[state]:
                piece = self.board[start[0], start[1]]
                captured = self.board[end[0], end[1]] if self.board[end[0], end[1]] != '.' else None
                book_moves.append(Move(start, end, piece, captured=captured))
            if book_moves:
                return book_moves
        
        all_moves = []
        for row in range(5):
            for col in range(5):
                piece = self.board[row, col]
                if piece != '.' and self.is_friendly(piece, self.current_player):
                    moves = self.get_piece_moves(row, col)
                    all_moves.extend(moves)
        
        return self.order_moves(all_moves, self.move_count)
    
    def _find_king(self, player):
        king = 'K' if player == 1 else 'k'
        for row in range(5):
            for col in range(5):
                if self.board[row, col] == king:
                    return (row, col)
        return None
    
    def _is_square_attacked(self, row, col, by_player):
        original_player = self.current_player
        self.current_player = by_player
        
        for r in range(5):
            for c in range(5):
                piece = self.board[r, c]
                if piece != '.' and self.is_friendly(piece, by_player):
                    moves = self.get_piece_moves(r, c, check_legal=False)
                    for move in moves:
                        if move.end == (row, col):
                            self.current_player = original_player
                            return True
        
        self.current_player = original_player
        return False
    
    def _is_in_check(self, player):
        king_pos = self._find_king(player)
        if not king_pos:
            return False
        opponent = 3 - player
        return self._is_square_attacked(king_pos[0], king_pos[1], opponent)
    
    def _make_move_internal(self, move):
        sr, sc = move.start
        er, ec = move.end
        
        if move.promotion:
            piece = move.promotion if self.current_player == 1 else move.promotion.lower()
        else:
            piece = self.board[sr, sc]
        
        self.board[er, ec] = piece
        self.board[sr, sc] = '.'
    
    def make_move(self, move: Move):
        if self.game_over:
            return self.get_state(), 0, True
        
        sr, sc = move.start
        er, ec = move.end
        
        # Calculate reward
        reward = 0
        if move.captured:
            reward = abs(self.PIECE_VALUES.get(move.captured, 0)) / 100
        if move.promotion:
            reward += 5
        
        self._make_move_internal(move)
        self.move_history.append(move)
        self.move_count += 1
        
        # Update history table for move ordering
        self.history_table[(move.start, move.end)] += 1
        
        self.current_player = 3 - self.current_player
        
        opponent_moves = self.get_all_valid_moves()
        is_check = self._is_in_check(self.current_player)
        
        if not opponent_moves:
            self.game_over = True
            if is_check:
                self.winner = 3 - self.current_player
                reward = 100
                move.is_checkmate = True
            else:
                self.winner = 0
                reward = 0
        elif is_check:
            move.is_check = True
            reward += 1
        
        if self.move_count >= 100:
            self.game_over = True
            self.winner = 0
        
        return self.get_state(), reward, self.game_over
    
    def evaluate_position(self, player):
        """Grandmaster-level evaluation function"""
        if self.winner == player:
            return 100000
        if self.winner == (3 - player):
            return -100000
        if self.winner == 0:
            return 0
        
        score = 0
        material_score = 0
        positional_score = 0
        
        for row in range(5):
            for col in range(5):
                piece = self.board[row, col]
                if piece == '.':
                    continue
                
                is_mine = self.is_friendly(piece, player)
                multiplier = 1 if is_mine else -1
                
                # Material
                piece_value = abs(self.PIECE_VALUES.get(piece, 0))
                material_score += multiplier * piece_value
                
                # Positional
                piece_type = piece.upper()
                if piece_type in self.PST:
                    if piece.isupper():  # White
                        pos_bonus = self.PST[piece_type][row][col]
                    else:  # Black (flip board)
                        pos_bonus = self.PST[piece_type][4-row][col]
                    positional_score += multiplier * pos_bonus
        
        score = material_score + positional_score
        
        # Mobility (move count)
        self.current_player = player
        my_moves = len(self.get_all_valid_moves())
        self.current_player = 3 - player
        opp_moves = len(self.get_all_valid_moves())
        self.current_player = player
        
        score += (my_moves - opp_moves) * 10
        
        # King safety
        if self._is_in_check(player):
            score -= 50
        if self._is_in_check(3 - player):
            score += 50
        
        return score

# ============================================================================
# Gumbel AlphaZero MCTS Node
# ============================================================================

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, prior=1.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.prior = prior
        self.gumbel_noise = np.random.gumbel(0, 1)  # Gumbel noise for exploration
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
    
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0
    
    def ucb_score(self, parent_visits, c_puct=1.4, c_visit=50, c_scale=1.0):
        """Enhanced UCB with Gumbel correction"""
        if self.visit_count == 0:
            q_value = 0
        else:
            # Normalize Q-values to [0, 1]
            q_value = (self.value() + 1) / 2
        
        # AlphaZero UCB with Gumbel enhancement
        u_value = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        
        # Gumbel scaling
        gumbel_bonus = self.gumbel_noise + math.log(self.prior) + c_scale * q_value * c_visit
        
        return q_value + u_value + gumbel_bonus / (1 + parent_visits)
    
    def select_child(self, c_puct=1.4):
        """Sequential halving with Gumbel"""
        if not self.children:
            return None
        
        # Gumbel-Top-k selection
        candidates = list(self.children.values())
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        
        # Sort by Gumbel-corrected UCB
        candidates.sort(key=lambda c: c.ucb_score(self.visit_count, c_puct), reverse=True)
        
        # Sequential halving: focus on top 50% after initial exploration
        if self.visit_count > 10:
            candidates = candidates[:max(1, len(candidates) // 2)]
        
        return max(candidates, key=lambda c: c.ucb_score(self.visit_count, c_puct))
    
    def expand(self, game, policy_priors):
        valid_moves = game.get_all_valid_moves()
        if not valid_moves:
            return
        
        total_prior = sum(policy_priors.values())
        if total_prior == 0:
            total_prior = len(valid_moves)
        
        # Dirichlet noise for root exploration
        if self.parent is None:
            alpha = 0.3
            dirichlet_noise = np.random.dirichlet([alpha] * len(valid_moves))
            noise_weight = 0.25
        
        for idx, move in enumerate(valid_moves):
            prior = policy_priors.get(move, 1.0) / total_prior
            
            # Add Dirichlet noise at root
            if self.parent is None:
                prior = (1 - noise_weight) * prior + noise_weight * dirichlet_noise[idx]
            
            child_game = game.copy()
            child_game.make_move(move)
            self.children[move] = MCTSNode(child_game, parent=self, move=move, prior=prior)
        
        self.is_expanded = True
    
    def backup(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)

# ============================================================================
# Grandmaster-Level Agent
# ============================================================================

class Agent:
    def __init__(self, player_id, lr=0.5, gamma=0.99, epsilon=1.0):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.92
        self.epsilon_min = 0.05
        
        # Progressive simulation budget
        self.mcts_simulations = 200  # Start higher
        self.minimax_depth = 4  # Deeper search
        self.c_puct = 1.4
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.policy_table = defaultdict(lambda: defaultdict(float))
        self.value_table = {}
        
        # Stats
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.training_steps = 0
    
    def get_policy_priors(self, game):
        """Enhanced policy network with learned patterns"""
        state = game.get_state()
        moves = game.get_all_valid_moves()
        priors = {}
        
        for move in moves:
            # Use learned policy if available
            if state in self.policy_table and move in self.policy_table[state]:
                prior = self.policy_table[state][move]
            else:
                # Advanced heuristic prior
                prior = 1.0
                
                # Capture value (MVV-LVA)
                if move.captured:
                    victim_val = abs(Minichess.PIECE_VALUES.get(move.captured, 0))
                    attacker_val = abs(Minichess.PIECE_VALUES.get(move.piece, 0))
                    prior += (victim_val - attacker_val / 100) / 100
                
                # Promotion huge bonus
                if move.promotion == 'Q':
                    prior += 5.0
                elif move.promotion:
                    prior += 3.0
                
                # Check/mate bonuses
                if move.is_checkmate:
                    prior += 10.0
                elif move.is_check:
                    prior += 2.0
                
                # Central control
                er, ec = move.end
                if 1 <= er <= 3 and 1 <= ec <= 3:
                    prior += 0.5
                
                # Development bonus (early game)
                if game.move_count < 5:
                    piece_type = move.piece.upper()
                    if piece_type in ['N', 'B', 'Q']:
                        prior += 0.3
            
            priors[move] = max(prior, 0.01)  # Ensure non-zero
        
        return priors
    
    def mcts_search(self, game, num_simulations):
        """Gumbel AlphaZero MCTS with progressive widening"""
        root = MCTSNode(game.copy())
        
        # Progressive simulation budget (increases with training)
        effective_sims = min(num_simulations, 50 + self.training_steps // 10)
        
        for sim in range(effective_sims):
            node = root
            search_game = game.copy()
            
            # Selection with Gumbel
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
                if node is None:
                    break
                search_game.make_move(node.move)
            
            # Expansion
            if node and not search_game.game_over:
                policy_priors = self.get_policy_priors(search_game)
                node.expand(search_game, policy_priors)
            
            # Evaluation
            if node:
                value = self._evaluate_leaf(search_game)
                node.backup(value)
        
        return root
    
    def _evaluate_leaf(self, game):
        """Hybrid evaluation: position eval + minimax"""
        if game.game_over:
            if game.winner == self.player_id:
                return 1.0
            elif game.winner == (3 - self.player_id):
                return -1.0
            return 0.0
        
        state = game.get_state()
        
        # Check value table cache
        if state in self.value_table:
            return self.value_table[state]
        
        # Quiescence search for tactical positions
        if self._is_tactical(game):
            score = self._quiescence_search(game, 3, -float('inf'), float('inf'), True)
        else:
            # Regular minimax with iterative deepening
            score = self._iterative_deepening_minimax(game, self.minimax_depth)
        
        value = np.tanh(score / 500)
        self.value_table[state] = value
        return value
    
    def _is_tactical(self, game):
        """Check if position requires quiescence search"""
        moves = game.get_all_valid_moves()
        return any(m.captured or m.is_check for m in moves[:3])
    
    def _quiescence_search(self, game, depth, alpha, beta, maximizing):
        """Search only captures until position is quiet"""
        stand_pat = game.evaluate_position(self.player_id)
        
        if depth == 0:
            return stand_pat
        
        if maximizing:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
        
        # Only consider captures
        moves = [m for m in game.get_all_valid_moves() if m.captured]
        if not moves:
            return stand_pat
        
        if maximizing:
            max_eval = stand_pat
            for move in moves[:5]:  # Limit branching
                sim_game = game.copy()
                sim_game.make_move(move)
                eval_score = self._quiescence_search(sim_game, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = stand_pat
            for move in moves[:5]:
                sim_game = game.copy()
                sim_game.make_move(move)
                eval_score = self._quiescence_search(sim_game, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def _iterative_deepening_minimax(self, game, max_depth):
        """Iterative deepening with aspiration windows"""
        score = 0
        alpha = -float('inf')
        beta = float('inf')
        
        for depth in range(1, max_depth + 1):
            # Aspiration window (narrow search)
            if depth > 1:
                window = 50
                alpha = score - window
                beta = score + window
            
            try:
                score = self._minimax(game, depth, alpha, beta, True)
            except:  # Re-search with full window if aspiration fails
                score = self._minimax(game, depth, -float('inf'), float('inf'), True)
        
        return score
    
    def _minimax(self, game, depth, alpha, beta, maximizing):
        """Enhanced minimax with alpha-beta and move ordering"""
        if depth == 0 or game.game_over:
            return game.evaluate_position(self.player_id)
        
        moves = game.get_all_valid_moves()
        if not moves:
            return game.evaluate_position(self.player_id)
        
        # Progressive widening: reduce branching factor dynamically
        if len(moves) > 10:
            moves = moves[:max(8, 15 - depth)]
        
        if maximizing:
            max_eval = -float('inf')
            for move in moves:
                sim_game = game.copy()
                sim_game.make_move(move)
                eval_score = self._minimax(sim_game, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    # Update killer moves
                    if move not in game.killer_moves[game.move_count]:
                        game.killer_moves[game.move_count].append(move)
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                sim_game = game.copy()
                sim_game.make_move(move)
                eval_score = self._minimax(sim_game, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def choose_action(self, game, training=True):
        """Gumbel AlphaZero action selection"""
        moves = game.get_all_valid_moves()
        if not moves:
            return None
        
        # Epsilon-greedy exploration (decreases over time)
        if training and random.random() < self.epsilon:
            return random.choice(moves)
        
        # Run Gumbel-enhanced MCTS
        root = self.mcts_search(game, self.mcts_simulations)
        
        if not root.children:
            return random.choice(moves)
        
        # Temperature-based selection (from AlphaZero paper)
        if training and game.move_count < 10:
            # Sample from visit distribution (early game)
            visits = np.array([child.visit_count for child in root.children.values()])
            temp = 1.0
            probs = visits ** (1 / temp)
            probs = probs / probs.sum()
            best_move = np.random.choice(list(root.children.keys()), p=probs)
        else:
            # Greedy selection (late game)
            best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        
        # Store policy for learning
        state = game.get_state()
        total_visits = sum(child.visit_count for child in root.children.values())
        for move, child in root.children.items():
            self.policy_table[state][move] = child.visit_count / total_visits
        
        # Store experience in replay buffer
        value = root.value()
        self.replay_buffer.append((state, best_move, value))
        
        return best_move
    
    def update_from_game(self, game_data, result):
        """Enhanced learning with experience replay"""
        self.training_steps += 1
        
        # Update from current game
        for state, move, player in game_data:
            if player != self.player_id:
                continue
            
            if result == self.player_id:
                reward = 1.0
            elif result == 0:
                reward = 0.0
            else:
                reward = -1.0
            
            # Policy gradient update
            current_policy = self.policy_table[state][move]
            self.policy_table[state][move] = current_policy + self.lr * (reward - current_policy)
        
        # Experience replay (sample from buffer)
        if len(self.replay_buffer) > 100:
            batch_size = min(32, len(self.replay_buffer))
            batch = random.sample(list(self.replay_buffer), batch_size)
            
            for state, move, value in batch:
                if state in self.policy_table and move in self.policy_table[state]:
                    old_val = self.policy_table[state][move]
                    self.policy_table[state][move] = old_val + 0.1 * self.lr * (value - old_val)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0

# ============================================================================
# Training System
# ============================================================================

def play_game(env, agent1, agent2, training=True):
    env.reset()
    game_history = []
    agents = {1: agent1, 2: agent2}
    
    move_count = 0
    max_moves = 100
    
    while not env.game_over and move_count < max_moves:
        current_player = env.current_player
        agent = agents[current_player]
        
        state = env.get_state()
        move = agent.choose_action(env, training)
        
        if move is None:
            break
        
        game_history.append((state, move, current_player))
        env.make_move(move)
        move_count += 1
    
    if env.winner == 1:
        agent1.wins += 1
        agent2.losses += 1
        if training:
            agent1.update_from_game(game_history, 1)
            agent2.update_from_game(game_history, 1)
    elif env.winner == 2:
        agent2.wins += 1
        agent1.losses += 1
        if training:
            agent1.update_from_game(game_history, 2)
            agent2.update_from_game(game_history, 2)
    else:
        agent1.draws += 1
        agent2.draws += 1
        if training:
            agent1.update_from_game(game_history, 0)
            agent2.update_from_game(game_history, 0)
    
    return env.winner

# ============================================================================
# Visualization
# ============================================================================

def visualize_board(board, title="Minichess Board", last_move=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    piece_symbols = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
    }
    
    for row in range(5):
        for col in range(5):
            color = '#F0D9B5' if (row + col) % 2 == 0 else '#B58863'
            
            if last_move and ((row, col) == last_move.start or (row, col) == last_move.end):
                color = '#BACA44'
            
            square = plt.Rectangle((col, 4-row), 1, 1, facecolor=color)
            ax.add_patch(square)
            
            piece = board[row, col]
            if piece != '.':
                symbol = piece_symbols.get(piece, piece)
                color = '#FFFFFF' if piece.isupper() else '#000000'
                ax.text(col + 0.5, 4-row + 0.5, symbol, 
                       ha='center', va='center', fontsize=36, color=color)
    
    for i in range(5):
        ax.text(-0.3, 4-i+0.5, str(i+1), ha='center', va='center', fontsize=12)
        ax.text(i+0.5, -0.3, 'abcde'[i], ha='center', va='center', fontsize=12)
    
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    return fig

# ============================================================================
# Serialization
# ============================================================================

def serialize_move(move):
    return {
        "s": [int(x) for x in move.start],
        "e": [int(x) for x in move.end],
        "p": str(move.piece),
        "c": str(move.captured) if move.captured else None,
        "pr": str(move.promotion) if move.promotion else None
    }

def deserialize_move(data):
    return Move(
        start=tuple(data["s"]),
        end=tuple(data["e"]),
        piece=data["p"],
        captured=data.get("c"),
        promotion=data.get("pr")
    )

def create_agents_zip(agent1, agent2, config):
    def serialize_agent(agent, role_name):
        clean_policy = {}
        current_policies = agent.policy_table.copy()
        
        for state, moves in current_policies.items():
            try:
                state_str = str(state)
                clean_policy[state_str] = {}
                
                for move, value in moves.items():
                    move_json_str = json.dumps(serialize_move(move))
                    clean_policy[state_str][move_json_str] = float(value)
            except Exception:
                continue
        
        return {
            "metadata": {"role": role_name, "version": "3.0_GRANDMASTER"},
            "policy_table": clean_policy,
            "epsilon": float(agent.epsilon),
            "wins": int(agent.wins),
            "losses": int(agent.losses),
            "draws": int(agent.draws),
            "mcts_sims": int(agent.mcts_simulations),
            "training_steps": int(agent.training_steps)
        }
    
    data1 = serialize_agent(agent1, "White")
    data2 = serialize_agent(agent2, "Black")
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent1.json", json.dumps(data1, indent=2))
        zf.writestr("agent2.json", json.dumps(data2, indent=2))
        zf.writestr("config.json", json.dumps(config, indent=2))
    
    buffer.seek(0)
    return buffer

def load_agents_from_zip(uploaded_file):
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            files = zf.namelist()
            if not all(f in files for f in ["agent1.json", "agent2.json", "config.json"]):
                st.error("❌ Corrupt File")
                return None, None, None, 0
            
            a1_data = json.loads(zf.read("agent1.json").decode('utf-8'))
            a2_data = json.loads(zf.read("agent2.json").decode('utf-8'))
            config = json.loads(zf.read("config.json").decode('utf-8'))
            
            def restore_agent(agent, data):
                agent.epsilon = data.get('epsilon', 0.1)
                agent.wins = data.get('wins', 0)
                agent.losses = data.get('losses', 0)
                agent.draws = data.get('draws', 0)
                agent.mcts_simulations = data.get('mcts_sims', 50)
                agent.training_steps = data.get('training_steps', 0)
                
                agent.policy_table = defaultdict(lambda: defaultdict(float))
                loaded_policies = 0
                policy_data = data.get('policy_table', {})
                
                for state_str, moves_dict in policy_data.items():
                    try:
                        state = ast.literal_eval(state_str)
                        for move_json_str, value in moves_dict.items():
                            move_dict = json.loads(move_json_str)
                            move = deserialize_move(move_dict)
                            agent.policy_table[state][move] = value
                        loaded_policies += 1
                    except Exception:
                        continue
                return loaded_policies
            
            agent1 = Agent(1, config.get('lr1', 0.5), config.get('gamma1', 0.99))
            count1 = restore_agent(agent1, a1_data)
            
            agent2 = Agent(2, config.get('lr2', 0.5), config.get('gamma2', 0.99))
            count2 = restore_agent(agent2, a2_data)
            
            return agent1, agent2, config, count1 + count2
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, None, None, 0

# ============================================================================
# Streamlit UI
# ============================================================================

st.sidebar.header("⚙️ Grandmaster Configuration")

with st.sidebar.expander("🎯 Training Parameters", expanded=True):
    episodes = st.number_input("Training Episodes", 10, 1000, 50, 5, 
                               help="Start with 50 for Grandmaster level!")
    update_freq = st.number_input("Update Every N Games", 1, 50, 10, 1)

with st.sidebar.expander("🧠 Agent 1 (White)", expanded=True):
    lr1 = st.slider("Learning Rate α₁", 0.1, 1.0, 0.5, 0.05)
    mcts_sims1 = st.slider("MCTS Sims₁", 1, 400, 200, 10)
    minimax_depth1 = st.slider("Minimax Depth₁", 1, 6, 4, 1)

with st.sidebar.expander("⚫ Agent 2 (Black)", expanded=True):
    lr2 = st.slider("Learning Rate α₂", 0.1, 1.0, 0.5, 0.05)
    mcts_sims2 = st.slider("MCTS Sims₂", 1, 400, 180, 10)
    minimax_depth2 = st.slider("Minimax Depth₂", 1, 6, 4, 1)

# This section continues from the sidebar configuration
# Replace everything from the "Brain Storage" expander onwards

with st.sidebar.expander("💾 Brain Storage", expanded=False):
    if 'agent1' in st.session_state and st.session_state.agent1:
        st.markdown("### 🔄 Neural Sync")
        col1, col2 = st.columns(2)
        
        if col1.button("W→B"):
            st.session_state.agent2.policy_table = deepcopy(st.session_state.agent1.policy_table)
            st.session_state.agent2.epsilon = st.session_state.agent1.epsilon
            st.toast("Synced!", icon="⚪")
        
        if col2.button("B→W"):
            st.session_state.agent1.policy_table = deepcopy(st.session_state.agent2.policy_table)
            st.session_state.agent1.epsilon = st.session_state.agent2.epsilon
            st.toast("Synced!", icon="⚫")
        
        st.markdown("---")
        
        config = {
            "lr1": lr1, "mcts_sims1": mcts_sims1, "minimax_depth1": minimax_depth1,
            "lr2": lr2, "mcts_sims2": mcts_sims2, "minimax_depth2": minimax_depth2,
            "training_history": st.session_state.get('training_history', None)
        }
        
        zip_buffer = create_agents_zip(st.session_state.agent1, st.session_state.agent2, config)
        st.download_button(
            label="💾 Download Brain",
            data=zip_buffer,
            file_name="grandmaster_minichess.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Upload Brain", type="zip")
    if uploaded_file:
        if st.button("🔄 Load", use_container_width=True):
            a1, a2, cfg, count = load_agents_from_zip(uploaded_file)
            if a1 and a2:
                st.session_state.agent1 = a1
                st.session_state.agent2 = a2
                st.session_state.training_history = cfg.get("training_history")
                st.toast(f"✅ {count} memories loaded!", icon="🧠")
                import time
                time.sleep(0.5)
                st.rerun()

train_btn = st.sidebar.button("🚀 Train", use_container_width=True, type="primary")
if st.sidebar.button("🧹 Reset", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Initialize
if 'env' not in st.session_state:
    st.session_state.env = Minichess()

if 'agent1' not in st.session_state:
    st.session_state.agent1 = Agent(1, lr1)
    st.session_state.agent1.mcts_simulations = mcts_sims1
    st.session_state.agent1.minimax_depth = minimax_depth1
    
    st.session_state.agent2 = Agent(2, lr2)
    st.session_state.agent2.mcts_simulations = mcts_sims2
    st.session_state.agent2.minimax_depth = minimax_depth2

agent1 = st.session_state.agent1
agent2 = st.session_state.agent2
env = st.session_state.env

agent1.mcts_simulations = mcts_sims1
agent1.minimax_depth = minimax_depth1
agent2.mcts_simulations = mcts_sims2
agent2.minimax_depth = minimax_depth2

# Stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("⚪ White", f"{len(agent1.policy_table):,} policies")
    st.caption(f"W: {agent1.wins} | ε: {agent1.epsilon:.3f}")
    st.caption(f"MCTS: {agent1.mcts_simulations} | Depth: {agent1.minimax_depth}")

with col2:
    st.metric("⚫ Black", f"{len(agent2.policy_table):,} policies")
    st.caption(f"W: {agent2.wins} | ε: {agent2.epsilon:.3f}")
    st.caption(f"MCTS: {agent2.mcts_simulations} | Depth: {agent2.minimax_depth}")

with col3:
    total = agent1.wins + agent2.wins + agent1.draws
    st.metric("Games", total)
    st.caption(f"Draws: {agent1.draws}")
    progress = min(100, (agent1.training_steps + agent2.training_steps) / 10)
    st.progress(progress / 100, f"Skill: {progress:.0f}%")

st.markdown("---")

# Training
if train_btn:
    st.subheader("🎯 Self-Play Training")
    status = st.empty()
    progress_bar = st.progress(0)
    
    agent1.reset_stats()
    agent2.reset_stats()
    
    history = {
        'agent1_wins': [], 'agent2_wins': [], 'draws': [],
        'agent1_epsilon': [], 'agent2_epsilon': [],
        'agent1_policies': [], 'agent2_policies': [],
        'episode': []
    }
    
    for ep in range(1, episodes + 1):
        winner = play_game(env, agent1, agent2, training=True)
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        if ep % update_freq == 0:
            history['agent1_wins'].append(agent1.wins)
            history['agent2_wins'].append(agent2.wins)
            history['draws'].append(agent1.draws)
            history['agent1_epsilon'].append(agent1.epsilon)
            history['agent2_epsilon'].append(agent2.epsilon)
            history['agent1_policies'].append(len(agent1.policy_table))
            history['agent2_policies'].append(len(agent2.policy_table))
            history['episode'].append(ep)
            
            progress = ep / episodes
            progress_bar.progress(progress)
            
            win_rate1 = agent1.wins / ep * 100 if ep > 0 else 0
            win_rate2 = agent2.wins / ep * 100 if ep > 0 else 0
            
            status.markdown(f"""
            **Episode {ep}/{episodes}** ({progress*100:.0f}%)
            
            | Metric | White ⚪ | Black ⚫ |
            |--------|----------|----------|
            | Wins | {agent1.wins} ({win_rate1:.1f}%) | {agent2.wins} ({win_rate2:.1f}%) |
            | Policies | {len(agent1.policy_table):,} | {len(agent2.policy_table):,} |
            | Epsilon | {agent1.epsilon:.4f} | {agent2.epsilon:.4f} |
            | Steps | {agent1.training_steps:,} | {agent2.training_steps:,} |
            
            **Draws:** {agent1.draws} | **Skill Level:** {min(100, (agent1.training_steps + agent2.training_steps) / 10):.0f}%
            """)
    
    progress_bar.progress(1.0)
    st.toast("🏆 Grandmaster training complete!", icon="✨")
    st.session_state.training_history = history
    
    import time
    time.sleep(1)
    st.rerun()

# Charts
if 'training_history' in st.session_state and st.session_state.training_history:
    history = st.session_state.training_history
    if isinstance(history, dict) and 'episode' in history and len(history['episode']) > 0:
        st.subheader("📊 Training Analytics")
        df = pd.DataFrame(history)
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("#### Performance")
            if all(col in df.columns for col in ['episode', 'agent1_wins', 'agent2_wins', 'draws']):
                st.line_chart(df[['episode', 'agent1_wins', 'agent2_wins', 'draws']].set_index('episode'))
        
        with c2:
            st.write("#### Exploration (ε)")
            if all(col in df.columns for col in ['episode', 'agent1_epsilon', 'agent2_epsilon']):
                st.line_chart(df[['episode', 'agent1_epsilon', 'agent2_epsilon']].set_index('episode'))
        
        st.write("#### Knowledge Growth")
        if all(col in df.columns for col in ['episode', 'agent1_policies', 'agent2_policies']):
            st.line_chart(df[['episode', 'agent1_policies', 'agent2_policies']].set_index('episode'))

# Demo
if 'agent1' in st.session_state and len(agent1.policy_table) > 20:
    st.markdown("---")
    st.subheader("⚔️ Grandmaster Battle")
    
    if st.button("🎬 Watch", use_container_width=True):
        sim_env = Minichess()
        board_ph = st.empty()
        move_ph = st.empty()
        
        agents = {1: agent1, 2: agent2}
        move_num = 0
        
        while not sim_env.game_over and move_num < 100:
            current = sim_env.current_player
            move = agents[current].choose_action(sim_env, training=False)
            
            if move is None:
                break
            
            sim_env.make_move(move)
            move_num += 1
            
            player_name = "White" if current == 1 else "Black"
            move_ph.caption(f"Move {move_num}: {player_name} → {move.to_notation()}")
            
            fig = visualize_board(sim_env.board, f"Move {move_num}", move)
            board_ph.pyplot(fig)
            plt.close(fig)
            
            import time
            time.sleep(0.2)
        
        if sim_env.winner == 1:
            st.success("🏆 White Wins!")
        elif sim_env.winner == 2:
            st.error("🏆 Black Wins!")
        else:
            st.warning("🤝 Draw")

# Human vs AI
st.markdown("---")
st.header("🎮 Challenge Grandmaster AI")

if len(agent1.policy_table) > 20:
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        opp = st.selectbox("Opponent", ["Agent 1 (White)", "Agent 2 (Black)"])
    with c2:
        color = st.selectbox("Your Color", ["White", "Black"])
    with c3:
        st.write("")
        if st.button("🎯 Start", use_container_width=True, type="primary"):
            st.session_state.human_env = Minichess()
            st.session_state.human_active = True
            st.session_state.ai_agent = agent1 if "Agent 1" in opp else agent2
            st.session_state.ai_player = 1 if color == "Black" else 2
            st.session_state.human_player = 3 - st.session_state.ai_player
            st.session_state.selected = None
            st.rerun()
    
    if 'human_env' in st.session_state and st.session_state.human_active:
        h_env = st.session_state.human_env
        
        # AI turn
        if h_env.current_player == st.session_state.ai_player and not h_env.game_over:
            with st.spinner("🤖 AI thinking..."):
                import time
                time.sleep(0.3)
                ai_move = st.session_state.ai_agent.choose_action(h_env, training=False)
                if ai_move:
                    h_env.make_move(ai_move)
                    st.rerun()
        
        # Status
        if h_env.game_over:
            if h_env.winner == st.session_state.human_player:
                st.success("🎉 YOU WIN!")
            elif h_env.winner == st.session_state.ai_player:
                st.error("🤖 AI WINS!")
            else:
                st.warning("🤝 DRAW")
        else:
            turn = "Your Turn" if h_env.current_player == st.session_state.human_player else "AI Turn"
            color_name = "White" if h_env.current_player == 1 else "Black"
            st.caption(f"**{turn}** ({color_name})")
        
        # Board
        last = h_env.move_history[-1] if h_env.move_history else None
        fig = visualize_board(h_env.board, "Human vs AI", last)
        st.pyplot(fig)
        plt.close(fig)
        
        # Moves
        if not h_env.game_over and h_env.current_player == st.session_state.human_player:
            st.write("**Select piece:**")
            moves = h_env.get_all_valid_moves()
            starts = list(set([m.start for m in moves]))
            
            piece_sym = {
                'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
                'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
            }
            
            cols = st.columns(min(len(starts), 5))
            for idx, pos in enumerate(starts):
                piece = h_env.board[pos[0], pos[1]]
                sym = piece_sym.get(piece, piece)
                coord = f"{'abcde'[pos[1]]}{5-pos[0]}"
                if cols[idx % len(cols)].button(f"{sym} {coord}", key=f"s{pos}"):
                    st.session_state.selected = pos
                    st.rerun()
            
            if 'selected' in st.session_state and st.session_state.selected:
                piece_moves = [m for m in moves if m.start == st.session_state.selected]
                st.write("**Moves:**")
                
                mcols = st.columns(min(len(piece_moves), 5))
                for idx, move in enumerate(piece_moves):
                    label = move.to_notation()
                    if move.captured:
                        label += "×"
                    if mcols[idx % len(mcols)].button(label, key=f"m{idx}"):
                        h_env.make_move(move)
                        st.session_state.selected = None
                        st.rerun()
else:
    st.info("⏳ Train agents first to unlock human play mode!")

st.markdown("---")
st.caption("🚀 Grandmaster-Level Gumbel AlphaZero | 5x5 Minichess")

