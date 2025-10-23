#!/usr/bin/env python3
"""
TicTacToe AI Trainer with Deep Q-Learning

This module implements a customizable TicTacToe game environment and a Deep Q-Learning agent
that learns to play the game through self-play. It includes:
- Customizable grid size and win streak length
- Configurable training parameters
- Console GUI for visualizing training progress
- Model save/load functionality
- Play mode to compete against the trained model
- Tuple-based position input for moves
"""

import numpy as np
import argparse
import json
import os
import sys
import time
import logging
from collections import deque
import random
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    print("PyTorch not found. Please install it with: pip install torch")
    sys.exit(1)

# colorama for colored console output
from colorama import Fore as color, Style as style, init as colorama_init
colorama_init()


class Logger:
    """Enhanced logging utility for training"""
    
    def __init__(self, config):
        self.config = config
        self.logger = None
        if config.get('logging', {}).get('enabled', False):
            self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_file = log_config.get('log_file', 'training.log')
        log_level = getattr(logging, log_config.get('log_level', 'INFO'))
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        log_path = os.path.join('logs', log_file)
        
        # Setup logger
        self.logger = logging.getLogger('tictactoe_training')
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        if self.logger:
            self.logger.info(message)
    
    def warning(self, message):
        if self.logger:
            self.logger.warning(message)
        else:
            print(f"WARNING: {message}")
    
    def error(self, message):
        if self.logger:
            self.logger.error(message)
        else:
            print(f"ERROR: {message}")


class CheckpointManager:
    """Manages model checkpointing and best model tracking"""
    
    def __init__(self, config):
        self.config = config
        self.checkpoint_config = config.get('checkpointing', {})
        self.enabled = self.checkpoint_config.get('enabled', False)
        self.checkpoint_dir = self.checkpoint_config.get('checkpoint_dir', 'checkpoints')
        self.keep_best = self.checkpoint_config.get('keep_best', True)
        self.metric = self.checkpoint_config.get('metric', 'win_rate')
        self.best_metric = -float('inf')
        self.best_model_path = None
        
        if self.enabled:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def should_save_checkpoint(self, episode, save_interval):
        """Check if we should save a checkpoint"""
        return self.enabled and (episode + 1) % save_interval == 0
    
    def save_checkpoint(self, agent, episode, metrics, logger=None):
        """Save model checkpoint"""
        if not self.enabled:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_ep{episode+1}_{timestamp}.pth"
        )
        
        # Save checkpoint
        agent.save(checkpoint_path)
        
        # Check if this is the best model
        current_metric = metrics.get(self.metric, -float('inf'))
        if self.keep_best and current_metric > self.best_metric:
            self.best_metric = current_metric
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)  # Remove old best model
            
            self.best_model_path = os.path.join(
                self.checkpoint_dir,
                f"best_model_{self.metric}_{current_metric:.3f}.pth"
            )
            agent.save(self.best_model_path)
            
            if logger:
                logger.info(f"New best model saved: {self.metric}={current_metric:.3f}")
        
        if logger:
            logger.info(f"Checkpoint saved: {checkpoint_path}")


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, config):
        self.config = config.get('early_stopping', {})
        self.enabled = self.config.get('enabled', False)
        self.patience = self.config.get('patience', 1000)
        self.min_delta = self.config.get('min_delta', 0.01)
        self.best_score = -float('inf')
        self.counter = 0
    
    def should_stop(self, current_score):
        """Check if training should stop early"""
        if not self.enabled:
            return False
        
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


class MetricsTracker:
    """Enhanced metrics tracking"""
    
    def __init__(self):
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'win_rates': deque(maxlen=100),
            'eval_win_rates': [],
            'eval_draw_rates': [],
            'eval_loss_rates': [],
            'learning_rates': [],
            'epsilon_values': []
        }
    
    def update(self, episode, reward, length, loss, result, eval_metrics=None, lr=None, epsilon=None):
        """Update all metrics"""
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_lengths'].append(length)
        if loss is not None:
            self.metrics['losses'].append(loss)
        
        # Track wins
        if result.get('winner') == 1:
            self.metrics['win_rates'].append(1)
        elif result.get('draw'):
            self.metrics['win_rates'].append(0.5)
        else:
            self.metrics['win_rates'].append(0)
        
        # Evaluation metrics
        if eval_metrics:
            self.metrics['eval_win_rates'].append(eval_metrics[0])
            self.metrics['eval_draw_rates'].append(eval_metrics[1])
            self.metrics['eval_loss_rates'].append(eval_metrics[2])
        
        # Training parameters
        if lr is not None:
            self.metrics['learning_rates'].append(lr)
        if epsilon is not None:
            self.metrics['epsilon_values'].append(epsilon)
    
    def get_current_metrics(self):
        """Get current metrics for checkpointing"""
        return {
            'win_rate': np.mean(list(self.metrics['win_rates'])) if self.metrics['win_rates'] else 0,
            'avg_reward': np.mean(self.metrics['episode_rewards'][-100:]) if self.metrics['episode_rewards'] else 0,
            'avg_loss': np.mean(self.metrics['losses'][-100:]) if self.metrics['losses'] else 0,
            'eval_win_rate': self.metrics['eval_win_rates'][-1] if self.metrics['eval_win_rates'] else 0
        }

class TicTacToe:
    """Customizable TicTacToe game environment"""
    
    def __init__(self, grid_size=3, win_length=3):
        self.grid_size = grid_size
        self.win_length = win_length
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.current_player = 1
        
    def reset(self):
        """Reset the game board"""
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.current_player = 1
        return self.board.copy()
    
    def get_valid_moves(self):
        """Get list of valid (row, col) positions"""
        valid_moves = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.board[row, col] == 0:
                    valid_moves.append((row, col))
        return valid_moves
    
    def make_move(self, position):
        """
        Make a move at the specified position (row, col)
        Returns: (new_state, reward, done, info)
        """
        if isinstance(position, int):
            # Convert flat index to (row, col)
            row, col = divmod(position, self.grid_size)
        else:
            row, col = position
        
        if (row, col) not in self.get_valid_moves():
            return self.board.copy(), -10, True, {"invalid_move": True}
        
        # Place the piece
        self.board[row, col] = self.current_player
        
        # Check for win
        if self._check_win(self.current_player):
            return self.board.copy(), 1, True, {"winner": self.current_player}
        
        # Check for draw
        if len(self.get_valid_moves()) == 0:
            return self.board.copy(), 0, True, {"draw": True}
        
        # Switch player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        return self.board.copy(), 0, False, {}
    
    def _check_win(self, player):
        """Check if the specified player has won"""
        # Check all possible directions for win_length in a row
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal down-right
            (1, -1),  # diagonal down-left
        ]
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.board[row, col] == player:
                    for dr, dc in directions:
                        if self._check_line(row, col, dr, dc, player):
                            return True
        return False
    
    def _check_line(self, start_row, start_col, dr, dc, player):
        """Check if there's a winning line starting from (start_row, start_col) in direction (dr, dc)"""
        count = 0
        row, col = start_row, start_col
        
        while (0 <= row < self.grid_size and 0 <= col < self.grid_size and 
               self.board[row, col] == player):
            count += 1
            if count >= self.win_length:
                return True
            row += dr
            col += dc
        
        return False

    def get_winning_positions(self, player):
        """Return list of (row,col) tuples for a winning alignment, or [] if none."""
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal down-right
            (1, -1),  # diagonal down-left
        ]
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.board[row, col] == player:
                    for dr, dc in directions:
                        positions = self._get_line_positions(row, col, dr, dc, player)
                        if len(positions) >= self.win_length:
                            return positions[:self.win_length]
        return []
    
    def _get_line_positions(self, start_row, start_col, dr, dc, player):
        """Get all consecutive positions in a line for the given player"""
        positions = []
        row, col = start_row, start_col
        
        while (0 <= row < self.grid_size and 0 <= col < self.grid_size and 
               self.board[row, col] == player):
            positions.append((row, col))
            row += dr
            col += dc
        
        return positions
    
    def render(self):
        """Render the game board to console with colored tokens"""
        # detect winning positions for highlighting
        win_pos_p1 = self.get_winning_positions(1)
        win_pos_p2 = self.get_winning_positions(2)
        win_set = set(win_pos_p1 + win_pos_p2)
        
        print()
        # Print column headers
        header = "   "
        for col in range(self.grid_size):
            header += f"{col:2} "
        print(header)
        
        # Print separator
        print("  " + "-" * (self.grid_size * 3 + 1))
        
        for row in range(self.grid_size):
            row_str = f"{row} |"
            for col in range(self.grid_size):
                cell = self.board[row, col]
                token = " "
                if cell == 1:
                    token = "X"
                    token = color.RED + token + style.RESET_ALL
                elif cell == 2:
                    token = "O"
                    token = color.BLUE + token + style.RESET_ALL
                # override if part of winning line
                if (row, col) in win_set:
                    # color winning tokens green regardless of X/O color
                    token = color.GREEN + (self.board[row, col] and ("X" if self.board[row, col] == 1 else "O")) + style.RESET_ALL
                row_str += f" {token} "
            row_str += "|"
            print(row_str)
        
        # Print bottom separator
        print("  " + "-" * (self.grid_size * 3 + 1))
        print()


class DQN(nn.Module):
    """Deep Q-Network for TicTacToe"""
    
    def __init__(self, input_size, hidden_size=128, output_size=9):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Learning Agent for TicTacToe"""
    
    def __init__(self, config, grid_size=3):
        self.config = config
        self.grid_size = grid_size
        self.input_size = grid_size * grid_size
        self.output_size = grid_size * grid_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQN(self.input_size, output_size=self.output_size).to(self.device)
        self.target_net = DQN(self.input_size, output_size=self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Training parameters
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        # Enhanced learning rate scheduling
        lr_config = config.get('learning_rate_schedule', {})
        if lr_config.get('enabled', False):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=lr_config.get('factor', 0.8), 
                patience=lr_config.get('patience', 100),
                min_lr=lr_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        self.losses = []
        
        # Replay memory
        self.memory = deque(maxlen=config.get('replay_memory_size', 10000))
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.gamma = config.get('gamma', 0.99)
        self.batch_size = config.get('batch_size', 64)
        self.target_update = config.get('target_update', 10)

        # Stability
        self.loss_fn = nn.SmoothL1Loss()
        self.gradient_clip = config.get('gradient_clip', 1.0)
    
    def _state_to_input(self, state):
        """Normalize board to -1/0/+1 tensor: player1=+1, player2=-1, empty=0"""
        p1 = (state == 1).astype(np.float32)
        p2 = (state == 2).astype(np.float32)
        mapped = p1 - p2
        return torch.from_numpy(mapped.flatten()).unsqueeze(0).to(self.device)

    def _position_to_index(self, position):
        """Convert (row, col) position to flat index"""
        if isinstance(position, tuple):
            row, col = position
            return row * self.grid_size + col
        return position

    def _index_to_position(self, index):
        """Convert flat index to (row, col) position"""
        return divmod(index, self.grid_size)

    def _check_win_board(self, board, player, win_length):
        """Quick inline win check for a numpy board"""
        grid_size = board.shape[0]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for row in range(grid_size):
            for col in range(grid_size):
                if board[row, col] == player:
                    for dr, dc in directions:
                        count = 0
                        r, c = row, col
                        while (0 <= r < grid_size and 0 <= c < grid_size and 
                               board[r, c] == player):
                            count += 1
                            if count >= win_length:
                                return True
                            r += dr
                            c += dc
        return False

    def _find_winning_move_local(self, board, player, valid_moves, win_length):
        """Return a winning position for player if exists, else None"""
        for position in valid_moves:
            row, col = position
            test = board.copy()
            test[row, col] = player
            if self._check_win_board(test, player, win_length):
                return position
        return None

    def _count_immediate_wins(self, board, player, win_length):
        """Count how many immediate winning moves `player` has on `board`."""
        count = 0
        grid_size = board.shape[0]
        for row in range(grid_size):
            for col in range(grid_size):
                if board[row, col] != 0:
                    continue
                test = board.copy()
                test[row, col] = player
                if self._check_win_board(test, player, win_length):
                    count += 1
        return count

    def _find_fork_moves(self, board, player, valid_moves, win_length):
        """Return list of positions where player would create a fork (>=2 immediate wins)."""
        forks = []
        for position in valid_moves:
            row, col = position
            test = board.copy()
            test[row, col] = player
            wins_after = self._count_immediate_wins(test, player, win_length)
            if wins_after >= 2:
                forks.append(position)
        return forks

    def _find_center_moves(self, valid_moves):
        """Find center positions (prefer center for opening strategy)"""
        center = self.grid_size // 2
        center_moves = []
        for position in valid_moves:
            row, col = position
            distance_from_center = abs(row - center) + abs(col - center)
            center_moves.append((position, distance_from_center))
        center_moves.sort(key=lambda x: x[1])
        return [pos for pos, _ in center_moves]

    def get_action(self, state, valid_moves, training=True, win_length=3):
        """Get action using epsilon-greedy policy with tactical reasoning"""
        # 1) Winning move for us
        win_move = self._find_winning_move_local(state, 1, valid_moves, win_length)
        if win_move is not None:
            return self._position_to_index(win_move)

        # 2) Block opponent immediate win
        block_move = self._find_winning_move_local(state, 2, valid_moves, win_length)
        if block_move is not None:
            return self._position_to_index(block_move)

        # 3) Block opponent forks
        opponent_forks = self._find_fork_moves(state, 2, valid_moves, win_length)
        if opponent_forks:
            return self._position_to_index(opponent_forks[0])

        # 4) Create our own fork
        my_forks = self._find_fork_moves(state, 1, valid_moves, win_length)
        if my_forks:
            return self._position_to_index(my_forks[0])

        # 5) Center preference
        center_moves = self._find_center_moves(valid_moves)
        if center_moves and ((not training) or random.random() < 0.7):
            return self._position_to_index(center_moves[0])

        # 6) Exploration
        if training and random.random() < self.epsilon:
            position = random.choice(valid_moves)
            return self._position_to_index(position)

        # 7) Model action
        with torch.no_grad():
            state_tensor = self._state_to_input(state)
            was_training = self.policy_net.training
            self.policy_net.eval()
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            if was_training:
                self.policy_net.train()

        # Mask invalid moves
        masked_q = np.full(self.output_size, -np.inf, dtype=np.float32)
        for position in valid_moves:
            index = self._position_to_index(position)
            masked_q[index] = q_values[index]

        return int(np.argmax(masked_q))
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        reward = max(-1.0, min(1.0, float(reward)))
        self.memory.append((state.copy(), action, reward, next_state.copy(), done))
    
    def replay(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        
        states_np = np.array([ (s == 1).astype(np.float32) - (s == 2).astype(np.float32) for s, _, _, _, _ in batch ])
        next_states_np = np.array([ (s == 1).astype(np.float32) - (s == 2).astype(np.float32) for _, _, _, s, _ in batch ])
        
        states = torch.FloatTensor(states_np.reshape(self.batch_size, -1)).to(self.device)
        actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(self.device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(self.device)
        next_states = torch.FloatTensor(next_states_np.reshape(self.batch_size, -1)).to(self.device)
        dones = torch.FloatTensor([float(d) for _, _, _, _, d in batch]).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.loss_fn(current_q, target_q.detach())
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("Detected NaN/Inf loss -- skipping step")
            return float('nan')
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        if self.scheduler and len(self.losses) >= 10 and len(self.losses) % 10 == 0:
            avg_loss = sum(self.losses[-10:]) / 10
            try:
                self.scheduler.step(avg_loss)
            except Exception as e:
                print(f"[WARN] scheduler.step failed: {e}")
        
        # Value scaling
        with torch.no_grad():
            scale_factor = 100.0
            if current_q.abs().max().item() > scale_factor:
                scale = scale_factor / current_q.abs().max().item()
                for param in self.policy_net.parameters():
                    param.data.mul_(scale)
                for param in self.target_net.parameters():
                    param.data.mul_(scale)
                print(f"[INFO] Rescaled network weights by factor {scale:.5f}")
        
        return loss.item()
    
    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath):
        """Save the model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'grid_size': self.grid_size,
        }, filepath)
    
    def load(self, filepath):
        """Load the model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        # Check if grid_size matches
        saved_grid_size = checkpoint.get('grid_size', 3)
        if saved_grid_size != self.grid_size:
            print(f"Warning: Model was trained on {saved_grid_size}x{saved_grid_size} grid, but current grid is {self.grid_size}x{self.grid_size}")


class TrainingVisualizer:
    """Console-based training progress visualizer"""
    
    def __init__(self):
        self.metrics_tracker = MetricsTracker()
        
    def update(self, episode, reward, length, loss, result, eval_metrics=None, lr=None, epsilon=None):
        """Update statistics"""
        self.metrics_tracker.update(episode, reward, length, loss, result, eval_metrics, lr, epsilon)
    
    def render(self, episode, epsilon, total_episodes, time_info=None, grid_size=3, win_length=3, current_lr=None):
        """Render training progress to console"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 60)
        print(f"TICTACTOE AI ({grid_size}x{grid_size}, win={win_length}) - TRAINING PROGRESS")
        print("=" * 60)
        print(f"\nEpisode: {episode + 1}/{total_episodes}")
        print(f"Progress: [{'#' * int(50 * (episode + 1) / total_episodes):<50}] {100 * (episode + 1) / total_episodes:.1f}%")
        print(f"\nEpsilon (Exploration Rate): {epsilon:.4f}")
        
        if current_lr:
            print(f"Learning Rate: {current_lr:.2e}")
        
        if time_info:
            print(f"\nTime: {time_info}")
        
        metrics = self.metrics_tracker.metrics
        if len(metrics['episode_rewards']) > 0:
            recent_rewards = metrics['episode_rewards'][-100:]
            recent_lengths = metrics['episode_lengths'][-100:]
            recent_losses = metrics['losses'][-100:] if metrics['losses'] else [0]
            
            print(f"\nRecent 100 Episodes Statistics:")
            print(f"  Average Reward: {np.mean(recent_rewards):.2f}")
            print(f"  Average Length: {np.mean(recent_lengths):.2f}")
            print(f"  Average Loss: {np.mean(recent_losses):.4f}")
            
            if len(metrics['win_rates']) > 0:
                win_rate = np.mean(list(metrics['win_rates']))
                print(f"  Win Rate: {win_rate * 100:.1f}%")
                
                # Simple bar chart
                bar_length = int(win_rate * 40)
                print(f"\n  Win Rate Visualization:")
                print(f"  0% [{'█' * bar_length}{'-' * (40 - bar_length)}] 100%")
            
            # Show evaluation metrics if available
            if metrics['eval_win_rates']:
                latest_eval = metrics['eval_win_rates'][-1]
                print(f"\n  Latest Evaluation Win Rate: {latest_eval * 100:.1f}%")
        
        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop training")
        print("=" * 60)


def heuristic_opponent_action(state, env, player=1):
    """Simple greedy opponent for TicTacToe"""
    # Check winning move for player
    win_move = env._find_winning_move_local if hasattr(env, '_find_winning_move_local') else None
    if win_move:
        win = win_move(state, player, env.get_valid_moves(), env.win_length)
        if win is not None:
            return win
    
    # Block opponent
    other = 3 - player
    if win_move:
        block = win_move(state, other, env.get_valid_moves(), env.win_length)
        if block is not None:
            return block
    
    # Center preference
    valid_moves = env.get_valid_moves()
    center = env.grid_size // 2
    if (center, center) in valid_moves:
        return (center, center)
    
    # Corners
    corners = [(0, 0), (0, env.grid_size-1), (env.grid_size-1, 0), (env.grid_size-1, env.grid_size-1)]
    corner_moves = [pos for pos in corners if pos in valid_moves]
    if corner_moves:
        return random.choice(corner_moves)
    
    # Random fallback
    return random.choice(valid_moves)


def evaluate_agent(agent, env, games=50):
    """Evaluate agent against heuristic opponent"""
    agent_wins = 0
    draws = 0
    losses = 0

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    was_training = agent.policy_net.training
    agent.policy_net.eval()

    for _ in range(games):
        state = env.reset()
        done = False
        while not done:
            if env.current_player == 1:
                # heuristic opponent move
                position = heuristic_opponent_action(state, env, player=1)
                state, reward, done, info = env.make_move(position)
            else:
                # AI move (player 2)
                flipped = state.copy()
                flipped[flipped == 1] = 3
                flipped[flipped == 2] = 1
                flipped[flipped == 3] = 2
                valid = env.get_valid_moves()
                action_index = agent.get_action(flipped, valid, training=False, win_length=env.win_length)
                position = agent._index_to_position(action_index)
                state, reward, done, info = env.make_move(position)
        
        if info.get('winner') == 2:
            agent_wins += 1
        elif info.get('winner') == 1:
            losses += 1
        else:
            draws += 1

    agent.epsilon = original_epsilon
    agent.policy_net.train(was_training)

    win_rate = agent_wins / games
    draw_rate = draws / games
    loss_rate = losses / games
    return win_rate, draw_rate, loss_rate


def train(config):
    """Train the TicTacToe AI"""
    grid_size = config.get('grid_size', 3)
    win_length = config.get('win_length', 3)
    
    print("Initializing training...")
    print(f"Grid size: {grid_size}x{grid_size}, Win length: {win_length}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize utilities
    logger = Logger(config)
    checkpoint_manager = CheckpointManager(config)
    early_stopping = EarlyStopping(config)
    
    # Create environment and agent
    env = TicTacToe(grid_size=grid_size, win_length=win_length)
    agent = DQNAgent(config, grid_size=grid_size)
    visualizer = TrainingVisualizer()

    # Training settings
    num_episodes = config.get('num_episodes', 1000)
    save_interval = config.get('save_interval', 100)
    model_path = config.get('model_path', 'tictactoe_model.pth')
    warmup_episodes = config.get('warmup_episodes', 500)
    
    # Evaluation settings
    eval_interval = config.get('eval_interval', 500)
    eval_games = config.get('eval_games', 50)
    eval_history = deque(maxlen=20)
    
    start_time = time.time()
    logger.info(f"Starting TicTacToe training: {grid_size}x{grid_size}, win_length={win_length}, {num_episodes} episodes")
    
    def _fmt_time(seconds):
        s = int(max(0, seconds))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        if h > 0:
            return f"{h}h {m:02d}m {sec:02d}s"
        if m > 0:
            return f"{m}m {sec:02d}s"
        return f"{sec}s"
    
    try:
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            episode_length = 0
            done = False
            episode_losses = []
            
            while not done:
                valid_moves = env.get_valid_moves()
                
                # Agent 1 (learning agent)
                action_index = agent.get_action(state, valid_moves, win_length=win_length)
                position = agent._index_to_position(action_index)
                next_state, reward, done, info = env.make_move(position)
                
                # Store experience from agent 1's perspective
                agent.remember(state, action_index, reward, next_state, done)
                total_reward += reward
                episode_length += 1
                
                if done:
                    break
                
                # Agent 2 (opponent - also the learning agent)
                flipped_state = next_state.copy()
                flipped_state[flipped_state == 1] = 3
                flipped_state[flipped_state == 2] = 1
                flipped_state[flipped_state == 3] = 2
                
                valid_moves = env.get_valid_moves()
                opponent_action_index = agent.get_action(flipped_state, valid_moves, win_length=win_length)
                opponent_position = agent._index_to_position(opponent_action_index)
                next_state, opponent_reward, done, info = env.make_move(opponent_position)
                
                # Store experience from agent 2's perspective
                flipped_next_state = next_state.copy()
                flipped_next_state[flipped_next_state == 1] = 3
                flipped_next_state[flipped_next_state == 2] = 1
                flipped_next_state[flipped_next_state == 3] = 2
                
                agent.remember(flipped_state, opponent_action_index, opponent_reward, flipped_next_state, done)
                
                # If opponent wins, agent 1 gets negative reward
                if done and info.get('winner') == 2:
                    total_reward -= 1
                
                state = next_state
            
            # Train the agent
            loss = agent.replay()
            if loss:
                episode_losses.append(loss)
            
            agent.update_epsilon()
            
            # Update target network
            if episode % agent.target_update == 0:
                agent.update_target_network()
            
            # Get current learning rate
            current_lr = agent.optimizer.param_groups[0]['lr'] if agent.scheduler else None
            
            # Update visualizer
            avg_loss = np.mean(episode_losses) if episode_losses else None
            eval_metrics = None
            
            # Time bookkeeping and ETA
            elapsed = time.time() - start_time
            episodes_done = episode + 1
            avg_per_ep = elapsed / episodes_done
            remaining_eps = max(0, num_episodes - episodes_done)
            est_remaining = avg_per_ep * remaining_eps
            time_info = f"{_fmt_time(elapsed)} / {_fmt_time(est_remaining)}"
            
            # Periodic evaluation (skip during warmup)
            if (episode + 1) % eval_interval == 0 and episode >= warmup_episodes:
                win_rate, draw_rate, loss_rate = evaluate_agent(agent, env, games=eval_games)
                eval_history.append(win_rate)
                eval_metrics = (win_rate, draw_rate, loss_rate)
                smoothed = float(np.mean(eval_history))
                
                # Log evaluation results
                logger.info(f"Evaluation ep {episode+1}: win={win_rate:.3f}, draw={draw_rate:.3f}, loss={loss_rate:.3f}")
                print(f"\n[EVAL] Episodes {episode+1-eval_interval+1}-{episode+1}: win={win_rate:.3f}, draw={draw_rate:.3f}, loss={loss_rate:.3f}")
                print(f"[EVAL] Smoothed win rate (last {len(eval_history)} evals): {smoothed:.3f}")
                
                # Check early stopping
                if early_stopping.should_stop(smoothed):
                    logger.info(f"Early stopping triggered at episode {episode+1}")
                    print(f"\nEarly stopping triggered! Best smoothed win rate: {early_stopping.best_score:.3f}")
                    break
                
                time.sleep(0.2)
            
            # Update visualizer with all metrics
            visualizer.update(episode, total_reward, episode_length, avg_loss, info, eval_metrics, current_lr, agent.epsilon)
            
            # Render progress
            visualizer.render(episode, agent.epsilon, num_episodes, time_info, grid_size, win_length, current_lr)
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                agent.save(model_path)
                logger.info(f"Model saved at episode {episode + 1}")
                print(f"\nModel saved at episode {episode + 1}")
                time.sleep(0.5)
            
            # Checkpoint management
            if checkpoint_manager.should_save_checkpoint(episode, save_interval):
                current_metrics = visualizer.metrics_tracker.get_current_metrics()
                checkpoint_manager.save_checkpoint(agent, episode, current_metrics, logger)
        
        # Final save and metrics
        agent.save(model_path)
        final_metrics = visualizer.metrics_tracker.get_current_metrics()
        logger.info(f"Training completed. Final metrics: {final_metrics}")
        print(f"\n\nTraining completed! Model saved to {model_path}")
        print(f"Final win rate: {final_metrics['win_rate']:.3f}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        agent.save(model_path)
        final_metrics = visualizer.metrics_tracker.get_current_metrics()
        logger.info(f"Training interrupted. Final metrics: {final_metrics}")
        print(f"Model saved to {model_path}")


def play_against_ai(model_path, grid_size=3, win_length=3):
    """Play TicTacToe against the trained AI"""
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train a model first using the --train option.")
        return
    
    print("Loading AI model...")
    env = TicTacToe(grid_size=grid_size, win_length=win_length)
    config = {'epsilon_start': 0.0}
    agent = DQNAgent(config, grid_size=grid_size)
    agent.load(model_path)
    print("AI model loaded successfully!\n")
    
    print("=" * 60)
    print(f"TICTACTOE ({grid_size}x{grid_size}, win={win_length}) - PLAY AGAINST AI")
    print("=" * 60)
    print("\nYou are 'X' (Player 1)")
    print("AI is 'O' (Player 2)")
    print("Enter moves as (row, col) tuples, e.g., (0, 1) or 0,1\n")
    
    while True:
        state = env.reset()
        done = False
        
        while not done:
            env.render()
            
            if env.current_player == 1:
                # Human player
                valid_moves = env.get_valid_moves()
                print(f"Valid moves: {valid_moves}")
                
                while True:
                    try:
                        move_input = input("Your move (row, col): ").strip()
                        # Parse different input formats
                        if ',' in move_input:
                            if move_input.startswith('(') and move_input.endswith(')'):
                                move_input = move_input[1:-1]
                            parts = move_input.split(',')
                            row, col = int(parts[0].strip()), int(parts[1].strip())
                        else:
                            # Single number input
                            idx = int(move_input)
                            row, col = divmod(idx, grid_size)
                        
                        position = (row, col)
                        if position in valid_moves:
                            break
                        else:
                            print("Invalid move. Position is occupied or out of range.")
                    except (ValueError, KeyboardInterrupt):
                        print("\nExiting game...")
                        return
                    except Exception as e:
                        print(f"Invalid input format. Please use (row, col) format. Error: {e}")
                
                state, reward, done, info = env.make_move(position)
            else:
                # AI player
                print("AI is thinking...")
                time.sleep(0.5)
                
                # Flip the board for AI's perspective
                flipped_state = state.copy()
                flipped_state[flipped_state == 1] = 3
                flipped_state[flipped_state == 2] = 1
                flipped_state[flipped_state == 3] = 2
                
                valid_moves = env.get_valid_moves()
                action_index = agent.get_action(flipped_state, valid_moves, training=False, win_length=win_length)
                position = agent._index_to_position(action_index)
                print(f"AI plays position: {position}")
                
                state, reward, done, info = env.make_move(position)
        
        # Game over
        env.render()
        
        if info.get('winner') == 1:
            print("🎉 Congratulations! You win!")
        elif info.get('winner') == 2:
            print("🤖 AI wins! Better luck next time.")
        else:
            print("🤝 It's a draw!")
        
        play_again = input("\nPlay again? (y/n): ").lower()
        if play_again != 'y':
            break
    
    print("\nThanks for playing!")


def load_config(config_file):
    """Load configuration from JSON file"""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}


def create_default_config(config_file):
    """Create a default configuration file"""
    default_config = {
        "grid_size": 3,
        "win_length": 3,
        "num_episodes": 1000,
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "replay_memory_size": 10000,
        "batch_size": 64,
        "target_update": 5,
        "save_interval": 100,
        "model_path": "tictactoe_model.pth",
        "gradient_clip": 0.5,
        "weight_decay": 1e-4,
        "eval_interval": 500,
        "eval_games": 50,
        "warmup_episodes": 250,
        "learning_rate_schedule": {
            "enabled": True,
            "patience": 50,
            "factor": 0.8,
            "min_lr": 1e-6
        },
        "early_stopping": {
            "enabled": False,
            "patience": 1000,
            "min_delta": 0.01
        },
        "logging": {
            "enabled": True,
            "log_file": "tictactoe_training.log",
            "log_level": "INFO"
        },
        "checkpointing": {
            "enabled": True,
            "checkpoint_dir": "tictactoe_checkpoints",
            "keep_best": True,
            "metric": "win_rate"
        },
        "visualization": {
            "plot_training": False,
            "save_plots": False,
            "plot_dir": "tictactoe_plots"
        },
        "tournament": {
            "enabled": False,
            "opponents": ["random", "minimax", "heuristic"],
            "games_per_opponent": 100
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Default configuration created at {config_file}")
    return default_config


def main():
    parser = argparse.ArgumentParser(
        description="TicTacToe AI Trainer using Deep Q-Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration (3x3, win=3)
  python tictactoe_ai.py --train
  
  # Train 5x5 grid with win length 4
  python tictactoe_ai.py --train --grid-size 5 --win-length 4
  
  # Train with custom configuration file
  python tictactoe_ai.py --train --config my_config.json
  
  # Play against trained AI
  python tictactoe_ai.py --play
  
  # Play 4x4 grid with win length 3
  python tictactoe_ai.py --play --grid-size 4 --win-length 3
  
  # Evaluate trained model
  python tictactoe_ai.py --evaluate --model my_model.pth --grid-size 4 --win-length 3
  
  # Create default configuration file
  python tictactoe_ai.py --create-config
        """
    )
    
    parser.add_argument('--train', action='store_true',
                       help='Train the AI model')
    parser.add_argument('--play', action='store_true',
                       help='Play against the trained AI')
    parser.add_argument('--config', type=str, default='tictactoe_config.json',
                       help='Path to configuration file (default: tictactoe_config.json)')
    parser.add_argument('--model', type=str, default='tictactoe_model.pth',
                       help='Path to model file (default: tictactoe_model.pth)')
    parser.add_argument('--grid-size', type=int, default=3,
                       help='Grid size (default: 3 for 3x3)')
    parser.add_argument('--win-length', type=int, default=3,
                       help='Win streak length (default: 3)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create a default configuration file')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate a trained model against heuristic opponent')
    parser.add_argument('--eval-games', type=int, default=100,
                       help='Number of games for evaluation (default: 100)')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config(args.config)
        return
    
    if args.train:
        # Load or create configuration
        if os.path.exists(args.config):
            config = load_config(args.config)
            print(f"Loaded configuration from {args.config}")
        else:
            print(f"Configuration file '{args.config}' not found.")
            print("Creating default configuration...")
            config = create_default_config(args.config)
        
        # Override with command line arguments
        config['grid_size'] = args.grid_size
        config['win_length'] = args.win_length
        if args.model != 'tictactoe_model.pth':
            config['model_path'] = args.model
        
        train(config)
    
    elif args.play:
        play_against_ai(args.model, args.grid_size, args.win_length)
    
    elif args.evaluate:
        # Evaluate trained model
        if not os.path.exists(args.model):
            print(f"Error: Model file '{args.model}' not found.")
            return
        
        print(f"Loading model from {args.model}")
        env = TicTacToe(grid_size=args.grid_size, win_length=args.win_length)
        agent = DQNAgent({'epsilon_start': 0.0}, grid_size=args.grid_size)
        agent.load(args.model)
        
        print(f"Evaluating {args.grid_size}x{args.grid_size} (win={args.win_length}) model over {args.eval_games} games...")
        win_rate, draw_rate, loss_rate = evaluate_agent(agent, env, games=args.eval_games)
        
        print(f"\nEvaluation Results:")
        print(f"  Win Rate:  {win_rate * 100:.1f}% ({int(win_rate * args.eval_games)} games)")
        print(f"  Draw Rate: {draw_rate * 100:.1f}% ({int(draw_rate * args.eval_games)} games)")
        print(f"  Loss Rate: {loss_rate * 100:.1f}% ({int(loss_rate * args.eval_games)} games)")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()