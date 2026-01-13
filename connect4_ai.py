#!/usr/bin/env python3
"""
Connect 4 AI Trainer with Deep Q-Learning

This module implements a Connect 4 game environment and a Deep Q-Learning agent
that learns to play the game through self-play. It includes:
- Configurable training parameters
- Console GUI for visualizing training progress
- Model save/load functionality
- Play mode to compete against the trained model
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

# Optimize CPU usage - use all available cores
import multiprocessing
num_cpus = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(num_cpus)
os.environ['MKL_NUM_THREADS'] = str(num_cpus)
torch.set_num_threads(num_cpus)
print(f"Configured to use {num_cpus} CPU cores")

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
        self.logger = logging.getLogger('training')
        self.logger.setLevel(log_level)
        
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

class Connect4:
    """Connect 4 game environment"""
    
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        
    def reset(self):
        """Reset the game board"""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self.board.copy()
    
    def get_valid_moves(self):
        """Get list of valid column indices"""
        return [col for col in range(self.cols) if self.board[0, col] == 0]
    
    def make_move(self, col):
        """
        Make a move in the specified column
        Returns: (new_state, reward, done, info)
        """
        if col not in self.get_valid_moves():
            return self.board.copy(), -10, True, {"invalid_move": True}
        
        # Drop the piece
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                break
        
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
        # Check horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row, col + i] == player for i in range(4)):
                    return True
        
        # Check vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if all(self.board[row + i, col] == player for i in range(4)):
                    return True
        
        # Check diagonal (down-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(self.board[row + i, col + i] == player for i in range(4)):
                    return True
        
        # Check diagonal (up-right)
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row - i, col + i] == player for i in range(4)):
                    return True
        
        return False

    def get_winning_positions(self, player):
        """Return list of 4 (row,col) tuples for a winning alignment, or [] if none."""
        # horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row, col + i] == player for i in range(4)):
                    return [(row, col + i) for i in range(4)]
        # vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if all(self.board[row + i, col] == player for i in range(4)):
                    return [(row + i, col) for i in range(4)]
        # down-right diagonal
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(self.board[row + i, col + i] == player for i in range(4)):
                    return [(row + i, col + i) for i in range(4)]
        # up-right diagonal
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row - i, col + i] == player for i in range(4)):
                    return [(row - i, col + i) for i in range(4)]
        return []
    
    def render(self):
        """Render the game board to console with colored tokens"""
        # detect winning positions for highlighting
        win_pos_p1 = self.get_winning_positions(1)
        win_pos_p2 = self.get_winning_positions(2)
        win_set = set(win_pos_p1 + win_pos_p2)
        
        print("\n|" + "|".join(str(i) for i in range(self.cols)) + "|")
        print("-" * (self.cols * 2 + 1))
        for row in range(self.rows):
            row_str = "|"
            for col in range(self.cols):
                cell = self.board[row, col]
                token = " "
                if cell == 1:
                    token = "X"
                    token = color.RED + token + style.RESET_ALL
                elif cell == 2:
                    token = "O"
                    token = color.BLUE + token + style.RESET_ALL
                # override if part of winning four
                if (row, col) in win_set:
                    # color winning tokens green regardless of X/O color
                    token = color.GREEN + (self.board[row, col] and ("X" if self.board[row, col] == 1 else "O")) + style.RESET_ALL
                row_str += token + "|"
            print(row_str)
        print("-" * (self.cols * 2 + 1))
        print()


# Modified network architecture
class DQN(nn.Module):
    """Deep Q-Network for Connect 4"""
    
    def __init__(self, input_size=42, hidden_size=128, output_size=7):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # LayerNorm instead of BatchNorm
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)  # LayerNorm instead of BatchNorm
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Learning Agent"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Enable optimized CPU operations
        if self.device.type == 'cpu':
            # Use inference mode optimizations where possible
            torch.set_float32_matmul_precision('high')
        
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
        # track recent training losses for scheduling/diagnostics
        self.losses = []
        # Replay memory (deque) for experience replay
        self.memory = deque(maxlen=config.get('replay_memory_size', 10000))
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.gamma = config.get('gamma', 0.99)
        self.batch_size = config.get('batch_size', 64)
        self.target_update = config.get('target_update', 10)

        # Stability / loss
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss is more robust than MSE
        self.gradient_clip = config.get('gradient_clip', 1.0)
        
        # Performance optimization: enable/disable expensive tactical checks
        self.use_advanced_tactics = config.get('use_advanced_tactics', True)
    
    def _state_to_input(self, state):
        """Normalize board to -1/0/+1 tensor: player1=+1, player2=-1, empty=0"""
        # state: numpy array with values {0,1,2}
        p1 = (state == 1).astype(np.float32)
        p2 = (state == 2).astype(np.float32)
        mapped = p1 - p2
        return torch.from_numpy(mapped.flatten()).unsqueeze(0).to(self.device)

    def _check_win_board(self, board, player):
        """Quick inline win check for a numpy board (used by tactical rules)."""
        rows, cols = board.shape
        # horiz
        for r in range(rows):
            for c in range(cols - 3):
                if all(board[r, c + i] == player for i in range(4)):
                    return True
        # vert
        for r in range(rows - 3):
            for c in range(cols):
                if all(board[r + i, c] == player for i in range(4)):
                    return True
        # down-right
        for r in range(rows - 3):
            for c in range(cols - 3):
                if all(board[r + i, c + i] == player for i in range(4)):
                    return True
        # up-right
        for r in range(3, rows):
            for c in range(cols - 3):
                if all(board[r - i, c + i] == player for i in range(4)):
                    return True
        return False

    def _find_winning_move_local(self, board, player, valid_moves):
        """Return a winning column for player if exists, else None"""
        for col in valid_moves:
            test = board.copy()
            for row in range(test.shape[0] - 1, -1, -1):
                if test[row, col] == 0:
                    test[row, col] = player
                    break
            if self._check_win_board(test, player):
                return col
        return None
    
    def _drop(self, board, col, player):
        """Simulate dropping a piece into col for player. Return new board or None if column full."""
        if board[0, col] != 0:
            return None
        b = board.copy()
        for r in range(b.shape[0] - 1, -1, -1):
            if b[r, col] == 0:
                b[r, col] = player
                return b
        return None

    def _count_immediate_wins_fast(self, board, player, valid_moves):
        """Fast version: only check valid columns"""
        count = 0
        for col in valid_moves:
            b = self._drop(board, col, player)
            if b is not None and self._check_win_board(b, player):
                count += 1
        return count
    
    def _find_fork_moves_fast(self, board, player, valid_moves):
        """Optimized fork detection: return columns where player creates >=2 immediate wins"""
        forks = []
        for col in valid_moves:
            b = self._drop(board, col, player)
            if b is None:
                continue
            # Count wins after this move
            wins_after = self._count_immediate_wins_fast(b, player, [c for c in range(board.shape[1]) if b[0, c] == 0])
            if wins_after >= 2:
                forks.append(col)
        return forks

    def _filter_safe_moves_fast(self, board, valid_moves):
        """Fast safe-move filter: avoid moves giving opponent immediate win"""
        safe = []
        for c in valid_moves:
            b1 = self._drop(board, c, 1)
            if b1 is None:
                continue
            # Quick check: does opponent have immediate win?
            opponent_valid = [col for col in range(board.shape[1]) if b1[0, col] == 0]
            if self._find_winning_move_local(b1, 2, opponent_valid) is not None:
                continue
            safe.append(c)
        return safe

    def get_action(self, state, valid_moves, training=True):
        """Get action using epsilon-greedy policy with optimized tactical reasoning"""
        # 1) Our immediate win
        win_move = self._find_winning_move_local(state, 1, valid_moves)
        if win_move is not None:
            return win_move

        # 2) Block opponent immediate win
        block_move = self._find_winning_move_local(state, 2, valid_moves)
        if block_move is not None:
            return block_move

        # Skip expensive checks during early training (warmup phase)
        if self.use_advanced_tactics and (not training or self.epsilon < 0.5):
            # 2b) Block opponent forks (>=2 immediate wins)
            opponent_forks = self._find_fork_moves_fast(state, 2, valid_moves)
            if opponent_forks:
                return opponent_forks[0]  # Just take first fork column

            # 2c) Filter safe moves (don't give away immediate win)
            safe_moves = self._filter_safe_moves_fast(state, valid_moves)
            if safe_moves:
                # Try to create our fork among safe moves
                my_forks = self._find_fork_moves_fast(state, 1, safe_moves)
                if my_forks:
                    center = state.shape[1] // 2
                    if center in my_forks:
                        return center
                    return my_forks[0]
                
                # Center preference within safe moves
                center = state.shape[1] // 2
                if center in safe_moves and ((not training) or random.random() < 0.7):
                    return center
                
                # Use safe moves as candidates
                valid_moves = safe_moves

        # 3) Exploration
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)

        # 4) Model action (fallback)
        with torch.no_grad():
            state_tensor = self._state_to_input(state)
            was_training = self.policy_net.training
            self.policy_net.eval()
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            if was_training:
                self.policy_net.train()

        masked_q = np.full(7, -np.inf, dtype=np.float32)
        for move in valid_moves:
            masked_q[move] = q_values[move]
        return int(np.argmax(masked_q))
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory (clip reward)"""
        # clip reward to [-1, 1] to avoid large targets
        reward = max(-1.0, min(1.0, float(reward)))
        self.memory.append((state.copy(), action, reward, next_state.copy(), done))
    
    def replay(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        
        states_np = np.array([ (s == 1).astype(np.float32) - (s == 2).astype(np.float32) for s, _, _, _, _ in batch ])
        actions_np = np.array([ a for _, a, _, _, _ in batch ], dtype=np.int64)
        rewards_np = np.array([ r for _, _, r, _, _ in batch ], dtype=np.float32)
        next_states_np = np.array([ (ns == 1).astype(np.float32) - (ns == 2).astype(np.float32) for _, _, _, ns, _ in batch ])
        dones_np = np.array([ d for _, _, _, _, d in batch ], dtype=np.float32)
        
        # Optimize tensor creation with non_blocking for faster CPU operations
        states = torch.from_numpy(states_np).to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions_np).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(rewards_np).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states_np).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones_np).to(self.device, non_blocking=True)
        
        # Reshape for the network
        states = states.reshape(self.batch_size, -1)
        next_states = next_states.reshape(self.batch_size, -1)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: policy_net selects next action, target_net evaluates it
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Diagnostics for large Q magnitudes
        try:
            max_q = current_q.abs().max().item()
            max_target = target_q.abs().max().item()
            if max_q > 1e3 or max_target > 1e3:
                print(f"[WARN] Large Q magnitudes: max_q={max_q:.1f}, max_target={max_target:.1f}")
        except Exception:
            pass
        
        # Compute robust loss
        loss = self.loss_fn(current_q, target_q.detach())
        
        # Detect pathological values
        if torch.isnan(loss) or torch.isinf(loss):
            print("Detected NaN/Inf loss -- skipping step and dumping diagnostics")
            try:
                print("loss:", loss.item())
                print("current_q min/max:", current_q.min().item(), current_q.max().item())
                print("target_q min/max:", target_q.min().item(), target_q.max().item())
            except Exception:
                pass
            return float('nan')
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        # Optional: detect huge gradients
        max_grad = 0.0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                max_grad = max(max_grad, p.grad.abs().max().item())
        if max_grad > 1e3:
            print(f"[WARN] huge grad detected: {max_grad:.1f}")
        self.optimizer.step()
        
        # record loss for scheduler/monitoring
        self.losses.append(loss.item())
        # Learning rate scheduling every 10 steps (requires at least 10 entries)
        if self.scheduler and len(self.losses) >= 10 and len(self.losses) % 10 == 0:
            avg_loss = sum(self.losses[-10:]) / 10
            try:
                self.scheduler.step(avg_loss)
            except Exception as e:
                print(f"[WARN] scheduler.step failed: {e}")
        
        # Add value scaling - prevent Q-values from growing too large
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
        }, filepath)
    
    def load(self, filepath):
        """Load the model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)


class TrainingVisualizer:
    """Console-based training progress visualizer"""
    
    def __init__(self):
        self.metrics_tracker = MetricsTracker()
        
    def update(self, episode, reward, length, loss, result, eval_metrics=None, lr=None, epsilon=None):
        """Update statistics"""
        self.metrics_tracker.update(episode, reward, length, loss, result, eval_metrics, lr, epsilon)
    
    def render(self, episode, epsilon, total_episodes, time_info=None, current_lr=None):
        """Render training progress to console"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 60)
        print("CONNECT 4 AI - TRAINING PROGRESS")
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


def _find_winning_move(board, player, env):
    """Return a winning column for player if exists, else None"""
    for col in env.get_valid_moves():
        test = board.copy()
        for row in range(env.rows - 1, -1, -1):
            if test[row, col] == 0:
                test[row, col] = player
                break
        # temporary check
        # reuse Connect4._check_win by creating a lightweight check
        # we'll inline same logic to avoid instantiating new env
        def check_win(b, p):
            rows, cols = b.shape
            # horiz
            for r in range(rows):
                for c in range(cols - 3):
                    if all(b[r, c + i] == p for i in range(4)):
                        return True
            # vert
            for r in range(rows - 3):
                for c in range(cols):
                    if all(b[r + i, c] == p for i in range(4)):
                        return True
            # down-right
            for r in range(rows - 3):
                for c in range(cols - 3):
                    if all(b[r + i, c + i] == p for i in range(4)):
                        return True
            # up-right
            for r in range(3, rows):
                for c in range(cols - 3):
                    if all(b[r - i, c + i] == p for i in range(4)):
                        return True
            return False
        if check_win(test, player):
            return col
    return None

def heuristic_opponent_action(state, env, player=1):
    """Simple greedy opponent:
       - Win if possible
       - Block opponent immediate win
       - Play center if available
       - Else random valid move
    """
    # player is 1 or 2 representing the moving side in env coordinates
    # check winning move for player
    win = _find_winning_move(state, player, env)
    if win is not None:
        return win
    # block opponent
    other = 3 - player
    block = _find_winning_move(state, other, env)
    if block is not None:
        return block
    # center preference
    center = env.cols // 2
    if center in env.get_valid_moves():
        return center
    # fallback random
    return random.choice(env.get_valid_moves())

def evaluate_agent(agent, env, games=50):
    """Evaluate agent against heuristic opponent.
       Agent will play as Player 2 (AI) and heuristic as Player 1.
       Returns win_rate (AI wins / games), draw_rate, loss_rate.
    """
    agent_wins = 0
    draws = 0
    losses = 0

    # ensure deterministic policy (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    # put policy in eval mode for safety
    was_training = agent.policy_net.training
    agent.policy_net.eval()

    for _ in range(games):
        state = env.reset()
        done = False
        while not done:
            if env.current_player == 1:
                # heuristic opponent move
                col = heuristic_opponent_action(state, env, player=1)
                state, reward, done, info = env.make_move(col)
            else:
                # AI move (player 2) - flip perspective as training uses
                flipped = state.copy()
                flipped[flipped == 1] = 3
                flipped[flipped == 2] = 1
                flipped[flipped == 3] = 2
                valid = env.get_valid_moves()
                col = agent.get_action(flipped, valid, training=False)
                state, reward, done, info = env.make_move(col)
        if info.get('winner') == 2:
            agent_wins += 1
        elif info.get('winner') == 1:
            losses += 1
        else:
            draws += 1

    # restore
    agent.epsilon = original_epsilon
    agent.policy_net.train(was_training)

    win_rate = agent_wins / games
    draw_rate = draws / games
    loss_rate = losses / games
    return win_rate, draw_rate, loss_rate


def train(config):
    """Train the Connect 4 AI"""
    print("Initializing training...")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize utilities
    logger = Logger(config)
    checkpoint_manager = CheckpointManager(config)
    early_stopping = EarlyStopping(config)
    
    # Create environment and agent
    env = Connect4()
    agent = DQNAgent(config)
    visualizer = TrainingVisualizer()

    # Training settings
    num_episodes = config.get('num_episodes', 1000)
    save_interval = config.get('save_interval', 100)
    model_path = config.get('model_path', 'connect4_model.pth')
    warmup_episodes = config.get('warmup_episodes', 1000)
    
    # Evaluation settings
    eval_interval = config.get('eval_interval', 500)
    eval_games = config.get('eval_games', 50)
    eval_history = deque(maxlen=20)
    
    start_time = time.time()
    logger.info(f"Starting training with {num_episodes} episodes")
    
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
                action = agent.get_action(state, valid_moves)
                next_state, reward, done, info = env.make_move(action)
                
                # Store experience from agent 1's perspective
                agent.remember(state, action, reward, next_state, done)
                total_reward += reward
                episode_length += 1
                
                if done:
                    break
                
                # Agent 2 (opponent - also the learning agent)
                # Flip the board perspective for agent 2
                flipped_state = next_state.copy()
                flipped_state[flipped_state == 1] = 3
                flipped_state[flipped_state == 2] = 1
                flipped_state[flipped_state == 3] = 2
                
                valid_moves = env.get_valid_moves()
                opponent_action = agent.get_action(flipped_state, valid_moves)
                next_state, opponent_reward, done, info = env.make_move(opponent_action)
                
                # Store experience from agent 2's perspective
                # Flip next_state for storage
                flipped_next_state = next_state.copy()
                flipped_next_state[flipped_next_state == 1] = 3
                flipped_next_state[flipped_next_state == 2] = 1
                flipped_next_state[flipped_next_state == 3] = 2
                
                agent.remember(flipped_state, opponent_action, opponent_reward, flipped_next_state, done)
                
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
            visualizer.render(episode, agent.epsilon, num_episodes, time_info, current_lr)
            
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


def play_against_ai(model_path):
    """Play Connect 4 against the trained AI"""
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train a model first using the --train option.")
        return
    
    print("Loading AI model...")
    env = Connect4()
    config = {'epsilon_start': 0.0}  # No exploration when playing
    agent = DQNAgent(config)
    agent.load(model_path)
    print("AI model loaded successfully!\n")
    
    print("=" * 60)
    print("CONNECT 4 - PLAY AGAINST AI")
    print("=" * 60)
    print("\nYou are 'X' (Player 1)")
    print("AI is 'O' (Player 2)")
    print("Enter column number (0-6) to make a move\n")
    
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
                        col = int(input("Your move (column 0-6): "))
                        if col in valid_moves:
                            break
                        else:
                            print("Invalid move. Column is full or out of range.")
                    except (ValueError, KeyboardInterrupt):
                        print("\nExiting game...")
                        return
                
                state, reward, done, info = env.make_move(col)
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
                col = agent.get_action(flipped_state, valid_moves, training=False)
                print(f"AI plays column: {col}")
                
                state, reward, done, info = env.make_move(col)
        
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
        "model_path": "connect4_model.pth",
        "gradient_clip": 0.5,
        "weight_decay": 1e-4,
        "eval_interval": 500,
        "eval_games": 50,
        "warmup_episodes": 500,
        "learning_rate_schedule": {
            "enabled": True,
            "patience": 100,
            "factor": 0.8,
            "min_lr": 1e-6
        },
        "early_stopping": {
            "enabled": False,
            "patience": 2000,
            "min_delta": 0.01
        },
        "logging": {
            "enabled": True,
            "log_file": "connect4_training.log",
            "log_level": "INFO"
        },
        "checkpointing": {
            "enabled": True,
            "checkpoint_dir": "checkpoints",
            "keep_best": True,
            "metric": "win_rate"
        },
        "visualization": {
            "plot_training": False,
            "save_plots": False,
            "plot_dir": "plots"
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Default configuration created at {config_file}")
    return default_config


def main():
    parser = argparse.ArgumentParser(
        description="Connect 4 AI Trainer using Deep Q-Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python connect4_ai.py --train
  
  # Train with custom configuration file
  python connect4_ai.py --train --config my_config.json
  
  # Play against trained AI
  python connect4_ai.py --play
  
  # Play against AI with custom model
  python connect4_ai.py --play --model my_model.pth
  
  # Evaluate trained model
  python connect4_ai.py --evaluate --model my_model.pth --eval-games 200
  
  # Create default configuration file
  python connect4_ai.py --create-config
        """
    )
    
    parser.add_argument('--train', action='store_true',
                       help='Train the AI model')
    parser.add_argument('--play', action='store_true',
                       help='Play against the trained AI')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file (default: config.json)')
    parser.add_argument('--model', type=str, default='connect4_model.pth',
                       help='Path to model file (default: connect4_model.pth)')
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
        
        # Override model path if specified
        if args.model != 'connect4_model.pth':
            config['model_path'] = args.model
        
        train(config)
    
    elif args.play:
        play_against_ai(args.model)
    
    elif args.evaluate:
        # Evaluate trained model
        if not os.path.exists(args.model):
            print(f"Error: Model file '{args.model}' not found.")
            return
        
        print(f"Loading model from {args.model}")
        env = Connect4()
        agent = DQNAgent({'epsilon_start': 0.0})
        agent.load(args.model)
        
        print(f"Evaluating model over {args.eval_games} games...")
        win_rate, draw_rate, loss_rate = evaluate_agent(agent, env, games=args.eval_games)
        
        print(f"\nEvaluation Results:")
        print(f"  Win Rate:  {win_rate * 100:.1f}% ({int(win_rate * args.eval_games)} games)")
        print(f"  Draw Rate: {draw_rate * 100:.1f}% ({int(draw_rate * args.eval_games)} games)")
        print(f"  Loss Rate: {loss_rate * 100:.1f}% ({int(loss_rate * args.eval_games)} games)")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
