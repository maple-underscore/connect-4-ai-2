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
from collections import deque
import random

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    print("PyTorch not found. Please install it with: pip install torch")
    sys.exit(1)


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
    
    def render(self):
        """Render the game board to console"""
        print("\n  " + " ".join(str(i) for i in range(self.cols)))
        print("  " + "-" * (self.cols * 2 - 1))
        for row in range(self.rows):
            row_str = "|"
            for col in range(self.cols):
                cell = self.board[row, col]
                if cell == 0:
                    row_str += " "
                elif cell == 1:
                    row_str += "X"
                else:
                    row_str += "O"
                row_str += "|"
            print(row_str)
        print("  " + "-" * (self.cols * 2 - 1))
        print()


class DQN(nn.Module):
    """Deep Q-Network for Connect 4"""
    
    def __init__(self, input_size=42, hidden_size=128, output_size=7):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                    lr=config.get('learning_rate', 0.001))
        self.memory = deque(maxlen=config.get('replay_memory_size', 10000))
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.gamma = config.get('gamma', 0.99)
        self.batch_size = config.get('batch_size', 64)
        self.target_update = config.get('target_update', 10)
        
    def get_action(self, state, valid_moves, training=True):
        """Get action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # Mask invalid moves
            masked_q = np.full(7, -np.inf)
            for move in valid_moves:
                masked_q[move] = q_values[move]
            
            return int(np.argmax(masked_q))
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([s.flatten() for s, _, _, _, _ in batch])).to(self.device)
        actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(self.device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([s.flatten() for _, _, _, s, _ in batch])).to(self.device)
        dones = torch.FloatTensor([d for _, _, _, _, d in batch]).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
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
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.win_rates = deque(maxlen=100)
        
    def update(self, episode, reward, length, loss, result):
        """Update statistics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if loss is not None:
            self.losses.append(loss)
        
        # Track wins
        if result.get('winner') == 1:
            self.win_rates.append(1)
        elif result.get('draw'):
            self.win_rates.append(0.5)
        else:
            self.win_rates.append(0)
    
    def render(self, episode, epsilon, total_episodes):
        """Render training progress to console"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 60)
        print("CONNECT 4 AI - TRAINING PROGRESS")
        print("=" * 60)
        print(f"\nEpisode: {episode + 1}/{total_episodes}")
        print(f"Progress: [{'#' * int(50 * (episode + 1) / total_episodes):<50}] {100 * (episode + 1) / total_episodes:.1f}%")
        print(f"\nEpsilon (Exploration Rate): {epsilon:.4f}")
        
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:]
            recent_lengths = self.episode_lengths[-100:]
            recent_losses = self.losses[-100:] if self.losses else [0]
            
            print(f"\nRecent 100 Episodes Statistics:")
            print(f"  Average Reward: {np.mean(recent_rewards):.2f}")
            print(f"  Average Length: {np.mean(recent_lengths):.2f}")
            print(f"  Average Loss: {np.mean(recent_losses):.4f}")
            
            if len(self.win_rates) > 0:
                win_rate = np.mean(list(self.win_rates))
                print(f"  Win Rate: {win_rate * 100:.1f}%")
                
                # Simple bar chart
                bar_length = int(win_rate * 40)
                print(f"\n  Win Rate Visualization:")
                print(f"  0% [{'█' * bar_length}{'-' * (40 - bar_length)}] 100%")
        
        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop training")
        print("=" * 60)


def train(config):
    """Train the Connect 4 AI"""
    print("Initializing training...")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create environment and agent
    env = Connect4()
    agent = DQNAgent(config)
    visualizer = TrainingVisualizer()
    
    num_episodes = config.get('num_episodes', 1000)
    save_interval = config.get('save_interval', 100)
    model_path = config.get('model_path', 'connect4_model.pth')
    
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
            
            # Update visualizer
            avg_loss = np.mean(episode_losses) if episode_losses else None
            visualizer.update(episode, total_reward, episode_length, avg_loss, info)
            
            # Render progress every episode
            visualizer.render(episode, agent.epsilon, num_episodes)
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                agent.save(model_path)
                print(f"\nModel saved at episode {episode + 1}")
                time.sleep(0.5)
        
        # Final save
        agent.save(model_path)
        print(f"\n\nTraining completed! Model saved to {model_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        agent.save(model_path)
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
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "replay_memory_size": 10000,
        "batch_size": 64,
        "target_update": 10,
        "save_interval": 100,
        "model_path": "connect4_model.pth"
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
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
