# Connect 4 AI Trainer

A Deep Q-Learning (DQN) implementation that learns to play Connect 4 through self-play. This project includes configurable training parameters, a console-based visualization of training progress, and the ability to play against the trained AI.

## Features

- **Deep Q-Learning AI**: Uses a neural network to learn optimal Connect 4 strategies
- **Self-Play Training**: The AI learns by playing against itself
- **Configurable Training**: All training parameters can be customized via JSON configuration
- **Training Visualization**: Real-time console GUI showing training progress, win rates, and statistics
- **Model Persistence**: Save and load trained models
- **Interactive Play Mode**: Play against the trained AI in the console

## Installation

### Requirements
- Python 3.7+
- PyTorch
- NumPy

### Setup

1. Clone the repository:
```bash
git clone https://github.com/maple-underscore/connect-4-ai-2.git
cd connect-4-ai-2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Create Default Configuration

Generate a default configuration file with recommended parameters:

```bash
python connect4_ai.py --create-config
```

This creates `config.json` with default settings:
- `num_episodes`: 1000 - Number of training episodes
- `learning_rate`: 0.001 - Neural network learning rate
- `gamma`: 0.99 - Discount factor for future rewards
- `epsilon_start`: 1.0 - Initial exploration rate
- `epsilon_min`: 0.01 - Minimum exploration rate
- `epsilon_decay`: 0.995 - Exploration decay rate per episode
- `replay_memory_size`: 10000 - Size of experience replay buffer
- `batch_size`: 64 - Training batch size
- `target_update`: 10 - Episodes between target network updates
- `save_interval`: 100 - Episodes between model saves
- `model_path`: connect4_model.pth - Path to save the model

### Train the AI

Train with default configuration:
```bash
python connect4_ai.py --train
```

Train with custom configuration:
```bash
python connect4_ai.py --train --config my_config.json
```

Train and save to a custom model file:
```bash
python connect4_ai.py --train --model my_model.pth
```

During training, you'll see a real-time console visualization showing:
- Current episode and progress bar
- Exploration rate (epsilon)
- Average reward over recent episodes
- Average episode length
- Training loss
- Win rate with visualization

Press `Ctrl+C` to stop training early (the model will be saved automatically).

### Play Against the AI

Play against the trained AI:
```bash
python connect4_ai.py --play
```

Play against a specific model:
```bash
python connect4_ai.py --play --model my_model.pth
```

In play mode:
- You are Player 1 (X)
- The AI is Player 2 (O)
- Enter column numbers (0-6) to make your moves
- The board is displayed after each move

## How It Works

### Game Environment
The `Connect4` class implements the game logic:
- 6x7 board (standard Connect 4 dimensions)
- Two players alternate placing pieces
- Win conditions: 4 in a row horizontally, vertically, or diagonally
- Draw condition: Board is full with no winner

### Deep Q-Learning Agent
The AI uses Deep Q-Learning with the following components:

1. **Neural Network**: A 3-layer feedforward network that takes the board state (42 values) and outputs Q-values for each action (7 columns)

2. **Experience Replay**: Stores past experiences and samples random batches for training to break correlation between consecutive samples

3. **Target Network**: A separate network updated periodically to stabilize training

4. **Epsilon-Greedy Exploration**: Balances exploration (random moves) and exploitation (best known moves) with decaying epsilon

### Training Process
1. The agent plays against itself
2. Both players learn from their experiences
3. The board is flipped for Player 2's perspective to reuse the same network
4. Rewards:
   - +1 for winning
   - -1 for losing
   - 0 for draw
   - -10 for invalid moves
5. The network learns to maximize future rewards

## Configuration Examples

### Quick Training (Testing)
```json
{
  "num_episodes": 100,
  "epsilon_decay": 0.95,
  "save_interval": 25
}
```

### Extended Training (Better Results)
```json
{
  "num_episodes": 5000,
  "learning_rate": 0.0005,
  "epsilon_decay": 0.998,
  "replay_memory_size": 50000,
  "batch_size": 128
}
```

## Tips for Better Training

1. **More Episodes**: Train for at least 1000 episodes for decent results, 5000+ for better performance
2. **Slower Decay**: Use higher epsilon_decay (e.g., 0.998) for longer exploration
3. **Larger Memory**: Increase replay_memory_size for more diverse training samples
4. **Learning Rate**: Adjust learning_rate if training is unstable (lower) or too slow (higher)

## Example Output

During training:
```
============================================================
CONNECT 4 AI - TRAINING PROGRESS
============================================================

Episode: 250/1000
Progress: [#########################                         ] 25.0%

Epsilon (Exploration Rate): 0.2193

Recent 100 Episodes Statistics:
  Average Reward: 0.34
  Average Length: 18.45
  Average Loss: 0.0234
  Win Rate: 52.5%

  Win Rate Visualization:
  0% [█████████████████████-----------------] 100%

============================================================
Press Ctrl+C to stop training
============================================================
```

## License

MIT License - feel free to use and modify as needed.