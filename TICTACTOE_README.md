# TicTacToe AI with Deep Q-Learning

This script implements a customizable TicTacToe game with an AI that learns through Deep Q-Learning. It features:

- **Customizable grid size** (e.g., 3x3, 4x4, 5x5, etc.)
- **Customizable win streak length** (e.g., 3, 4, 5 in a row)
- **Colored console output** using colorama (red X, blue O, green for winning pieces)
- **Training GUI** with real-time progress visualization
- **Tuple-based input** for move positions
- **Self-play training** where the AI learns by playing against itself

## Installation

Make sure you have the required dependencies:
```bash
pip install torch colorama numpy
```

## Usage

### 1. Create Configuration File
```bash
python tictactoe_ai.py --create-config
```
This creates `tictactoe_config.json` with default settings.

### 2. Train the AI

#### Standard 3x3 TicTacToe (win length 3):
```bash
python tictactoe_ai.py --train
```

#### Custom grid size with different win length:
```bash
# 4x4 grid with win length 3
python tictactoe_ai.py --train --grid-size 4 --win-length 3

# 5x5 grid with win length 4
python tictactoe_ai.py --train --grid-size 5 --win-length 4
```

#### With custom configuration:
```bash
python tictactoe_ai.py --train --config my_config.json
```

### 3. Play Against the AI

#### Standard 3x3:
```bash
python tictactoe_ai.py --play
```

#### Custom grid size:
```bash
# Play 4x4 with win length 3
python tictactoe_ai.py --play --grid-size 4 --win-length 3

# Play 5x5 with win length 4  
python tictactoe_ai.py --play --grid-size 5 --win-length 4
```

#### With custom model:
```bash
python tictactoe_ai.py --play --model my_model.pth --grid-size 4 --win-length 3
```

## Input Format

When playing against the AI, enter moves using tuples:
- `(0,0)` - Top-left corner
- `(1,1)` - Center (for 3x3)
- `0,1` - Row 0, Column 1 (parentheses optional)
- `4` - Single number (converted to row,col automatically)

## Training Progress

The training GUI shows:
- Episode progress bar
- Real-time statistics (win rate, average reward, loss)
- Time elapsed and estimated remaining time
- Win rate visualization bar

## Configuration Options

The configuration file supports:
- `grid_size`: Board dimensions (default: 3)
- `win_length`: Number in a row to win (default: 3)
- `num_episodes`: Training episodes (default: 1000)
- `learning_rate`: Neural network learning rate (default: 0.0001)
- `epsilon_start/min/decay`: Exploration parameters
- `batch_size`: Training batch size (default: 64)
- `model_path`: Where to save the trained model

## Examples

### Quick Start (3x3 standard):
```bash
# Train for default 1000 episodes
python tictactoe_ai.py --train

# Play against trained AI
python tictactoe_ai.py --play
```

### Advanced (5x5 with 4 in a row):
```bash
# Create custom config
python tictactoe_ai.py --create-config --config big_config.json

# Edit big_config.json to set "grid_size": 5, "win_length": 4, "num_episodes": 2000

# Train
python tictactoe_ai.py --train --config big_config.json --grid-size 5 --win-length 4

# Play
python tictactoe_ai.py --play --grid-size 5 --win-length 4 --model tictactoe_model.pth
```

## Game Features

- **Colored display**: 
  - Red X (Player 1)
  - Blue O (Player 2/AI)  
  - Green highlighting for winning pieces
- **Smart AI**: Uses tactical reasoning (wins, blocks, forks) combined with learned strategy
- **Flexible rules**: Any grid size and win length combination
- **Real-time training visualization**: Watch the AI improve as it learns

The AI learns through self-play, gradually improving its strategy over thousands of games against itself. The training GUI provides real-time feedback on the learning progress.