# Quick Start Guide

## Installation
```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Create a configuration file
```bash
python connect4_ai.py --create-config
```
This creates `config.json` with default training parameters.

### 2. Train the AI
```bash
python connect4_ai.py --train
```

**Training Tips:**
- Training 1000 episodes takes approximately 10-20 minutes depending on your hardware
- You can interrupt training with Ctrl+C (model will be saved)
- The AI learns by playing against itself
- Watch the win rate increase as training progresses

### 3. Play against the AI
```bash
python connect4_ai.py --play
```

**Playing Tips:**
- You are Player 1 (X), AI is Player 2 (O)
- Enter column numbers 0-6 to drop your piece
- The AI will think for a moment before making its move
- Play as many games as you want!

## Quick Training Test (50 episodes)

Use the example configuration for a quick test:
```bash
python connect4_ai.py --train --config example_config.json
```

This completes in about 1-2 minutes and creates `test_model.pth`.

To play against the quick-trained model:
```bash
python connect4_ai.py --play --model test_model.pth
```

## Configuration Parameters

Edit `config.json` to customize training:

- **num_episodes**: How many games to train (default: 1000)
- **learning_rate**: How fast the AI learns (default: 0.001)
- **epsilon_start**: Initial exploration rate (default: 1.0 = 100% random)
- **epsilon_min**: Minimum exploration rate (default: 0.01 = 1% random)
- **epsilon_decay**: How quickly to reduce exploration (default: 0.995)

For better results, train longer:
```json
{
  "num_episodes": 5000,
  "epsilon_decay": 0.998
}
```

## Troubleshooting

**Problem**: "PyTorch not found"
**Solution**: `pip install torch`

**Problem**: "Model file not found"
**Solution**: Train a model first with `--train`

**Problem**: Training too slow
**Solution**: Reduce `num_episodes` in config.json or use example_config.json

**Problem**: AI plays poorly
**Solution**: Train for more episodes (5000+) or adjust learning parameters
