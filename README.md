# Qwen2.5-Checkers
Qwen2.5:14b but as checkers. This comes with a simple interface that user can interact with and play checkers



![image](https://github.com/user-attachments/assets/cb12023c-7672-4fb5-a091-63504cbc911b)




# Qwen Checkers

A sophisticated checkers game implementation using PyQt5 with an AI opponent powered by Qwen 2.5 LLM. Features a modern dark-themed UI and professional game mechanics.

## Features

- **Modern Dark Theme UI**
  - Custom-styled window with frameless design
  - Draggable title bar
  - Professional dark color palette
  - Smooth animations and visual feedback

- **Advanced Game Mechanics**
  - Complete checkers ruleset implementation
  - Mandatory capture moves
  - Multiple jump sequences
  - King piece promotion
  - Move validation and error handling

- **AI Opponent**
  - Powered by Qwen 2.5 LLM
  - Sophisticated position evaluation
  - Multiple fallback strategies
  - Intelligent move parsing
  - Error recovery system

## Technical Details

### Core Components

- **GameState**: Manages the game's core logic, board state, and move validation
- **CheckerSquare**: Custom QPushButton subclass for individual board squares
- **AIWorkerThread**: Handles AI move generation in a separate thread
- **TitleBar**: Custom window title bar implementation
- **CheckersGame**: Main game window and controller

### AI Implementation

The AI system uses a multi-layered approach:

1. **Primary Analysis**:
   - Detailed board evaluation
   - Material counting
   - Spatial control assessment
   - King creation opportunities
   - Tactical pattern recognition

2. **Fallback System**:
   - Simplified position analysis
   - Basic move validation
   - Random valid move selection as last resort

### Dependencies

- PyQt5
- Ollama
- Qwen 2.5 model

## Installation Requirements

```bash
pip install PyQt5 ollama
# Qwen 2.5 model must be available through Ollama
```

## Usage

Run the main script to start the game:

```bash
python checkers_game.py
```

## Game Rules

- Red pieces move upward, Black (AI) pieces move downward
- Regular pieces move diagonally forward only
- Kings can move diagonally in any direction
- Captures are mandatory when available
- Multiple captures must be completed in the same turn
- Pieces become kings when reaching the opposite end
- Game ends when a player has no valid moves or loses all pieces

## Technical Implementation

The game uses a robust architecture with several key classes:

- **PieceType Enum**: 
  - EMPTY = 0
  - BLACK = 1
  - BLACK_KING = 2
  - RED = 3
  - RED_KING = 4

- **Board Representation**: 8x8 grid using nested lists
- **Move Validation**: Comprehensive checking system for legal moves
- **State Management**: Complete game state tracking including move history

## Contributing

Feel free to submit issues and pull requests. Areas for potential improvement:

- Additional AI models support
- Network multiplayer functionality
- Save/load game feature
- Move replay system
- Difficulty levels


## Acknowledgments

- PyQt5 for the UI framework
- Qwen team for the LLM model
- Ollama for model serving

## Known issues

- King moves, the sequencial move is not implimented fully. The groundwork is there, but I just never got around to executing. Sorry
