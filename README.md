# Catan RL - Reinforcement Learning Environment for Settlers of Catan

A Python implementation of Settlers of Catan as a reinforcement learning environment, complete with agents, game visualization, and interactive Jupyter notebooks.

## 📁 Project Structure

```
catanRL/
├── catan_env.py              # Core board representation with NetworkX graph
├── game_constants.py         # Game rules, costs, and constants
├── actions.py                # Action definitions and types
├── game_state.py             # Full game state management
├── action_manager.py         # Action validation and execution
├── game_visualization.py     # Jupyter visualization utilities
├── play_game.py              # Example game loop
├── agents/                   # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py        # Abstract base agent
│   ├── random_agent.py      # Random action baseline
│   └── human_agent.py       # Interactive human player
├── agent_demo.ipynb          # Interactive Jupyter demo
└── README.md                 # This file
```

## 🚀 Quick Start

### Installation

```bash
pip install networkx matplotlib numpy jupyter
```

### Run a Game

```python
from game_state import GameState
from action_manager import ActionManager
from agents import RandomAgent
from play_game import play_game

# Create agents
agents = [
    RandomAgent(0, "Alice"),
    RandomAgent(1, "Bob"),
    RandomAgent(2, "Charlie"),
    RandomAgent(3, "Diana"),
]

# Play a game
winner = play_game(agents, max_turns=100, verbose=True)
```

Or run the example directly:

```bash
python play_game.py
```

### Interactive Jupyter Notebook

```bash
jupyter notebook agent_demo.ipynb
```

## 🎮 Game Components

### Board Representation

The board uses NetworkX graphs for efficient spatial relationships:
- **Vertices** (54 total): Settlement/city placement locations
- **Edges** (72 total): Road placement locations
- **Tiles** (19 total): Resource hexagons with number tokens

### Game State

`GameState` manages:
- Player resources and development cards
- Turn progression and game phases
- Dice rolling and resource distribution
- Victory point tracking
- Robber management

### Actions

All possible actions are defined in `actions.py`:
- **Building**: Settlement, City, Road
- **Trading**: Bank (4:1), Port (3:1 or 2:1), Player-to-player
- **Development Cards**: Buy and play (Knight, Road Building, Year of Plenty, Monopoly)
- **Turn Management**: Roll dice, End turn, Discard cards

### Agents

#### BaseAgent (Abstract)
```python
class BaseAgent(ABC):
    @abstractmethod
    def choose_action(self, observation: dict, legal_actions: List[Action]) -> Action:
        pass
```

#### RandomAgent
Selects actions uniformly at random - useful for baseline testing.

#### HumanAgent
Interactive terminal-based player for debugging and manual testing.

### Observation Space

Each agent receives observations with:
- **Full visibility**: Own resources, development cards, victory points
- **Partial visibility**: Opponent resource counts, public VP, buildings
- **Game state**: Current turn, dice roll, board state, robber position

## 📊 Visualization

### Game State Visualization

```python
from game_visualization import plot_game_state
import matplotlib.pyplot as plt

plot_game_state(game)
plt.show()
```

Features:
- Board with colored resource hexagons
- Number tokens with 6/8 highlighting
- Player settlements (circles) and cities (squares)
- Roads connecting settlements
- Player statistics panel
- Game info panel

### Player Resources

```python
from game_visualization import plot_player_resources

plot_player_resources(player)
plt.show()
```

## 🎯 Example Usage

### Manual Step-by-Step Play

```python
from game_state import GameState
from action_manager import ActionManager

# Create game
game = GameState(num_players=4)
action_mgr = ActionManager(game)

# Get legal actions
legal_actions = action_mgr.get_legal_actions(game.current_player.player_id)

# Execute an action
action = legal_actions[0]
success = action_mgr.execute_action(game.current_player.player_id, action)

# Check for victory
winner = game.check_victory()
```

### Custom Agent

```python
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def choose_action(self, observation, legal_actions):
        # Your AI logic here
        # Return an Action from legal_actions
        return legal_actions[0]

# Use your agent
agents = [MyAgent(i, f"AI {i}") for i in range(4)]
winner = play_game(agents)
```

## 🔧 Implementation Status

### ✅ Completed
- Core board graph structure
- Game state management
- Turn progression and phases
- Setup phase (2 settlements + 2 roads per player)
- Dice rolling and resource distribution
- Building placement (settlements, cities, roads)
- Basic trading (bank 4:1)
- Development card deck
- Victory conditions
- Agent interface
- Random and human agents
- Visualization tools
- Interactive Jupyter notebook

### 🚧 To Do
- Port trading (3:1, 2:1)
- Player-to-player trading
- All development card actions
- Longest road calculation (BFS/DFS)
- Robber steal mechanics
- Discard logic when 7 is rolled
- Advanced agents (rule-based, Q-learning, DQN, PPO)
- Training loops for RL
- Action masking for RL algorithms
- Reward shaping experiments

## 🧪 Testing

Run a quick test:

```bash
python -c "from play_game import main; main()"
```

Or use the Jupyter notebook for interactive testing and visualization.

## 📝 Files Overview

**Keep these files:**
- ✅ `catan_env.py` - Core board logic with NetworkX graph
- ✅ `game_constants.py` - Game rules and costs
- ✅ `actions.py` - Action definitions
- ✅ `game_state.py` - Game state and player management
- ✅ `action_manager.py` - Action validation and execution
- ✅ `game_visualization.py` - Visualization for Jupyter
- ✅ `play_game.py` - Example game loop
- ✅ `agent_demo.ipynb` - Interactive demo
- ✅ `agents/` - Agent implementations

**Previously removed (redundant):**
- ❌ `catan_board.py` - Old implementation
- ❌ `fixed_visualization.py` - Temporary helper

## 🤝 Contributing

Ideas for contribution:
1. Implement missing features (ports, trading, dev cards)
2. Create more sophisticated agents
3. Add comprehensive test suite
4. Optimize action space for RL
5. Create training environments
6. Add multiplayer web interface

## 📚 Resources

- [Catan Rules](https://www.catan.com/understand-catan/game-rules)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [OpenAI Gym](https://gymnasium.farama.org/) - For creating RL environments

## License

MIT License - Feel free to use for research and learning!
