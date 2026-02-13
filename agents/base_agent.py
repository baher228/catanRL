"""
Base agent interface for Catan agents.
"""

from abc import ABC, abstractmethod
from typing import List
from actions import Action


class BaseAgent(ABC):
    """Abstract base class for all Catan agents."""
    
    def __init__(self, player_id: int, name: str):
        """
        Initialize the agent.
        
        Args:
            player_id: The player ID this agent controls
            name: Name of the agent
        """
        self.player_id = player_id
        self.name = name
    
    @abstractmethod
    def choose_action(self, observation: dict, legal_actions: List[Action]) -> Action:
        """
        Choose an action given the current observation and legal actions.
        
        Args:
            observation: Dictionary containing game state information
                - phase: Current game phase
                - current_player_idx: Index of current player
                - board: BoardState object
                - resources: Player's resources
                - dev_cards: Player's development cards
                - victory_points: Player's total VP
                - opponents: List of opponent info
                - pending_discards: Whether player needs to discard
                - pending_robber_move: Whether robber needs to be moved
                
            legal_actions: List of legal actions available
        
        Returns:
            Action to take
        """
        pass
    
    def on_game_start(self, observation: dict):
        """
        Called when the game starts.
        Useful for agents that need to initialize state.
        """
        pass
    
    def on_turn_start(self, observation: dict):
        """Called at the start of the agent's turn."""
        pass
    
    def on_turn_end(self, observation: dict):
        """Called at the end of the agent's turn."""
        pass
    
    def on_opponent_action(self, player_id: int, action: Action, observation: dict):
        """
        Called when an opponent takes an action.
        Useful for learning agents.
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name} (Player {self.player_id})"
    
    def __repr__(self) -> str:
        return self.__str__()
