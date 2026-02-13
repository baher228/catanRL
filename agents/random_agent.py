"""Random agent that selects actions uniformly at random."""

import random
from typing import List
from agents.base_agent import BaseAgent
from actions import Action


class RandomAgent(BaseAgent):
    """Agent that selects random legal actions."""
    
    def __init__(self, player_id: int, name: str = None):
        """
        Initialize random agent.
        
        Args:
            player_id: Player ID
            name: Optional name, defaults to "Random Agent {player_id}"
        """
        if name is None:
            name = f"Random Agent {player_id}"
        super().__init__(player_id, name)
    
    def choose_action(self, observation: dict, legal_actions: List[Action]) -> Action:
        """Randomly select from legal actions."""
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        return random.choice(legal_actions)
