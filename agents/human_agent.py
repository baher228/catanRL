"""Human agent for interactive play and debugging."""

from typing import List
from agents.base_agent import BaseAgent
from actions import Action


class HumanAgent(BaseAgent):
    """Interactive agent that prompts for human input."""
    
    def __init__(self, player_id: int, name: str = None):
        """
        Initialize human agent.
        
        Args:
            player_id: Player ID
            name: Optional name, defaults to "Human Player {player_id}"
        """
        if name is None:
            name = f"Human Player {player_id}"
        super().__init__(player_id, name)
    
    def choose_action(self, observation: dict, legal_actions: List[Action]) -> Action:
        """Prompt user to select an action."""
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        print(f"\n{'='*60}")
        print(f"{self.name}'s turn")
        print(f"{'='*60}")
        
        # Display current state
        self._display_observation(observation)
        
        # Display legal actions
        print(f"\nLegal actions ({len(legal_actions)}):")
        for i, action in enumerate(legal_actions):
            print(f"  {i}: {action}")
        
        # Get user input
        while True:
            try:
                choice = input(f"\nSelect action (0-{len(legal_actions)-1}): ").strip()
                idx = int(choice)
                if 0 <= idx < len(legal_actions):
                    return legal_actions[idx]
                else:
                    print(f"Invalid choice. Please enter 0-{len(legal_actions)-1}")
            except (ValueError, KeyboardInterrupt):
                print("Invalid input. Please enter a number.")
    
    def _display_observation(self, observation: dict):
        """Display current game state in a human-readable format."""
        print(f"\nTurn: {observation.get('turn_number', 'Setup')}")
        print(f"Phase: {observation.get('phase', 'Unknown')}")
        
        if observation.get('dice_roll'):
            print(f"Last dice roll: {observation['dice_roll']}")
        
        # Resources
        resources = observation.get('resources', {})
        print(f"\nYour resources:")
        for resource, count in resources.items():
            if count > 0:
                print(f"  {resource.value}: {count}")
        
        # Development cards
        dev_cards = observation.get('dev_cards', {})
        dev_card_total = sum(dev_cards.values())
        if dev_card_total > 0:
            print(f"\nYour development cards: {dev_card_total}")
            for card, count in dev_cards.items():
                if count > 0:
                    print(f"  {card.value}: {count}")
        
        # Victory points
        vp = observation.get('victory_points', 0)
        print(f"\nVictory Points: {vp}")
        
        # Opponents
        opponents = observation.get('opponents', [])
        if opponents:
            print(f"\nOpponents:")
            for opp in opponents:
                print(f"  {opp['name']}: {opp['resource_count']} resources, "
                      f"{opp['public_vp']} public VP, {opp['knights_played']} knights")
