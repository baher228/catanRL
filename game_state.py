"""
Game state management for Catan.
Manages players, resources, turn progression, and game phases.
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum, auto

from catan_env import BoardState, generate_random_board
from game_constants import (
    Resource, DevelopmentCard, Port,
    SETTLEMENT_COST, CITY_COST, ROAD_COST, DEVELOPMENT_CARD_COST,
    VP_PER_SETTLEMENT, VP_PER_CITY, VP_LONGEST_ROAD, VP_LARGEST_ARMY, VP_TO_WIN,
    MIN_ROAD_LENGTH_FOR_LONGEST, MIN_KNIGHTS_FOR_LARGEST,
    MAX_RESOURCES_BEFORE_DISCARD, DEV_CARD_DECK_COMPOSITION,
    INITIAL_SETTLEMENTS, INITIAL_CITIES, INITIAL_ROADS,
    can_afford, deduct_resources, add_resources, total_resources
)
from actions import Action, ActionType


class GamePhase(Enum):
    """Game phases."""
    SETUP = auto()  # Initial settlement/road placement
    MAIN = auto()   # Normal gameplay
    GAME_OVER = auto()


@dataclass
class Player:
    """Represents a player in the game."""
    player_id: int
    name: str
    
    # Resources
    resources: Dict[Resource, int] = field(default_factory=lambda: {r: 0 for r in Resource})
    
    # Development cards
    dev_cards: Dict[DevelopmentCard, int] = field(default_factory=lambda: {d: 0 for d in DevelopmentCard})
    dev_cards_played_this_turn: Set[DevelopmentCard] = field(default_factory=set)
    dev_cards_bought_this_turn: List[DevelopmentCard] = field(default_factory=list)  # Can't play on same turn
    
    # Building counts (remaining pieces)
    settlements_remaining: int = INITIAL_SETTLEMENTS
    cities_remaining: int = INITIAL_CITIES
    roads_remaining: int = INITIAL_ROADS
    
    # Special achievements
    knights_played: int = 0
    has_longest_road: bool = False
    has_largest_army: bool = False
    
    # Victory points (excluding hidden VP cards)
    public_victory_points: int = 0
    
    def get_total_victory_points(self) -> int:
        """Calculate total victory points including hidden VP cards."""
        vp = self.public_victory_points
        vp += self.dev_cards.get(DevelopmentCard.VICTORY_POINT, 0)
        return vp
    
    def add_resource(self, resource: Resource, amount: int = 1):
        """Add resources to player's hand."""
        self.resources[resource] = self.resources.get(resource, 0) + amount
    
    def remove_resource(self, resource: Resource, amount: int = 1) -> bool:
        """Remove resources from player's hand. Returns False if insufficient."""
        if self.resources.get(resource, 0) < amount:
            return False
        self.resources[resource] -= amount
        return True
    
    def can_afford(self, cost: Dict[Resource, int]) -> bool:
        """Check if player can afford a cost."""
        return can_afford(self.resources, cost)
    
    def pay_cost(self, cost: Dict[Resource, int]) -> bool:
        """Pay a cost. Returns False if insufficient resources."""
        if not self.can_afford(cost):
            return False
        deduct_resources(self.resources, cost)
        return True
    
    def total_resource_count(self) -> int:
        """Get total number of resource cards."""
        return total_resources(self.resources)
    
    def needs_to_discard(self) -> bool:
        """Check if player needs to discard (>7 cards when 7 is rolled)."""
        return self.total_resource_count() > MAX_RESOURCES_BEFORE_DISCARD


class GameState:
    """
    Manages the full game state including board, players, and turn progression.
    """
    
    def __init__(self, num_players: int = 4, board: Optional[BoardState] = None):
        """
        Initialize a new game.
        
        Args:
            num_players: Number of players (2-4 for base game)
            board: Optional pre-generated board, otherwise creates random board
        """
        if not 2 <= num_players <= 4:
            raise ValueError("Catan supports 2-4 players")
        
        # Board
        self.board = board if board else generate_random_board()
        
        # Players
        self.num_players = num_players
        self.players = [
            Player(player_id=i, name=f"Player {i}")
            for i in range(num_players)
        ]
        
        # Game state
        self.phase = GamePhase.SETUP
        self.current_player_idx = 0
        self.turn_number = 0
        self.dice_roll: Optional[int] = None
        self.winner: Optional[int] = None
        
        # Setup phase tracking
        self.setup_round = 0  # 0 = first round, 1 = second round (reverse order)
        self.setup_placements: Dict[int, Dict[str, int]] = {
            i: {'settlements': 0, 'roads': 0} for i in range(num_players)
        }
        # Track the last settlement placed by each player during setup
        self.last_setup_settlement: Dict[int, Optional[int]] = {
            i: None for i in range(num_players)
        }
        
        # Robber
        self.robber_tile_id = self._find_desert_tile()
        
        # Development card deck
        self.dev_card_deck = self._create_dev_card_deck()
        random.shuffle(self.dev_card_deck)
        
        # Ports (to be configured based on board layout)
        self.ports: List[Port] = []
        
        # Pending actions (e.g., after rolling 7, need to discard/move robber)
        self.pending_discards: Set[int] = set()  # Player IDs who need to discard
        self.pending_robber_move = False
        
        # Trade offers
        self.active_trade_offer: Optional[dict] = None
    
    def _find_desert_tile(self) -> int:
        """Find the desert tile ID for initial robber placement."""
        for tile in self.board.tiles:
            if tile.resource == "desert":
                return tile.id
        return 0  # Fallback to first tile if no desert found
    
    def _create_dev_card_deck(self) -> List[DevelopmentCard]:
        """Create the development card deck."""
        deck = []
        for card_type, count in DEV_CARD_DECK_COMPOSITION.items():
            deck.extend([card_type] * count)
        return deck
    
    @property
    def current_player(self) -> Player:
        """Get the current player."""
        return self.players[self.current_player_idx]
    
    def next_turn(self):
        """Advance to the next turn."""
        # Clear turn-specific state
        self.current_player.dev_cards_played_this_turn.clear()
        self.current_player.dev_cards_bought_this_turn.clear()
        self.dice_roll = None
        
        # Advance turn
        if self.phase == GamePhase.SETUP:
            self._advance_setup_turn()
        else:
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players
            self.turn_number += 1
    
    def _advance_setup_turn(self):
        """Advance turn during setup phase."""
        if self.setup_round == 0:
            # First round: go forward 0 -> 1 -> 2 -> 3
            self.current_player_idx += 1
            if self.current_player_idx >= self.num_players:
                # Start second round in reverse
                self.setup_round = 1
                self.current_player_idx = self.num_players - 1
        else:
            # Second round: go backward 3 -> 2 -> 1 -> 0
            self.current_player_idx -= 1
            if self.current_player_idx < 0:
                # Setup complete, start main game
                self.phase = GamePhase.MAIN
                self.current_player_idx = 0
                self.turn_number = 1
    
    def roll_dice(self) -> int:
        """Roll two dice and return the sum."""
        roll = random.randint(1, 6) + random.randint(1, 6)
        self.dice_roll = roll
        
        if roll == 7:
            # Handle robber logic
            self._handle_seven_rolled()
        else:
            # Distribute resources
            self._distribute_resources(roll)
        
        return roll
    
    def _handle_seven_rolled(self):
        """Handle the case when 7 is rolled."""
        # Check which players need to discard
        self.pending_discards = {
            p.player_id for p in self.players if p.needs_to_discard()
        }
        self.pending_robber_move = True
    
    def _distribute_resources(self, roll: int):
        """Distribute resources based on dice roll."""
        # Find all tiles with this number token
        for tile in self.board.tiles:
            if tile.token == roll and tile.id != self.robber_tile_id:
                # Find vertices on this tile
                vertices = self.board.get_vertices_for_tile(tile.id)
                
                # Give resources to players with settlements/cities on these vertices
                for vertex_id in vertices:
                    vertex_data = self.board.get_vertex_data(vertex_id)
                    owner = vertex_data.get('owner')
                    
                    if owner is not None:
                        resource = Resource(tile.resource)
                        amount = 2 if vertex_data.get('is_city') else 1
                        self.players[owner].add_resource(resource, amount)
    
    def check_victory(self) -> Optional[int]:
        """Check if any player has won. Returns player_id of winner or None."""
        for player in self.players:
            if player.get_total_victory_points() >= VP_TO_WIN:
                self.phase = GamePhase.GAME_OVER
                self.winner = player.player_id
                return player.player_id
        return None
    
    def update_longest_road(self):
        """Update who has the longest road."""
        # This requires BFS/DFS to calculate road lengths
        # Simplified implementation - to be expanded
        pass
    
    def update_largest_army(self):
        """Update who has the largest army."""
        eligible_players = [
            p for p in self.players 
            if p.knights_played >= MIN_KNIGHTS_FOR_LARGEST
        ]
        
        if not eligible_players:
            return
        
        # Find player with most knights
        max_knights = max(p.knights_played for p in eligible_players)
        leaders = [p for p in eligible_players if p.knights_played == max_knights]
        
        # If there's a tie and someone currently has it, they keep it
        current_holder = next((p for p in self.players if p.has_largest_army), None)
        if len(leaders) > 1 and current_holder in leaders:
            return
        
        # Update largest army
        for p in self.players:
            old_status = p.has_largest_army
            p.has_largest_army = (p == leaders[0])
            
            # Update VP
            if old_status and not p.has_largest_army:
                p.public_victory_points -= VP_LARGEST_ARMY
            elif not old_status and p.has_largest_army:
                p.public_victory_points += VP_LARGEST_ARMY
    
    def get_observation(self, player_id: int) -> dict:
        """
        Get observation for a specific player.
        Includes full info for the player, partial info for opponents.
        """
        player = self.players[player_id]
        
        return {
            'phase': self.phase,
            'current_player_idx': self.current_player_idx,
            'turn_number': self.turn_number,
            'dice_roll': self.dice_roll,
            
            # Board state
            'board': self.board,
            'robber_tile_id': self.robber_tile_id,
            
            # Own player info (full visibility)
            'resources': player.resources.copy(),
            'dev_cards': player.dev_cards.copy(),
            'victory_points': player.get_total_victory_points(),
            
            # Opponent info (partial visibility)
            'opponents': [
                {
                    'player_id': p.player_id,
                    'name': p.name,
                    'resource_count': p.total_resource_count(),
                    'dev_card_count': sum(p.dev_cards.values()),
                    'public_vp': p.public_victory_points,
                    'knights_played': p.knights_played,
                    'settlements_remaining': p.settlements_remaining,
                    'cities_remaining': p.cities_remaining,
                    'roads_remaining': p.roads_remaining,
                }
                for p in self.players if p.player_id != player_id
            ],
            
            # Pending actions
            'pending_discards': player_id in self.pending_discards,
            'pending_robber_move': self.pending_robber_move,
        }
    
    def get_observation_vector(self, player_id: int) -> np.ndarray:
        """
        Returns a fixed-size flat numpy array for neural network input.
        
        Observation structure (total: 404 dimensions):
        - Player resources (5 values) - normalized by /10
        - Player dev cards (5 types) - normalized by /5
        - Player stats (7 values) - settlements, cities, roads, knights, VPs, achievements
        - Opponent stats (3 opponents × 7 features each = 21)
        - Board vertices (54 × 3 = 162) - [owner_id, is_city, is_empty]
        - Board edges (72 × 2 = 144) - [owner_id, is_empty]
        - Board tiles (19 × 3 = 57) - [resource_type, token_number, has_robber]
        - Game state (3 values) - current_player, turn_number, dice_roll
        """
        obs = []
        player = self.players[player_id]
        
        # Player resources (5)
        for resource in Resource:
            obs.append(player.resources.get(resource, 0) / 10.0)
        
        # Player dev cards (5)
        for card in DevelopmentCard:
            obs.append(player.dev_cards.get(card, 0) / 5.0)
        
        # Player stats (7)
        obs.extend([
            player.settlements_remaining / 5.0,
            player.cities_remaining / 4.0,
            player.roads_remaining / 15.0,
            player.knights_played / 10.0,
            player.public_victory_points / 10.0,
            float(player.has_longest_road),
            float(player.has_largest_army)
        ])
        
        # Opponents (3 × 7 = 21)
        for p in self.players:
            if p.player_id != player_id:
                obs.extend([
                    p.total_resource_count() / 10.0,
                    sum(p.dev_cards.values()) / 5.0,
                    p.settlements_remaining / 5.0,
                    p.cities_remaining / 4.0,
                    p.roads_remaining / 15.0,
                    p.knights_played / 10.0,
                    p.public_victory_points / 10.0
                ])
        
        # Board vertices (54 × 3 = 162)
        for v_id in range(54):
            v_data = self.board.get_vertex_data(v_id)
            owner = v_data['owner']
            obs.extend([
                (owner + 1) / 4.0 if owner is not None else 0,  # normalize owner (0-4 range)
                float(v_data['is_city']),
                float(owner is None)  # is_empty
            ])
        
        # Board edges (72 × 2 = 144)
        for edge in self.board._edge_list_cache:
            e_data = self.board.get_edge_data(edge)
            owner = e_data['owner']
            obs.extend([
                (owner + 1) / 4.0 if owner is not None else 0,
                float(owner is None)
            ])
        
        # Board tiles (19 × 3 = 57)
        resource_map = {'desert': 0, 'wood': 1, 'brick': 2, 'wheat': 3, 'sheep': 4, 'ore': 5}
        for tile in self.board.tiles:
            obs.extend([
                resource_map[tile.resource] / 5.0,
                (tile.token or 0) / 12.0,
                float(self.robber_tile_id == tile.id)
            ])
        
        # Game state (3)
        obs.extend([
            self.current_player_idx / 3.0,
            self.turn_number / 100.0,
            (self.dice_roll or 0) / 12.0
        ])
        
        return np.array(obs, dtype=np.float32)
