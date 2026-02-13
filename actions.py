"""
Action definitions for Catan agents.
Defines all possible actions and their parameters.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Tuple
from game_constants import Resource, DevelopmentCard


class ActionType(Enum):
    """All possible action types in Catan."""
    # Turn management
    ROLL_DICE = auto()
    END_TURN = auto()
    
    # Building actions
    BUILD_SETTLEMENT = auto()
    BUILD_CITY = auto()
    BUILD_ROAD = auto()
    BUY_DEVELOPMENT_CARD = auto()
    
    # Trading actions
    TRADE_BANK = auto()  # 4:1 trade with bank
    TRADE_PORT = auto()  # 3:1 or 2:1 trade with port
    TRADE_OFFER = auto()  # Propose trade to other players
    TRADE_ACCEPT = auto()  # Accept a trade offer
    TRADE_DECLINE = auto()  # Decline a trade offer
    
    # Development card actions
    PLAY_KNIGHT = auto()
    PLAY_ROAD_BUILDING = auto()
    PLAY_YEAR_OF_PLENTY = auto()
    PLAY_MONOPOLY = auto()
    
    # Special actions
    DISCARD_CARDS = auto()  # When 7 is rolled and player has >7 cards
    MOVE_ROBBER = auto()  # After rolling 7 or playing knight
    STEAL_RESOURCE = auto()  # After moving robber


@dataclass
class Action:
    """
    Represents a specific action with its parameters.
    
    Different action types require different parameters:
    - BUILD_SETTLEMENT: vertex_id
    - BUILD_CITY: vertex_id
    - BUILD_ROAD: edge (tuple of two vertex IDs)
    - TRADE_BANK: give_resource, receive_resource
    - TRADE_PORT: give_resource, receive_resource, port_vertices
    - PLAY_KNIGHT: tile_id (to move robber), target_player_id (to steal from)
    - DISCARD_CARDS: resources (dict of Resource -> count to discard)
    - etc.
    """
    action_type: ActionType
    
    # Building parameters
    vertex_id: Optional[int] = None
    edge: Optional[Tuple[int, int]] = None
    
    # Trading parameters
    give_resource: Optional[Resource] = None
    receive_resource: Optional[Resource] = None
    give_amount: int = 0
    receive_amount: int = 0
    
    # Trade offer parameters
    target_player_id: Optional[int] = None
    offer_id: Optional[int] = None  # For accepting/declining trades
    
    # Port trade parameters
    port_vertices: Optional[Tuple[int, int]] = None
    
    # Robber/Knight parameters
    tile_id: Optional[int] = None
    
    # Discard/resource selection parameters
    resources: Optional[dict] = None  # Resource -> count
    
    # Road building parameters (play 2 roads)
    edges: Optional[List[Tuple[int, int]]] = None
    
    def __str__(self) -> str:
        """Human-readable action description."""
        base = f"{self.action_type.name}"
        
        if self.action_type == ActionType.BUILD_SETTLEMENT:
            return f"{base} at vertex {self.vertex_id}"
        elif self.action_type == ActionType.BUILD_CITY:
            return f"{base} at vertex {self.vertex_id}"
        elif self.action_type == ActionType.BUILD_ROAD:
            return f"{base} on edge {self.edge}"
        elif self.action_type == ActionType.TRADE_BANK:
            return f"{base}: give {self.give_amount} {self.give_resource.value} for {self.receive_amount} {self.receive_resource.value}"
        elif self.action_type == ActionType.TRADE_PORT:
            return f"{base}: give {self.give_amount} {self.give_resource.value} for {self.receive_amount} {self.receive_resource.value}"
        elif self.action_type == ActionType.PLAY_KNIGHT:
            return f"{base}: move robber to tile {self.tile_id}, steal from player {self.target_player_id}"
        elif self.action_type == ActionType.DISCARD_CARDS:
            return f"{base}: {self.resources}"
        elif self.action_type == ActionType.MOVE_ROBBER:
            return f"{base} to tile {self.tile_id}"
        elif self.action_type == ActionType.STEAL_RESOURCE:
            return f"{base} from player {self.target_player_id}"
        else:
            return base
    
    def __repr__(self) -> str:
        return self.__str__()


# Action factory functions for cleaner code
def build_settlement_action(vertex_id: int) -> Action:
    """Create a build settlement action."""
    return Action(ActionType.BUILD_SETTLEMENT, vertex_id=vertex_id)


def build_city_action(vertex_id: int) -> Action:
    """Create a build city action."""
    return Action(ActionType.BUILD_CITY, vertex_id=vertex_id)


def build_road_action(edge: Tuple[int, int]) -> Action:
    """Create a build road action."""
    return Action(ActionType.BUILD_ROAD, edge=edge)


def bank_trade_action(give_resource: Resource, receive_resource: Resource, 
                      give_amount: int = 4, receive_amount: int = 1) -> Action:
    """Create a bank trade action."""
    return Action(
        ActionType.TRADE_BANK,
        give_resource=give_resource,
        receive_resource=receive_resource,
        give_amount=give_amount,
        receive_amount=receive_amount
    )


def port_trade_action(give_resource: Resource, receive_resource: Resource,
                      port_vertices: Tuple[int, int], give_amount: int, 
                      receive_amount: int = 1) -> Action:
    """Create a port trade action."""
    return Action(
        ActionType.TRADE_PORT,
        give_resource=give_resource,
        receive_resource=receive_resource,
        port_vertices=port_vertices,
        give_amount=give_amount,
        receive_amount=receive_amount
    )


def roll_dice_action() -> Action:
    """Create a roll dice action."""
    return Action(ActionType.ROLL_DICE)


def end_turn_action() -> Action:
    """Create an end turn action."""
    return Action(ActionType.END_TURN)


def buy_dev_card_action() -> Action:
    """Create a buy development card action."""
    return Action(ActionType.BUY_DEVELOPMENT_CARD)


def play_knight_action(tile_id: int, target_player_id: Optional[int] = None) -> Action:
    """Create a play knight action."""
    return Action(ActionType.PLAY_KNIGHT, tile_id=tile_id, target_player_id=target_player_id)


def discard_cards_action(resources: dict) -> Action:
    """Create a discard cards action."""
    return Action(ActionType.DISCARD_CARDS, resources=resources)
