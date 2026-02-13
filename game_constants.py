"""
Game constants for Settlers of Catan.
Defines resources, building costs, development cards, and game rules.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict


class Resource(Enum):
    """Resource types in Catan."""
    WOOD = "wood"
    BRICK = "brick"
    WHEAT = "wheat"
    SHEEP = "sheep"
    ORE = "ore"


class DevelopmentCard(Enum):
    """Development card types."""
    KNIGHT = "knight"
    VICTORY_POINT = "victory_point"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"


# Building costs (resource type -> count)
SETTLEMENT_COST = {
    Resource.WOOD: 1,
    Resource.BRICK: 1,
    Resource.WHEAT: 1,
    Resource.SHEEP: 1,
}

CITY_COST = {
    Resource.ORE: 3,
    Resource.WHEAT: 2,
}

ROAD_COST = {
    Resource.WOOD: 1,
    Resource.BRICK: 1,
}

DEVELOPMENT_CARD_COST = {
    Resource.ORE: 1,
    Resource.WHEAT: 1,
    Resource.SHEEP: 1,
}

# Development card deck composition (25 cards total)
DEV_CARD_DECK_COMPOSITION = {
    DevelopmentCard.KNIGHT: 14,
    DevelopmentCard.VICTORY_POINT: 5,
    DevelopmentCard.ROAD_BUILDING: 2,
    DevelopmentCard.YEAR_OF_PLENTY: 2,
    DevelopmentCard.MONOPOLY: 2,
}

# Victory points
VP_PER_SETTLEMENT = 1
VP_PER_CITY = 2
VP_LONGEST_ROAD = 2
VP_LARGEST_ARMY = 2
VP_TO_WIN = 10

# Road requirements
MIN_ROAD_LENGTH_FOR_LONGEST = 5

# Army requirements
MIN_KNIGHTS_FOR_LARGEST = 3

# Trade ratios
BANK_TRADE_RATIO = 4  # 4 of same resource for 1 of any other
GENERIC_PORT_RATIO = 3  # 3 of same resource for 1 of any other
SPECIFIC_PORT_RATIO = 2  # 2 of specific resource for 1 of any other

# Resource limits
MAX_RESOURCES_BEFORE_DISCARD = 7  # Must discard half if 7 is rolled and you have more than this

# Initial pieces per player
INITIAL_SETTLEMENTS = 5
INITIAL_CITIES = 4
INITIAL_ROADS = 15

# Number of resource cards
RESOURCE_CARDS_PER_TYPE = 19

# Dice probabilities (for reference, not used in code)
DICE_PROBABILITIES = {
    2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
    7: 6/36,
    8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
}


@dataclass
class Port:
    """Represents a port on the board."""
    vertices: tuple[int, int]  # The two vertices where this port is located
    resource: Resource | None  # None for generic 3:1 port, specific resource for 2:1 port
    
    @property
    def trade_ratio(self) -> int:
        """Returns the trade ratio for this port."""
        return SPECIFIC_PORT_RATIO if self.resource else GENERIC_PORT_RATIO


# Helper functions for resource management
def can_afford(resources: Dict[Resource, int], cost: Dict[Resource, int]) -> bool:
    """Check if player has enough resources to pay a cost."""
    for resource, amount in cost.items():
        if resources.get(resource, 0) < amount:
            return False
    return True


def deduct_resources(resources: Dict[Resource, int], cost: Dict[Resource, int]) -> None:
    """Deduct cost from resources (modifies in place)."""
    for resource, amount in cost.items():
        resources[resource] = resources.get(resource, 0) - amount


def add_resources(resources: Dict[Resource, int], gains: Dict[Resource, int]) -> None:
    """Add resources (modifies in place)."""
    for resource, amount in gains.items():
        resources[resource] = resources.get(resource, 0) + amount


def total_resources(resources: Dict[Resource, int]) -> int:
    """Count total number of resource cards."""
    return sum(resources.values())
