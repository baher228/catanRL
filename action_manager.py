"""
Action manager for validating and executing actions.
"""

from typing import List, Optional, Tuple
from game_state import GameState, GamePhase
from actions import Action, ActionType
from game_constants import (
    Resource, SETTLEMENT_COST, CITY_COST, ROAD_COST, DEVELOPMENT_CARD_COST,
    VP_PER_SETTLEMENT, VP_PER_CITY
)


class ActionManager:
    """Validates and executes actions on game state."""
    
    def __init__(self, game_state: GameState):
        self.game = game_state
    
    def get_legal_actions(self, player_id: int) -> List[Action]:
        """Get all legal actions for a player."""
        if self.game.phase == GamePhase.GAME_OVER:
            return []
        
        # Special case: players can discard even if it's not their turn
        if player_id in self.game.pending_discards:
            return self._get_discard_actions(player_id)
        
        # Otherwise, must be current player's turn
        if self.game.current_player.player_id != player_id:
            return []  # Not this player's turn
        
        if self.game.phase == GamePhase.SETUP:
            return self._get_setup_actions(player_id)
        
        # Check for pending actions that require current player
        if self.game.pending_robber_move:
            return self._get_robber_actions(player_id)
        
        # Normal actions
        actions = []
        
        # If haven't rolled dice yet
        if self.game.dice_roll is None:
            actions.append(Action(ActionType.ROLL_DICE))
        else:
            # Can build
            actions.extend(self._get_build_actions(player_id))
            
            # Can trade
            actions.extend(self._get_trade_actions(player_id))
            
            # Can buy/play development cards
            actions.extend(self._get_dev_card_actions(player_id))
            
            # Can always end turn
            actions.append(Action(ActionType.END_TURN))
        
        return actions
    
    def _get_setup_actions(self, player_id: int) -> List[Action]:
        """Get legal actions during setup phase."""
        actions = []
        player = self.game.players[player_id]
        placements = self.game.setup_placements[player_id]
        
        if placements['settlements'] < (self.game.setup_round + 1):
            # Need to place settlement - can place on any valid vertex during setup
            for vertex_id in self.game.board.vertices:
                if self._can_place_settlement(player_id, vertex_id, setup=True):
                    actions.append(Action(ActionType.BUILD_SETTLEMENT, vertex_id=vertex_id))
        elif placements['roads'] < (self.game.setup_round + 1):
            # Need to place road adjacent to last settlement
            last_settlement = self._get_last_settlement_placed(player_id)
            if last_settlement is not None:
                # Only check edges adjacent to the last settlement
                adjacent_edges = self.game.board.get_adjacent_edges(last_settlement)
                for edge in adjacent_edges:
                    if self._can_place_road(player_id, edge, setup=True):
                        actions.append(Action(ActionType.BUILD_ROAD, edge=edge))
        
        return actions
    
    def _get_last_settlement_placed(self, player_id: int) -> Optional[int]:
        """Get the last settlement placed by a player (for setup phase)."""
        return self.game.last_setup_settlement.get(player_id)
    
    def _get_build_actions(self, player_id: int) -> List[Action]:
        """Get legal building actions."""
        actions = []
        player = self.game.players[player_id]
        
        # Settlements - only check vertices connected to player's roads
        if player.can_afford(SETTLEMENT_COST) and player.settlements_remaining > 0:
            candidate_vertices = self._get_connected_vertices(player_id)
            for vertex_id in candidate_vertices:
                if self._can_place_settlement(player_id, vertex_id):
                    actions.append(Action(ActionType.BUILD_SETTLEMENT, vertex_id=vertex_id))
        
        # Cities - upgrade existing settlements
        if player.can_afford(CITY_COST) and player.cities_remaining > 0:
            for vertex_id in self.game.board.player_settlements.get(player_id, []):
                if self._can_upgrade_to_city(player_id, vertex_id):
                    actions.append(Action(ActionType.BUILD_CITY, vertex_id=vertex_id))
        
        # Roads - only check edges connected to player's network
        if player.can_afford(ROAD_COST) and player.roads_remaining > 0:
            candidate_edges = self._get_connected_edges(player_id)
            for edge in candidate_edges:
                if self._can_place_road(player_id, edge):
                    actions.append(Action(ActionType.BUILD_ROAD, edge=edge))
        
        return actions
    
    def _get_trade_actions(self, player_id: int) -> List[Action]:
        """Get legal trade actions."""
        actions = []
        player = self.game.players[player_id]
        
        # Bank trades (4:1)
        for give_resource in Resource:
            if player.resources.get(give_resource, 0) >= 4:
                for receive_resource in Resource:
                    if give_resource != receive_resource:
                        actions.append(Action(
                            ActionType.TRADE_BANK,
                            give_resource=give_resource,
                            receive_resource=receive_resource,
                            give_amount=4,
                            receive_amount=1
                        ))
        
        # TODO: Port trades, player trades
        
        return actions
    
    def _get_dev_card_actions(self, player_id: int) -> List[Action]:
        """Get legal development card actions."""
        actions = []
        player = self.game.players[player_id]
        
        # Buy development card
        if player.can_afford(DEVELOPMENT_CARD_COST) and len(self.game.dev_card_deck) > 0:
            actions.append(Action(ActionType.BUY_DEVELOPMENT_CARD))
        
        # TODO: Play development cards
        
        return actions
    
    def _get_discard_actions(self, player_id: int) -> List[Action]:
        """Get legal discard actions when player has >7 cards."""
        player = self.game.players[player_id]
        total = player.total_resource_count()
        discard_count = total // 2
        
        # Generate valid discard combinations
        actions = []
        
        # Simple approach: generate one valid combination per resource type
        # Try to discard evenly from all resources the player has
        available_resources = [r for r in Resource if player.resources.get(r, 0) > 0]
        
        if not available_resources:
            # Player has no resources (shouldn't happen but handle gracefully)
            return [Action(ActionType.DISCARD_CARDS, resources={})]
        
        # Generate a simple valid discard action
        discard_resources = {}
        remaining = discard_count
        
        # Try to distribute discards across available resources
        while remaining > 0:
            for resource in available_resources:
                if remaining <= 0:
                    break
                available = player.resources.get(resource, 0) - discard_resources.get(resource, 0)
                if available > 0:
                    discard_resources[resource] = discard_resources.get(resource, 0) + 1
                    remaining -= 1
        
        actions.append(Action(ActionType.DISCARD_CARDS, resources=discard_resources))
        
        # Optionally generate a few more variations
        # For now, one valid action is sufficient for AI agents
        
        return actions
    
    def _get_robber_actions(self, player_id: int) -> List[Action]:
        """Get legal robber movement actions."""
        actions = []
        for tile in self.game.board.tiles:
            if tile.id != self.game.robber_tile_id:
                # Can move robber to any tile except current location
                # TODO: Add steal actions
                actions.append(Action(ActionType.MOVE_ROBBER, tile_id=tile.id))
        return actions
    
    def _can_place_settlement(self, player_id: int, vertex_id: int, setup: bool = False) -> bool:
        """Check if player can place settlement at vertex."""
        # Must be connected to player's road (unless setup)
        if not setup:
            connected = False
            for neighbor in self.game.board.get_adjacent_vertices(vertex_id):
                edge = tuple(sorted([vertex_id, neighbor]))
                edge_data = self.game.board.get_edge_data(edge)
                if edge_data.get('owner') == player_id:
                    connected = True
                    break
            if not connected:
                return False
        
        # Use board's validation rules (without modifying state)
        return self.game.board.can_place_settlement(player_id, vertex_id)
    
    def _can_upgrade_to_city(self, player_id: int, vertex_id: int) -> bool:
        """Check if player can upgrade settlement to city."""
        return self.game.board.can_upgrade_to_city(player_id, vertex_id)
    
    def _can_place_road(self, player_id: int, edge: Tuple[int, int], setup: bool = False) -> bool:
        """Check if player can place road on edge."""
        return self.game.board.can_place_road(player_id, edge=edge)
    
    def _get_connected_vertices(self, player_id: int) -> set:
        """Get all vertices where player could potentially place a settlement.
        Returns vertices that are endpoints of player's roads."""
        connected_vertices = set()
        
        # Get all vertices that are endpoints of player's roads
        player_roads = self.game.board.player_roads.get(player_id, set())
        for v_a, v_b in player_roads:
            connected_vertices.add(v_a)
            connected_vertices.add(v_b)
        
        return connected_vertices
    
    def _get_connected_edges(self, player_id: int) -> set:
        """Get all edges where player could potentially place a road.
        Returns edges that are adjacent to player's settlements or roads."""
        connected_edges = set()
        
        # Get edges adjacent to player's settlements/cities
        player_settlements = self.game.board.player_settlements.get(player_id, set())
        for vertex_id in player_settlements:
            adjacent_edges = self.game.board.get_adjacent_edges(vertex_id)
            connected_edges.update(adjacent_edges)
        
        # Get edges adjacent to player's roads
        player_roads = self.game.board.player_roads.get(player_id, set())
        for v_a, v_b in player_roads:
            # Add edges adjacent to v_a
            for neighbor in self.game.board.graph.neighbors(v_a):
                edge = tuple(sorted([v_a, neighbor]))
                connected_edges.add(edge)
            # Add edges adjacent to v_b
            for neighbor in self.game.board.graph.neighbors(v_b):
                edge = tuple(sorted([v_b, neighbor]))
                connected_edges.add(edge)
        
        return connected_edges
    
    def execute_action(self, player_id: int, action: Action) -> bool:
        """
        Execute an action. Returns True if successful, False otherwise.
        """
        # Special case: allow discards even if not current player's turn
        if action.action_type == ActionType.DISCARD_CARDS:
            if player_id not in self.game.pending_discards:
                return False
        elif player_id != self.game.current_player.player_id:
            # For all other actions, must be current player
            return False
        
        player = self.game.players[player_id]
        
        if action.action_type == ActionType.ROLL_DICE:
            self.game.roll_dice()
            return True
        
        elif action.action_type == ActionType.BUILD_SETTLEMENT:
            if self.game.phase == GamePhase.SETUP or player.pay_cost(SETTLEMENT_COST):
                if self.game.board.place_settlement(player_id, action.vertex_id):
                    player.settlements_remaining -= 1
                    player.public_victory_points += VP_PER_SETTLEMENT
                    
                    # Track setup placements
                    if self.game.phase == GamePhase.SETUP:
                        self.game.setup_placements[player_id]['settlements'] += 1
                        # Track the most recently placed settlement for road placement
                        self.game.last_setup_settlement[player_id] = action.vertex_id
                        
                        # In second setup round, get resources
                        if self.game.setup_round == 1:
                            vertex_data = self.game.board.get_vertex_data(action.vertex_id)
                            for tile_id in vertex_data['adjacent_tiles']:
                                tile = self.game.board.tiles[tile_id]
                                if tile.resource != "desert":
                                    player.add_resource(Resource(tile.resource), 1)
                    
                    return True
            return False
        
        elif action.action_type == ActionType.BUILD_CITY:
            if player.pay_cost(CITY_COST):
                if self.game.board.upgrade_to_city(player_id, action.vertex_id):
                    player.cities_remaining -= 1
                    player.settlements_remaining += 1  # Get settlement piece back
                    player.public_victory_points += VP_PER_CITY - VP_PER_SETTLEMENT
                    return True
            return False
        
        elif action.action_type == ActionType.BUILD_ROAD:
            if self.game.phase == GamePhase.SETUP or player.pay_cost(ROAD_COST):
                if self.game.board.place_road(player_id, edge=action.edge):
                    player.roads_remaining -= 1
                    
                    if self.game.phase == GamePhase.SETUP:
                        self.game.setup_placements[player_id]['roads'] += 1
                        # Check if player completed their setup placements for this round
                        placements = self.game.setup_placements[player_id]
                        if placements['settlements'] >= (self.game.setup_round + 1) and \
                           placements['roads'] >= (self.game.setup_round + 1):
                            # Automatically advance turn after completing setup placements
                            self.game.next_turn()
                    
                    # TODO: Update longest road
                    return True
            return False
        
        elif action.action_type == ActionType.TRADE_BANK:
            if player.resources.get(action.give_resource, 0) >= action.give_amount:
                player.remove_resource(action.give_resource, action.give_amount)
                player.add_resource(action.receive_resource, action.receive_amount)
                return True
            return False
        
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            if player.pay_cost(DEVELOPMENT_CARD_COST) and self.game.dev_card_deck:
                card = self.game.dev_card_deck.pop()
                player.dev_cards[card] += 1
                player.dev_cards_bought_this_turn.append(card)
                return True
            return False
        
        elif action.action_type == ActionType.DISCARD_CARDS:
            # Validate discard amount
            total = player.total_resource_count()
            required_discard = total // 2
            
            if action.resources is None:
                action.resources = {}
            
            actual_discard = sum(action.resources.values())
            
            # Check if player has the resources to discard
            for resource, count in action.resources.items():
                if player.resources.get(resource, 0) < count:
                    return False
            
            # Check if discarding the correct amount
            if actual_discard != required_discard:
                return False
            
            # Execute discard
            for resource, count in action.resources.items():
                player.remove_resource(resource, count)
            
            # Remove player from pending discards
            self.game.pending_discards.discard(player_id)
            
            return True
        
        elif action.action_type == ActionType.MOVE_ROBBER:
            self.game.robber_tile_id = action.tile_id
            self.game.pending_robber_move = False
            # TODO: Steal from adjacent players
            return True
        
        elif action.action_type == ActionType.END_TURN:
            self.game.next_turn()
            return True
        
        return False
