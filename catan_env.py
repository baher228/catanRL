import random
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Tile:
    id: int
    resource: str
    token: Optional[int]
    row: int
    col: int


# Resource distribution for standard Catan (19 tiles total)
RESOURCE_DISTRIBUTION = {
    "wood": 4,
    "brick": 3,
    "wheat": 4,
    "sheep": 4,
    "ore": 3,
    "desert": 1
}

# Number tokens for standard Catan (18 tokens, one per non-desert tile)
NUMBER_TOKENS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]

# Board layout: number of tiles per row (3-4-5-4-3)
ROW_SIZES = [3, 4, 5, 4, 3]


class BoardState:
    def __init__(self, tiles: list[Tile]):
        self.tiles = tiles
        self.graph = nx.Graph()
        self._edge_list_cache = [] 
        self._build_graph()
        self._edge_list_cache = list(self.graph.edges())
        
        # Performance optimization: O(1) connectivity checks
        self.player_roads = {}  # {player_id: set of edge tuples}
        self.player_settlements = {}  # {player_id: set of vertex IDs}
    
    def _build_graph(self):
        # Precise vertex counts for 54 nodes in a standard Catan hex grid
        vertex_rows = [3, 4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 3]
        
        vertex_id = 0
        for row_idx, num_vertices in enumerate(vertex_rows):
            for col_idx in range(num_vertices):
                self.graph.add_node(
                    vertex_id,
                    adjacent_tiles=[],
                    owner=None,
                    is_city=False
                )
                vertex_id += 1
        
        self.tile_to_vertices = self._compute_tile_vertex_mapping()
        edges_set = set()
        
        for tile_id, vertex_ids in self.tile_to_vertices.items():
            # Set adjacent_tiles for each vertex
            for vid in vertex_ids:
                if vid in self.graph:
                    self.graph.nodes[vid]['adjacent_tiles'].append(tile_id)
            
            # Create edges around tile perimeter (6 edges per tile)
            for i in range(6):
                v_a = vertex_ids[i]
                v_b = vertex_ids[(i + 1) % 6]
                
                edge_tuple = tuple(sorted([v_a, v_b]))
                if edge_tuple not in edges_set:
                    edges_set.add(edge_tuple)
                    self.graph.add_edge(edge_tuple[0], edge_tuple[1], owner=None)
    
    def _compute_tile_vertex_mapping(self) -> dict[int, list[int]]:
        mapping = {
            0: [0, 4, 8, 12, 7, 3],
            1: [1, 5, 9, 13, 8, 4],
            2: [2, 6, 10, 14, 9, 5],
            3: [7, 12, 17, 22, 16, 11],
            4: [8, 13, 18, 23, 17, 12],
            5: [9, 14, 19, 24, 18, 13],
            6: [10, 15, 20, 25, 19, 14],
            7: [16, 22, 28, 33, 27, 21],
            8: [17, 23, 29, 34, 28, 22],
            9: [18, 24, 30, 35, 29, 23],
            10: [19, 25, 31, 36, 30, 24],
            11: [20, 26, 32, 37, 31, 25],
            12: [28, 34, 39, 43, 38, 33],
            13: [29, 35, 40, 44, 39, 34],
            14: [30, 36, 41, 45, 40, 35],
            15: [31, 37, 42, 46, 41, 36],
            16: [39, 44, 48, 51, 47, 43],
            17: [40, 45, 49, 52, 48, 44],
            18: [41, 46, 50, 53, 49, 45],
        }
        
        return mapping
        
        return mapping
    
    def get_adjacent_vertices(self, vertex_id: int) -> list[int]:
        if vertex_id not in self.graph:
            return []
        return list(self.graph.neighbors(vertex_id))
    
    def get_adjacent_edges(self, vertex_id: int) -> list[tuple[int, int]]:
        if vertex_id not in self.graph:
            return []
        return [(vertex_id, neighbor) for neighbor in self.graph.neighbors(vertex_id)]
    
    def get_vertices_for_tile(self, tile_id: int) -> list[int]:
        return self.tile_to_vertices.get(tile_id, [])
    
    def can_place_settlement(self, player_id: int, vertex_id: int) -> bool:
        """Check if a settlement can be placed without modifying state."""
        if vertex_id not in self.graph:
            return False
        
        if self.graph.nodes[vertex_id]['owner'] is not None:
            return False
        
        for neighbor_id in self.graph.neighbors(vertex_id):
            if self.graph.nodes[neighbor_id]['owner'] is not None:
                return False
        
        return True
    
    def place_settlement(self, player_id: int, vertex_id: int) -> bool:
        if not self.can_place_settlement(player_id, vertex_id):
            return False
        
        self.graph.nodes[vertex_id]['owner'] = player_id
        self.graph.nodes[vertex_id]['is_city'] = False
        
        # Track settlement for fast connectivity checks
        if player_id not in self.player_settlements:
            self.player_settlements[player_id] = set()
        self.player_settlements[player_id].add(vertex_id)
        
        return True
    
    def can_upgrade_to_city(self, player_id: int, vertex_id: int) -> bool:
        """Check if a settlement can be upgraded to a city without modifying state."""
        if vertex_id not in self.graph:
            return False
        
        node = self.graph.nodes[vertex_id]
        
        if node['owner'] != player_id:
            return False
        if node['is_city']:
            return False
        
        return True
    
    def upgrade_to_city(self, player_id: int, vertex_id: int) -> bool:
        if not self.can_upgrade_to_city(player_id, vertex_id):
            return False
        
        self.graph.nodes[vertex_id]['is_city'] = True
        return True
    
    def can_place_road(self, player_id: int, edge_id: int = None, edge: tuple[int, int] = None) -> bool:
        """Check if a road can be placed without modifying state."""
        if edge_id is not None and edge is None:
            if edge_id >= len(self._edge_list_cache):
                return False
            edge = self._edge_list_cache[edge_id]

        if edge is None:
            return False
        
        edge = tuple(sorted(edge))
        v_a, v_b = edge
        
        if not self.graph.has_edge(v_a, v_b):
            return False
        
        if self.graph.edges[edge]['owner'] is not None:
            return False
        
        # O(1) connectivity check using player_settlements and player_roads
        connected = False
        
        # Check if either endpoint has player's settlement/city
        player_settlements = self.player_settlements.get(player_id, set())
        if v_a in player_settlements or v_b in player_settlements:
            connected = True
        
        # Check if any adjacent edge has player's road (O(1) set lookups)
        if not connected:
            player_roads = self.player_roads.get(player_id, set())
            
            # Check all edges incident to v_a
            for neighbor in self.graph.neighbors(v_a):
                adjacent_edge = tuple(sorted([v_a, neighbor]))
                if adjacent_edge in player_roads:
                    connected = True
                    break
            
            # Check all edges incident to v_b if not yet connected
            if not connected:
                for neighbor in self.graph.neighbors(v_b):
                    adjacent_edge = tuple(sorted([v_b, neighbor]))
                    if adjacent_edge in player_roads:
                        connected = True
                        break
        
        return connected
    
    def place_road(self, player_id: int, edge_id: int = None, edge: tuple[int, int] = None) -> bool:
        if not self.can_place_road(player_id, edge_id, edge):
            return False
        
        if edge_id is not None and edge is None:
            edge = self._edge_list_cache[edge_id]
        
        edge = tuple(sorted(edge))
        
        # Place road and track it
        self.graph.edges[edge]['owner'] = player_id
        
        if player_id not in self.player_roads:
            self.player_roads[player_id] = set()
        self.player_roads[player_id].add(edge)
        
        return True
    
    def get_vertex_data(self, vertex_id: int) -> dict:
        """Direct access to vertex data without object creation."""
        if vertex_id not in self.graph:
            raise KeyError(f"Vertex {vertex_id} not found")
        return self.graph.nodes[vertex_id]
    
    def get_edge_data(self, edge: tuple[int, int]) -> dict:
        """Direct access to edge data without object creation."""
        edge = tuple(sorted(edge))
        if not self.graph.has_edge(*edge):
            raise KeyError(f"Edge {edge} not found")
        return self.graph.edges[edge]
    
    @property
    def vertices(self):
        """Returns view of all vertex IDs. Use graph.nodes[id] for data access."""
        return self.graph.nodes
    
    @property
    def edges(self):
        """Returns view of all edges. Use graph.edges[edge] for data access."""
        return self.graph.edges


def generate_random_board() -> BoardState:
    resources = []
    for resource, count in RESOURCE_DISTRIBUTION.items():
        resources.extend([resource] * count)
    
    random.shuffle(resources)
    
    tokens = NUMBER_TOKENS.copy()
    random.shuffle(tokens)
    
    tiles = []
    tile_id = 0
    token_idx = 0
    
    for row_num, row_size in enumerate(ROW_SIZES):
        for col_num in range(row_size):
            resource = resources[tile_id]
            
            if resource == "desert":
                token = None
            else:
                token = tokens[token_idx]
                token_idx += 1
            
            tile = Tile(
                id=tile_id,
                resource=resource,
                token=token,
                row=row_num,
                col=col_num
            )
            tiles.append(tile)
            tile_id += 1
    
    return BoardState(tiles)


# ── Simplified RL Environment ─────────────────────────────────────────────────

class CatanEnv:
    """
    Simplified Catan environment for RL training.

    Simplified rules (to keep the action space tractable):
      - No dev cards (excluded from action space)
      - No robber / discard (auto-resolved when 7 is rolled)
      - No port trades (only 4:1 bank trades)

    Action space  (ACT_DIM = 202):
      0              ROLL_DICE
      1              END_TURN
      2  .. 55       BUILD_SETTLEMENT at vertex 0..53
      56 .. 109      BUILD_CITY       at vertex 0..53
      110.. 181      BUILD_ROAD       at edge   0..71
      182.. 201      TRADE_BANK       20 combos (5 give × 4 receive, give≠receive)

    Observation space (OBS_DIM = 404): see GameState.get_observation_vector()
    """

    NUM_VERTICES = 54
    NUM_EDGES    = 72
    OBS_DIM      = 404  # must match GameState.get_observation_vector()

    _NUM_TRADE = 20  # 5 resources × 4 receive options = 20 combos (give ≠ receive)

    ACT_DIM = 1 + 1 + NUM_VERTICES + NUM_VERTICES + NUM_EDGES + _NUM_TRADE  # = 202

    # ── index ranges (for readability) ───────────────────────────────
    _IDX_ROLL      = 0
    _IDX_END       = 1
    _IDX_SETTLE    = 2                                              # 2..55
    _IDX_CITY      = 2 + NUM_VERTICES                              # 56..109
    _IDX_ROAD      = 2 + 2 * NUM_VERTICES                          # 110..181
    _IDX_TRADE     = 2 + 2 * NUM_VERTICES + NUM_EDGES              # 182..201

    # ─────────────────────────────────────────────────────────────────

    def __init__(self, num_players: int = 4, max_turns: int = 300):
        self.num_players = num_players
        self.max_turns   = max_turns

        # lazy-imported to avoid circular imports at module load time
        from game_state    import GameState
        from action_manager import ActionManager
        from actions        import Action, ActionType
        from game_constants import Resource
        self._GameState     = GameState
        self._ActionManager = ActionManager
        self._Action        = Action
        self._ActionType    = ActionType
        self._Resource      = Resource

        # Trade bank combos: all (give, receive) pairs where give ≠ receive (20 total)
        self._TRADE_COMBOS = [(g, r) for g in Resource for r in Resource if g != r]
        assert len(self._TRADE_COMBOS) == self._NUM_TRADE

        self.game: Optional[object]   = None
        self.action_manager           = None
        self._edge_to_idx: dict       = {}

        self._init_game()

        # Verify edge count matches expected
        num_edges = len(self.game.board._edge_list_cache)
        assert num_edges == self.NUM_EDGES, (
            f"Expected {self.NUM_EDGES} edges, board has {num_edges}. "
            "Update CatanEnv.NUM_EDGES and ACT_DIM if the board layout changed."
        )

    # ── public API ────────────────────────────────────────────────────

    def reset(self) -> List[np.ndarray]:
        """Reset to a new game. Returns list of obs vectors, one per player."""
        self._init_game()
        return [self.game.get_observation_vector(i) for i in range(self.num_players)]

    @property
    def current_player(self) -> int:
        return self.game.current_player_idx

    def get_action_mask(self, player_id: int) -> np.ndarray:
        """Boolean mask over ACT_DIM actions for the given player."""
        mask = np.zeros(self.ACT_DIM, dtype=bool)
        for action in self.action_manager.get_legal_actions(player_id):
            idx = self._action_to_index(action)
            if idx is not None:
                mask[idx] = True
        if not mask.any():
            mask[self._IDX_END] = True  # safety fallback: END_TURN
        return mask

    def step(self, action_idx: int):
        """
        Execute *action_idx* for the current player.
        Returns (obs_list, rewards, done, info).
        """
        from game_state import GamePhase

        player_id  = self.current_player
        action     = self._index_to_action(action_idx)
        vp_before  = [p.get_total_victory_points() for p in self.game.players]
        res_before = [p.total_resource_count()      for p in self.game.players]

        # Execute; on failure fall back to END_TURN (only after dice roll)
        success = False
        if action is not None:
            success = self.action_manager.execute_action(player_id, action)

        if not success and self.game.phase == GamePhase.MAIN:
            if self.game.dice_roll is not None:
                self.action_manager.execute_action(
                    player_id, self._Action(self._ActionType.END_TURN)
                )

        # Auto-resolve robber / discard (simplified: skip those mechanics)
        self._auto_resolve_pending()

        vp_after  = [p.get_total_victory_points() for p in self.game.players]
        res_after = [p.total_resource_count()      for p in self.game.players]

        rewards = [(vp_after[i] - vp_before[i]) * 0.5 for i in range(self.num_players)]

        # Small shaping reward for collecting resources (encourages dice-rolling
        # and staying alive in the resource economy)
        for i in range(self.num_players):
            gained = res_after[i] - res_before[i]
            if gained > 0:
                rewards[i] += gained * 0.02

        # Small penalty for illegal / no-op actions (discourages lazy looping)
        if not success:
            rewards[player_id] -= 0.01

        winner = self.game.check_victory()
        done   = winner is not None or self.game.turn_number >= self.max_turns

        if winner is not None:
            rewards[winner] += 1.0
            for i in range(self.num_players):
                if i != winner:
                    rewards[i] -= 0.2

        obs  = [self.game.get_observation_vector(i) for i in range(self.num_players)]
        info = {"winner": winner, "turn": self.game.turn_number, "success": success}
        return obs, rewards, done, info

    # ── internals ─────────────────────────────────────────────────────

    def _init_game(self):
        from game_state     import GameState
        from action_manager import ActionManager
        self.game           = GameState(num_players=self.num_players)
        self.action_manager = ActionManager(self.game)
        # Build O(1) edge→index map (rebuilt each reset because board is random)
        self._edge_to_idx   = {
            edge: i for i, edge in enumerate(self.game.board._edge_list_cache)
        }

    def _auto_resolve_pending(self):
        """Auto-discard and auto-move robber so agents never see those actions."""
        # Auto-discard: strip half of each player's resources silently
        for pid in list(self.game.pending_discards):
            player      = self.game.players[pid]
            discard_n   = player.total_resource_count() // 2
            remaining   = discard_n
            for resource in list(self._Resource):
                while remaining > 0 and player.resources.get(resource, 0) > 0:
                    player.remove_resource(resource, 1)
                    remaining -= 1
        self.game.pending_discards.clear()

        # Auto-move robber to the first non-current tile
        if self.game.pending_robber_move:
            for tile in self.game.board.tiles:
                if tile.id != self.game.robber_tile_id:
                    self.game.robber_tile_id = tile.id
                    break
            self.game.pending_robber_move = False

    def _action_to_index(self, action) -> Optional[int]:
        """Convert an Action object → integer index (None if outside simplified space)."""
        at = action.action_type
        AT = self._ActionType
        if at == AT.ROLL_DICE:
            return self._IDX_ROLL
        if at == AT.END_TURN:
            return self._IDX_END
        if at == AT.BUILD_SETTLEMENT and action.vertex_id is not None:
            if 0 <= action.vertex_id < self.NUM_VERTICES:
                return self._IDX_SETTLE + action.vertex_id
        if at == AT.BUILD_CITY and action.vertex_id is not None:
            if 0 <= action.vertex_id < self.NUM_VERTICES:
                return self._IDX_CITY + action.vertex_id
        if at == AT.BUILD_ROAD and action.edge is not None:
            edge = tuple(sorted(action.edge))
            idx  = self._edge_to_idx.get(edge)
            if idx is not None:
                return self._IDX_ROAD + idx
        if at == AT.TRADE_BANK:
            combo = (action.give_resource, action.receive_resource)
            if combo in self._TRADE_COMBOS:
                return self._IDX_TRADE + self._TRADE_COMBOS.index(combo)
        return None

    def _index_to_action(self, idx: int) -> Optional[object]:
        """Convert integer index → Action object (None if out of range)."""
        AT = self._ActionType
        A  = self._Action
        if idx == self._IDX_ROLL:
            return A(AT.ROLL_DICE)
        if idx == self._IDX_END:
            return A(AT.END_TURN)
        if self._IDX_SETTLE <= idx < self._IDX_CITY:
            return A(AT.BUILD_SETTLEMENT, vertex_id=idx - self._IDX_SETTLE)
        if self._IDX_CITY <= idx < self._IDX_ROAD:
            return A(AT.BUILD_CITY, vertex_id=idx - self._IDX_CITY)
        if self._IDX_ROAD <= idx < self._IDX_TRADE:
            edge_i = idx - self._IDX_ROAD
            if edge_i < len(self.game.board._edge_list_cache):
                return A(AT.BUILD_ROAD, edge=self.game.board._edge_list_cache[edge_i])
        if self._IDX_TRADE <= idx < self.ACT_DIM:
            give, receive = self._TRADE_COMBOS[idx - self._IDX_TRADE]
            return A(AT.TRADE_BANK,
                     give_resource=give, receive_resource=receive,
                     give_amount=4, receive_amount=1)
        return None
