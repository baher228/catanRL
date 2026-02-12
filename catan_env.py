import random
import networkx as nx
from dataclasses import dataclass
from typing import Optional


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
        
        self._build_graph()
    
    def _build_graph(self):
        vertex_rows = [7, 9, 11, 11, 9, 7]
        
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
        

        for tile_id, vertex_ids in self.tile_to_vertices.items():
            for vid in vertex_ids:
                if vid in self.graph:
                    self.graph.nodes[vid]['adjacent_tiles'].append(tile_id)
        
        edges_set = set()
        
        for tile_id, vertex_ids in self.tile_to_vertices.items():
            for i in range(len(vertex_ids)):
                v_a = vertex_ids[i]
                v_b = vertex_ids[(i + 1) % len(vertex_ids)]
                
                edge_tuple = tuple(sorted([v_a, v_b]))
                if edge_tuple not in edges_set:
                    edges_set.add(edge_tuple)
                    self.graph.add_edge(edge_tuple[0], edge_tuple[1], owner=None)
    
    def _compute_tile_vertex_mapping(self) -> dict[int, list[int]]:
        mapping = {
            0: [0, 1, 8, 9, 7, 2],
            1: [2, 3, 10, 11, 9, 8],
            2: [3, 4, 12, 13, 11, 10],
            3: [7, 9, 17, 18, 16, 5],
            4: [9, 11, 19, 20, 18, 17],
            5: [11, 13, 21, 22, 20, 19],
            6: [13, 14, 23, 24, 22, 21],
            7: [16, 18, 27, 28, 26, 25],
            8: [18, 20, 29, 30, 28, 27],
            9: [20, 22, 31, 32, 30, 29],
            10: [22, 24, 33, 34, 32, 31],
            11: [24, 15, 35, 36, 34, 33],
            12: [26, 28, 38, 39, 37, 6],
            13: [28, 30, 40, 41, 39, 38],
            14: [30, 32, 42, 43, 41, 40],
            15: [32, 34, 44, 45, 43, 42],
            16: [37, 39, 47, 48, 46, 25],
            17: [39, 41, 49, 50, 48, 47],
            18: [41, 43, 51, 52, 50, 49],
        }
        
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
    
    def place_settlement(self, player_id: int, vertex_id: int) -> bool:
        if vertex_id not in self.graph:
            return False
        
        if self.graph.nodes[vertex_id]['owner'] is not None:
            return False
        
        for neighbor_id in self.graph.neighbors(vertex_id):
            if self.graph.nodes[neighbor_id]['owner'] is not None:
                return False
        
        self.graph.nodes[vertex_id]['owner'] = player_id
        self.graph.nodes[vertex_id]['is_city'] = False
        return True
    
    def upgrade_to_city(self, player_id: int, vertex_id: int) -> bool:
        if vertex_id not in self.graph:
            return False
        
        node = self.graph.nodes[vertex_id]
        
        if node['owner'] != player_id:
            return False
        if node['is_city']:
            return False
        
        self.graph.nodes[vertex_id]['is_city'] = True
        return True
    
    def place_road(self, player_id: int, edge_id: int = None, edge: tuple[int, int] = None) -> bool:
        if edge_id is not None and edge is None:
            edges_list = list(self.graph.edges())
            if edge_id >= len(edges_list):
                return False
            edge = edges_list[edge_id]
        
        if edge is None:
            return False
        
        edge = tuple(sorted(edge))
        v_a, v_b = edge
        
        if not self.graph.has_edge(v_a, v_b):
            return False
        
        if self.graph.edges[edge]['owner'] is not None:
            return False
        
        connected = False
        

        if self.graph.nodes[v_a]['owner'] == player_id:
            connected = True
        if self.graph.nodes[v_b]['owner'] == player_id:
            connected = True
        
        if not connected:
            for neighbor in self.graph.neighbors(v_a):
                edge_key = tuple(sorted([v_a, neighbor]))
                if self.graph.edges[edge_key]['owner'] == player_id:
                    connected = True
                    break
            
            if not connected:
                for neighbor in self.graph.neighbors(v_b):
                    edge_key = tuple(sorted([v_b, neighbor]))
                    if self.graph.edges[edge_key]['owner'] == player_id:
                        connected = True
                        break
        
        if not connected:
            return False
        
        self.graph.edges[edge]['owner'] = player_id
        return True
    
    @property
    def vertices(self):
        class VertexView:
            def __init__(self, graph):
                self.graph = graph
            
            def __len__(self):
                return len(self.graph.nodes)
            
            def __getitem__(self, idx):
                if idx not in self.graph.nodes:
                    raise IndexError(f"Vertex {idx} not found")
                
                class Vertex:
                    def __init__(self, vid, data):
                        self.id = vid
                        self.adjacent_tiles = data['adjacent_tiles']
                        self.owner = data['owner']
                        self.is_city = data['is_city']
                
                return Vertex(idx, self.graph.nodes[idx])
        
        return VertexView(self.graph)
    
    @property
    def edges(self):
        class EdgeView:
            def __init__(self, graph):
                self.graph = graph
                self._edges = list(graph.edges())
            
            def __len__(self):
                return len(self._edges)
            
            def __iter__(self):
                for idx, edge in enumerate(self._edges):
                    class Edge:
                        def __init__(self, edge_id, edge_tuple, data):
                            self.id = edge_id
                            self.vertex_a = edge_tuple[0]
                            self.vertex_b = edge_tuple[1]
                            self.owner = data['owner']
                    
                    yield Edge(idx, edge, self.graph.edges[edge])
        
        return EdgeView(self.graph)


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
