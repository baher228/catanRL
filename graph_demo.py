"""
Demo script showing the graph-based Catan board implementation.

This demonstrates how the NetworkX graph structure enables:
- Clean placement logic using graph operations
- Easy access to graph algorithms
- Efficient neighbor queries
"""

import catan_env

# Create a random board
print("Creating a random Catan board with graph structure...")
board = catan_env.generate_random_board()

print(f"\nBoard Statistics:")
print(f"  Tiles: {len(board.tiles)}")
print(f"  Vertices (nodes): {board.graph.number_of_nodes()}")
print(f"  Edges: {board.graph.number_of_edges()}")

# Demonstrate graph operations
print("\n" + "="*60)
print("GRAPH OPERATIONS DEMO")
print("="*60)

# Place a settlement
vertex_id = 10
print(f"\n1. Placing settlement for Player 0 at vertex {vertex_id}")
result = board.place_settlement(player_id=0, vertex_id=vertex_id)
print(f"   Result: {'SUCCESS' if result else 'FAILED'}")

# Show neighbors using graph
neighbors = list(board.graph.neighbors(vertex_id))
print(f"   Adjacent vertices (graph neighbors): {neighbors}")

# Try to place settlement at adjacent vertex (should fail)
if neighbors:
    print(f"\n2. Attempting to place settlement at adjacent vertex {neighbors[0]} (should fail)")
    result = board.place_settlement(player_id=1, vertex_id=neighbors[0])
    print(f"   Result: {'SUCCESS' if result else 'FAILED'} - Distance rule enforced via graph!")

# Place a road using graph edge
print(f"\n3. Placing road for Player 0 on an edge")
# Get edges incident to the settlement vertex
edges = list(board.graph.edges(vertex_id))
if edges:
    edge = edges[0]
    print(f"   Edge: {edge}")
    result = board.place_road(player_id=0, edge=edge)
    print(f"   Result: {'SUCCESS' if result else 'FAILED'}")

# Place another road extending from the first
print(f"\n4. Extending road network")
# Get neighbors of the second vertex in our road
if edges:
    second_vertex = edge[1] if edge[1] != vertex_id else edge[0]
    next_edges = [e for e in board.graph.edges(second_vertex) if board.graph.edges[e]['owner'] is None]
    if next_edges:
        next_edge = next_edges[0]
        print(f"   Edge: {next_edge}")
        result = board.place_road(player_id=0, edge=next_edge)
        print(f"   Result: {'SUCCESS' if result else 'FAILED'}")

# Demonstrate graph algorithm: find connected road network
print("\n" + "="*60)
print("GRAPH ALGORITHMS")
print("="*60)

# Create a subgraph of Player 0's roads
print("\n5. Analyzing Player 0's road network using graph algorithms")
player_0_edges = [(u, v) for u, v, data in board.graph.edges(data=True) if data['owner'] == 0]
if player_0_edges:
    print(f"   Player 0 has {len(player_0_edges)} roads")
    print(f"   Edges: {player_0_edges}")
    
    # Create subgraph
    road_subgraph = board.graph.edge_subgraph(player_0_edges)
    print(f"   Road network spans {road_subgraph.number_of_nodes()} vertices")

# Show vertex attributes via graph
print("\n6. Accessing vertex attributes via graph")
print(f"   Vertex {vertex_id} data: {board.graph.nodes[vertex_id]}")

# Show edge attributes
if player_0_edges:
    print(f"   Edge {player_0_edges[0]} data: {board.graph.edges[player_0_edges[0]]}")

print("\n" + "="*60)
print("BENEFITS OF GRAPH STRUCTURE")
print("="*60)
print("""
[+] Clean placement logic using graph.neighbors()
[+] Easy access to NetworkX algorithms (shortest paths, components, etc.)
[+] Efficient neighbor queries
[+] Natural representation of game structure
[+] Ready for RL features:
  - Connectivity analysis
  - Path finding
  - Resource region control
  - Strategic position evaluation
""")
