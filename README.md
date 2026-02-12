# Catan Board Visualization - Jupyter Notebook Update

## Files Overview

**Keep these files:**
- ✅ `catan_env.py` - Core game logic with NetworkX graph
- ✅ `catan_visualization.ipynb` - Jupyter notebook (needs updating)
- ✅ `graph_demo.py` - Demo of graph features
- ✅ `notebook_viz_code.py` - Updated visualization code

**Removed (redundant):**
- ❌ `catan_board.py` - Old implementation
- ❌ `fixed_visualization.py` - Temporary helper

## How to Update Your Notebook

### Step 1: Open the notebook
Open `catan_visualization.ipynb` in Jupyter

### Step 2: Update Cell 2 (Visualization Functions)
Replace the **second code cell** (the one with all the functions) with the contents of `notebook_viz_code.py`

The new code:
- Works with NetworkX graph structure
- Accesses vertex/edge data from `board.graph`
- Uses proper hex geometry for vertex positions

### Step 3: Update Example Cells (Optional)

The examples should work as-is, but you can also use the new graph-based API:

```python
# Old way (still works)
board.place_road(player_id=0, edge_id=5)

# New way (more explicit)
board.place_road(player_id=0, edge=(10, 11))

# Graph operations
neighbors = list(board.graph.neighbors(10))
edges_at_vertex = list(board.graph.edges(10))
```

### Step 4: Run the cells
All cells should work with the updated visualization code!

## Quick Test

Run this in a new notebook cell to verify everything works:

```python
import catan_env

# Create board
board = catan_env.generate_random_board()

# Place some pieces
board.place_settlement(0, 10)
board.place_settlement(1, 30)

# Get an edge connected to vertex 10
edges = list(board.graph.edges(10))
if edges:
    board.place_road(0, edge=edges[0])

# Visualize
plot_board(board)
```

## Dependencies

Make sure you have:
```bash
pip install networkx matplotlib numpy jupyter
```

## What Changed

- Board now uses NetworkX `graph` internally
- Vertices = graph nodes with attributes (owner, is_city, adjacent_tiles)
- Edges = graph edges with attributes (owner)
- Visualization accesses data via `board.graph.nodes` and `board.graph.edges`
- All placement logic uses graph operations (neighbors, etc.)

See `graph_demo.py` for examples of graph features!
