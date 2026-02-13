"""
Visualization utilities for Catan game state in Jupyter notebooks.
Extends the existing board visualization with game state info.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon, Circle
import numpy as np
from typing import Optional

from game_state import GameState, Player
from catan_env import BoardState, ROW_SIZES
import catan_env


# Player colors (matching notebook)
PLAYER_COLORS = ['#FF4444', '#4444FF', '#FFAA00', '#44FF44']


def plot_game_state(game: GameState, figsize=(16, 10)):
    """
    Plot the complete game state including board and player info.
    
    Args:
        game: GameState object
        figsize: Figure size (width, height)
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid: board on left (2/3), info on right (1/3)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    ax_board = fig.add_subplot(gs[:, :2])
    ax_info = fig.add_subplot(gs[0, 2])
    ax_players = fig.add_subplot(gs[1:, 2])
    
    # Plot board
    plot_board(game.board, ax=ax_board, show_legend=False)
    ax_board.set_title(f"Catan Board - Turn {game.turn_number}", fontsize=14, fontweight='bold')
    
    # Plot game info
    plot_game_info(game, ax=ax_info)
    
    # Plot player stats
    plot_player_stats(game, ax=ax_players)
    
    plt.tight_layout()
    return fig


def plot_game_info(game: GameState, ax=None):
    """Plot general game information."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    
    ax.axis('off')
    
    info_text = f"""
GAME INFO

Phase: {game.phase.name}
Turn: {game.turn_number}
Current Player: {game.current_player.name}
Last Dice Roll: {game.dice_roll if game.dice_roll else 'N/A'}

Robber Location: Tile {game.robber_tile_id}
Dev Cards Remaining: {len(game.dev_card_deck)}
"""
    
    ax.text(0.1, 0.5, info_text.strip(), fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return ax


def plot_player_stats(game: GameState, ax=None):
    """Plot player statistics."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    
    ax.axis('off')
    
    # Build player stats text
    stats_lines = ["PLAYER STATS\n" + "="*40]
    
    for player in game.players:
        color = PLAYER_COLORS[player.player_id % len(PLAYER_COLORS)]
        marker = "→ " if player.player_id == game.current_player_idx else "  "
        
        stats_lines.append(f"\n{marker}{player.name} (VP: {player.get_total_victory_points()})")
        stats_lines.append(f"  Resources: {player.total_resource_count()}")
        stats_lines.append(f"  Dev Cards: {sum(player.dev_cards.values())}")
        stats_lines.append(f"  Settlements: {5 - player.settlements_remaining}/5")
        stats_lines.append(f"  Cities: {4 - player.cities_remaining}/4")
        stats_lines.append(f"  Roads: {15 - player.roads_remaining}/15")
        
        if player.has_longest_road:
            stats_lines.append(f"  ★ Longest Road")
        if player.has_largest_army:
            stats_lines.append(f"  ★ Largest Army ({player.knights_played} knights)")
    
    stats_text = "\n".join(stats_lines)
    
    ax.text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    return ax


# Resource colors (matching notebook)
RESOURCE_COLORS = {
    'wood': '#228B22',
    'brick': '#8B4513',
    'wheat': '#FFD700',
    'sheep': '#90EE90',
    'ore': '#708090',
    'desert': '#F4A460',
}

# Hex geometry constants (matching notebook)
HEX_RADIUS = 1.0
HEX_WIDTH = np.sqrt(3) * HEX_RADIUS
HEX_HEIGHT = 2 * HEX_RADIUS


def get_tile_center(tile) -> tuple[float, float]:
    """Convert tile to 2D coordinates. Ported from catan_visualization.ipynb."""
    row, col = tile.row, tile.col
    y = -row * (HEX_HEIGHT * 0.75)
    row_width = catan_env.ROW_SIZES[row]
    row_offset = -(row_width - 1) * HEX_WIDTH / 2
    x = row_offset + col * HEX_WIDTH
    return x, y


def get_vertex_position(board, vertex_id: int) -> tuple[float, float]:
    """Get vertex position using proper hex geometry (at corners). Ported from catan_visualization.ipynb."""
    if vertex_id not in board.graph:
        return (0, 0)
    
    adjacent_tiles = board.graph.nodes[vertex_id]['adjacent_tiles']
    if not adjacent_tiles:
        return (0, 0)
    
    tile_id = adjacent_tiles[0]
    tile = board.tiles[tile_id]
    tile_center_x, tile_center_y = get_tile_center(tile)
    
    tile_vertices = board.get_vertices_for_tile(tile_id)
    try:
        vertex_index = tile_vertices.index(vertex_id)
    except ValueError:
        tile_centers = [get_tile_center(board.tiles[tid]) for tid in adjacent_tiles]
        return (sum(x for x, y in tile_centers) / len(tile_centers),
                sum(y for x, y in tile_centers) / len(tile_centers))
    
    angles = [np.pi/2, np.pi/6, -np.pi/6, -np.pi/2, -5*np.pi/6, 5*np.pi/6]
    angle = angles[vertex_index]
    return (tile_center_x + HEX_RADIUS * np.cos(angle),
            tile_center_y + HEX_RADIUS * np.sin(angle))


def plot_board(board: BoardState, ax=None, show_legend=True):
    """
    Plot the Catan board with tiles, roads, settlements, and cities.
    Ported from the working catan_visualization.ipynb.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_aspect('equal')
    
    # Draw tiles
    for tile in board.tiles:
        center_x, center_y = get_tile_center(tile)
        color = RESOURCE_COLORS.get(tile.resource, '#CCCCCC')
        
        # Hidden flat-top hexagon (keeps vertex/edge geometry intact)
        hidden_hex = RegularPolygon(
            (center_x, center_y), numVertices=6, radius=HEX_RADIUS,
            orientation=np.pi/6, facecolor='none', edgecolor='none',
            linewidth=0, alpha=0
        )
        ax.add_patch(hidden_hex)
        
        # Visible pointy-top hexagon (vertices pointing up)
        visible_hex = RegularPolygon(
            (center_x, center_y), numVertices=6, radius=HEX_RADIUS,
            orientation=0, facecolor=color, edgecolor='black',
            linewidth=2, alpha=0.7
        )
        ax.add_patch(visible_hex)
        
        if tile.token is not None:
            circle = Circle((center_x, center_y), radius=0.35,
                          facecolor='white', edgecolor='black',
                          linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            
            number_color = 'red' if tile.token in [6, 8] else 'black'
            ax.text(center_x, center_y, str(tile.token),
                   ha='center', va='center', fontsize=16,
                   fontweight='bold', color=number_color, zorder=4)
    
    # Draw roads (from graph edges)
    for v_a, v_b, data in board.graph.edges(data=True):
        if data['owner'] is not None:
            x1, y1 = get_vertex_position(board, v_a)
            x2, y2 = get_vertex_position(board, v_b)
            color = PLAYER_COLORS[data['owner'] % len(PLAYER_COLORS)]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=5,
                   solid_capstyle='round', zorder=5)
    
    # Draw settlements and cities (from graph nodes)
    for vertex_id, data in board.graph.nodes(data=True):
        if data['owner'] is not None:
            x, y = get_vertex_position(board, vertex_id)
            color = PLAYER_COLORS[data['owner'] % len(PLAYER_COLORS)]
            
            if data['is_city']:
                ax.scatter(x, y, s=500, marker='s', color=color,
                          edgecolors='black', linewidths=2.5, zorder=6)
            else:
                ax.scatter(x, y, s=250, marker='o', color=color,
                          edgecolors='black', linewidths=2.5, zorder=6)
    
    # Styling - explicit axis limits (critical for correct rendering)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-8, 2.5)
    ax.axis('off')
    
    if show_legend:
        # Legend
        resource_patches = [
            mpatches.Patch(color=color, label=resource.capitalize())
            for resource, color in RESOURCE_COLORS.items()
        ]
        
        active_players = set()
        for _, data in board.graph.nodes(data=True):
            if data['owner'] is not None:
                active_players.add(data['owner'])
        for _, _, data in board.graph.edges(data=True):
            if data['owner'] is not None:
                active_players.add(data['owner'])
        
        player_patches = [
            mpatches.Patch(color=PLAYER_COLORS[p % len(PLAYER_COLORS)], label=f'Player {p}')
            for p in sorted(active_players)
        ]
        
        all_patches = resource_patches
        if player_patches:
            all_patches.append(mpatches.Patch(color='white', label=''))
            all_patches.extend(player_patches)
        
        ax.legend(handles=all_patches, loc='upper left',
                 bbox_to_anchor=(1.02, 1), fontsize=10)
    
    return ax



def plot_player_resources(player: Player, ax=None):
    """Plot a player's resources as a bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    resources = [r.value for r in player.resources.keys()]
    counts = [player.resources[r] for r in player.resources.keys()]
    
    colors = ['#2d5016', '#8b4513', '#f4e285', '#90ee90', '#696969']
    
    ax.bar(resources, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Count')
    ax.set_title(f"{player.name}'s Resources (Total: {player.total_resource_count()})")
    ax.grid(axis='y', alpha=0.3)
    
    return ax
