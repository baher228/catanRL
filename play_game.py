"""
Example game loop demonstrating how to run a Catan game with agents.
"""

from game_state import GameState, GamePhase
from action_manager import ActionManager
from agents import RandomAgent, HumanAgent


def play_game(agents: list, max_turns: int = 100, verbose: bool = True):
    """
    Play a full game of Catan.
    
    Args:
        agents: List of agent instances (must match number of players)
        max_turns: Maximum number of turns before game is terminated
        verbose: Whether to print game progress
    
    Returns:
        tuple: (winner_id, game_state) - The player ID of the winner (or None) and the final game state
    """
    num_players = len(agents)
    game = GameState(num_players=num_players)
    action_mgr = ActionManager(game)
    
    # Notify agents of game start
    for agent in agents:
        obs = game.get_observation(agent.player_id)
        agent.on_game_start(obs)
    
    if verbose:
        print("="*60)
        print("CATAN GAME START")
        print("="*60)
        print(f"Players: {', '.join(str(a) for a in agents)}")
        print()
    
    turn_count = 0
    
    # Game loop
    while game.phase != GamePhase.GAME_OVER and turn_count < max_turns:
        # First, handle any pending discards from all players
        if game.pending_discards:
            if verbose:
                print(f"\n  Players must discard: {list(game.pending_discards)}")
            
            # Process discards for all players who need to
            for player_id in list(game.pending_discards):  # Use list() to avoid modification during iteration
                player = game.players[player_id]
                agent = agents[player_id]
                obs = game.get_observation(player_id)
                
                if verbose:
                    print(f"  {agent.name} must discard (has {player.total_resource_count()} cards)")
                
                # Get and execute discard action
                legal_actions = action_mgr.get_legal_actions(player_id)
                if legal_actions:
                    action = agent.choose_action(obs, legal_actions)
                    if verbose:
                        print(f"    Discarding: {action.resources}")
                    success = action_mgr.execute_action(player_id, action)
                    if not success:
                        if verbose:
                            print(f"    Discard failed!")
                else:
                    if verbose:
                        print(f"    No legal discard actions!")
                    # Force remove from pending discards to avoid infinite loop
                    game.pending_discards.discard(player_id)
        
        current_agent = agents[game.current_player_idx]
        current_player = game.current_player
        
        # Get observation
        obs = game.get_observation(current_player.player_id)
        
        # Notify agent of turn start
        current_agent.on_turn_start(obs)
        
        if verbose and game.phase == GamePhase.MAIN:
            print(f"\n--- Turn {game.turn_number}: {current_agent.name} ---")
            print(f"VP: {current_player.get_total_victory_points()}, "
                  f"Resources: {current_player.total_resource_count()}")
        
        # Agent takes actions until turn ends
        turn_active = True
        action_count = 0
        max_actions_per_turn = 20  # Safety limit
        
        while turn_active and action_count < max_actions_per_turn:
            # Get legal actions
            legal_actions = action_mgr.get_legal_actions(current_player.player_id)
            
            if not legal_actions:
                if verbose:
                    print(f"  No legal actions available! Ending turn.")
                # Force end turn to avoid infinite loop
                if game.phase == GamePhase.SETUP:
                    game.next_turn()
                turn_active = False
                break
            
            # Agent chooses action
            action = current_agent.choose_action(obs, legal_actions)
            
            if verbose and game.phase == GamePhase.MAIN:
                print(f"  Action: {action}")
            
            # Execute action
            success = action_mgr.execute_action(current_player.player_id, action)
            
            if not success:
                if verbose:
                    print(f"  Action failed!")
                break
            
            # Check if turn ended (either END_TURN action or automatic turn change during setup)
            if action.action_type.name == 'END_TURN':
                turn_active = False
            elif game.current_player_idx != current_player.player_id:
                # Turn was automatically advanced (e.g., during setup phase)
                turn_active = False
            
            # Update observation
            obs = game.get_observation(current_player.player_id)
            action_count += 1
        
        # Notify agent of turn end
        current_agent.on_turn_end(obs)
        
        # Check for victory
        winner = game.check_victory()
        if winner is not None:
            if verbose:
                print("\n" + "="*60)
                print(f"🎉 GAME OVER! {agents[winner].name} wins! 🎉")
                print("="*60)
                _print_final_scores(game, agents)
            return winner, game
        
        if game.phase == GamePhase.MAIN:
            turn_count += 1
    
    if verbose:
        print("\n" + "="*60)
        print("Game ended: Maximum turns reached")
        print("="*60)
        _print_final_scores(game, agents)
    
    return None, game


def _print_final_scores(game: GameState, agents: list):
    """Print final scores."""
    print("\nFinal Scores:")
    scores = [(p.get_total_victory_points(), p.player_id, agents[p.player_id].name) 
              for p in game.players]
    scores.sort(reverse=True)
    
    for i, (vp, pid, name) in enumerate(scores, 1):
        print(f"  {i}. {name}: {vp} VP")


def main():
    """Run a demo game."""
    # Create agents
    agents = [
        RandomAgent(0, "Alice (Random)"),
        RandomAgent(1, "Bob (Random)"),
        RandomAgent(2, "Charlie (Random)"),
        RandomAgent(3, "Diana (Random)"),
    ]
    
    # Play game
    winner, final_game = play_game(agents, max_turns=100, verbose=True)
    
    if winner is not None:
        print(f"\nWinner: Player {winner}")
    else:
        print("\nNo winner (max turns reached)")


if __name__ == "__main__":
    main()
