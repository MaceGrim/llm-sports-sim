#!/usr/bin/env python3
"""
Demo script to test the NBA simulation system.
"""

import os
from src.data import NBADataProcessor
from src.prompting import NBAPromptGenerator, GameContext, TeamLineup
from src.evaluation import parse_llm_output, EvaluationMetrics


def demo_data_processing():
    """Demo the data processing functionality."""
    print("=== NBA Data Processing Demo ===")
    
    data_dir = "nba_data"
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found")
        return
    
    processor = NBADataProcessor(data_dir)
    
    # Get available teams
    teams = processor.get_teams_in_dataset()
    print(f"Teams in dataset: {teams[:10]}... ({len(teams)} total)")
    
    # Get some game files
    game_files = processor.get_game_files()[:5]
    print(f"\nProcessing sample games: {game_files}")
    
    for filename in game_files:
        try:
            file_path = os.path.join(data_dir, filename)
            game_info = processor.process_game_file(file_path)
            
            print(f"\n{filename}:")
            print(f"  {game_info.away_team} @ {game_info.home_team}")
            print(f"  Final Score: {game_info.final_score}")
            print(f"  Date: {game_info.date}")
            
            # Show a few player stats
            if game_info.player_stats:
                top_players = sorted(
                    game_info.player_stats.items(), 
                    key=lambda x: x[1].get('points', 0), 
                    reverse=True
                )[:3]
                
                print("  Top Scorers:")
                for player, stats in top_players:
                    print(f"    {player}: {stats.get('points', 0)} pts, {stats.get('rebounds', 0)} reb, {stats.get('assists', 0)} ast")
                    
        except Exception as e:
            print(f"  Error processing {filename}: {e}")


def demo_prompt_generation():
    """Demo the prompt generation functionality."""
    print("\n=== Prompt Generation Demo ===")
    
    # Create sample game context
    home_team = TeamLineup(
        team_name="Boston Celtics",
        team_abbrev="BOS", 
        players=[
            {"name": "Jayson Tatum", "position": "SF"},
            {"name": "Jaylen Brown", "position": "SG"},
            {"name": "Marcus Smart", "position": "PG"},
            {"name": "Al Horford", "position": "C"},
            {"name": "Robert Williams", "position": "PF"}
        ],
        is_home=True
    )
    
    away_team = TeamLineup(
        team_name="Philadelphia 76ers",
        team_abbrev="PHI",
        players=[
            {"name": "Joel Embiid", "position": "C"},
            {"name": "James Harden", "position": "PG"},
            {"name": "Tyrese Maxey", "position": "SG"},
            {"name": "Tobias Harris", "position": "PF"},
            {"name": "P.J. Tucker", "position": "SF"}
        ],
        is_home=False
    )
    
    context = GameContext(
        home_team=home_team,
        away_team=away_team,
        date="2022-10-18"
    )
    
    generator = NBAPromptGenerator()
    
    # Generate different types of prompts
    strategies = ['basic', 'detailed', 'few_shot', 'chain_of_thought']
    
    for strategy in strategies:
        print(f"\n--- {strategy.title()} Prompt ---")
        
        if strategy == 'basic':
            prompt = generator.generate_basic_prompt(context)
        elif strategy == 'detailed':
            prompt = generator.generate_detailed_prompt(context)
        elif strategy == 'few_shot':
            prompt = generator.generate_few_shot_prompt(context)
        elif strategy == 'chain_of_thought':
            prompt = generator.generate_chain_of_thought_prompt(context)
        
        # Show first 300 characters
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)


def demo_evaluation():
    """Demo the evaluation functionality."""
    print("\n=== Evaluation Demo ===")
    
    # Sample LLM output (realistic format)
    sample_output = '''
    {
      "final_score": {"home": 108, "away": 102},
      "quarter_scores": [[28, 24], [26, 28], [29, 25], [25, 25]],
      "player_stats": {
        "Jayson Tatum": {"points": 32, "rebounds": 8, "assists": 5, "minutes": 38},
        "Joel Embiid": {"points": 28, "rebounds": 12, "assists": 3, "minutes": 36},
        "Jaylen Brown": {"points": 22, "rebounds": 6, "assists": 4, "minutes": 35},
        "James Harden": {"points": 18, "rebounds": 5, "assists": 9, "minutes": 34}
      }
    }
    '''
    
    # Parse the output
    predicted_result = parse_llm_output(sample_output)
    print("Parsed LLM Output:")
    print(f"  Final Score: {predicted_result.final_score}")
    print(f"  Quarter Scores: {predicted_result.quarter_scores}")
    print(f"  Number of Players: {len(predicted_result.player_stats)}")
    
    # Create fake "actual" result for comparison
    from src.evaluation import GameResult
    actual_result = GameResult(
        final_score={"home": 110, "away": 98},
        quarter_scores=[[30, 22], [28, 26], [27, 24], [25, 26]],
        player_stats={
            "Jayson Tatum": {"points": 35, "rebounds": 7, "assists": 6, "minutes": 40},
            "Joel Embiid": {"points": 25, "rebounds": 15, "assists": 2, "minutes": 38},
            "Jaylen Brown": {"points": 20, "rebounds": 5, "assists": 3, "minutes": 36},
            "James Harden": {"points": 15, "rebounds": 4, "assists": 11, "minutes": 32}
        }
    )
    
    # Evaluate
    evaluator = EvaluationMetrics()
    metrics = evaluator.evaluate_game(predicted_result, actual_result)
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")


def main():
    """Run all demos."""
    try:
        demo_data_processing()
        demo_prompt_generation()
        demo_evaluation()
        
        print("\n=== Demo Complete ===")
        print("To run a full benchmark, use: python run_benchmark.py --models gpt-3.5-turbo --num-games 5")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()