#!/usr/bin/env python3
"""
Main script to run NBA game simulation benchmarks.
"""

import argparse
import json
import os
from typing import List, Dict, Any
import time
from datetime import datetime

from src.models import LLMConfig, LLMFactory, LLMBenchmark, PREDEFINED_CONFIGS
from src.prompting import NBAPromptGenerator, GameContext, TeamLineup
from src.evaluation import DatasetEvaluator, parse_llm_output, GameResult
from src.data import NBADataProcessor


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_results(results: Dict[str, Any], output_path: str):
    """Save benchmark results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def run_prompting_benchmark(
    models: List[str],
    test_cases: List[tuple],
    prompt_strategy: str = 'detailed',
    output_dir: str = 'results'
) -> Dict[str, Any]:
    """Run prompting-based simulation benchmark."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    prompt_generator = NBAPromptGenerator()
    evaluator = DatasetEvaluator()
    
    # Prepare LLM configs
    llm_configs = []
    for model_name in models:
        if model_name in PREDEFINED_CONFIGS:
            llm_configs.append(PREDEFINED_CONFIGS[model_name])
        else:
            print(f"Warning: Unknown model {model_name}, skipping...")
    
    if not llm_configs:
        raise ValueError("No valid LLM configurations found")
    
    # Create benchmark runner
    benchmark = LLMBenchmark(llm_configs)
    
    # Generate prompts for all test cases
    prompts = []
    actual_results = []
    
    for context, actual_result in test_cases:
        if prompt_strategy == 'basic':
            prompt = prompt_generator.generate_basic_prompt(context)
        elif prompt_strategy == 'detailed':
            prompt = prompt_generator.generate_detailed_prompt(context)
        elif prompt_strategy == 'few_shot':
            prompt = prompt_generator.generate_few_shot_prompt(context)
        elif prompt_strategy == 'chain_of_thought':
            prompt = prompt_generator.generate_chain_of_thought_prompt(context)
        else:
            raise ValueError(f"Unknown prompt strategy: {prompt_strategy}")
        
        prompts.append(prompt)
        actual_results.append(actual_result)
    
    print(f"Running benchmark with {len(llm_configs)} models on {len(prompts)} test cases...")
    
    # Run benchmark
    raw_results = benchmark.run_benchmark(prompts)
    
    # Process results and evaluate
    benchmark_results = {}
    
    for model_name, model_outputs in raw_results.items():
        print(f"\nEvaluating {model_name}...")
        
        # Parse LLM outputs
        predicted_results = []
        for output_data in model_outputs:
            predicted_result = parse_llm_output(output_data['output'])
            predicted_results.append(predicted_result)
        
        # Evaluate predictions
        evaluation_metrics = evaluator.evaluate_dataset(predicted_results, actual_results)
        
        # Store results
        benchmark_results[model_name] = {
            'evaluation_metrics': evaluation_metrics,
            'raw_outputs': model_outputs,
            'parsed_predictions': [
                {
                    'final_score': pred.final_score,
                    'quarter_scores': pred.quarter_scores,
                    'player_stats': pred.player_stats,
                    'metadata': pred.metadata
                } for pred in predicted_results
            ]
        }
        
        # Print summary
        print(f"  Overall Score: {evaluation_metrics.get('overall_score', 0):.3f}")
        print(f"  Score MAE: {evaluation_metrics.get('score_mae_total_mean', 0):.2f}")
        print(f"  Format Compliance: {evaluation_metrics.get('format_compliance_mean', 0):.3f}")
        print(f"  Basketball Realism: {evaluation_metrics.get('basketball_realism_mean', 0):.3f}")
    
    # Add metadata
    benchmark_results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'prompt_strategy': prompt_strategy,
        'num_test_cases': len(test_cases),
        'models_tested': models
    }
    
    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description='Run NBA simulation benchmark')
    parser.add_argument('--models', nargs='+', default=['gpt-3.5-turbo'], 
                       help='Models to benchmark')
    parser.add_argument('--data-dir', default='nba_data', 
                       help='Directory containing NBA CSV files')
    parser.add_argument('--num-games', type=int, default=10, 
                       help='Number of games to test')
    parser.add_argument('--prompt-strategy', default='detailed',
                       choices=['basic', 'detailed', 'few_shot', 'chain_of_thought'],
                       help='Prompting strategy to use')
    parser.add_argument('--output-dir', default='results', 
                       help='Output directory for results')
    parser.add_argument('--config', help='Path to custom config file')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load custom config if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Initialize data processor
    print(f"Loading NBA data from {args.data_dir}...")
    data_processor = NBADataProcessor(args.data_dir)
    
    # Create test cases
    print(f"Creating {args.num_games} test cases...")
    test_cases = data_processor.create_test_cases(
        num_games=args.num_games,
        random_seed=args.seed
    )
    
    if not test_cases:
        print("Error: No test cases could be created from the data")
        return
    
    print(f"Created {len(test_cases)} test cases")
    
    # Run benchmark
    results = run_prompting_benchmark(
        models=args.models,
        test_cases=test_cases,
        prompt_strategy=args.prompt_strategy,
        output_dir=args.output_dir
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output_dir, f'benchmark_results_{timestamp}.json')
    save_results(results, output_file)
    
    print(f"\nBenchmark complete! Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for model_name, model_results in results.items():
        if model_name == 'metadata':
            continue
        
        metrics = model_results['evaluation_metrics']
        print(f"\n{model_name}:")
        print(f"  Overall Score: {metrics.get('overall_score', 0):.3f}")
        print(f"  Avg Score Error: {metrics.get('score_mae_total_mean', 0):.1f} points")
        print(f"  Format Compliance: {metrics.get('format_compliance_mean', 0):.1%}")
        print(f"  Basketball Realism: {metrics.get('basketball_realism_mean', 0):.1%}")
        print(f"  Winner Prediction: {metrics.get('winner_correct_mean', 0):.1%}")


if __name__ == '__main__':
    main()