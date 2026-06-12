# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM Sports Simulation project focused on NBA basketball data analysis and simulation. The project implements three different approaches to LLM-based NBA game simulation:

1. **Prompting-Only**: Using existing LLMs with zero-shot/few-shot prompting
2. **Fine-Tuning**: Fine-tuned models on NBA data
3. **Structured Decoding**: Constrained generation ensuring valid basketball outputs

## Project Structure

```
llm-sports-sim/
├── nba_data/                    # NBA game CSV files (900+ games)
├── src/
│   ├── data/                    # Data processing utilities
│   │   └── nba_data_processor.py
│   ├── evaluation/              # Evaluation metrics and benchmarking
│   │   └── metrics.py
│   ├── models/                  # LLM interfaces (API and local)
│   │   └── llm_interface.py
│   └── prompting/               # Prompt generation strategies
│       └── nba_prompts.py
├── run_benchmark.py             # Main benchmarking script
├── demo.py                      # Demo/testing script
├── requirements.txt             # Python dependencies
└── config_example.json          # Example configuration
```

## Data Structure

### NBA Data Format
- Location: `nba_data/` directory
- File naming convention: `[YYYY-MM-DD]-[game_id]-[AWAY]@[HOME].csv`
- Example: `[2022-10-18]-0022200001-PHI@BOS.csv`
- Contains play-by-play data with player positions, events, coordinates, and game state

### CSV Schema
Each game file contains detailed play-by-play data with these key columns:
- Game metadata: `game_id`, `date`, `period`, `remaining_time`
- Player lineups: `a1-a5` (away team), `h1-h5` (home team)
- Event data: `event_type`, `player`, `team`, `description`
- Basketball metrics: `shot_distance`, `points`, `possession`
- Court coordinates: `original_x/y`, `converted_x/y`

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run Demo
```bash
python demo.py
```

### Run Benchmark
```bash
# Basic benchmark with GPT-3.5
python run_benchmark.py --models gpt-3.5-turbo --num-games 10

# Multiple models with custom config
python run_benchmark.py --config config_example.json

# Local models (requires appropriate installations)
python run_benchmark.py --models ollama-llama2 --num-games 5
```

### Environment Variables
- `OPENAI_API_KEY`: For OpenAI models (gpt-3.5-turbo, gpt-4)
- `ANTHROPIC_API_KEY`: For Anthropic models (claude-3-sonnet)

## Evaluation Framework

The project includes comprehensive evaluation across multiple dimensions:
- **Score Accuracy**: Final score and quarter-by-quarter prediction accuracy
- **Statistical Realism**: Player statistics similarity to actual performance
- **Format Compliance**: JSON structure and required field validation
- **Basketball Realism**: Adherence to basketball rules and realistic ranges

## Supported LLM Types

1. **OpenAI API**: GPT-3.5, GPT-4
2. **Anthropic API**: Claude models
3. **HuggingFace Local**: Llama-2, Mistral, etc.
4. **Ollama Local**: Local model server

## Data Source & Licensing

Play-by-play data is from BigDataBall (paid). Per their support: do NOT cite
BigDataBall publicly in the Sloan submission unless academically affiliated —
reference the official NBA API instead (the data closely mirrors it). Never
commit or redistribute `nba_data/`. Details and open questions in `TODO.md`.

## Data Coverage

The dataset spans six complete NBA seasons — 2018-19 through 2022-23 plus 2024-25 (2023-24 not acquired), ~7,590 game files including regular season, play-in, and playoffs, distinguishable by game_id prefix (002 = regular season, 005 = play-in, 004 = playoffs) — with comprehensive play-by-play data for all 30 NBA teams. All seasons share one CSV schema and filename convention.