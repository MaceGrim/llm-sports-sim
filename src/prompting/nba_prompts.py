"""
NBA game simulation prompts and prompt engineering strategies.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json


@dataclass 
class TeamLineup:
    """Team lineup with players and positions."""
    team_name: str
    team_abbrev: str
    players: List[Dict[str, str]]  # [{"name": "Player Name", "position": "PG"}]
    is_home: bool


@dataclass
class GameContext:
    """Context for NBA game simulation."""
    home_team: TeamLineup
    away_team: TeamLineup
    date: Optional[str] = None
    season: Optional[str] = "2022-23"
    game_type: str = "Regular Season"


class NBAPromptGenerator:
    """Generate prompts for NBA game simulation."""
    
    def __init__(self):
        self.base_json_schema = {
            "final_score": {"home": "int", "away": "int"},
            "quarter_scores": [["int", "int"], ["int", "int"], ["int", "int"], ["int", "int"]],
            "player_stats": {
                "Player Name": {
                    "points": "int",
                    "rebounds": "int", 
                    "assists": "int",
                    "minutes": "int"
                }
            }
        }
    
    def generate_basic_prompt(self, context: GameContext) -> str:
        """Generate basic simulation prompt."""
        prompt = f"""Generate a realistic NBA game simulation between:

Home Team: {context.home_team.team_name} ({context.home_team.team_abbrev})
Starting Lineup:"""
        
        for player in context.home_team.players:
            prompt += f"\n- {player['name']} ({player['position']})"
        
        prompt += f"\n\nAway Team: {context.away_team.team_name} ({context.away_team.team_abbrev})\nStarting Lineup:"
        
        for player in context.away_team.players:
            prompt += f"\n- {player['name']} ({player['position']})"
        
        prompt += f"""

Return the result in this exact JSON format:
{json.dumps(self.base_json_schema, indent=2)}

Make the simulation realistic based on player abilities and team strengths. Include reasonable statistics that reflect actual NBA performance levels."""
        
        return prompt
    
    def generate_detailed_prompt(self, context: GameContext) -> str:
        """Generate detailed prompt with more context."""
        prompt = f"""You are an NBA analyst tasked with simulating a realistic {context.game_type} game.

GAME SETUP:
- Date: {context.date or 'TBD'}
- Season: {context.season}
- Home Team: {context.home_team.team_name} ({context.home_team.team_abbrev})
- Away Team: {context.away_team.team_name} ({context.away_team.team_abbrev})

STARTING LINEUPS:

{context.home_team.team_name} (Home):"""
        
        for player in context.home_team.players:
            prompt += f"\n- {player['name']} ({player['position']})"
        
        prompt += f"\n\n{context.away_team.team_name} (Away):"
        
        for player in context.away_team.players:
            prompt += f"\n- {player['name']} ({player['position']})"
        
        prompt += """

SIMULATION REQUIREMENTS:
1. Generate realistic final scores (typical NBA range: 90-130 points)
2. Create quarter-by-quarter progression that makes sense
3. Assign individual player statistics that reflect their real-world abilities
4. Consider factors like home court advantage, player matchups, and team playing styles
5. Ensure statistics are internally consistent (team totals should roughly match individual totals)

OUTPUT FORMAT:
Return ONLY a valid JSON object with this exact structure:

{
  "final_score": {"home": 108, "away": 102},
  "quarter_scores": [[28, 24], [26, 28], [29, 25], [25, 25]],
  "player_stats": {
    "Player Name": {
      "points": 25,
      "rebounds": 8,
      "assists": 6,
      "minutes": 36
    }
  }
}

Include stats for all starting players. Make sure the quarter scores add up to the final score."""
        
        return prompt
    
    def generate_few_shot_prompt(self, context: GameContext, examples: List[Dict] = None) -> str:
        """Generate prompt with few-shot examples."""
        if examples is None:
            examples = self._get_default_examples()
        
        prompt = "You are simulating NBA games. Here are some examples:\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"EXAMPLE {i}:\n"
            prompt += f"Teams: {example['teams']}\n"
            prompt += f"Result: {json.dumps(example['result'], indent=2)}\n\n"
        
        prompt += "Now simulate this game:\n\n"
        prompt += self.generate_basic_prompt(context)
        
        return prompt
    
    def generate_chain_of_thought_prompt(self, context: GameContext) -> str:
        """Generate prompt that encourages step-by-step reasoning."""
        prompt = f"""Simulate an NBA game between {context.home_team.team_name} (home) and {context.away_team.team_name} (away).

Starting lineups:
Home ({context.home_team.team_abbrev}): {', '.join([p['name'] for p in context.home_team.players])}
Away ({context.away_team.team_abbrev}): {', '.join([p['name'] for p in context.away_team.players])}

Think through this step by step:

1. TEAM ANALYSIS: Consider each team's strengths, weaknesses, and playing style
2. KEY MATCHUPS: Identify important player matchups that could decide the game  
3. GAME FLOW: Think about how the game might progress quarter by quarter
4. FINAL PREDICTION: Based on your analysis, predict the final score and key stats

Then provide your simulation in this JSON format:
{json.dumps(self.base_json_schema, indent=2)}"""
        
        return prompt
    
    def _get_default_examples(self) -> List[Dict]:
        """Get default few-shot examples."""
        return [
            {
                "teams": "Lakers vs Warriors",
                "result": {
                    "final_score": {"home": 112, "away": 108},
                    "quarter_scores": [[28, 26], [30, 28], [26, 24], [28, 30]],
                    "player_stats": {
                        "LeBron James": {"points": 28, "rebounds": 9, "assists": 7, "minutes": 38},
                        "Stephen Curry": {"points": 32, "rebounds": 5, "assists": 8, "minutes": 36}
                    }
                }
            }
        ]


class PromptTemplate:
    """Template system for customizable prompts."""
    
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        return self.template.format(**kwargs)


# Predefined prompt templates
PROMPT_TEMPLATES = {
    'basic': """Simulate NBA game: {home_team} vs {away_team}
Home lineup: {home_players}
Away lineup: {away_players}

Return JSON with final_score, quarter_scores, and player_stats.""",
    
    'detailed': """NBA Simulation Task:
Teams: {home_team} (home) vs {away_team} (away)
Date: {date}

Starting Lineups:
Home: {home_players}
Away: {away_players}

Generate realistic game with:
- Final scores (90-130 range)
- Quarter progression  
- Player stats reflecting real abilities
- Home court advantage consideration

Output: JSON format with final_score, quarter_scores, player_stats""",
    
    'analytical': """As an NBA analyst, simulate: {home_team} vs {away_team}

Consider:
- Team playing styles and pace
- Star player performances
- Bench contributions
- Home court advantage
- Recent form and matchups

Lineups:
Home: {home_players}
Away: {away_players}

Provide realistic simulation in JSON format."""
}


def create_game_context_from_data(home_data: Dict, away_data: Dict, date: str = None) -> GameContext:
    """Create GameContext from data extracted from NBA CSV files."""
    home_lineup = TeamLineup(
        team_name=home_data.get('team_name', 'Home Team'),
        team_abbrev=home_data.get('team_abbrev', 'HOME'),
        players=home_data.get('players', []),
        is_home=True
    )
    
    away_lineup = TeamLineup(
        team_name=away_data.get('team_name', 'Away Team'),
        team_abbrev=away_data.get('team_abbrev', 'AWAY'),
        players=away_data.get('players', []),
        is_home=False
    )
    
    return GameContext(
        home_team=home_lineup,
        away_team=away_lineup,
        date=date
    )