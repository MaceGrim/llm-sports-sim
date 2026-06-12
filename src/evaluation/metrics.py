"""
Evaluation metrics for NBA game simulation.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import cosine
import re


@dataclass
class GameResult:
    """Standardized game result format."""
    final_score: Dict[str, int]  # {"home": 108, "away": 102}
    quarter_scores: List[List[int]]  # [[28, 24], [26, 28], [29, 25], [25, 25]]
    player_stats: Dict[str, Dict[str, Any]]  # {"Player": {"points": 32, "rebounds": 8, ...}}
    metadata: Dict[str, Any] = None


class EvaluationMetrics:
    """Comprehensive evaluation metrics for NBA game simulation."""
    
    def __init__(self):
        self.basketball_rules = {
            'min_team_score': 50,
            'max_team_score': 200,
            'max_player_minutes': 48,
            'max_player_points': 70,
            'max_player_rebounds': 30,
            'max_player_assists': 20,
            'typical_total_score_range': (160, 260)
        }
    
    def evaluate_game(self, predicted: GameResult, actual: GameResult) -> Dict[str, float]:
        """Evaluate a single game prediction against ground truth."""
        metrics = {}
        
        # Score accuracy
        metrics.update(self._evaluate_scores(predicted, actual))
        
        # Statistical similarity
        metrics.update(self._evaluate_player_stats(predicted, actual))
        
        # Format compliance
        metrics.update(self._evaluate_format_compliance(predicted))
        
        # Basketball realism
        metrics.update(self._evaluate_basketball_realism(predicted))
        
        return metrics
    
    def _evaluate_scores(self, predicted: GameResult, actual: GameResult) -> Dict[str, float]:
        """Evaluate score prediction accuracy."""
        pred_home = predicted.final_score.get('home', 0)
        pred_away = predicted.final_score.get('away', 0)
        actual_home = actual.final_score.get('home', 0)
        actual_away = actual.final_score.get('away', 0)
        
        return {
            'score_mae_home': abs(pred_home - actual_home),
            'score_mae_away': abs(pred_away - actual_away),
            'score_mae_total': abs((pred_home + pred_away) - (actual_home + actual_away)),
            'score_mae_diff': abs((pred_home - pred_away) - (actual_home - actual_away)),
            'winner_correct': (pred_home > pred_away) == (actual_home > actual_away)
        }
    
    def _evaluate_player_stats(self, predicted: GameResult, actual: GameResult) -> Dict[str, float]:
        """Evaluate player statistics similarity."""
        if not predicted.player_stats or not actual.player_stats:
            return {'stat_similarity': 0.0}
        
        # Find common players
        common_players = set(predicted.player_stats.keys()) & set(actual.player_stats.keys())
        if not common_players:
            return {'stat_similarity': 0.0}
        
        stat_similarities = []
        stat_maes = {'points': [], 'rebounds': [], 'assists': []}
        
        for player in common_players:
            pred_stats = predicted.player_stats[player]
            actual_stats = actual.player_stats[player]
            
            # Calculate MAE for key stats
            for stat in ['points', 'rebounds', 'assists']:
                if stat in pred_stats and stat in actual_stats:
                    mae = abs(pred_stats[stat] - actual_stats[stat])
                    stat_maes[stat].append(mae)
        
        # Average MAEs
        result = {}
        for stat, maes in stat_maes.items():
            if maes:
                result[f'{stat}_mae'] = np.mean(maes)
        
        return result
    
    def _evaluate_format_compliance(self, predicted: GameResult) -> Dict[str, float]:
        """Evaluate format and structure compliance."""
        score = 0.0
        total_checks = 5
        
        # Check final score format
        if isinstance(predicted.final_score, dict) and 'home' in predicted.final_score and 'away' in predicted.final_score:
            score += 1
        
        # Check quarter scores format
        if (isinstance(predicted.quarter_scores, list) and 
            len(predicted.quarter_scores) == 4 and
            all(isinstance(q, list) and len(q) == 2 for q in predicted.quarter_scores)):
            score += 1
        
        # Check player stats format
        if isinstance(predicted.player_stats, dict) and predicted.player_stats:
            score += 1
            
            # Check if player stats have required fields
            required_fields = {'points', 'rebounds', 'assists'}
            has_required = any(
                required_fields.issubset(set(stats.keys())) 
                for stats in predicted.player_stats.values()
            )
            if has_required:
                score += 1
        
        # Check numeric values
        try:
            home_score = int(predicted.final_score.get('home', 0))
            away_score = int(predicted.final_score.get('away', 0))
            if home_score >= 0 and away_score >= 0:
                score += 1
        except (ValueError, TypeError):
            pass
        
        return {'format_compliance': score / total_checks}
    
    def _evaluate_basketball_realism(self, predicted: GameResult) -> Dict[str, float]:
        """Evaluate basketball realism constraints."""
        violations = 0
        total_checks = 0
        
        # Check team score ranges
        total_checks += 2
        home_score = predicted.final_score.get('home', 0)
        away_score = predicted.final_score.get('away', 0)
        
        if not (self.basketball_rules['min_team_score'] <= home_score <= self.basketball_rules['max_team_score']):
            violations += 1
        if not (self.basketball_rules['min_team_score'] <= away_score <= self.basketball_rules['max_team_score']):
            violations += 1
        
        # Check total score range
        total_checks += 1
        total_score = home_score + away_score
        if not (self.basketball_rules['typical_total_score_range'][0] <= total_score <= self.basketball_rules['typical_total_score_range'][1]):
            violations += 1
        
        # Check player stat ranges
        for player, stats in predicted.player_stats.items():
            for stat, limit in [
                ('points', self.basketball_rules['max_player_points']),
                ('rebounds', self.basketball_rules['max_player_rebounds']),
                ('assists', self.basketball_rules['max_player_assists']),
                ('minutes', self.basketball_rules['max_player_minutes'])
            ]:
                if stat in stats:
                    total_checks += 1
                    if stats[stat] > limit or stats[stat] < 0:
                        violations += 1
        
        realism_score = 1.0 - (violations / max(total_checks, 1))
        return {'basketball_realism': max(0.0, realism_score)}


class DatasetEvaluator:
    """Evaluate simulation approaches across multiple games."""
    
    def __init__(self):
        self.metrics = EvaluationMetrics()
    
    def evaluate_dataset(self, predictions: List[GameResult], actuals: List[GameResult]) -> Dict[str, Any]:
        """Evaluate predictions across a dataset."""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        all_metrics = []
        for pred, actual in zip(predictions, actuals):
            game_metrics = self.metrics.evaluate_game(pred, actual)
            all_metrics.append(game_metrics)
        
        # Aggregate metrics
        aggregated = {}
        metric_names = all_metrics[0].keys() if all_metrics else []
        
        for metric in metric_names:
            values = [m[metric] for m in all_metrics if metric in m and m[metric] is not None]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_median'] = np.median(values)
        
        # Add summary metrics
        aggregated['num_games'] = len(predictions)
        aggregated['overall_score'] = self._calculate_overall_score(aggregated)
        
        return aggregated
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score (0-1, higher is better)."""
        # Weight different metric categories
        weights = {
            'score_accuracy': 0.3,
            'stat_accuracy': 0.2,
            'format_compliance': 0.2,
            'basketball_realism': 0.3
        }
        
        score = 0.0
        
        # Score accuracy (lower MAE is better, normalize to 0-1)
        score_mae = metrics.get('score_mae_total_mean', 100)
        score_accuracy = max(0, 1 - (score_mae / 50))  # 50 point error = 0 score
        score += weights['score_accuracy'] * score_accuracy
        
        # Stat accuracy (lower MAE is better)
        points_mae = metrics.get('points_mae_mean', 30)
        stat_accuracy = max(0, 1 - (points_mae / 20))  # 20 point error = 0 score
        score += weights['stat_accuracy'] * stat_accuracy
        
        # Format compliance (already 0-1)
        format_score = metrics.get('format_compliance_mean', 0)
        score += weights['format_compliance'] * format_score
        
        # Basketball realism (already 0-1)
        realism_score = metrics.get('basketball_realism_mean', 0)
        score += weights['basketball_realism'] * realism_score
        
        return score


def parse_llm_output(raw_output: str) -> GameResult:
    """Parse LLM output into standardized GameResult format."""
    try:
        # Try to extract JSON from the output
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)
            
            return GameResult(
                final_score=data.get('final_score', {}),
                quarter_scores=data.get('quarter_scores', []),
                player_stats=data.get('player_stats', {}),
                metadata={'raw_output': raw_output}
            )
        else:
            # Fallback parsing for non-JSON outputs
            return _parse_non_json_output(raw_output)
            
    except Exception as e:
        print(f"Error parsing LLM output: {e}")
        return GameResult(
            final_score={},
            quarter_scores=[],
            player_stats={},
            metadata={'raw_output': raw_output, 'parse_error': str(e)}
        )


def _parse_non_json_output(output: str) -> GameResult:
    """Fallback parser for non-JSON formatted outputs."""
    # Basic regex patterns for score extraction
    score_pattern = r'(\d+)\s*-\s*(\d+)'
    scores = re.findall(score_pattern, output)
    
    if scores:
        final_score = {'home': int(scores[-1][0]), 'away': int(scores[-1][1])}
    else:
        final_score = {}
    
    return GameResult(
        final_score=final_score,
        quarter_scores=[],
        player_stats={},
        metadata={'raw_output': output, 'parsed_with_fallback': True}
    )