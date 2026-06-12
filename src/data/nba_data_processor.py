"""
Process NBA CSV data to extract game information and create test cases.
"""

import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from collections import defaultdict

from ..evaluation.metrics import GameResult
from ..prompting.nba_prompts import GameContext, TeamLineup


@dataclass
class GameInfo:
    """Extracted game information from CSV data."""
    game_id: str
    date: str
    home_team: str
    away_team: str
    home_players: List[str]
    away_players: List[str]
    final_score: Dict[str, int]
    quarter_scores: List[List[int]]
    player_stats: Dict[str, Dict[str, int]]
    file_path: str


class NBADataProcessor:
    """Process NBA CSV files to extract game data."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.team_mapping = self._create_team_mapping()
    
    def _create_team_mapping(self) -> Dict[str, str]:
        """Create mapping from team abbreviations to full names."""
        return {
            'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
            'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
            'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
            'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
            'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
            'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
            'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
            'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
            'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
            'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
        }
    
    def parse_filename(self, filename: str) -> Tuple[str, str, str, str]:
        """Parse game filename to extract date, game_id, and team abbreviations."""
        # Format: [YYYY-MM-DD]-XXXXXXXXXX-AWAY@HOME.csv
        pattern = r'\[(\d{4}-\d{2}-\d{2})\]-(\d+)-([A-Z]{3})@([A-Z]{3})\.csv'
        match = re.match(pattern, filename)
        
        if match:
            date, game_id, away_team, home_team = match.groups()
            return date, game_id, away_team, home_team
        else:
            raise ValueError(f"Cannot parse filename: {filename}")
    
    def process_game_file(self, file_path: str) -> GameInfo:
        """Process a single NBA game CSV file."""
        filename = os.path.basename(file_path)
        date, game_id, away_abbrev, home_abbrev = self.parse_filename(filename)
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Extract team names
        home_team = self.team_mapping.get(home_abbrev, home_abbrev)
        away_team = self.team_mapping.get(away_abbrev, away_abbrev)
        
        # Extract starting lineups
        home_players = self._extract_starting_lineup(df, is_home=True)
        away_players = self._extract_starting_lineup(df, is_home=False)
        
        # Extract final score
        final_score = self._extract_final_score(df)
        
        # Extract quarter scores
        quarter_scores = self._extract_quarter_scores(df)
        
        # Extract player statistics
        player_stats = self._extract_player_stats(df)
        
        return GameInfo(
            game_id=game_id,
            date=date,
            home_team=home_team,
            away_team=away_team,
            home_players=home_players,
            away_players=away_players,
            final_score=final_score,
            quarter_scores=quarter_scores,
            player_stats=player_stats,
            file_path=file_path
        )
    
    def _extract_starting_lineup(self, df: pd.DataFrame, is_home: bool) -> List[str]:
        """Extract starting lineup from game data."""
        if is_home:
            columns = ['h1', 'h2', 'h3', 'h4', 'h5']
        else:
            columns = ['a1', 'a2', 'a3', 'a4', 'a5']
        
        # Get first row (starting lineup)
        if len(df) > 0:
            lineup = []
            for col in columns:
                if col in df.columns:
                    player = df[col].iloc[0]
                    if pd.notna(player) and str(player).strip():
                        lineup.append(str(player).strip())
            return lineup
        return []
    
    def _extract_final_score(self, df: pd.DataFrame) -> Dict[str, int]:
        """Extract final score from game data."""
        if len(df) > 0:
            # Get the last recorded scores
            final_row = df.iloc[-1]
            home_score = final_row.get('home_score', 0)
            away_score = final_row.get('away_score', 0)
            
            try:
                return {
                    'home': int(home_score) if pd.notna(home_score) else 0,
                    'away': int(away_score) if pd.notna(away_score) else 0
                }
            except (ValueError, TypeError):
                return {'home': 0, 'away': 0}
        return {'home': 0, 'away': 0}
    
    def _extract_quarter_scores(self, df: pd.DataFrame) -> List[List[int]]:
        """Extract quarter-by-quarter scores."""
        quarter_scores = []
        
        for quarter in [1, 2, 3, 4]:
            quarter_df = df[df['period'] == quarter]
            if len(quarter_df) > 0:
                # Get the last score of the quarter
                last_row = quarter_df.iloc[-1]
                home_score = last_row.get('home_score', 0)
                away_score = last_row.get('away_score', 0)
                
                try:
                    quarter_scores.append([int(home_score), int(away_score)])
                except (ValueError, TypeError):
                    quarter_scores.append([0, 0])
            else:
                quarter_scores.append([0, 0])
        
        # Convert to quarter-by-quarter differences
        qtr_scores = []
        prev_home, prev_away = 0, 0
        
        for home_total, away_total in quarter_scores:
            qtr_home = home_total - prev_home
            qtr_away = away_total - prev_away
            qtr_scores.append([qtr_home, qtr_away])
            prev_home, prev_away = home_total, away_total
        
        return qtr_scores
    
    def _extract_player_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Extract player statistics from game data."""
        player_stats = defaultdict(lambda: {'points': 0, 'rebounds': 0, 'assists': 0, 'minutes': 0})
        
        # Aggregate stats from play-by-play data
        for _, row in df.iterrows():
            player = row.get('player')
            if pd.notna(player) and str(player).strip():
                player_name = str(player).strip()
                
                # Points
                if row.get('event_type') == 'shot' and row.get('result') == 'made':
                    points = row.get('points', 0)
                    if pd.notna(points):
                        try:
                            player_stats[player_name]['points'] += int(points)
                        except (ValueError, TypeError):
                            pass
                
                # Rebounds
                if row.get('event_type') == 'rebound':
                    player_stats[player_name]['rebounds'] += 1
                
                # Assists
                if pd.notna(row.get('assist')) and str(row.get('assist')).strip():
                    assist_player = str(row.get('assist')).strip()
                    player_stats[assist_player]['assists'] += 1
        
        # Estimate minutes (simplified - assume starters play more)
        starting_players = self._extract_starting_lineup(df, True) + self._extract_starting_lineup(df, False)
        for player in player_stats:
            if player in starting_players:
                player_stats[player]['minutes'] = 36  # Typical starter minutes
            else:
                player_stats[player]['minutes'] = 20  # Bench player minutes
        
        return dict(player_stats)
    
    def create_test_cases(self, num_games: int = 10, random_seed: int = 42) -> List[Tuple[GameContext, GameResult]]:
        """Create test cases from NBA data."""
        import random
        random.seed(random_seed)
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if num_games > len(csv_files):
            num_games = len(csv_files)
        
        # Randomly sample games
        selected_files = random.sample(csv_files, num_games)
        
        test_cases = []
        for filename in selected_files:
            try:
                file_path = os.path.join(self.data_dir, filename)
                game_info = self.process_game_file(file_path)
                
                # Create GameContext
                home_lineup = TeamLineup(
                    team_name=game_info.home_team,
                    team_abbrev=game_info.home_team.split()[-1][:3].upper() if game_info.home_team else 'HOME',
                    players=[{'name': player, 'position': 'Unknown'} for player in game_info.home_players],
                    is_home=True
                )
                
                away_lineup = TeamLineup(
                    team_name=game_info.away_team,
                    team_abbrev=game_info.away_team.split()[-1][:3].upper() if game_info.away_team else 'AWAY',
                    players=[{'name': player, 'position': 'Unknown'} for player in game_info.away_players],
                    is_home=False
                )
                
                context = GameContext(
                    home_team=home_lineup,
                    away_team=away_lineup,
                    date=game_info.date
                )
                
                # Create GameResult (ground truth)
                actual_result = GameResult(
                    final_score=game_info.final_score,
                    quarter_scores=game_info.quarter_scores,
                    player_stats=game_info.player_stats
                )
                
                test_cases.append((context, actual_result))
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        return test_cases
    
    def get_game_files(self) -> List[str]:
        """Get list of all game CSV files."""
        return [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
    
    def get_teams_in_dataset(self) -> List[str]:
        """Get list of all teams in the dataset."""
        teams = set()
        for filename in self.get_game_files():
            try:
                _, _, away_abbrev, home_abbrev = self.parse_filename(filename)
                teams.add(away_abbrev)
                teams.add(home_abbrev)
            except ValueError:
                continue
        return sorted(list(teams))