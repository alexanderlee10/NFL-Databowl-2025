"""
Create GIF with ALL players on field (like CPP visualization)

This version uses the RouteDominanceScorer's all_frames_df to show
all players on the field, not just the receiver.
"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.route_dominance_scoring import RouteDominanceScorer
from src.create_dominance_gif import create_gif_for_play

print("="*80)
print("CREATING GIF WITH ALL PLAYERS ON FIELD")
print("="*80)

# Load your data
print("\n1. Loading data...")
try:
    # Load input and output data (needed for scorer)
    input_df = pd.read_csv('../data/input_2023_w01.csv')
    output_df = pd.read_csv('../data/output_2023_w01.csv')
    supp_df = pd.read_csv('../data/Supplementary.csv')
    training_df = pd.read_csv('route_dominance_training_data.csv')
    
    print(f"   Input data: {len(input_df):,} rows")
    print(f"   Output data: {len(output_df):,} rows")
    print(f"   Training data: {len(training_df):,} rows")
except FileNotFoundError as e:
    print(f"   Error: {e}")
    print("\n   Make sure you have:")
    print("     - ../data/input_2023_w01.csv")
    print("     - ../data/output_2023_w01.csv")
    print("     - ../data/Supplementary.csv")
    print("     - route_dominance_training_data.csv")
    raise

# Initialize scorer (this has all_frames_df with ALL players)
print("\n2. Initializing RouteDominanceScorer...")
print("   (This creates all_frames_df with all players on field)")
scorer = RouteDominanceScorer(input_df, output_df, supp_df)
print(f"   Total frames in all_frames_df: {len(scorer.all_frames_df):,}")

# Get example play
example_game = training_df['game_id'].iloc[0]
example_play = training_df['play_id'].iloc[0]
receiver_nfl_id = training_df['nfl_id'].iloc[0]

print(f"\n3. Creating GIF for Game {example_game}, Play {example_play}...")
print(f"   Receiver: {receiver_nfl_id}")
print(f"   This will show ALL players on the field!")

# Create GIF with scorer (this will show all players)
gif_path = create_gif_for_play(
    training_df,
    game_id=example_game,
    play_id=example_play,
    fps=5,
    scorer=scorer  # Pass scorer to get all players
)

print(f"\n" + "="*80)
print("GIF CREATED WITH ALL PLAYERS!")
print("="*80)
print(f"\nLocation: {gif_path}")
print("\nThe GIF now shows:")
print("  - ALL players on the field (like CPP visualization)")
print("  - Green star = Target receiver")
print("  - Blue circles = All defenders")
print("  - Red circles = All other offensive players")
print("  - Purple contour plot = Dominance regions")
print("  - Yellow X = Ball landing")
print("\nOpening GIF...")

# Open the GIF
import os
import sys
if sys.platform == "win32":
    os.startfile(os.path.abspath(gif_path))
else:
    print(f"Please open: {os.path.abspath(gif_path)}")

