"""
Launch Interactive Play Viewer

This script launches an interactive viewer where you can navigate between plays
using arrow keys to see the field visualization for each play.
"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.route_dominance_scoring import RouteDominanceScorer
from interactive_play_viewer import launch_interactive_viewer

print("="*80)
print("LAUNCHING INTERACTIVE PLAY VIEWER")
print("="*80)

# Load your data
print("\n1. Loading data...")
try:
    # Load input and output data (needed for scorer)
    input_df = pd.read_csv('../data/input_2023_w01.csv')
    output_df = pd.read_csv('../data/output_2023_w01.csv')
    supp_df = pd.read_csv('../data/Supplementary.csv')
    training_df = pd.read_csv('route_dominance_training_data.csv')
    
    print(f"   ✓ Input data: {len(input_df):,} rows")
    print(f"   ✓ Output data: {len(output_df):,} rows")
    print(f"   ✓ Training data: {len(training_df):,} rows")
except FileNotFoundError as e:
    print(f"   ✗ Error: {e}")
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
print(f"   ✓ Total frames in all_frames_df: {len(scorer.all_frames_df):,}")

# Get unique plays count
unique_plays = training_df[['game_id', 'play_id']].drop_duplicates()
print(f"\n3. Found {len(unique_plays):,} unique plays in training data")

# Launch interactive viewer
print("\n4. Launching Interactive Play Viewer...")
print("\n" + "="*80)
print("CONTROLS:")
print("  ← Left Arrow  : Previous play")
print("  → Right Arrow : Next play")
print("  'g'           : Generate GIF for current play")
print("  'q' or Escape : Quit")
print("="*80)
print("\n⚠ Click on the plot window and use arrow keys to navigate!")
print("="*80 + "\n")

# Launch viewer
viewer = launch_interactive_viewer(training_df, scorer)

