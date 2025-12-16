"""
Create Receiver Dominance GIF - Run this to create animated GIFs!

Features used from your dataframe:
- receiver_x, receiver_y: Receiver position on field
- nearest_defender_x, nearest_defender_y: Nearest defender position
- sep_nearest: Separation distance (yards)
- receiver_speed: Receiver speed (yards/second)
- receiver_accel: Receiver acceleration
- leverage_angle: Leverage angle (degrees)
- dist_to_ball: Distance to ball landing
- ball_land_x_std, ball_land_y_std: Ball landing coordinates
- receiver_dominance: Dominance score (0-1)
- continuous_frame or frame_id: Frame number for sequencing
"""

import sys
import os
import pandas as pd

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.create_dominance_gif import create_gif_for_play

print("="*80)
print("RECEIVER DOMINANCE GIF CREATOR")
print("="*80)

print("\nFeatures used from your dataframe:")
print("  Position:")
print("    - receiver_x, receiver_y: Receiver position")
print("    - nearest_defender_x, nearest_defender_y: Defender position")
print("    - ball_land_x_std, ball_land_y_std: Ball landing position")
print("  Metrics:")
print("    - sep_nearest: Separation distance")
print("    - receiver_speed: Receiver speed")
print("    - leverage_angle: Leverage angle")
print("    - receiver_dominance: Dominance score (if calculated)")
print("  Sequencing:")
print("    - continuous_frame or frame_id: Frame number")
print()

# Load your dataframe
try:
    training_df = pd.read_csv('route_dominance_training_data.csv')
    print(f"Loaded {len(training_df):,} rows")
except FileNotFoundError:
    print("CSV not found. Please load your dataframe:")
    print("  training_df = pd.read_csv('your_file.csv')")
    print("  OR use your existing training_df variable")
    training_df = None

if training_df is not None:
    # Get example play
    if 'game_id' in training_df.columns and 'play_id' in training_df.columns:
        example_game = training_df['game_id'].iloc[0]
        example_play = training_df['play_id'].iloc[0]
        
        print(f"\nCreating GIF for Game {example_game}, Play {example_play}...")
        print("  (This will create an animated GIF showing dominance frame-by-frame)")
        
        try:
            gif_path = create_gif_for_play(
                training_df,
                game_id=example_game,
                play_id=example_play,
                fps=5  # 5 frames per second
            )
            
            print(f"\nGIF created successfully!")
            print(f"  Location: {gif_path}")
            print(f"\nThe GIF shows:")
            print("  - Receiver (green star) moving through the play")
            print("  - Defender (blue circle) tracking")
            print("  - Purple contour plot showing dominance regions")
            print("  - Dominance score updating frame-by-frame")
            print("  - Separation circle and stats")
            
        except Exception as e:
            print(f"\nError creating GIF: {e}")
            import traceback
            traceback.print_exc()
            print("\nMake sure your dataframe has these columns:")
            print("  - receiver_x, receiver_y")
            print("  - nearest_defender_x, nearest_defender_y")
            print("  - sep_nearest")
            print("  - receiver_speed")
            print("  - ball_land_x_std, ball_land_y_std")
            print("  - continuous_frame or frame_id")
    else:
        print("\nDataframe missing 'game_id' or 'play_id' columns")
        print("  Please specify game_id and play_id manually:")
        print("  gif_path = create_gif_for_play(training_df, game_id=2023090700, play_id=101)")

print("\n" + "="*80)
print("USAGE")
print("="*80)
print("""
To create a GIF for any play:

from src.create_dominance_gif import create_gif_for_play

gif_path = create_gif_for_play(
    training_df,
    game_id=2023090700,  # Your game ID
    play_id=101,          # Your play ID
    fps=5                  # Frames per second (5 = smooth, 10 = faster)
)

The GIF will be saved in 'dominance_gifs/' folder.
""")


