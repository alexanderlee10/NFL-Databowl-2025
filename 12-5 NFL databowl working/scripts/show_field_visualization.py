"""
Show the CPP-style field visualization with players on the field!

This creates the exact visualization you saw in the CPP code - players positioned
on the field with the purple contour plot showing dominance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from receiver_dominance_field_viz import (
    visualize_receiver_dominance_field,
    create_field_visualization_from_dataframe
)

print("="*80)
print("RECEIVER DOMINANCE FIELD VISUALIZATION")
print("="*80)
print("\nThis creates the CPP-style field view with:")
print("  - All players on the field")
print("  - Purple contour plot showing dominance")
print("  - Dominance indicator in corner")
print("  - Field markings and yard lines")
print()

# ============================================================================
# OPTION 1: If you have full player tracking data
# ============================================================================
print("OPTION 1: Using full player tracking data")
print("-" * 80)

# Load your data
try:
    training_df = pd.read_csv('route_dominance_training_data.csv')
    print(f"✓ Loaded {len(training_df):,} rows")
except FileNotFoundError:
    print("⚠ CSV not found. Make sure you have your dataframe loaded.")
    print("  You can use: training_df = pd.read_csv('your_file.csv')")
    training_df = None

if training_df is not None:
    # Example: Visualize a specific play
    # You'll need to adjust these based on your data
    example_game_id = training_df['game_id'].iloc[0] if 'game_id' in training_df.columns else None
    example_play_id = training_df['play_id'].iloc[0] if 'play_id' in training_df.columns else None
    
    if example_game_id and example_play_id:
        print(f"\nCreating visualization for Game {example_game_id}, Play {example_play_id}...")
        
        try:
            # This requires full player tracking data (all players on field)
            # If your dataframe only has receiver data, see OPTION 2 below
            fig = create_field_visualization_from_dataframe(
                training_df,
                game_id=example_game_id,
                play_id=example_play_id
            )
            plt.savefig('field_visualization.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: field_visualization.png")
            plt.show()
            plt.close()
        except Exception as e:
            print(f"⚠ Could not create full visualization: {e}")
            print("  This usually means you need full player tracking data.")
            print("  See OPTION 2 for a simplified version.")

# ============================================================================
# OPTION 2: Simplified version with just receiver and nearest defender
# ============================================================================
print("\n" + "="*80)
print("OPTION 2: Simplified version (receiver + nearest defender)")
print("-" * 80)

# Create a minimal player coordinates dataframe from your training data
if training_df is not None and len(training_df) > 0:
    # Get a sample frame
    sample_frame = training_df.iloc[0]
    
    # Extract receiver info
    receiver_nfl_id = sample_frame.get('nfl_id', None)
    receiver_x = sample_frame.get('receiver_x', sample_frame.get('x', 0))
    receiver_y = sample_frame.get('receiver_y', sample_frame.get('y', 0))
    receiver_speed = sample_frame.get('receiver_speed', 0)
    receiver_dir = sample_frame.get('dir', 0) if 'dir' in sample_frame else 0
    
    # Extract nearest defender info
    def_x = sample_frame.get('nearest_defender_x', receiver_x + 3)
    def_y = sample_frame.get('nearest_defender_y', receiver_y)
    
    # Ball landing
    ball_land_x = sample_frame.get('ball_land_x_std', sample_frame.get('ball_land_x', receiver_x + 20))
    ball_land_y = sample_frame.get('ball_land_y_std', sample_frame.get('ball_land_y', receiver_y))
    
    # Dominance score
    dominance_score = sample_frame.get('receiver_dominance', 0.6)
    
    # Create minimal player coordinates dataframe
    player_coords = pd.DataFrame([
        {
            'nfl_id': receiver_nfl_id,
            'x': receiver_x,
            'y': receiver_y,
            's': receiver_speed,
            'dir': receiver_dir,
            'player_side': 'Offense',
            'jerseyNumber': receiver_nfl_id % 100 if receiver_nfl_id else 1
        },
        {
            'nfl_id': 99999,  # Dummy defender ID
            'x': def_x,
            'y': def_y,
            's': 0,
            'dir': 0,
            'player_side': 'Defense',
            'jerseyNumber': 24
        }
    ])
    
    print(f"\nCreating simplified visualization...")
    print(f"  Receiver: ({receiver_x:.1f}, {receiver_y:.1f})")
    print(f"  Defender: ({def_x:.1f}, {def_y:.1f})")
    print(f"  Dominance: {dominance_score:.2f}")
    
    try:
        fig = visualize_receiver_dominance_field(
            player_coords,
            receiver_nfl_id=int(receiver_nfl_id) if receiver_nfl_id else 1,
            ball_land_x=float(ball_land_x),
            ball_land_y=float(ball_land_y),
            dominance_score=float(dominance_score),
            label_numbers=True,
            show_arrow=False
        )
        plt.savefig('field_visualization_simplified.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: field_visualization_simplified.png")
        plt.show()
        print("\n✓ Visualization displayed!")
        plt.close()
    except Exception as e:
        print(f"⚠ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# INSTRUCTIONS
# ============================================================================
print("\n" + "="*80)
print("HOW TO USE WITH YOUR DATA")
print("="*80)
print("""
To create the full field visualization, you need:

1. Player tracking data with ALL players on the field (not just receiver)
   - Each player's x, y position
   - Player's speed (s) and direction (dir)
   - Player side (Offense/Defense) or team
   - Jersey numbers (optional)

2. Then use:
   from receiver_dominance_field_viz import visualize_receiver_dominance_field
   
   fig = visualize_receiver_dominance_field(
       player_coordinates,  # DataFrame with all players
       receiver_nfl_id=44930,
       ball_land_x=66.5,
       ball_land_y=41.6,
       dominance_score=0.75
   )
   plt.show()

3. The visualization shows:
   - Green star = Target receiver (like QB in CPP)
   - Blue circles = Defenders
   - Red circles = Other offensive players
   - Purple contour = Dominance regions
   - Yellow X = Ball landing position
   - Dominance % indicator in corner
""")

