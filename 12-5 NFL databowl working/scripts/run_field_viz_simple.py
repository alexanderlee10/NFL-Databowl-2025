"""
SIMPLE FIELD VISUALIZATION - Run this to see the CPP-style field view!

This creates the exact visualization from CPP - players on field with purple contour plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from receiver_dominance_field_viz import visualize_receiver_dominance_field

# Create sample player data (you can replace this with your actual data)
print("Creating sample field visualization...")

# Sample player coordinates (all players on field for one frame)
player_coords = pd.DataFrame([
    # Target receiver (will be highlighted in green)
    {'nfl_id': 44930, 'x': 41.0, 'y': 41.1, 's': 5.0, 'dir': 45, 
     'player_side': 'Offense', 'jerseyNumber': 11, 'receiver_speed': 5.0, 
     'sep_nearest': 3.5, 'leverage_angle': 60},
    
    # Nearest defender
    {'nfl_id': 50001, 'x': 44.4, 'y': 41.6, 's': 4.5, 'dir': 225,
     'player_side': 'Defense', 'jerseyNumber': 24},
    
    # Another defender
    {'nfl_id': 50002, 'x': 45.0, 'y': 38.0, 's': 3.0, 'dir': 180,
     'player_side': 'Defense', 'jerseyNumber': 21},
    
    # QB
    {'nfl_id': 40001, 'x': 10.0, 'y': 26.6, 's': 0, 'dir': 0,
     'player_side': 'Offense', 'jerseyNumber': 9},
    
    # Other offensive players
    {'nfl_id': 40002, 'x': 12.0, 'y': 30.0, 's': 2.0, 'dir': 90,
     'player_side': 'Offense', 'jerseyNumber': 85},
])

# Ball landing position
ball_land_x = 66.5
ball_land_y = 41.6

# Dominance score (0-1, where 1 = high receiver dominance)
dominance_score = 0.75

print(f"  Receiver: Player {player_coords.iloc[0]['nfl_id']}")
print(f"  Ball landing: ({ball_land_x}, {ball_land_y})")
print(f"  Dominance: {dominance_score:.0%}")

# Create the visualization
print("\nGenerating field visualization...")
fig = visualize_receiver_dominance_field(
    player_coords,
    receiver_nfl_id=44930,
    ball_land_x=ball_land_x,
    ball_land_y=ball_land_y,
    dominance_score=dominance_score,
    label_numbers=True,
    show_arrow=False
)

# Save and show
plt.savefig('cpp_style_field_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cpp_style_field_visualization.png")
print("\n✓ Displaying visualization...")
plt.show()

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print("\nWhat you see:")
print("  - Dark green field with white yard lines")
print("  - Green star = Target receiver (like QB in CPP)")
print("  - Blue circles = Defenders")
print("  - Red circles = Other offensive players")
print("  - Purple contour plot = Dominance regions")
print("  - Yellow X = Ball landing position")
print("  - Dominance % indicator in top-left corner")
print("\nTo use with YOUR data:")
print("  1. Replace 'player_coords' with your player tracking data")
print("  2. Update receiver_nfl_id, ball_land_x, ball_land_y")
print("  3. Run the same function!")

