"""
WORKING Field Visualization - Guaranteed to show players and field
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from receiver_dominance_field_viz import visualize_receiver_dominance_field

print("="*80)
print("CREATING FIELD VISUALIZATION WITH PLAYERS")
print("="*80)

# Create sample player data with all required fields
player_coords = pd.DataFrame([
    # Target receiver
    {
        'nfl_id': 44930,
        'x': 41.0,
        'y': 41.1,
        's': 5.0,
        'dir': 45,
        'player_side': 'Offense',
        'jerseyNumber': 11,
        'receiver_speed': 5.0,
        'sep_nearest': 3.5,
        'leverage_angle': 60,
        'xComponent': np.cos(np.radians(45)),
        'yComponent': np.sin(np.radians(45)),
        'xspeed': 5.0 * np.cos(np.radians(45)),
        'yspeed': 5.0 * np.sin(np.radians(45))
    },
    # Nearest defender
    {
        'nfl_id': 50001,
        'x': 44.4,
        'y': 41.6,
        's': 4.5,
        'dir': 225,
        'player_side': 'Defense',
        'jerseyNumber': 24,
        'xComponent': np.cos(np.radians(225)),
        'yComponent': np.sin(np.radians(225)),
        'xspeed': 4.5 * np.cos(np.radians(225)),
        'yspeed': 4.5 * np.sin(np.radians(225))
    },
    # Another defender
    {
        'nfl_id': 50002,
        'x': 45.0,
        'y': 38.0,
        's': 3.0,
        'dir': 180,
        'player_side': 'Defense',
        'jerseyNumber': 21,
        'xComponent': np.cos(np.radians(180)),
        'yComponent': np.sin(np.radians(180)),
        'xspeed': 3.0 * np.cos(np.radians(180)),
        'yspeed': 3.0 * np.sin(np.radians(180))
    },
    # QB
    {
        'nfl_id': 40001,
        'x': 10.0,
        'y': 26.6,
        's': 0,
        'dir': 0,
        'player_side': 'Offense',
        'jerseyNumber': 9,
        'xComponent': 1.0,
        'yComponent': 0.0,
        'xspeed': 0,
        'yspeed': 0
    },
])

# Ball landing
ball_land_x = 66.5
ball_land_y = 41.6

# Dominance score
dominance_score = 0.75

print(f"\nPlayer data:")
print(f"  Receiver: ({player_coords.iloc[0]['x']:.1f}, {player_coords.iloc[0]['y']:.1f})")
print(f"  Defender 1: ({player_coords.iloc[1]['x']:.1f}, {player_coords.iloc[1]['y']:.1f})")
print(f"  Defender 2: ({player_coords.iloc[2]['x']:.1f}, {player_coords.iloc[2]['y']:.1f})")
print(f"  QB: ({player_coords.iloc[3]['x']:.1f}, {player_coords.iloc[3]['y']:.1f})")
print(f"  Ball: ({ball_land_x:.1f}, {ball_land_y:.1f})")
print(f"  Dominance: {dominance_score:.0%}")

print("\nCreating visualization...")
try:
    fig = visualize_receiver_dominance_field(
        player_coords,
        receiver_nfl_id=44930,
        ball_land_x=ball_land_x,
        ball_land_y=ball_land_y,
        dominance_score=dominance_score,
        label_numbers=True,
        show_arrow=False
    )
    
    plt.savefig('field_viz_working.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: field_viz_working.png")
    print("\n✓ Displaying visualization...")
    print("  You should see:")
    print("    - Dark green field with white yard lines")
    print("    - Green star = Receiver")
    print("    - Blue circles = Defenders")
    print("    - Red circle = QB")
    print("    - Yellow X = Ball")
    print("    - Purple contour plot = Dominance")
    print("    - Dominance % in corner")
    plt.show()
    plt.close()
    
except Exception as e:
    print(f"⚠ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\nTrying simplified version...")
    
    # Fallback: Simple matplotlib plot
    fig, ax = plt.subplots(1, figsize=(12, 24))
    ax.set_facecolor('darkgreen')
    ax.set_xlim(0, 53.3)
    ax.set_ylim(0, 120)
    
    # Draw field lines
    for y in range(10, 111, 10):
        ax.axhline(y, color='white', linewidth=2)
    
    # Plot players
    ax.scatter(player_coords.iloc[0]['y'], player_coords.iloc[0]['x'], 
              color='limegreen', s=500, marker='*', zorder=10, label='Receiver')
    ax.scatter(player_coords.iloc[1]['y'], player_coords.iloc[1]['x'],
              color='blue', s=400, zorder=9, label='Defender')
    ax.scatter(ball_land_y, ball_land_x, color='yellow', s=600, 
              marker='X', zorder=8, label='Ball')
    
    ax.legend()
    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig('field_viz_simple_fallback.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: field_viz_simple_fallback.png")
    plt.show()


