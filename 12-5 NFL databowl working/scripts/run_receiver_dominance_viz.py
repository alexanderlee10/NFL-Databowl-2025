"""
Quick script to run receiver dominance visualizations
Run this to see the visualizations in action!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from receiver_dominance_cpp import (
    calculate_receiver_dominance,
    visualize_receiver_dominance
)
from receiver_dominance_visualization import (
    plot_completion_vs_dominance,
    plot_dual_dominance_graphs,
    plot_pressure_events_vs_dominance
)
from receiver_dominance_example import (
    add_dominance_scores_to_dataframe,
    calculate_max_dominance_per_play
)

# ============================================================================
# STEP 1: Load your training dataframe
# ============================================================================
print("="*80)
print("LOADING DATA")
print("="*80)

# Option 1: Load from CSV
try:
    training_df = pd.read_csv('route_dominance_training_data.csv')
    print(f"✓ Loaded {len(training_df):,} rows from CSV")
except FileNotFoundError:
    print("⚠ CSV file not found. Using your existing training_df variable...")
    print("   Make sure you have 'training_df' defined in your notebook/script")
    # If running in notebook, uncomment this:
    # training_df = your_existing_dataframe_variable
    raise FileNotFoundError("Please load your training dataframe first!")

print(f"  Columns: {list(training_df.columns)[:10]}...")
print(f"  Shape: {training_df.shape}")

# ============================================================================
# STEP 2: Calculate dominance scores (if not already calculated)
# ============================================================================
print("\n" + "="*80)
print("CALCULATING DOMINANCE SCORES")
print("="*80)

if 'receiver_dominance' not in training_df.columns:
    print("Calculating dominance scores for all frames...")
    print("  (This may take a few minutes for large datasets)")
    
    # For faster testing, you can use a subset
    # training_df = training_df.head(100)  # Uncomment to test with first 100 rows
    
    training_df = add_dominance_scores_to_dataframe(training_df)
    print(f"✓ Calculated dominance scores")
else:
    print("✓ Dominance scores already exist in dataframe")

print(f"  Mean dominance: {training_df['receiver_dominance'].mean():.3f}")
print(f"  Dominance range: [{training_df['receiver_dominance'].min():.3f}, {training_df['receiver_dominance'].max():.3f}]")

# ============================================================================
# STEP 3: Calculate max dominance per play
# ============================================================================
print("\n" + "="*80)
print("CALCULATING MAX DOMINANCE PER PLAY")
print("="*80)

training_df = calculate_max_dominance_per_play(training_df)
print("✓ Added 'max_dominance' column")

# Get play-level data for analysis
if 'game_id' in training_df.columns and 'play_id' in training_df.columns:
    play_level_df = training_df.groupby(['game_id', 'play_id']).first().reset_index()
    print(f"  {len(play_level_df):,} unique plays")
else:
    play_level_df = training_df.drop_duplicates(subset=['play_id'] if 'play_id' in training_df.columns else [])
    print(f"  {len(play_level_df):,} unique plays")

# ============================================================================
# STEP 4: Create Visualizations
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Visualization 1: Completion % vs Maximum Dominance
if 'is_complete' in play_level_df.columns:
    print("\n1. Creating Completion % vs Dominance graph...")
    fig1 = plot_completion_vs_dominance(
        play_level_df,
        dominance_col='max_dominance',
        completion_col='is_complete'
    )
    plt.savefig('completion_vs_dominance.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: completion_vs_dominance.png")
    plt.show()  # Display the plot
    plt.close()

# Visualization 2: Events vs Dominance
print("\n2. Creating Events vs Dominance graph...")
fig2 = plot_pressure_events_vs_dominance(
    play_level_df,
    dominance_col='max_dominance'
)
plt.savefig('events_vs_dominance.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: events_vs_dominance.png")
plt.show()
plt.close()

# Visualization 3: Dual plot (both graphs side by side)
if 'is_complete' in play_level_df.columns:
    print("\n3. Creating dual plot (both graphs)...")
    fig3 = plot_dual_dominance_graphs(
        play_level_df,
        dominance_col='max_dominance',
        completion_col='is_complete'
    )
    plt.savefig('dual_dominance_graphs.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: dual_dominance_graphs.png")
    plt.show()
    plt.close()

# ============================================================================
# STEP 5: Field Visualization (Single Play Example)
# ============================================================================
print("\n" + "="*80)
print("CREATING FIELD VISUALIZATION (Example Play)")
print("="*80)

# Find a play with good data
example_play = training_df[
    (training_df['receiver_dominance'].notna()) &
    (training_df['receiver_dominance'] > 0)
].iloc[0] if len(training_df[training_df['receiver_dominance'].notna()]) > 0 else None

if example_play is not None and len(example_play) > 0:
    print(f"\n4. Visualizing example play...")
    print(f"   Game ID: {example_play.get('game_id', 'N/A')}")
    print(f"   Play ID: {example_play.get('play_id', 'N/A')}")
    print(f"   Dominance: {example_play.get('receiver_dominance', 0):.3f}")
    
    # Get all frames for this play
    if 'game_id' in training_df.columns and 'play_id' in training_df.columns:
        play_frames = training_df[
            (training_df['game_id'] == example_play['game_id']) &
            (training_df['play_id'] == example_play['play_id'])
        ]
    else:
        play_frames = training_df[training_df.index == example_play.name]
    
    if len(play_frames) > 0:
        # Get ball landing position
        ball_land_x = example_play.get('ball_land_x_std', example_play.get('ball_land_x', 0))
        ball_land_y = example_play.get('ball_land_y_std', example_play.get('ball_land_y', 0))
        receiver_nfl_id = example_play.get('nfl_id', example_play.get('receiver_nfl_id', None))
        dominance_score = example_play.get('receiver_dominance', 0.5)
        
        if receiver_nfl_id is not None:
            try:
                # Create field visualization
                fig4 = visualize_receiver_dominance(
                    play_frames.iloc[0:1],  # Use first frame for visualization
                    receiver_nfl_id=int(receiver_nfl_id),
                    ball_land_x=float(ball_land_x),
                    ball_land_y=float(ball_land_y),
                    dominance_score=float(dominance_score),
                    show_players=True,
                    label_numbers=True
                )
                plt.savefig('field_visualization_example.png', dpi=300, bbox_inches='tight')
                print("   ✓ Saved: field_visualization_example.png")
                plt.show()
                plt.close()
            except Exception as e:
                print(f"   ⚠ Could not create field visualization: {e}")
                print("   (This is okay - field visualization needs full player tracking data)")
else:
    print("\n4. Skipping field visualization (no suitable play found)")

# ============================================================================
# STEP 6: Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nFrame-level statistics:")
print(f"  Mean dominance: {training_df['receiver_dominance'].mean():.3f}")
print(f"  Std dominance:  {training_df['receiver_dominance'].std():.3f}")
print(f"  Min dominance:  {training_df['receiver_dominance'].min():.3f}")
print(f"  Max dominance:  {training_df['receiver_dominance'].max():.3f}")

if 'is_complete' in play_level_df.columns:
    print(f"\nPlay-level statistics (by completion):")
    completed = play_level_df[play_level_df['is_complete'] == 1]
    incomplete = play_level_df[play_level_df['is_complete'] == 0]
    
    if len(completed) > 0:
        print(f"  Completed plays:")
        print(f"    Mean max dominance: {completed['max_dominance'].mean():.3f}")
        print(f"    Count: {len(completed):,}")
    
    if len(incomplete) > 0:
        print(f"  Incomplete plays:")
        print(f"    Mean max dominance: {incomplete['max_dominance'].mean():.3f}")
        print(f"    Count: {len(incomplete):,}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - completion_vs_dominance.png")
print("  - events_vs_dominance.png")
print("  - dual_dominance_graphs.png")
if example_play is not None:
    print("  - field_visualization_example.png")
print("\nAll visualizations have been displayed and saved!")

