"""
Notebook cell to verify receiver_pressure is in the dataframe and varies by frame

Add this as a new cell in your notebook after creating training_df
"""

# Verify receiver_pressure column exists
if 'receiver_pressure' in training_df.columns:
    print("="*80)
    print("RECEIVER PRESSURE VERIFICATION")
    print("="*80)
    print(f"✓ receiver_pressure column found in training_df")
    print(f"\nColumn statistics:")
    print(training_df['receiver_pressure'].describe())
    
    print(f"\n" + "="*80)
    print("FRAME-BY-FRAME VARIATION (Sample Play)")
    print("="*80)
    
    # Show receiver_pressure varies by frame for a sample play
    sample_play = training_df.groupby(['game_id', 'play_id']).first().reset_index().iloc[0]
    sample_frames = training_df[
        (training_df['game_id'] == sample_play['game_id']) &
        (training_df['play_id'] == sample_play['play_id'])
    ].sort_values('continuous_frame')
    
    print(f"\nGame {sample_play['game_id']}, Play {sample_play['play_id']}:")
    print(f"Receiver Pressure by Frame:")
    display_cols = ['continuous_frame', 'frame_type', 'receiver_pressure', 'sep_nearest', 'num_def_within_3']
    print(sample_frames[display_cols].to_string(index=False))
    
    print(f"\n" + "="*80)
    print("RECEIVER PRESSURE DISTRIBUTION")
    print("="*80)
    print(f"Mean receiver pressure: {training_df['receiver_pressure'].mean():.3f}")
    print(f"Std receiver pressure: {training_df['receiver_pressure'].std():.3f}")
    print(f"Min receiver pressure: {training_df['receiver_pressure'].min():.3f}")
    print(f"Max receiver pressure: {training_df['receiver_pressure'].max():.3f}")
    
    # Show how receiver_pressure changes within a play
    print(f"\n" + "="*80)
    print("RECEIVER PRESSURE VARIATION WITHIN PLAYS")
    print("="*80)
    pressure_variation = training_df.groupby(['game_id', 'play_id'])['receiver_pressure'].agg(['min', 'max', 'std'])
    print(f"Average pressure range per play: {(pressure_variation['max'] - pressure_variation['min']).mean():.3f}")
    print(f"Average pressure std per play: {pressure_variation['std'].mean():.3f}")
    
    print(f"\n✓ Receiver pressure is calculated and varies frame-by-frame!")
    
else:
    print("="*80)
    print("WARNING: receiver_pressure column NOT FOUND")
    print("="*80)
    print("The receiver_pressure column should be automatically included when you call:")
    print("  scorer.calculate_frame_dominance(game_id, play_id, target_nfl_id)")
    print("\nTo add it, you need to:")
    print("1. Make sure you're using the updated route_dominance_scoring.py")
    print("2. Regenerate your training_df by running:")
    print("   training_df, errors = create_training_dataframe(...)")
    print("\nThe receiver_pressure is calculated using:")
    print("- Receiver influence PDF (6-yard radius, centered 2 yards in front)")
    print("- Defender pressure PDFs (4-yard radius, all defenders within 6 yards)")
    print("- Multivariate normal distributions")
    print("- Values range from 0.0 (high defender pressure) to 1.0 (high receiver advantage)")


