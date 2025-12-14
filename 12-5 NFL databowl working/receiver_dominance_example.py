"""
Example: Using Receiver Dominance (CRP) with Your Training DataFrame

This script demonstrates how to:
1. Calculate receiver dominance scores for your dataframe
2. Create visualizations similar to CPP
3. Analyze dominance patterns

Adapted from the 2023 NFL Data Bowl winner's CPP approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from receiver_dominance_cpp import (
    calculate_receiver_dominance,
    calculate_play_dominance_sequence,
    visualize_receiver_dominance
)
from receiver_dominance_visualization import (
    plot_completion_vs_dominance,
    plot_dual_dominance_graphs,
    bucket_dominance_score
)


def add_dominance_scores_to_dataframe(
    training_df: pd.DataFrame,
    receiver_nfl_id_col: str = 'nfl_id',
    ball_land_x_col: str = 'ball_land_x_std',
    ball_land_y_col: str = 'ball_land_y_std'
) -> pd.DataFrame:
    """
    Add receiver dominance scores to your training dataframe
    
    Args:
        training_df: Your existing training dataframe
        receiver_nfl_id_col: Column name for receiver NFL ID
        ball_land_x_col: Column name for ball landing X coordinate
        ball_land_y_col: Column name for ball landing Y coordinate
    
    Returns:
        DataFrame with added 'receiver_dominance' column
    """
    df = training_df.copy()
    df['receiver_dominance'] = np.nan
    
    # Group by play to process efficiently
    if 'game_id' in df.columns and 'play_id' in df.columns:
        grouped = df.groupby(['game_id', 'play_id'])
    else:
        # If no grouping columns, process row by row
        grouped = [(None, df)]
    
    for (game_id, play_id), group in grouped:
        # Get ball landing position (should be same for all frames in a play)
        first_row = group.iloc[0]
        ball_land_x = first_row.get(ball_land_x_col, first_row.get('ball_land_x', 0))
        ball_land_y = first_row.get(ball_land_y_col, first_row.get('ball_land_y', 0))
        
        # Get receiver NFL ID
        receiver_nfl_id = first_row.get(receiver_nfl_id_col, first_row.get('nfl_id', None))
        
        if receiver_nfl_id is None or pd.isna(receiver_nfl_id):
            continue
        
        # Calculate dominance for each frame in this play
        for idx, row in group.iterrows():
            try:
                # Create a mini dataframe with just this frame's data
                # You may need to adjust this based on your dataframe structure
                frame_data = group.loc[[idx]]
                
                # If your dataframe has all players per frame, use it directly
                # Otherwise, you might need to merge with player tracking data
                dominance = calculate_receiver_dominance(
                    frame_data,
                    int(receiver_nfl_id),
                    float(ball_land_x),
                    float(ball_land_y)
                )
                df.loc[idx, 'receiver_dominance'] = dominance
            except Exception as e:
                print(f"Error calculating dominance for frame {idx}: {e}")
                continue
    
    return df


def calculate_max_dominance_per_play(
    df: pd.DataFrame,
    dominance_col: str = 'receiver_dominance'
) -> pd.DataFrame:
    """
    Calculate maximum dominance score per play (similar to maximas in CPP)
    
    Args:
        df: DataFrame with dominance scores
        dominance_col: Column name for dominance scores
    
    Returns:
        DataFrame with 'max_dominance' column added
    """
    df_result = df.copy()
    
    # Group by play
    if 'game_id' in df.columns and 'play_id' in df.columns:
        df_result['max_dominance'] = df_result.groupby(['game_id', 'play_id'])[dominance_col].transform('max')
    else:
        df_result['max_dominance'] = df_result[dominance_col]
    
    return df_result


def create_dominance_analysis(
    training_df: pd.DataFrame,
    output_dir: str = './dominance_analysis'
) -> None:
    """
    Create comprehensive dominance analysis similar to CPP analysis
    
    Args:
        training_df: Your training dataframe
        output_dir: Directory to save outputs
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Add dominance scores
    print("Step 1: Calculating receiver dominance scores...")
    df_with_dominance = add_dominance_scores_to_dataframe(training_df)
    
    # Step 2: Calculate max dominance per play
    print("Step 2: Calculating maximum dominance per play...")
    df_with_max = calculate_max_dominance_per_play(df_with_dominance)
    
    # Step 3: Create visualizations
    print("Step 3: Creating visualizations...")
    
    # Get one row per play (for play-level analysis)
    if 'game_id' in df_with_max.columns and 'play_id' in df_with_max.columns:
        play_level_df = df_with_max.groupby(['game_id', 'play_id']).first().reset_index()
    else:
        play_level_df = df_with_max.drop_duplicates(subset=['play_id'] if 'play_id' in df_with_max.columns else [])
    
    # Plot completion vs dominance
    if 'is_complete' in play_level_df.columns:
        fig1 = plot_completion_vs_dominance(
            play_level_df,
            dominance_col='max_dominance',
            completion_col='is_complete'
        )
        fig1.savefig(f'{output_dir}/completion_vs_dominance.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/completion_vs_dominance.png")
        plt.close(fig1)
        
        # Dual plot
        fig2 = plot_dual_dominance_graphs(
            play_level_df,
            dominance_col='max_dominance',
            completion_col='is_complete'
        )
        fig2.savefig(f'{output_dir}/dual_dominance_graphs.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/dual_dominance_graphs.png")
        plt.close(fig2)
    
    # Step 4: Save dataframe with dominance scores
    print("Step 4: Saving results...")
    df_with_max.to_csv(f'{output_dir}/training_df_with_dominance.csv', index=False)
    print(f"  Saved: {output_dir}/training_df_with_dominance.csv")
    
    # Step 5: Summary statistics
    print("\n=== Dominance Statistics ===")
    print(f"Mean dominance: {df_with_max['receiver_dominance'].mean():.3f}")
    print(f"Max dominance: {df_with_max['receiver_dominance'].max():.3f}")
    print(f"Min dominance: {df_with_max['receiver_dominance'].min():.3f}")
    
    if 'is_complete' in play_level_df.columns:
        print(f"\nMean dominance (completed): {play_level_df[play_level_df['is_complete']==1]['max_dominance'].mean():.3f}")
        print(f"Mean dominance (incomplete): {play_level_df[play_level_df['is_complete']==0]['max_dominance'].mean():.3f}")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")


# Example usage
if __name__ == "__main__":
    # Load your training dataframe
    # training_df = pd.read_csv('route_dominance_training_data.csv')
    
    # Or use your existing dataframe variable
    # create_dominance_analysis(training_df)
    
    print("""
    To use this with your dataframe:
    
    1. Load your training dataframe:
       training_df = pd.read_csv('route_dominance_training_data.csv')
       # or use your existing dataframe
    
    2. Run the analysis:
       create_dominance_analysis(training_df)
    
    3. Or calculate dominance for specific plays:
       from receiver_dominance_cpp import calculate_receiver_dominance
       
       # For a single frame
       dominance = calculate_receiver_dominance(
           frame_data,
           receiver_nfl_id=44930,
           ball_land_x=66.5,
           ball_land_y=41.6
       )
    """)

