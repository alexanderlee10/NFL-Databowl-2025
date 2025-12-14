"""
Create Animated GIF of Receiver Dominance (like CPP GIFs)

This creates an animated GIF showing receiver dominance evolving frame-by-frame
through a play, just like the CPP pressure GIFs.

Features used from your dataframe:
- receiver_x, receiver_y: Receiver position
- nearest_defender_x, nearest_defender_y: Defender position  
- sep_nearest: Separation distance
- receiver_speed, receiver_accel: Receiver motion
- leverage_angle: Leverage angle
- dist_to_ball: Distance to ball
- ball_land_x_std, ball_land_y_std: Ball landing position
- receiver_dominance: Dominance score (if calculated)
- continuous_frame or frame_id: Frame number
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import multivariate_normal
import imageio
import os
import math
from typing import Optional


def create_dominance_gif_from_dataframe(
    training_df: pd.DataFrame,
    game_id: int,
    play_id: int,
    receiver_nfl_id: Optional[int] = None,
    output_filename: str = None,
    fps: int = 5,
    scorer: Optional[object] = None
) -> str:
    """
    Create animated GIF of receiver dominance for a play
    
    Args:
        training_df: Your training dataframe
        game_id: Game ID
        play_id: Play ID
        receiver_nfl_id: Receiver NFL ID (if None, uses first receiver in play)
        output_filename: Output GIF filename (if None, auto-generates)
        fps: Frames per second for GIF
    
    Returns:
        Path to created GIF file
    
    Features used from dataframe:
    - receiver_x, receiver_y: Receiver position
    - nearest_defender_x, nearest_defender_y: Defender position
    - sep_nearest: Separation
    - receiver_speed: Speed
    - leverage_angle: Leverage
    - ball_land_x_std, ball_land_y_std: Ball landing
    - receiver_dominance: Dominance score
    - continuous_frame: Frame number
    """
    print("="*80)
    print(f"CREATING DOMINANCE GIF FOR GAME {game_id}, PLAY {play_id}")
    print("="*80)
    
    # Get play data from training dataframe (for receiver metrics)
    play_data = training_df[
        (training_df['game_id'] == game_id) &
        (training_df['play_id'] == play_id)
    ].copy()
    
    if len(play_data) == 0:
        raise ValueError(f"Play {game_id}-{play_id} not found in dataframe")
    
    # Sort by frame
    if 'continuous_frame' in play_data.columns:
        play_data = play_data.sort_values('continuous_frame')
        frame_col = 'continuous_frame'
    elif 'frame_id' in play_data.columns:
        play_data = play_data.sort_values('frame_id')
        frame_col = 'frame_id'
    else:
        play_data = play_data.sort_index()
        frame_col = None
    
    # Get receiver NFL ID
    if receiver_nfl_id is None:
        receiver_nfl_id = play_data['nfl_id'].iloc[0]
    
    # Get ALL players on field from scorer (if available)
    all_players_data = None
    if scorer is not None and hasattr(scorer, 'all_frames_df'):
        all_players_data = scorer.all_frames_df[
            (scorer.all_frames_df['game_id'] == game_id) &
            (scorer.all_frames_df['play_id'] == play_id)
        ].copy()
        print(f"\nUsing ALL players from scorer.all_frames_df")
        if len(all_players_data) > 0:
            players_per_frame = all_players_data.groupby('frame_id').size()
            if len(players_per_frame) > 0:
                print(f"  - Total players per frame: ~{players_per_frame.iloc[0]}")
            else:
                print(f"  - Total players: {len(all_players_data)}")
        else:
            print(f"  - No player data found")
    else:
        print(f"\nNote: Scorer not provided - will use limited player data")
        print(f"  To show all players, pass scorer object to create_gif_for_play()")
    
    print(f"\nFeatures used from dataframe:")
    print(f"  - Receiver: nfl_id={receiver_nfl_id}")
    print(f"  - Frames: {len(play_data)} frames")
    
    # Get ball landing (should be same for all frames) - use standardized coordinates
    first_frame = play_data.iloc[0]
    # ball_land_x_std and ball_land_y_std are already standardized in training_df
    ball_land_x = first_frame.get('ball_land_x_std', first_frame.get('ball_land_x', 0))
    ball_land_y = first_frame.get('ball_land_y_std', first_frame.get('ball_land_y', 0))
    print(f"  - Ball landing: ({ball_land_x:.1f}, {ball_land_y:.1f})")
    
    # Verify coordinates are standardized (check first frame receiver position)
    sample_receiver_x = first_frame.get('receiver_x', first_frame.get('x_std', np.nan))
    sample_receiver_y = first_frame.get('receiver_y', first_frame.get('y_std', np.nan))
    print(f"  - Sample receiver position: x={sample_receiver_x:.1f}, y={sample_receiver_y:.1f} (should use x_std/y_std)")
    
    # Create output directory
    output_dir = "dominance_gifs"
    os.makedirs(output_dir, exist_ok=True)
    
    if output_filename is None:
        output_filename = f"dominance_game{game_id}_play{play_id}.gif"
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Create images for each frame
    print(f"\nCreating {len(play_data)} frame visualizations...")
    image_files = []
    
    for idx, (frame_idx, frame_row) in enumerate(play_data.iterrows()):
        frame_num = frame_row.get(frame_col, idx + 1) if frame_col else idx + 1
        
        # Get dominance for this frame (for debugging)
        frame_dom = frame_row.get('receiver_dominance', None)
        if frame_dom is not None and not pd.isna(frame_dom):
            print(f"  Frame {idx+1}/{len(play_data)} (frame {frame_num}) - Dominance: {frame_dom:.3f}...", end='\r')
        else:
            print(f"  Frame {idx+1}/{len(play_data)} (frame {frame_num})...", end='\r')
        
        # Get all players for this frame (if available)
        frame_players = None
        if all_players_data is not None:
            if frame_col == 'continuous_frame':
                # Map continuous_frame to frame_id and frame_type
                frame_id = frame_row.get('frame_id', frame_num)
                frame_type = frame_row.get('frame_type', 'input')
                frame_players = all_players_data[
                    (all_players_data['frame_id'] == frame_id) &
                    (all_players_data['frame_type'] == frame_type)
                ]
            elif 'frame_id' in frame_row:
                frame_id = frame_row['frame_id']
                frame_type = frame_row.get('frame_type', 'input')
                frame_players = all_players_data[
                    (all_players_data['frame_id'] == frame_id) &
                    (all_players_data['frame_type'] == frame_type)
                ]
        
        # Create visualization for this frame
        fig = create_single_frame_visualization(
            frame_row,
            receiver_nfl_id,
            ball_land_x,
            ball_land_y,
            frame_num=frame_num,
            total_frames=len(play_data),
            all_players_frame=frame_players  # Pass all players if available
        )
        
        # Save frame
        frame_filename = os.path.join(output_dir, f"frame_{frame_num:03d}.png")
        plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
        image_files.append(frame_filename)
        plt.close(fig)
    
    print(f"\n  Created {len(image_files)} frame images")
    
    # Create GIF
    print(f"\nCreating GIF from {len(image_files)} frames...")
    images = []
    for filename in image_files:
        images.append(imageio.imread(filename))
    
    imageio.mimsave(output_path, images, fps=fps)
    print(f"  Saved GIF: {output_path}")
    
    # Clean up individual frame images
    for filename in image_files:
        try:
            os.remove(filename)
        except:
            pass
    
    print(f"\nGIF creation complete!")
    print(f"  Output: {output_path}")
    print(f"  Frames: {len(images)}")
    print(f"  FPS: {fps}")
    
    return output_path


def create_single_frame_visualization(
    frame_row: pd.Series,
    receiver_nfl_id: int,
    ball_land_x: float,
    ball_land_y: float,
    frame_num: int = 1,
    total_frames: int = 1,
    all_players_frame: Optional[pd.DataFrame] = None
) -> plt.Figure:
    """
    Create single frame visualization (one frame of the GIF)
    
    Uses these features from frame_row:
    - receiver_x, receiver_y
    - nearest_defender_x, nearest_defender_y
    - sep_nearest
    - receiver_speed
    - leverage_angle
    - receiver_dominance
    """
    # Extract features from dataframe (use standardized coordinates)
    # IMPORTANT: In the notebook, receiver_x and receiver_y are set from x_std and y_std
    # So receiver_x = x_std (length, 0-120) and receiver_y = y_std (width, 0-53.3)
    # When plotting: x-axis = width (y_std), y-axis = length (x_std)
    receiver_x = frame_row.get('receiver_x', frame_row.get('x_std', frame_row.get('x', 0)))
    receiver_y = frame_row.get('receiver_y', frame_row.get('y_std', frame_row.get('y', 0)))
    def_x = frame_row.get('nearest_defender_x', frame_row.get('x_std', receiver_x + 3))
    def_y = frame_row.get('nearest_defender_y', frame_row.get('y_std', receiver_y))
    sep_nearest = frame_row.get('sep_nearest', 3.0)
    receiver_speed = frame_row.get('receiver_speed', 0)
    leverage_angle = frame_row.get('leverage_angle', np.nan)
    
    # Get dominance score from THIS FRAME (this changes frame-by-frame!)
    # If not available, calculate it from the current frame's separation and positions
    dominance_score = frame_row.get('receiver_dominance', None)
    if dominance_score is None or pd.isna(dominance_score):
        # Fallback: estimate from separation (closer = lower dominance)
        if sep_nearest < np.inf:
            # Simple heuristic: more separation = higher dominance
            dominance_score = min(0.95, max(0.05, 0.5 + (sep_nearest - 3.0) / 20.0))
        else:
            dominance_score = 0.5
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(10.66, 24))
    
    # Draw field (EXACT CPP STYLE)
    rect = patches.Rectangle((0, 0), 53.3, 120, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)
    ax.add_patch(rect)
    
    # Field lines
    plt.plot([0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             [10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             color='white', linewidth=1.5)
    
    # End zones
    home_endzone = patches.Rectangle((0, 0), 53.3, 10,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='purple',
                                    alpha=0.2,
                                    zorder=10)
    away_endzone = patches.Rectangle((0, 110), 53.3, 10,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='purple',
                                    alpha=0.2,
                                    zorder=10)
    ax.add_patch(home_endzone)
    ax.add_patch(away_endzone)
    
    # Yard markers
    for y in range(20, 110, 10):
        numb = y
        if y > 50:
            numb = 120 - y
        plt.text(5, y-1.5, str(numb - 10),
                 horizontalalignment='center',
                 fontsize=18,
                 color='white', rotation=270, fontweight='bold')
        plt.text(53.3 - 5, y - 0.95, str(numb - 10),
                 horizontalalignment='center',
                 fontsize=18,
                 color='white', rotation=90, fontweight='bold')
    
    # Hash lines
    for y in range(11, 110):
        ax.plot([0.7, 0.4], [y, y], color='white', linewidth=0.5)
        ax.plot([53.0, 52.5], [y, y], color='white', linewidth=0.5)
        ax.plot([22.91, 23.57], [y, y], color='white', linewidth=0.5)
        ax.plot([29.73, 30.39], [y, y], color='white', linewidth=0.5)
    
    # Create contour plot (dominance regions)
    x, y = np.mgrid[0:53.3:0.5, 0:120:0.5]
    
    # Receiver influence PDF
    receiver_pdf = multivariate_normal([receiver_y, receiver_x], [[8, 0], [0, 8]]).pdf(np.dstack((x, y)))
    
    # Defender pressure PDF (weighted by separation)
    if sep_nearest < np.inf:
        def_weight = 1.0 / (1.0 + sep_nearest / 5.0)
    else:
        def_weight = 0.5
    
    defender_pdf = multivariate_normal([def_y, def_x], [[6, 0], [0, 6]]).pdf(np.dstack((x, y))) * def_weight
    
    # Calculate dominance
    total_pdf = receiver_pdf + defender_pdf + 1e-10
    dominance_pdf = receiver_pdf / total_pdf
    
    # Draw contour (Purple colormap like CPP)
    ax.contourf(x, y, dominance_pdf, cmap='Purples', alpha=0.7, levels=15, zorder=1)
    
    # Plot ALL players if available (like CPP shows all players)
    if all_players_frame is not None and len(all_players_frame) > 0:
        # Plot all players on field
        for idx, player in all_players_frame.iterrows():
            player_x = player.get('x_std', player.get('x', 0))
            player_y = player.get('y_std', player.get('y', 0))
            player_nfl_id = player.get('nfl_id', 0)
            player_side = player.get('player_side', 'Unknown')
            player_pos = player.get('player_position', player.get('officialPosition', ''))
            jersey_num = player.get('jerseyNumber', player_nfl_id % 100)
            
            is_receiver = player_nfl_id == receiver_nfl_id
            
            if is_receiver:
                # Highlight receiver (green star - like QB in CPP)
                ax.scatter(player_y, player_x, color='limegreen', s=500,
                          marker='*', edgecolors='black', linewidths=3, zorder=10)
                ax.annotate(str(int(jersey_num)) if pd.notna(jersey_num) else 'WR', 
                           (player_y, player_x),
                           xytext=(player_y-0.5, player_x-0.5),
                           color='white', fontweight='bold', fontsize=12)
            elif player_side == 'Defense':
                # Defenders (blue circles - like away team in CPP)
                ax.scatter(player_y, player_x, color='blue', s=300,
                          edgecolors='white', linewidths=2, zorder=9)
                if pd.notna(jersey_num):
                    ax.annotate(str(int(jersey_num)), 
                               (player_y, player_x),
                               xytext=(player_y-0.5, player_x-0.5),
                               color='white', fontsize=10)
            else:
                # Other offensive players (red circles - like home team in CPP)
                ax.scatter(player_y, player_x, color='red', s=300,
                          edgecolors='white', linewidths=2, zorder=9)
                if pd.notna(jersey_num):
                    ax.annotate(str(int(jersey_num)), 
                               (player_y, player_x),
                               xytext=(player_y-0.5, player_x-0.5),
                               color='white', fontsize=10)
    else:
        # Fallback: Just plot receiver and nearest defender
        # Plot receiver (green star - like QB in CPP)
        ax.scatter(receiver_y, receiver_x, color='limegreen', s=500,
                  marker='*', edgecolors='black', linewidths=3, zorder=10)
        ax.annotate('WR', (receiver_y, receiver_x),
                   xytext=(receiver_y-1, receiver_x-1),
                   color='white', fontweight='bold', fontsize=14)
        
        # Plot defender (blue circle)
        ax.scatter(def_y, def_x, color='blue', s=400,
                  edgecolors='white', linewidths=2, zorder=9)
        ax.annotate('CB', (def_y, def_x),
                   xytext=(def_y-1, def_x-1),
                   color='white', fontsize=12)
    
    # Plot ball landing (yellow X)
    ax.scatter(ball_land_y, ball_land_x, color='yellow', s=600,
              marker='X', edgecolors='black', linewidths=3, zorder=8)
    
    # Draw line from receiver to ball
    ax.plot([receiver_y, ball_land_y], [receiver_x, ball_land_x],
           color='yellow', linewidth=2, linestyle='--', alpha=0.5, zorder=2)
    
    # Draw separation circle
    if sep_nearest < np.inf:
        circle = patches.Circle((receiver_y, receiver_x), sep_nearest,
                               fill=False, edgecolor='cyan', linewidth=2,
                               linestyle=':', alpha=0.7, zorder=3)
        ax.add_patch(circle)
    
    # Dominance indicator
    dominance_percent = int(dominance_score * 100)
    
    if dominance_score >= 0.8:
        indicator_color = '#00FF00'  # Green
    elif dominance_score >= 0.65:
        indicator_color = '#FFFF00'  # Yellow
    elif dominance_score >= 0.5:
        indicator_color = '#FFA500'  # Orange
    else:
        indicator_color = '#FF0000'  # Red
    
    # Frame info
    frame_text = f"Frame {frame_num}/{total_frames}"
    ax.text(2, 115, frame_text,
           fontsize=16, fontweight='bold', color='white',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Dominance score
    ax.text(2, 112, f"Dominance: {dominance_percent}%",
           fontsize=22, fontweight='bold', color='white',
           bbox=dict(boxstyle='round,pad=1', facecolor=indicator_color,
                    alpha=0.8, edgecolor='black', linewidth=2))
    
    # Stats box
    stats_text = f"Sep: {sep_nearest:.1f}yd\nSpeed: {receiver_speed:.1f}yd/s"
    if not pd.isna(leverage_angle):
        stats_text += f"\nLeverage: {leverage_angle:.0f}°"
    
    ax.text(2, 5, stats_text,
           fontsize=12, fontweight='bold', color='white',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
           verticalalignment='bottom')
    
    # Set axis
    ax.set_xlim(0, 53.3)
    ax.set_ylim(0, 120)
    ax.set_aspect('equal')
    plt.axis('off')
    
    return fig


# Quick usage function
def create_gif_for_play(
    training_df: pd.DataFrame,
    game_id: int,
    play_id: int,
    fps: int = 5,
    scorer: Optional[object] = None
) -> str:
    """
    Quick function to create GIF for a play
    
    Args:
        training_df: Your training dataframe
        game_id: Game ID
        play_id: Play ID
        fps: Frames per second
        scorer: RouteDominanceScorer object (optional - if provided, shows ALL players on field)
    
    Example:
        # With all players (recommended):
        from route_dominance_scoring import RouteDominanceScorer
        scorer = RouteDominanceScorer(input_df, output_df, supp_df)
        gif_path = create_gif_for_play(training_df, 2023090700, 101, scorer=scorer)
        
        # Without scorer (limited players):
        gif_path = create_gif_for_play(training_df, 2023090700, 101)
    """
    return create_dominance_gif_from_dataframe(
        training_df, game_id, play_id, fps=fps, scorer=scorer
    )


if __name__ == "__main__":
    print("""
    To create a GIF from your dataframe:
    
    from create_dominance_gif import create_gif_for_play
    
    # Load your dataframe
    training_df = pd.read_csv('route_dominance_training_data.csv')
    
    # Create GIF for a play
    gif_path = create_gif_for_play(training_df, 2023090700, 101)
    
    # The GIF will be saved in 'dominance_gifs/' folder
    """)


