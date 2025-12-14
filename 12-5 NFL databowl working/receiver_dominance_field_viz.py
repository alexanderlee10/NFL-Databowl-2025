"""
Receiver Dominance Field Visualization
Exact match to CPP field visualization style - shows players on field with contour plots

This creates the full field view with:
- All players positioned on the field
- Purple contour plot showing dominance regions
- Dominance indicator in corner
- Field markings and yard lines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import multivariate_normal
import math
from typing import Optional, Tuple


def visualize_receiver_dominance_field(
    player_coordinates: pd.DataFrame,
    receiver_nfl_id: int,
    ball_land_x: float,
    ball_land_y: float,
    dominance_score: Optional[float] = None,
    img_size: Tuple[float, float] = (10.66, 24),
    field_color: str = 'darkgreen',
    endzone_color: str = 'purple',
    label_numbers: bool = True,
    show_arrow: bool = False
) -> plt.Figure:
    """
    Full field visualization matching CPP style exactly
    
    Shows:
    - Football field with yard lines and markings
    - All players positioned (receiver highlighted, defenders, offense)
    - Purple contour plot showing dominance regions
    - Dominance indicator in corner (like pressure indicator in CPP)
    
    Args:
        player_coordinates: DataFrame with all player positions for one frame
        receiver_nfl_id: NFL ID of target receiver
        ball_land_x: X coordinate of ball landing
        ball_land_y: Y coordinate of ball landing
        dominance_score: Pre-calculated dominance (0-1). If None, will calculate.
        img_size: Figure size tuple
        field_color: Field background color
        endzone_color: Endzone color
        label_numbers: Whether to show jersey numbers
        show_arrow: Whether to show direction arrows
    
    Returns:
        matplotlib Figure
    """
    # Draw the football field (EXACT CPP STYLE)
    rect = patches.Rectangle((0, 0), 53.3, 120, linewidth=0.1,
                             edgecolor='r', facecolor=field_color, zorder=0)
    fig, ax = plt.subplots(1, figsize=img_size)
    
    # Field lines (EXACT CPP STYLE)
    plt.plot([0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             [10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             color='white', linewidth=1.5)
    
    # End zones (EXACT CPP STYLE)
    home_endzone = patches.Rectangle((0, 0), 53.3, 10,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor=endzone_color,
                                    alpha=0.2,
                                    zorder=10)
    away_endzone = patches.Rectangle((0, 110), 53.3, 10,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor=endzone_color,
                                    alpha=0.2,
                                    zorder=10)
    ax.add_patch(home_endzone)
    ax.add_patch(away_endzone)
    
    # Yard markers (EXACT CPP STYLE)
    for y in range(20, 110, 10):
        numb = y
        if y > 50:
            numb = 120 - y
        plt.text(5, y-1.5, str(numb - 10),
                 horizontalalignment='center',
                 fontsize=20,
                 color='white', rotation=270, fontweight='bold')
        plt.text(53.3 - 5, y - 0.95, str(numb - 10),
                 horizontalalignment='center',
                 fontsize=20,
                 color='white', rotation=90, fontweight='bold')
    
    # Hash lines (EXACT CPP STYLE)
    for y in range(11, 110):
        ax.plot([0.7, 0.4], [y, y], color='white', linewidth=0.5)
        ax.plot([53.0, 52.5], [y, y], color='white', linewidth=0.5)
        ax.plot([22.91, 23.57], [y, y], color='white', linewidth=0.5)
        ax.plot([29.73, 30.39], [y, y], color='white', linewidth=0.5)
    
    # Get receiver position
    receiver_row = player_coordinates[player_coordinates['nfl_id'] == receiver_nfl_id]
    if len(receiver_row) == 0:
        # Try alternative column names
        receiver_row = player_coordinates[
            (player_coordinates.get('player_role') == 'Targeted Receiver') |
            (player_coordinates.get('officialPosition') == 'WR') |
            (player_coordinates.get('officialPosition') == 'TE')
        ]
    
    if len(receiver_row) == 0:
        raise ValueError(f"Receiver {receiver_nfl_id} not found in player_coordinates")
    
    receiver_row = receiver_row.iloc[0]
    receiver_x = receiver_row.get('receiver_x', receiver_row.get('x', 0))
    receiver_y = receiver_row.get('receiver_y', receiver_row.get('y', 0))
    
    # Calculate speed components for all players (like CPP)
    player_coords = player_coordinates.copy()
    
    # Add distance from receiver
    player_coords['distanceFromReceiver'] = np.sqrt(
        (player_coords.get('x', player_coords.get('receiver_x', 0)) - receiver_x)**2 +
        (player_coords.get('y', player_coords.get('receiver_y', 0)) - receiver_y)**2
    )
    
    # Calculate direction components (like CPP)
    if 'dir' in player_coords.columns:
        player_coords['radiansDirection'] = player_coords['dir'].astype(float).apply(math.radians)
        player_coords['xComponent'] = player_coords['radiansDirection'].astype(float).apply(math.cos)
        player_coords['yComponent'] = player_coords['radiansDirection'].astype(float).apply(math.sin)
    else:
        player_coords['xComponent'] = 1.0
        player_coords['yComponent'] = 0.0
    
    # Speed components
    if 's' in player_coords.columns:
        player_coords['xspeed'] = player_coords['xComponent'] * player_coords['s']
        player_coords['yspeed'] = player_coords['yComponent'] * player_coords['s']
    else:
        player_coords['xspeed'] = 0
        player_coords['yspeed'] = 0
    
    # Define field domain for PDFs
    x, y = np.mgrid[0:53.3:1, 0:120:1]
    locations = np.dstack((x, y))
    
    # Initialize PDFs (like CPP)
    defense_pdf_is_none = True
    receiver_pdf_is_none = True
    defense_pdf = None
    receiver_pdf = None
    receiver_pos_x = 0
    receiver_pos_y = 0
    
    # Get defenders and offensive players
    defenders = player_coords[
        (player_coords.get('player_side') == 'Defense') |
        (player_coords.get('team') != player_coords.get('possessionTeam', ''))
    ].copy()
    
    # Generate defensive player PDFs (like CPP defensive players)
    for idx, row in defenders.iterrows():
        def_x = row.get('x', row.get('nearest_defender_x', 0))
        def_y = row.get('y', row.get('nearest_defender_y', 0))
        def_speed = row.get('s', 0)
        
        distance_from_receiver = row.get('distanceFromReceiver', 
                                        np.sqrt((def_x - receiver_x)**2 + (def_y - receiver_y)**2))
        
        # CPP-style covariance matrix calculation
        speed_ratio = (def_speed**2) / 100 if def_speed > 0 else 0.01
        top_left_s = (distance_from_receiver + distance_from_receiver * speed_ratio) / 2
        bottom_right_s = (distance_from_receiver - distance_from_receiver * speed_ratio) / 2
        top_left_s = max(top_left_s, 0.0001)
        bottom_right_s = max(bottom_right_s, 0.0001)
        
        x_comp = row.get('xComponent', 1.0)
        y_comp = row.get('yComponent', 0.0)
        
        r_matrix = np.array([[x_comp, -y_comp], [y_comp, x_comp]])
        s_matrix = np.array([[top_left_s, 0], [0, bottom_right_s]])
        
        try:
            inv_r = np.linalg.inv(r_matrix)
            covariance_matrix = r_matrix @ s_matrix @ s_matrix @ inv_r
        except:
            covariance_matrix = np.array([[top_left_s, 0], [0, bottom_right_s]])
        
        mu_x = def_y + row.get('yspeed', 0) * 0.5
        mu_y = def_x + row.get('xspeed', 0) * 0.5
        mu = [mu_x, mu_y]
        
        try:
            player_pdf = multivariate_normal(mu, covariance_matrix).pdf(locations)
        except:
            player_pdf = multivariate_normal([mu_x, mu_y], [[4, 0], [0, 4]]).pdf(locations)
        
        if defense_pdf_is_none:
            defense_pdf = player_pdf
            defense_pdf_is_none = False
        else:
            defense_pdf = defense_pdf + player_pdf
    
    # Generate receiver advantage PDF (like CPP offensive line protection)
    receiver_speed = receiver_row.get('receiver_speed', 0)
    sep_nearest = receiver_row.get('sep_nearest', np.inf)
    leverage_angle = receiver_row.get('leverage_angle', np.nan)
    
    # Receiver advantage factors
    if sep_nearest < np.inf:
        separation_advantage = min(sep_nearest / 10.0, 1.0)
    else:
        separation_advantage = 0.5
    
    if not pd.isna(leverage_angle):
        leverage_advantage = leverage_angle / 180.0
    else:
        leverage_advantage = 0.5
    
    speed_advantage = min(receiver_speed / 8.0, 1.0) if receiver_speed > 0 else 0.0
    receiver_advantage_factor = 0.4 * separation_advantage + 0.3 * leverage_advantage + 0.3 * speed_advantage
    
    # Receiver PDF
    speed_ratio = (receiver_speed**2) / 100 if receiver_speed > 0 else 0.01
    distance_to_ball = receiver_row.get('dist_to_ball', 
                                       np.sqrt((receiver_x - ball_land_x)**2 + (receiver_y - ball_land_y)**2))
    
    top_left_s = (distance_to_ball + distance_to_ball * speed_ratio) / 2
    bottom_right_s = (distance_to_ball - distance_to_ball * speed_ratio) / 2
    top_left_s = max(top_left_s, 0.0001)
    bottom_right_s = max(bottom_right_s, 0.0001)
    
    x_comp = receiver_row.get('xComponent', 1.0) if 'xComponent' in receiver_row else 1.0
    y_comp = receiver_row.get('yComponent', 0.0) if 'yComponent' in receiver_row else 0.0
    
    r_matrix = np.array([[x_comp, -y_comp], [y_comp, x_comp]])
    s_matrix = np.array([[top_left_s, 0], [0, bottom_right_s]])
    
    try:
        inv_r = np.linalg.inv(r_matrix)
        covariance_matrix = r_matrix @ s_matrix @ s_matrix @ inv_r
    except:
        covariance_matrix = np.array([[top_left_s, 0], [0, bottom_right_s]])
    
    mu_x = receiver_y + receiver_row.get('yspeed', 0) * 0.5
    mu_y = receiver_x + receiver_row.get('xspeed', 0) * 0.5
    mu = [mu_x, mu_y]
    
    try:
        receiver_player_pdf = multivariate_normal(mu, covariance_matrix).pdf(locations)
    except:
        receiver_player_pdf = multivariate_normal([mu_x, mu_y], [[4, 0], [0, 4]]).pdf(locations)
    
    if receiver_pdf_is_none:
        receiver_pdf = receiver_advantage_factor * receiver_player_pdf
        receiver_pdf_is_none = False
    
    # Calculate dominance PDF (like CPP pressure PDF)
    if defense_pdf is None:
        defense_pdf = np.zeros_like(receiver_pdf)
    if receiver_pdf is None:
        receiver_pdf = np.zeros_like(defense_pdf)
    
    pdf = np.array(defense_pdf) / (np.array(defense_pdf) + np.array(receiver_pdf) + 1e-10)
    is_def = defense_pdf > 0.01
    display_pdf = is_def * pdf
    
    # Receiver area PDF (like QB pocket PDF in CPP)
    receiver_pos_x = receiver_y  # Note: x and y are swapped in field coordinates
    receiver_pos_y = receiver_x
    receiver_area_pdf = multivariate_normal([receiver_pos_x, receiver_pos_y], 
                                           [[6, 0], [0, 6]]).pdf(locations)
    
    # Calculate dominance score if not provided
    if dominance_score is None:
        dominance_pdf = receiver_area_pdf * (1 - pdf)  # Invert: higher receiver advantage = higher dominance
        dominance_score = np.sum(dominance_pdf) / np.sum(receiver_area_pdf)
        dominance_score = (dominance_score - 0.50) / (0.80 - 0.50)
        dominance_score = max(0, min(1, dominance_score))
    
    # Ensure we have valid PDFs for contour
    if defense_pdf is None or np.all(defense_pdf == 0):
        # Create a default defense PDF if none exists
        defense_pdf = np.zeros_like(receiver_area_pdf)
        for idx, row in defenders.iterrows():
            def_x = row.get('x', row.get('nearest_defender_x', receiver_x + 3))
            def_y = row.get('y', row.get('nearest_defender_y', receiver_y))
            def_pdf = multivariate_normal([def_y, def_x], [[6, 0], [0, 6]]).pdf(locations)
            defense_pdf = defense_pdf + def_pdf
    
    if receiver_pdf is None or np.all(receiver_pdf == 0):
        # Create a default receiver PDF
        receiver_pdf = multivariate_normal([receiver_y, receiver_x], [[8, 0], [0, 8]]).pdf(locations)
    
    # Calculate dominance PDF
    pdf = np.array(defense_pdf) / (np.array(defense_pdf) + np.array(receiver_pdf) + 1e-10)
    is_def = defense_pdf > 0.01
    display_pdf = is_def * pdf
    
    # Draw contour plot (EXACT CPP STYLE - Purple colormap)
    # Ensure display_pdf has valid values
    if np.any(display_pdf > 0):
        ax.contourf(x, y, display_pdf, cmap='Purples', alpha=0.7, levels=15, zorder=1)
    else:
        # Fallback: show receiver area
        ax.contourf(x, y, receiver_area_pdf, cmap='Purples', alpha=0.5, levels=10, zorder=1)
    
    # Plot all players (EXACT CPP STYLE)
    for idx, row in player_coords.iterrows():
        player_x = row.get('x', row.get('receiver_x', 0))
        player_y = row.get('y', row.get('receiver_y', 0))
        is_receiver = row.get('nfl_id') == receiver_nfl_id
        
        if is_receiver:
            # Highlight receiver (like QB in CPP - limegreen)
            plt.scatter(player_y, player_x, color='limegreen', s=400, 
                       marker='*', edgecolors='black', linewidths=3, zorder=5)
            if label_numbers:
                jersey = row.get('jerseyNumber', row.get('nfl_id', ''))
                plt.annotate(int(jersey) if pd.notna(jersey) else '', 
                           (player_y, player_x), 
                           xytext=(player_y-0.5, player_x-0.5),
                           color='white', fontweight='bold', fontsize=14)
        elif row.get('player_side') == 'Defense' or row.get('team') != row.get('possessionTeam', ''):
            # Defenders (blue, like away team in CPP)
            plt.scatter(player_y, player_x, color='blue', s=300, 
                       edgecolors='white', linewidths=2, zorder=4)
            if show_arrow:
                xspeed = row.get('xspeed', 0)
                yspeed = row.get('yspeed', 0)
                plt.arrow(player_y, player_x, yspeed*0.1, xspeed*0.1,
                         color='orange', width=0.1, zorder=3)
            if label_numbers:
                jersey = row.get('jerseyNumber', row.get('nfl_id', ''))
                plt.annotate(int(jersey) if pd.notna(jersey) else '', 
                           (player_y, player_x),
                           xytext=(player_y-0.5, player_x-0.5),
                           color='white', fontsize=12)
        else:
            # Other offensive players (red, like home team in CPP)
            plt.scatter(player_y, player_x, color='red', s=300,
                       edgecolors='white', linewidths=2, zorder=4)
            if show_arrow:
                xspeed = row.get('xspeed', 0)
                yspeed = row.get('yspeed', 0)
                plt.arrow(player_y, player_x, yspeed*0.1, xspeed*0.1,
                         color='green', width=0.1, zorder=3)
            if label_numbers:
                jersey = row.get('jerseyNumber', row.get('nfl_id', ''))
                plt.annotate(int(jersey) if pd.notna(jersey) else '', 
                           (player_y, player_x),
                           xytext=(player_y-0.5, player_x-0.5),
                           color='white', fontsize=12)
    
    # Plot ball landing position
    plt.scatter(ball_land_y, ball_land_x, color='yellow', s=500,
               marker='X', edgecolors='black', linewidths=3, zorder=6)
    
    # Dominance indicator (like pressure indicator in CPP)
    dominance_percent = int(dominance_score * 100)
    
    # Create text box (like CPP pressure indicator)
    if dominance_score >= 0.8:
        indicator_color = '#00FF00'  # Green
        indicator_text = 'HIGH'
    elif dominance_score >= 0.65:
        indicator_color = '#FFFF00'  # Yellow
        indicator_text = 'MED-HIGH'
    elif dominance_score >= 0.5:
        indicator_color = '#FFA500'  # Orange
        indicator_text = 'MED'
    else:
        indicator_color = '#FF0000'  # Red
        indicator_text = 'LOW'
    
    # Add dominance text box (like CPP pressure indicator position)
    fontsize = 66 if dominance_score >= 0.8 else 50
    ax.text(2, 112, f"Dominance: {dominance_percent}%",
           fontsize=25, fontweight='bold', color='white',
           bbox=dict(boxstyle='round,pad=1', facecolor=indicator_color, 
                    alpha=0.8, edgecolor='black', linewidth=2))
    
    # Set axis limits to ensure field is visible
    ax.set_xlim(0, 53.3)
    ax.set_ylim(0, 120)
    ax.set_aspect('equal')
    plt.axis('off')
    
    # Force redraw
    plt.tight_layout()
    
    return fig


def create_field_visualization_from_dataframe(
    training_df: pd.DataFrame,
    game_id: int,
    play_id: int,
    frame_id: Optional[int] = None,
    receiver_nfl_id: Optional[int] = None
) -> plt.Figure:
    """
    Create field visualization from your training dataframe
    
    Args:
        training_df: Your training dataframe
        game_id: Game ID
        play_id: Play ID
        frame_id: Specific frame (if None, uses first frame)
        receiver_nfl_id: Receiver NFL ID (if None, tries to find from dataframe)
    
    Returns:
        matplotlib Figure
    """
    # Get play data
    play_data = training_df[
        (training_df['game_id'] == game_id) &
        (training_df['play_id'] == play_id)
    ]
    
    if len(play_data) == 0:
        raise ValueError(f"Play {game_id}-{play_id} not found")
    
    # Get specific frame or first frame
    if frame_id is not None:
        frame_data = play_data[play_data.get('frame_id', play_data.get('continuous_frame', 0)) == frame_id]
    else:
        frame_data = play_data.iloc[0:1]
    
    if len(frame_data) == 0:
        frame_data = play_data.iloc[0:1]
    
    frame_row = frame_data.iloc[0]
    
    # Get receiver NFL ID
    if receiver_nfl_id is None:
        receiver_nfl_id = frame_row.get('nfl_id', frame_row.get('receiver_nfl_id', None))
    
    if receiver_nfl_id is None:
        raise ValueError("Could not determine receiver NFL ID")
    
    # Get ball landing
    ball_land_x = frame_row.get('ball_land_x_std', frame_row.get('ball_land_x', 0))
    ball_land_y = frame_row.get('ball_land_y_std', frame_row.get('ball_land_y', 0))
    
    # Get dominance score
    dominance_score = frame_row.get('receiver_dominance', None)
    
    # Create player coordinates dataframe
    # Note: Your dataframe might need adjustment based on structure
    # This assumes you have player positions in the frame
    player_coords = frame_data.copy()
    
    # Visualize
    fig = visualize_receiver_dominance_field(
        player_coords,
        int(receiver_nfl_id),
        float(ball_land_x),
        float(ball_land_y),
        dominance_score=float(dominance_score) if dominance_score is not None else None,
        label_numbers=True,
        show_arrow=False
    )
    
    return fig

