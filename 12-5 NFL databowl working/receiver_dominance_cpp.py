"""
Receiver Dominance - Continuous Receiver Pressure (CRP)
Adapted from the 2023 NFL Data Bowl winner's Continuous Pocket Pressure (CPP) approach

This module calculates and visualizes receiver dominance using a similar methodology:
- Uses multivariate normal distributions to model defender and receiver influence
- Calculates a dominance score based on defensive pressure vs receiver advantage
- Provides visualizations matching the CPP style
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import multivariate_normal
import math
from typing import Dict, List, Tuple, Optional


def calculate_receiver_dominance(
    frame_data: pd.DataFrame,
    receiver_nfl_id: int,
    ball_land_x: float,
    ball_land_y: float,
    starting_separations: Optional[Dict[int, float]] = None
) -> float:
    """
    Calculate Continuous Receiver Pressure (CRP) for a single frame
    
    Similar to CPP but adapted for receiver analysis:
    - Models defender influence using multivariate normal distributions
    - Models receiver advantage based on separation, leverage angle, and speed
    - Calculates dominance ratio similar to pressure ratio
    
    Args:
        frame_data: DataFrame with player positions and metrics for one frame
        receiver_nfl_id: NFL ID of the target receiver
        ball_land_x: X coordinate of ball landing position
        ball_land_y: Y coordinate of ball landing position
        starting_separations: Dict mapping defender nfl_id to initial separation (for normalization)
    
    Returns:
        Receiver dominance score (0-1, where 1 = high dominance/receiver advantage)
    """
    # Get receiver position and metrics
    receiver_row = frame_data[frame_data['nfl_id'] == receiver_nfl_id].iloc[0]
    receiver_x = receiver_row['receiver_x'] if 'receiver_x' in receiver_row else receiver_row['x']
    receiver_y = receiver_row['receiver_y'] if 'receiver_y' in receiver_row else receiver_row['y']
    
    # Get receiver metrics
    receiver_speed = receiver_row.get('receiver_speed', 0)
    receiver_accel = receiver_row.get('receiver_accel', 0)
    sep_nearest = receiver_row.get('sep_nearest', np.inf)
    leverage_angle = receiver_row.get('leverage_angle', np.nan)
    
    # Calculate receiver direction components
    if 'dir' in receiver_row and not pd.isna(receiver_row['dir']):
        receiver_dir_rad = math.radians(receiver_row['dir'])
        receiver_xspeed = receiver_speed * math.cos(receiver_dir_rad)
        receiver_yspeed = receiver_speed * math.sin(receiver_dir_rad)
    else:
        # Estimate direction from position change or use default
        receiver_xspeed = 0
        receiver_yspeed = 0
    
    # Get all defenders
    defenders = frame_data[
        (frame_data['player_side'] == 'Defense') | 
        (frame_data.get('team', '') != frame_data.get('possessionTeam', ''))
    ].copy()
    
    # Define field domain (same as CPP)
    x, y = np.mgrid[0:53.3:1, 0:120:1]
    locations = np.dstack((x, y))
    
    # Initialize PDFs
    defense_pdf_is_none = True
    receiver_pdf_is_none = True
    defense_pdf = None
    receiver_pdf = None
    
    # Calculate distance from receiver for all players
    frame_data['distanceFromReceiver'] = np.sqrt(
        (frame_data['x'] - receiver_x)**2 + (frame_data['y'] - receiver_y)**2
    )
    
    # Generate defensive player influence PDFs
    for idx, def_row in defenders.iterrows():
        def_x = def_row.get('x', def_row.get('nearest_defender_x', 0))
        def_y = def_row.get('y', def_row.get('nearest_defender_y', 0))
        
        # Get defender speed and direction
        def_speed = def_row.get('s', 0)
        def_dir = def_row.get('dir', 0)
        
        if pd.isna(def_dir) or def_dir == 0:
            def_xspeed = 0
            def_yspeed = 0
        else:
            def_dir_rad = math.radians(def_dir)
            def_xspeed = def_speed * math.cos(def_dir_rad)
            def_yspeed = def_speed * math.sin(def_dir_rad)
        
        # Distance from receiver
        distance_from_receiver = def_row.get('distanceFromReceiver', 
                                            np.sqrt((def_x - receiver_x)**2 + (def_y - receiver_y)**2))
        
        # Similar to CPP: speed ratio affects covariance matrix
        speed_ratio = (def_speed**2) / 100 if def_speed > 0 else 0.01
        top_left_s = (distance_from_receiver + distance_from_receiver * speed_ratio) / 2
        bottom_right_s = (distance_from_receiver - distance_from_receiver * speed_ratio) / 2
        
        # Ensure matrix is invertible
        top_left_s = max(top_left_s, 0.0001)
        bottom_right_s = max(bottom_right_s, 0.0001)
        
        # Rotation matrix based on direction
        if def_speed > 0:
            x_component = def_xspeed / def_speed
            y_component = def_yspeed / def_speed
        else:
            x_component = 1
            y_component = 0
        
        r_matrix = np.array([[x_component, -y_component], [y_component, x_component]])
        s_matrix = np.array([[top_left_s, 0], [0, bottom_right_s]])
        
        # Calculate covariance matrix (similar to CPP)
        try:
            inv_r = np.linalg.inv(r_matrix)
            covariance_matrix = r_matrix @ s_matrix @ s_matrix @ inv_r
        except:
            # Fallback to simple covariance
            covariance_matrix = np.array([[top_left_s, 0], [0, bottom_right_s]])
        
        # Mean position (projected forward based on speed)
        mu_x = def_y + def_yspeed * 0.5  # Note: y and x are swapped in field coordinates
        mu_y = def_x + def_xspeed * 0.5
        mu = [mu_x, mu_y]
        
        # Generate player PDF
        try:
            player_pdf = multivariate_normal(mu, covariance_matrix).pdf(locations)
        except:
            # Fallback to simple normal distribution
            player_pdf = multivariate_normal([mu_x, mu_y], [[4, 0], [0, 4]]).pdf(locations)
        
        # Weight by separation (closer defenders have more influence)
        if sep_nearest < np.inf and distance_from_receiver > 0:
            separation_weight = 1.0 / (1.0 + distance_from_receiver / 5.0)  # Decay with distance
        else:
            separation_weight = 1.0
        
        if defense_pdf_is_none:
            defense_pdf = player_pdf * separation_weight
            defense_pdf_is_none = False
        else:
            defense_pdf = defense_pdf + player_pdf * separation_weight
    
    # Generate receiver advantage PDF
    # Receiver advantage is based on:
    # 1. Separation from defenders
    # 2. Leverage angle (larger angle = better position)
    # 3. Speed toward ball
    # 4. Number of defenders nearby
    
    # Calculate receiver advantage factors
    if sep_nearest < np.inf:
        separation_advantage = min(sep_nearest / 10.0, 1.0)  # Normalize to 0-1
    else:
        separation_advantage = 0.5
    
    if not pd.isna(leverage_angle):
        leverage_advantage = leverage_angle / 180.0  # Normalize to 0-1
    else:
        leverage_advantage = 0.5
    
    speed_advantage = min(receiver_speed / 8.0, 1.0) if receiver_speed > 0 else 0.0
    
    # Number of defenders nearby (inverse - fewer is better)
    num_def_nearby = receiver_row.get('num_def_within_3', 0)
    pressure_advantage = 1.0 - min(num_def_nearby / 5.0, 1.0)
    
    # Combined receiver advantage factor
    receiver_advantage_factor = (
        0.3 * separation_advantage +
        0.25 * leverage_advantage +
        0.25 * speed_advantage +
        0.2 * pressure_advantage
    )
    
    # Create receiver influence PDF (similar to offensive line protection in CPP)
    speed_ratio = (receiver_speed**2) / 100 if receiver_speed > 0 else 0.01
    distance_to_ball = receiver_row.get('dist_to_ball', 
                                       np.sqrt((receiver_x - ball_land_x)**2 + (receiver_y - ball_land_y)**2))
    
    top_left_s = (distance_to_ball + distance_to_ball * speed_ratio) / 2
    bottom_right_s = (distance_to_ball - distance_to_ball * speed_ratio) / 2
    top_left_s = max(top_left_s, 0.0001)
    bottom_right_s = max(bottom_right_s, 0.0001)
    
    if receiver_speed > 0:
        x_comp = receiver_xspeed / receiver_speed
        y_comp = receiver_yspeed / receiver_speed
    else:
        x_comp = 1
        y_comp = 0
    
    r_matrix = np.array([[x_comp, -y_comp], [y_comp, x_comp]])
    s_matrix = np.array([[top_left_s, 0], [0, bottom_right_s]])
    
    try:
        inv_r = np.linalg.inv(r_matrix)
        covariance_matrix = r_matrix @ s_matrix @ s_matrix @ inv_r
    except:
        covariance_matrix = np.array([[top_left_s, 0], [0, bottom_right_s]])
    
    mu_x = receiver_y + receiver_yspeed * 0.5
    mu_y = receiver_x + receiver_xspeed * 0.5
    mu = [mu_x, mu_y]
    
    try:
        receiver_player_pdf = multivariate_normal(mu, covariance_matrix).pdf(locations)
    except:
        receiver_player_pdf = multivariate_normal([mu_x, mu_y], [[4, 0], [0, 4]]).pdf(locations)
    
    # Weight receiver PDF by advantage factors
    if receiver_pdf_is_none:
        receiver_pdf = receiver_advantage_factor * receiver_player_pdf
        receiver_pdf_is_none = False
    else:
        receiver_pdf = receiver_pdf + receiver_advantage_factor * receiver_player_pdf
    
    # Calculate dominance ratio (similar to pressure ratio in CPP)
    # Higher defense_pdf = more defensive pressure
    # Higher receiver_pdf = more receiver advantage
    if defense_pdf is None:
        defense_pdf = np.zeros_like(receiver_pdf)
    if receiver_pdf is None:
        receiver_pdf = np.zeros_like(defense_pdf)
    
    # Dominance = receiver advantage / (defense pressure + receiver advantage)
    # This gives us a 0-1 scale where 1 = complete receiver dominance
    total_influence = defense_pdf + receiver_pdf
    dominance_pdf = np.divide(receiver_pdf, total_influence, 
                             out=np.zeros_like(receiver_pdf), 
                             where=total_influence!=0)
    
    # Create receiver area PDF (similar to QB pocket PDF in CPP)
    receiver_area_pdf = multivariate_normal([receiver_y, receiver_x], [[6, 0], [0, 6]]).pdf(locations)
    
    # Calculate final dominance score
    # Multiply dominance PDF by receiver area to focus on receiver's immediate area
    receiver_dominance_pdf = receiver_area_pdf * dominance_pdf
    
    # Calculate dominance value (similar to pressure_val in CPP)
    dominance_val = np.sum(receiver_dominance_pdf) / np.sum(receiver_area_pdf)
    
    # Normalize similar to CPP: (val - 0.5) / (0.8 - 0.5)
    # But invert so higher = more receiver dominance (opposite of pressure)
    dominance_val = (dominance_val - 0.50) / (0.80 - 0.50)
    
    # Clamp to [0, 1]
    if dominance_val >= 1:
        dominance_val = 1
    elif dominance_val <= 0:
        dominance_val = 0
    
    return dominance_val


def visualize_receiver_dominance(
    frame_data: pd.DataFrame,
    receiver_nfl_id: int,
    ball_land_x: float,
    ball_land_y: float,
    dominance_score: float,
    img_size: Tuple[float, float] = (10.66, 24),
    field_color: str = 'darkgreen',
    endzone_color: str = 'purple',
    show_players: bool = True,
    label_numbers: bool = True
) -> plt.Figure:
    """
    Visualize receiver dominance on a football field
    
    Similar style to CPP visualizations:
    - Field background with yard lines
    - Contour plot showing dominance regions
    - Pressure indicator (adapted for dominance)
    - Player positions
    
    Args:
        frame_data: DataFrame with player positions
        receiver_nfl_id: NFL ID of target receiver
        ball_land_x: X coordinate of ball landing
        ball_land_y: Y coordinate of ball landing
        dominance_score: Pre-calculated dominance score (0-1)
        img_size: Figure size tuple
        field_color: Field background color
        endzone_color: Endzone color
        show_players: Whether to show player positions
        label_numbers: Whether to label jersey numbers
    
    Returns:
        matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(1, figsize=img_size)
    
    # Draw field (same as CPP)
    rect = patches.Rectangle((0, 0), 53.3, 120, linewidth=0.1,
                             edgecolor='r', facecolor=field_color, zorder=0)
    ax.add_patch(rect)
    
    # Field lines
    plt.plot([0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             [10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             color='white', linewidth=1)
    
    # End zones
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
    
    # Yard markers
    for y in range(20, 110, 10):
        numb = y
        if y > 50:
            numb = 120 - y
        plt.text(5, y-1.5, str(numb - 10),
                 horizontalalignment='center',
                 fontsize=20,
                 color='white', rotation=270)
        plt.text(53.3 - 5, y - 0.95, str(numb - 10),
                 horizontalalignment='center',
                 fontsize=20,
                 color='white', rotation=90)
    
    # Hash lines
    for y in range(11, 110):
        ax.plot([0.7, 0.4], [y, y], color='white', linewidth=0.5)
        ax.plot([53.0, 52.5], [y, y], color='white', linewidth=0.5)
        ax.plot([22.91, 23.57], [y, y], color='white', linewidth=0.5)
        ax.plot([29.73, 30.39], [y, y], color='white', linewidth=0.5)
    
    # Get receiver position
    receiver_row = frame_data[frame_data['nfl_id'] == receiver_nfl_id].iloc[0]
    receiver_x = receiver_row.get('receiver_x', receiver_row.get('x', 0))
    receiver_y = receiver_row.get('receiver_y', receiver_row.get('y', 0))
    
    # Calculate dominance visualization (similar to pressure visualization)
    x, y = np.mgrid[0:53.3:1, 0:120:1]
    
    # Create a simple dominance heatmap centered on receiver
    receiver_area_pdf = multivariate_normal([receiver_y, receiver_x], [[8, 0], [0, 8]]).pdf(np.dstack((x, y)))
    
    # Scale by dominance score
    display_pdf = receiver_area_pdf * dominance_score
    
    # Contour plot (using 'Purples' colormap like CPP, but inverted for dominance)
    # Higher dominance = lighter colors
    ax.contourf(x, y, display_pdf, cmap='Purples', alpha=0.6, levels=10)
    
    # Plot players if requested
    if show_players:
        for idx, row in frame_data.iterrows():
            player_x = row.get('x', 0)
            player_y = row.get('y', 0)
            is_receiver = row.get('nfl_id') == receiver_nfl_id
            
            if is_receiver:
                # Highlight receiver
                ax.scatter(player_y, player_x, color='limegreen', s=400, 
                          marker='*', edgecolors='black', linewidths=2, zorder=5)
                if label_numbers:
                    jersey = row.get('jerseyNumber', row.get('nfl_id', ''))
                    ax.annotate(str(jersey), (player_y, player_x),
                               xytext=(player_y-0.5, player_x-0.5),
                               color='white', fontweight='bold', fontsize=12)
            elif row.get('player_side') == 'Defense' or row.get('team') != row.get('possessionTeam', ''):
                # Defenders
                ax.scatter(player_y, player_x, color='blue', s=300, 
                          edgecolors='white', linewidths=1, zorder=4)
                if label_numbers:
                    jersey = row.get('jerseyNumber', row.get('nfl_id', ''))
                    ax.annotate(str(jersey), (player_y, player_x),
                               xytext=(player_y-0.5, player_x-0.5),
                               color='white', fontsize=10)
            else:
                # Other offensive players
                ax.scatter(player_y, player_x, color='red', s=300,
                          edgecolors='white', linewidths=1, zorder=4)
                if label_numbers:
                    jersey = row.get('jerseyNumber', row.get('nfl_id', ''))
                    ax.annotate(str(jersey), (player_y, player_x),
                               xytext=(player_y-0.5, player_x-0.5),
                               color='white', fontsize=10)
    
    # Plot ball landing position
    ax.scatter(ball_land_y, ball_land_x, color='yellow', s=500,
              marker='X', edgecolors='black', linewidths=2, zorder=6)
    
    # Dominance indicator (similar to pressure indicator in CPP)
    # Use color scheme: Green (high dominance) -> Yellow -> Orange -> Red (low dominance)
    dominance_percent = int(dominance_score * 100)
    
    # Create text box for dominance score
    if dominance_score >= 0.8:
        indicator_color = 'green'
    elif dominance_score >= 0.65:
        indicator_color = 'yellow'
    elif dominance_score >= 0.5:
        indicator_color = 'orange'
    else:
        indicator_color = 'red'
    
    # Add dominance text (similar to pressure text in CPP)
    ax.text(2, 112, f"Dominance: {dominance_percent}%",
           fontsize=25, fontweight='bold', color='white',
           bbox=dict(boxstyle='round', facecolor=indicator_color, alpha=0.7))
    
    plt.axis('off')
    return fig


def calculate_play_dominance_sequence(
    play_data: pd.DataFrame,
    receiver_nfl_id: int,
    ball_land_x: float,
    ball_land_y: float,
    game_id: Optional[int] = None,
    play_id: Optional[int] = None
) -> List[float]:
    """
    Calculate dominance sequence for an entire play (similar to extractPocketPressureArray)
    
    Args:
        play_data: DataFrame with all frames for a play
        receiver_nfl_id: NFL ID of target receiver
        ball_land_x: X coordinate of ball landing
        ball_land_y: Y coordinate of ball landing
        game_id: Optional game ID for filtering
        play_id: Optional play ID for filtering
    
    Returns:
        List of dominance scores for each frame in the play
    """
    if game_id is not None and play_id is not None:
        play_data = play_data[
            (play_data['game_id'] == game_id) & 
            (play_data['play_id'] == play_id)
        ]
    
    # Sort by frame
    if 'continuous_frame' in play_data.columns:
        play_data = play_data.sort_values('continuous_frame')
    elif 'frame_id' in play_data.columns:
        play_data = play_data.sort_values('frame_id')
    
    dominance_sequence = []
    
    # Get starting separations for normalization (optional)
    first_frame = play_data.iloc[0] if len(play_data) > 0 else None
    starting_separations = None
    
    # Calculate dominance for each frame
    for frame_idx in play_data.index.unique():
        frame_data = play_data.loc[[frame_idx]]
        
        if len(frame_data) == 0:
            continue
        
        try:
            dominance = calculate_receiver_dominance(
                frame_data,
                receiver_nfl_id,
                ball_land_x,
                ball_land_y,
                starting_separations
            )
            dominance_sequence.append(dominance)
        except Exception as e:
            # If calculation fails, use NaN or previous value
            if len(dominance_sequence) > 0:
                dominance_sequence.append(dominance_sequence[-1])
            else:
                dominance_sequence.append(0.5)  # Neutral value
    
    return dominance_sequence

