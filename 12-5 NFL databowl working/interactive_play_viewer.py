"""
Interactive Play Viewer - Navigate between plays with arrow keys

Navigate play-by-play through your dataset using arrow keys:
- Left Arrow: Previous play
- Right Arrow: Next play
- 'q' or Escape: Quit
- 'g': Generate GIF for current play
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from scipy.stats import multivariate_normal
from route_dominance_scoring import RouteDominanceScorer
from create_dominance_gif import create_gif_for_play
import sys
import os


class InteractivePlayViewer:
    """Interactive play-by-play viewer for receiver dominance field visualization"""
    
    def __init__(self, training_df, scorer, start_game_id=None, start_play_id=None):
        """
        Initialize interactive play viewer
        
        Args:
            training_df: Your training dataframe
            scorer: RouteDominanceScorer object (for all_frames_df)
            start_game_id: Starting game ID (if None, uses first play)
            start_play_id: Starting play ID (if None, uses first play)
        """
        self.training_df = training_df
        self.scorer = scorer
        
        # Get unique plays
        plays = training_df[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
        plays = plays.sort_values(['game_id', 'play_id'])
        self.plays_list = plays.to_dict('records')
        self.total_plays = len(self.plays_list)
        
        # Find starting play index
        if start_game_id is not None and start_play_id is not None:
            start_idx = next((i for i, p in enumerate(self.plays_list) 
                             if p['game_id'] == start_game_id and p['play_id'] == start_play_id), 0)
        else:
            start_idx = 0
        
        self.current_play_idx = start_idx
        self.current_frame_idx = 0  # Track current frame within play
        
        # Create figure with field on left, info panel on right
        self.fig = plt.figure(figsize=(20, 24))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        try:
            self.fig.canvas.set_window_title('Interactive Play Viewer - Left/Right: Plays | Up/Down: Frames')
        except:
            pass  # Some matplotlib backends don't support set_window_title
        
        # Initial render
        self.update_visualization()
        plt.tight_layout(pad=2.0)
        plt.show()
    
    def get_current_play(self):
        """Get current play info"""
        return self.plays_list[self.current_play_idx]
    
    def update_visualization(self):
        """Update the field visualization for current play"""
        play = self.get_current_play()
        game_id = play['game_id']
        play_id = play['play_id']
        receiver_nfl_id = play['nfl_id']
        
        # Clear figure and create subplots: field on left, info panel on right
        self.fig.clear()
        # Field takes up left 60%, info panel takes right 40%
        ax_field = plt.subplot2grid((1, 10), (0, 0), colspan=6)
        ax_info = plt.subplot2grid((1, 10), (0, 6), colspan=4)
        ax_info.axis('off')  # Turn off axes for info panel
        ax = ax_field
        
        # Get play data
        play_data = self.training_df[
            (self.training_df['game_id'] == game_id) &
            (self.training_df['play_id'] == play_id)
        ]
        
        if len(play_data) == 0:
            ax.text(0.5, 0.5, f"No data for Game {game_id}, Play {play_id}",
                   ha='center', va='center', fontsize=20)
            ax_info.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=16)
            self.fig.canvas.draw()
            return
        
        # Sort play data by frame
        if 'continuous_frame' in play_data.columns:
            play_data = play_data.sort_values('continuous_frame')
            frame_col = 'continuous_frame'
        elif 'frame_id' in play_data.columns:
            play_data = play_data.sort_values('frame_id')
            frame_col = 'frame_id'
        else:
            play_data = play_data.sort_index()
            frame_col = None
        
        # Get current frame (allow frame-by-frame navigation)
        total_frames = len(play_data)
        if self.current_frame_idx >= total_frames:
            self.current_frame_idx = total_frames - 1
        if self.current_frame_idx < 0:
            self.current_frame_idx = 0
        
        current_frame_row = play_data.iloc[self.current_frame_idx]
        frame_id = current_frame_row.get('frame_id', self.current_frame_idx + 1)
        frame_type = current_frame_row.get('frame_type', 'input')
        
        # Get all players for this frame
        all_players_frame = self.scorer.all_frames_df[
            (self.scorer.all_frames_df['game_id'] == game_id) &
            (self.scorer.all_frames_df['play_id'] == play_id) &
            (self.scorer.all_frames_df['frame_id'] == frame_id) &
            (self.scorer.all_frames_df['frame_type'] == frame_type)
        ]
        
        # Get receiver position (standardized) from current frame
        receiver_x = current_frame_row.get('receiver_x', current_frame_row.get('x_std', 0))
        receiver_y = current_frame_row.get('receiver_y', current_frame_row.get('y_std', 0))
        
        # Get ball landing (same for all frames in a play)
        ball_land_x = current_frame_row.get('ball_land_x_std', current_frame_row.get('ball_land_x', 0))
        ball_land_y = current_frame_row.get('ball_land_y_std', current_frame_row.get('ball_land_y', 0))
        
        # Get dominance score from current frame (this changes frame-by-frame!)
        dominance_score = current_frame_row.get('receiver_dominance', 0.5)
        
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
        
        # Get nearest defender for contour (from current frame)
        def_x = current_frame_row.get('nearest_defender_x', receiver_x + 3)
        def_y = current_frame_row.get('nearest_defender_y', receiver_y)
        sep_nearest = current_frame_row.get('sep_nearest', 3.0)
        
        # ===== DOMINANCE CALCULATION (Detailed) =====
        # Create grid across entire field (0.5 yard resolution)
        # Step 1: Receiver Influence PDF (Gaussian distribution)
        # - Center: Receiver's position [receiver_y, receiver_x]
        # - Covariance: [[8, 0], [0, 8]] = 8 yard radius influence zone
        # - Represents receiver's "space" or area of control
        receiver_pdf = multivariate_normal([receiver_y, receiver_x], [[8, 0], [0, 8]]).pdf(np.dstack((x, y)))
        
        # Step 2: Defender Pressure PDF (Gaussian distribution)
        # - Center: Nearest defender's position [def_y, def_x]
        # - Covariance: [[6, 0], [0, 6]] = 6 yard radius pressure zone
        # - Weight: Inversely proportional to separation distance
        #   Formula: weight = 1.0 / (1.0 + separation / 5.0)
        #   Closer defender = higher weight (more pressure)
        if sep_nearest < np.inf:
            def_weight = 1.0 / (1.0 + sep_nearest / 5.0)
        else:
            def_weight = 0.5
        
        defender_pdf = multivariate_normal([def_y, def_x], [[6, 0], [0, 6]]).pdf(np.dstack((x, y))) * def_weight
        
        # Step 3: Calculate Dominance Ratio
        # For each point on field: dominance = receiver_influence / (receiver_influence + defender_influence)
        # Result: 0.0 = defender dominance, 1.0 = receiver dominance, 0.5 = balanced
        total_pdf = receiver_pdf + defender_pdf + 1e-10  # Add small epsilon to prevent division by zero
        dominance_pdf = receiver_pdf / total_pdf
        
        # Step 4: Draw Purple Contour Plot
        # - Dark Purple = High receiver dominance (receiver has advantage in that area)
        # - Light Purple = Low receiver dominance (defender has advantage)
        # - No Purple = Areas with minimal influence from either side
        # The contour shows the "spatial dominance map" - which areas are controlled by receiver vs defenders
        contour = ax.contourf(x, y, dominance_pdf, cmap='Purples', alpha=0.7, levels=15, zorder=1)
        
        # Add colorbar to show dominance scale - make it smaller and less intrusive
        cbar = plt.colorbar(contour, ax=ax, fraction=0.015, pad=0.005)
        cbar.set_label('Dom\n(1=Rec\n0=Def)', rotation=270, labelpad=8, color='white', fontsize=6)
        cbar.ax.tick_params(colors='white', labelsize=5)
        
        # Plot ALL players if available
        if len(all_players_frame) > 0:
            for idx, player in all_players_frame.iterrows():
                player_x = player.get('x_std', player.get('x', 0))
                player_y = player.get('y_std', player.get('y', 0))
                player_nfl_id = player.get('nfl_id', 0)
                player_side = player.get('player_side', 'Unknown')
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
            # Fallback: Just plot receiver
            ax.scatter(receiver_y, receiver_x, color='limegreen', s=500,
                      marker='*', edgecolors='black', linewidths=3, zorder=10)
        
        # Plot ball landing (yellow X)
        if ball_land_x > 0 or ball_land_y > 0:  # Only plot if valid
            ax.scatter(ball_land_y, ball_land_x, color='yellow', s=600,
                      marker='X', edgecolors='black', linewidths=3, zorder=8)
        
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
        
        # Single compact info box - Top left corner (no overlapping)
        info_text = f"Play {self.current_play_idx + 1}/{self.total_plays} | G{game_id} P{play_id} | Frame {self.current_frame_idx + 1}/{total_frames}"
        ax.text(1, 118, info_text,
               fontsize=8, fontweight='bold', color='white',
               bbox=dict(boxstyle='square,pad=3', facecolor='black', alpha=0.8))
        
        # Dominance score - Top right corner (compact)
        ax.text(52.3, 118, f"Dom: {dominance_percent}%",
               fontsize=11, fontweight='bold', color='white', ha='right',
               bbox=dict(boxstyle='square,pad=3', facecolor=indicator_color,
                        alpha=0.85, edgecolor='black', linewidth=1))
        
        # Navigation hint - Bottom right corner (small, unobtrusive)
        nav_hint = "←→ Plays | ↑↓ Frames | 'g' GIF | 'q' Quit"
        ax.text(52.3, 2, nav_hint,
               fontsize=7, color='white', ha='right',
               bbox=dict(boxstyle='square,pad=2', facecolor='black', alpha=0.7))
        
        # Set axis
        ax.set_xlim(0, 53.3)
        ax.set_ylim(0, 120)
        ax.set_aspect('equal')
        plt.axis('off')
        
        # ===== INFO PANEL =====
        self.draw_info_panel(ax_info, current_frame_row, play_data)
        
        self.fig.canvas.draw()
    
    def draw_info_panel(self, ax, frame_row, play_data):
        """Draw simple organized info panel - simple box format"""
        
        # Organize columns into logical groups
        sections = {
            'PLAY INFO': ['game_id', 'play_id', 'nfl_id', 'target_name'],
            'RECEIVER METRICS': ['receiver_speed', 'receiver_accel', 'dist_to_ball', 
                                'receiver_x', 'receiver_y'],
            'SEPARATION': ['sep_nearest', 'sep_second', 
                          'num_def_within_2', 'num_def_within_3', 'num_def_within_5'],
            'LEVERAGE': ['leverage_angle', 'initial_leverage'],
            'DEFENDER POSITION': ['nearest_defender_x', 'nearest_defender_y',
                                 'abs_dx_to_nearest_defender', 'abs_dy_to_nearest_defender'],
            'ROUTE INFO': ['route', 'pass_result', 'is_complete'],
            'GAME CONTEXT': ['offense_formation', 'receiver_alignment', 'coverage_type',
                           'down', 'yards_to_go', 'pass_length'],
            'FRAME INFO': ['continuous_frame', 'throw_status'],
            'ROUTE BREAK': ['play_break_frame', 'has_break', 'is_break_frame',
                           'frames_until_break', 'frames_since_break']
        }
        
        fontsize = 7
        
        # Build text content - simple format
        text_lines = []
        
        # Add purple contour instructions at top
        text_lines.append("PURPLE CONTOUR:")
        text_lines.append("Dark Purple = Receiver Dominance (advantage)")
        text_lines.append("Light Purple = Defender Pressure")
        text_lines.append("Size = Influence Zone (8yd receiver, 6yd defender)")
        text_lines.append("Changes frame-by-frame as players move")
        text_lines.append("")
        text_lines.append("=" * 40)
        text_lines.append("")
        
        # Display each section
        for section_name, columns in sections.items():
            text_lines.append(section_name)
            text_lines.append("-" * 35)
            
            for col in columns:
                if col in frame_row:
                    value = frame_row[col]
                    
                    # Format value based on type
                    if pd.isna(value):
                        display_value = "N/A"
                    elif isinstance(value, (int, np.integer)):
                        display_value = str(int(value))
                    elif isinstance(value, (float, np.floating)):
                        if abs(value) < 0.01:
                            display_value = "0.00"
                        elif abs(value) < 1:
                            display_value = f"{value:.3f}"
                        elif abs(value) < 100:
                            display_value = f"{value:.2f}"
                        else:
                            display_value = f"{value:.1f}"
                    elif isinstance(value, bool):
                        display_value = "Yes" if value else "No"
                    else:
                        display_value = str(value)
                    
                    # Truncate long strings
                    if len(display_value) > 18:
                        display_value = display_value[:15] + "..."
                    
                    # Column name (formatted)
                    col_name = col.replace('_', ' ').title()
                    text_lines.append(f"  {col_name:.<28} {display_value}")
                else:
                    col_name = col.replace('_', ' ').title()
                    text_lines.append(f"  {col_name:.<28} N/A")
            
            text_lines.append("")  # Empty line between sections
        
        # Draw as simple text box - no fancy styling
        full_text = "\n".join(text_lines)
        ax.text(0.05, 0.98, full_text, fontsize=fontsize, 
               family='monospace', color='white',
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='square,pad=8', facecolor='black', alpha=0.9, 
                        edgecolor='#888888', linewidth=1))
        
        # Set background to dark grey
        ax.set_facecolor('#2a2a2a')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        if event.key == 'right' or event.key == 'd':
            # Next play
            if self.current_play_idx < self.total_plays - 1:
                self.current_play_idx += 1
                self.current_frame_idx = 0  # Reset to first frame of new play
                self.update_visualization()
                print(f"→ Play {self.current_play_idx + 1}/{self.total_plays}")
        
        elif event.key == 'left' or event.key == 'a':
            # Previous play
            if self.current_play_idx > 0:
                self.current_play_idx -= 1
                self.current_frame_idx = 0  # Reset to first frame of new play
                self.update_visualization()
                print(f"← Play {self.current_play_idx + 1}/{self.total_plays}")
        
        elif event.key == 'up':
            # Previous frame (within current play)
            if self.current_frame_idx > 0:
                self.current_frame_idx -= 1
                self.update_visualization()
                play = self.get_current_play()
                play_data = self.training_df[
                    (self.training_df['game_id'] == play['game_id']) &
                    (self.training_df['play_id'] == play['play_id'])
                ]
                total_frames = len(play_data)
                print(f"↑ Frame {self.current_frame_idx + 1}/{total_frames}")
        
        elif event.key == 'down':
            # Next frame (within current play)
            play = self.get_current_play()
            play_data = self.training_df[
                (self.training_df['game_id'] == play['game_id']) &
                (self.training_df['play_id'] == play['play_id'])
            ]
            total_frames = len(play_data)
            if self.current_frame_idx < total_frames - 1:
                self.current_frame_idx += 1
                self.update_visualization()
                print(f"↓ Frame {self.current_frame_idx + 1}/{total_frames}")
        
        elif event.key == 'g':
            # Generate GIF for current play
            play = self.get_current_play()
            print(f"\nGenerating GIF for Game {play['game_id']}, Play {play['play_id']}...")
            try:
                gif_path = create_gif_for_play(
                    self.training_df,
                    game_id=play['game_id'],
                    play_id=play['play_id'],
                    fps=5,
                    scorer=self.scorer
                )
                print(f"✓ GIF created: {gif_path}")
                # Open GIF
                if sys.platform == "win32":
                    os.startfile(os.path.abspath(gif_path))
            except Exception as e:
                print(f"✗ Error creating GIF: {e}")
        
        elif event.key == 'q' or event.key == 'escape':
            # Quit
            print("\nClosing viewer...")
            plt.close(self.fig)
            sys.exit(0)


def launch_interactive_viewer(training_df, scorer, start_game_id=None, start_play_id=None):
    """
    Launch the interactive play viewer
    
    Args:
        training_df: Your training dataframe
        scorer: RouteDominanceScorer object
        start_game_id: Starting game ID (optional)
        start_play_id: Starting play ID (optional)
    
    Example:
        from route_dominance_scoring import RouteDominanceScorer
        from interactive_play_viewer import launch_interactive_viewer
        
        scorer = RouteDominanceScorer(input_df, output_df, supp_df)
        viewer = launch_interactive_viewer(training_df, scorer)
    """
    viewer = InteractivePlayViewer(training_df, scorer, start_game_id, start_play_id)
    return viewer


if __name__ == "__main__":
    print("""
    Interactive Play Viewer
    
    To use:
    
    from route_dominance_scoring import RouteDominanceScorer
    from interactive_play_viewer import launch_interactive_viewer
    
    # Load your data
    training_df = pd.read_csv('route_dominance_training_data.csv')
    scorer = RouteDominanceScorer(input_df, output_df, supp_df)
    
    # Launch viewer
    viewer = launch_interactive_viewer(training_df, scorer)
    
    Controls:
    - Left Arrow / 'a': Previous play
    - Right Arrow / 'd': Next play
    - 'g': Generate GIF for current play
    - 'q' or Escape: Quit
    """)

