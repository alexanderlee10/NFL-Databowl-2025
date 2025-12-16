"""
Interactive Route Dominance Visualizer

Navigate frame-by-frame through a play using arrow keys:
- Left Arrow: Previous frame
- Right Arrow: Next frame
- Up Arrow: Jump to first frame
- Down Arrow: Jump to last frame
- 'q' or Escape: Quit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Arc, Arrow
from .route_dominance_scoring import RouteDominanceScorer
import sys

class InteractiveRouteDominanceViewer:
    """Interactive frame-by-frame viewer for route dominance"""
    
    def __init__(self, scorer, game_id, play_id, target_nfl_id):
        self.scorer = scorer
        self.game_id = game_id
        self.play_id = play_id
        self.target_nfl_id = target_nfl_id
        
        # Calculate frame metrics
        print("Calculating frame-by-frame dominance...")
        self.frame_metrics = scorer.calculate_frame_dominance(game_id, play_id, target_nfl_id)
        
        # Get all frames for the play (all players for visualization)
        self.play_frames = scorer.all_frames_df[
            (scorer.all_frames_df["game_id"] == game_id) &
            (scorer.all_frames_df["play_id"] == play_id)
        ]
        
        # Get frames for targeted receiver only (for sequencing)
        target_frames = self.play_frames[
            self.play_frames["nfl_id"] == target_nfl_id
        ]
        
        # Separate input and output frames for targeted receiver
        input_frames = target_frames[target_frames["frame_type"] == "input"].sort_values("frame_id")
        output_frames = target_frames[target_frames["frame_type"] == "output"].sort_values("frame_id")
        
        # Create continuous frame list
        self.frames_list = []
        for _, row in input_frames.iterrows():
            self.frames_list.append((row["frame_id"], "input", row.name))
        for _, row in output_frames.iterrows():
            self.frames_list.append((row["frame_id"], "output", row.name))
        
        self.current_frame_idx = 0
        self.total_frames = len(self.frames_list)
        
        # Get play context
        supp_row = scorer.supp_df[
            (scorer.supp_df["game_id"] == game_id) &
            (scorer.supp_df["play_id"] == play_id)
        ]
        if not supp_row.empty:
            self.route = supp_row.iloc[0].get("route_of_targeted_receiver", "UNKNOWN")
            self.pass_result = supp_row.iloc[0].get("pass_result", "UNKNOWN")
            self.offense_formation = supp_row.iloc[0].get("offense_formation", "UNKNOWN")
            self.receiver_alignment = supp_row.iloc[0].get("receiver_alignment", "UNKNOWN")
            self.coverage_type = supp_row.iloc[0].get("team_coverage_type", "UNKNOWN")
            self.pass_length = supp_row.iloc[0].get("pass_length", "UNKNOWN")
            self.down = supp_row.iloc[0].get("down", "UNKNOWN")
            self.yards_to_go = supp_row.iloc[0].get("yards_to_go", "UNKNOWN")
        else:
            self.route = "UNKNOWN"
            self.pass_result = "UNKNOWN"
            self.offense_formation = "UNKNOWN"
            self.receiver_alignment = "UNKNOWN"
            self.coverage_type = "UNKNOWN"
            self.pass_length = "UNKNOWN"
            self.down = "UNKNOWN"
            self.yards_to_go = "UNKNOWN"
        
        # Get ball landing
        self.ball_land_x = self.play_frames["ball_land_x_std"].iloc[0]
        self.ball_land_y = self.play_frames["ball_land_y_std"].iloc[0]
        
        # Get target name
        target_info = self.play_frames[
            (self.play_frames["nfl_id"] == target_nfl_id) &
            (self.play_frames["frame_type"] == "input")
        ]
        if not target_info.empty:
            self.target_name = target_info["player_name"].iloc[0]
        else:
            self.target_name = f"Player {target_nfl_id}"
        
        # Create figure with more space for info panel
        self.fig = plt.figure(figsize=(22, 14))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Create subplots - field on left, metrics on right, info panel below field
        self.ax_field = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
        self.ax_score = plt.subplot2grid((4, 4), (0, 3))
        self.ax_running = plt.subplot2grid((4, 4), (1, 3))
        self.ax_info = plt.subplot2grid((4, 4), (3, 0), colspan=4)  # Full width at bottom
        self.ax_info.axis('off')
        
        # Draw initial frame
        self.update_display()
        
        # Instructions
        print("\n" + "="*80)
        print("INTERACTIVE ROUTE DOMINANCE VIEWER")
        print("="*80)
        print("Controls:")
        print("  Left Arrow  : Previous frame")
        print("  Right Arrow : Next frame")
        print("  Up Arrow    : Jump to first frame")
        print("  Down Arrow  : Jump to last frame")
        print("  'q' or Esc  : Quit")
        print("="*80)
        print(f"\nTotal frames: {self.total_frames}")
        print(f"Current frame: {self.current_frame_idx + 1}/{self.total_frames}")
        print("\nClick on the plot window and use arrow keys to navigate!")
        print("="*80)
        
        plt.tight_layout()
        plt.show()
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        if event.key == 'left' or event.key == 'backspace':
            self.current_frame_idx = max(0, self.current_frame_idx - 1)
            self.update_display()
        elif event.key == 'right' or event.key == ' ':
            self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 1)
            self.update_display()
        elif event.key == 'up':
            self.current_frame_idx = 0
            self.update_display()
        elif event.key == 'down':
            self.current_frame_idx = self.total_frames - 1
            self.update_display()
        elif event.key == 'q' or event.key == 'escape':
            plt.close(self.fig)
            sys.exit(0)
    
    def draw_field(self):
        """Draw the football field"""
        self.ax_field.clear()
        self.ax_field.set_xlim(-5, 120 + 5)
        self.ax_field.set_ylim(-5, 53.3 + 5)
        self.ax_field.set_aspect('equal')
        self.ax_field.set_facecolor('#0d5f20')
        
        # End zones
        endzone1 = Rectangle((0, 0), 10, 53.3, facecolor='navy', alpha=0.5)
        endzone2 = Rectangle((110, 0), 10, 53.3, facecolor='navy', alpha=0.5)
        self.ax_field.add_patch(endzone1)
        self.ax_field.add_patch(endzone2)
        
        # Yard lines
        for yard in range(10, 110 + 1, 5):
            self.ax_field.axvline(x=yard, color='white', linewidth=0.5, alpha=0.3)
            if yard % 10 == 0:
                self.ax_field.text(yard, 53.3/2, str(yard), ha='center', va='center',
                                 color='white', fontsize=8, fontweight='bold',
                                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    def update_display(self):
        """Update the display for current frame"""
        frame_id, frame_type, row_idx = self.frames_list[self.current_frame_idx]
        
        # Draw field
        self.draw_field()
        
        # Get frame data
        frame_data = self.play_frames[
            (self.play_frames["frame_id"] == frame_id) & 
            (self.play_frames["frame_type"] == frame_type)
        ]
        
        frame_metric = self.frame_metrics[
            (self.frame_metrics["frame_id"] == frame_id) & 
            (self.frame_metrics["frame_type"] == frame_type)
        ]
        
        if not frame_metric.empty:
            current_sep = frame_metric.iloc[0]["sep_nearest"]
            running_avg = self.frame_metrics.iloc[:self.current_frame_idx+1]["sep_nearest"].mean()
        else:
            current_sep = 0.0
            running_avg = 0.0
        
        # Plot all players
        # First, identify which players are in this frame
        players_in_frame = frame_data["nfl_id"].unique()
        
        for _, player in frame_data.iterrows():
            is_target = player["nfl_id"] == self.target_nfl_id
            
            # Get player side (may be NaN for output-only players, default to Defense)
            player_side = player.get("player_side", "Defense")
            if pd.isna(player_side):
                # If player_side is missing, try to infer from player_role or default
                player_role = player.get("player_role", "")
                if "Receiver" in str(player_role) or "Offense" in str(player_role):
                    player_side = "Offense"
                else:
                    player_side = "Defense"
            
            if is_target:
                # Color-code targeted receiver by separation
                if not frame_metric.empty:
                    sep = frame_metric.iloc[0]["sep_nearest"]
                    if sep < 2.0:
                        target_color = 'red'
                    elif sep < 5.0:
                        target_color = 'yellow'
                    else:
                        target_color = 'lime'
                else:
                    target_color = 'yellow'
                    sep = 0.0
                
                # Highlight targeted receiver (same size as other players)
                marker_size = 150
                if frame_type == "output":
                    marker_size = 180
                
                self.ax_field.scatter(player["x_std"], player["y_std"], 
                                   c=target_color, s=marker_size, marker='o',
                                   edgecolors='black', linewidths=3, zorder=10)
                
                # Add direction arrow for targeted receiver
                player_vx = player.get("vx", 0)
                player_vy = player.get("vy", 0)
                if pd.notna(player_vx) and pd.notna(player_vy) and (player_vx != 0 or player_vy != 0):
                    # Scale arrow length based on speed (normalize to reasonable size)
                    speed = np.sqrt(player_vx**2 + player_vy**2)
                    arrow_length = min(speed * 0.3, 2.0)  # Max 2 yards
                    if arrow_length > 0.1:  # Only draw if moving
                        arrow_dx = (player_vx / speed) * arrow_length
                        arrow_dy = (player_vy / speed) * arrow_length
                        arrow = FancyArrowPatch(
                            (player["x_std"], player["y_std"]),
                            (player["x_std"] + arrow_dx, player["y_std"] + arrow_dy),
                            arrowstyle='->', mutation_scale=15, linewidth=2.5,
                            color='black', alpha=0.9, zorder=11
                        )
                        self.ax_field.add_patch(arrow)
            else:
                # Regular players - show all players from output file
                color = 'orange' if player_side == "Offense" else 'blue'
                marker_size = 150
                
                # Make output frame players slightly larger/more visible
                if frame_type == "output":
                    marker_size = 180
                
                self.ax_field.scatter(player["x_std"], player["y_std"],
                                   c=color, s=marker_size, alpha=0.8, zorder=5,
                                   edgecolors='white', linewidths=1)
                
                # Add direction arrow for regular players
                player_vx = player.get("vx", 0)
                player_vy = player.get("vy", 0)
                if pd.notna(player_vx) and pd.notna(player_vy) and (player_vx != 0 or player_vy != 0):
                    # Scale arrow length based on speed (normalize to reasonable size)
                    speed = np.sqrt(player_vx**2 + player_vy**2)
                    arrow_length = min(speed * 0.25, 1.5)  # Max 1.5 yards, smaller than target
                    if arrow_length > 0.1:  # Only draw if moving
                        arrow_dx = (player_vx / speed) * arrow_length
                        arrow_dy = (player_vy / speed) * arrow_length
                        arrow = FancyArrowPatch(
                            (player["x_std"], player["y_std"]),
                            (player["x_std"] + arrow_dx, player["y_std"] + arrow_dy),
                            arrowstyle='->', mutation_scale=12, linewidth=2,
                            color='white', alpha=0.8, zorder=6
                        )
                        self.ax_field.add_patch(arrow)
                
                # Optionally add player ID label for output frames
                if frame_type == "output":
                    player_name = player.get("player_name", f"ID:{player['nfl_id']}")
                    if pd.isna(player_name) or player_name == "":
                        player_name = f"ID:{player['nfl_id']}"
                    self.ax_field.annotate(player_name.split()[-1] if len(str(player_name).split()) > 1 else str(player_name)[:4],
                                        (player["x_std"], player["y_std"]),
                                        xytext=(0, -15), textcoords='offset points',
                                        fontsize=8, color='white',
                                        bbox=dict(boxstyle='round,pad=0.3', 
                                                facecolor=color, alpha=0.7,
                                                edgecolor='white', linewidth=1),
                                        ha='center', zorder=6)
        
        # Plot ball landing
        self.ax_field.scatter(self.ball_land_x, self.ball_land_y, c='yellow', s=600,
                           marker='X', edgecolors='black', linewidths=3, zorder=9,
                           label='Ball Landing')
        
        # Draw separation circle, defender line, and leverage angle
        if not frame_metric.empty:
            target_row = frame_data[frame_data["nfl_id"] == self.target_nfl_id]
            if not target_row.empty:
                target_x = target_row.iloc[0]["x_std"]
                target_y = target_row.iloc[0]["y_std"]
                sep = frame_metric.iloc[0]["sep_nearest"]
                leverage_angle = frame_metric.iloc[0].get("leverage_angle", np.nan)
                
                if sep < np.inf:
                    # Color circle based on separation
                    if sep < 2.0:
                        circle_color = 'red'
                    elif sep < 5.0:
                        circle_color = 'orange'
                    else:
                        circle_color = 'green'
                    
                    circle = Circle((target_x, target_y), sep, fill=False,
                                  edgecolor=circle_color, linewidth=3, 
                                  linestyle='--', alpha=0.7)
                    self.ax_field.add_patch(circle)
                    
                    # Draw line to nearest defender
                    defenders = frame_data[frame_data["player_side"] == "Defense"]
                    if not defenders.empty:
                        def_dists = np.sqrt(
                            (defenders["x_std"] - target_x)**2 +
                            (defenders["y_std"] - target_y)**2
                        )
                        nearest_idx = def_dists.idxmin()
                        nearest_def = defenders.loc[nearest_idx]
                        nearest_def_x = nearest_def["x_std"]
                        nearest_def_y = nearest_def["y_std"]
                        
                        # Draw line from defender to receiver
                        self.ax_field.plot([nearest_def_x, target_x], 
                                         [nearest_def_y, target_y],
                                         color=circle_color, linewidth=2.5, 
                                         linestyle=':', alpha=0.7, zorder=1,
                                         label='Defender-to-Receiver')
                        
                        # Draw line from receiver to ball
                        self.ax_field.plot([target_x, self.ball_land_x], 
                                         [target_y, self.ball_land_y],
                                         color='yellow', linewidth=2.5, 
                                         linestyle='-', alpha=0.8, zorder=1,
                                         label='Receiver-to-Ball')
                        
                        # Calculate and display leverage angle
                        if not np.isnan(leverage_angle):
                            # Calculate angles for arc drawing (in degrees, measured from positive x-axis)
                            # Angle from defender-to-receiver vector (from defender to receiver)
                            def_to_rec_angle_deg = np.rad2deg(np.arctan2(target_y - nearest_def_y, 
                                                                        target_x - nearest_def_x))
                            # Angle from receiver-to-ball vector
                            rec_to_ball_angle_deg = np.rad2deg(np.arctan2(self.ball_land_y - target_y,
                                                                          self.ball_land_x - target_x))
                            
                            # Normalize angles to [0, 360) for Arc
                            def_to_rec_angle_deg = def_to_rec_angle_deg % 360
                            rec_to_ball_angle_deg = rec_to_ball_angle_deg % 360
                            
                            # Draw arc to visualize the angle at receiver position
                            arc_radius = min(sep * 0.4, 4.0)  # Scale arc size
                            
                            # Determine which angle is smaller to draw the correct arc
                            if abs(rec_to_ball_angle_deg - def_to_rec_angle_deg) > 180:
                                # Need to wrap around
                                if rec_to_ball_angle_deg < def_to_rec_angle_deg:
                                    theta1 = rec_to_ball_angle_deg
                                    theta2 = def_to_rec_angle_deg
                                else:
                                    theta1 = def_to_rec_angle_deg
                                    theta2 = rec_to_ball_angle_deg
                            else:
                                theta1 = min(def_to_rec_angle_deg, rec_to_ball_angle_deg)
                                theta2 = max(def_to_rec_angle_deg, rec_to_ball_angle_deg)
                            
                            arc = Arc((target_x, target_y), arc_radius*2, arc_radius*2,
                                     angle=0, theta1=theta1, theta2=theta2,
                                     color='cyan', linewidth=3, alpha=0.9, zorder=2)
                            self.ax_field.add_patch(arc)
                            
                            # Leverage angle is shown in info panel, not on field to save space
        
        # Title with formation info
        continuous_frame_num = self.current_frame_idx + 1
        title_text = (
            f"{self.target_name} | Frame {continuous_frame_num}/{self.total_frames} ({frame_type.upper()})\n"
            f"Route: {self.route} | Formation: {self.offense_formation} {self.receiver_alignment} | "
            f"Coverage: {self.coverage_type} | Result: {self.pass_result}"
        )
        self.ax_field.set_title(
            title_text,
            fontsize=12, fontweight='bold', color='white', pad=10
        )
        self.ax_field.set_xlabel('X Position (yards)', fontsize=11, color='white')
        self.ax_field.set_ylabel('Y Position (yards)', fontsize=11, color='white')
        
        # Update score plot
        self.ax_score.clear()
        frame_metrics_plot = self.frame_metrics.copy()
        frame_metrics_plot['continuous_frame'] = range(1, len(frame_metrics_plot) + 1)
        
        self.ax_score.plot(frame_metrics_plot["continuous_frame"], frame_metrics_plot["sep_nearest"],
                         'b-', linewidth=2.5, marker='o', markersize=4, label='Separation')
        self.ax_score.axvline(x=continuous_frame_num, color='r', linestyle='--', linewidth=3)
        self.ax_score.scatter([continuous_frame_num], [current_sep], 
                            c='red', s=200, marker='*', zorder=5, label='Current')
        self.ax_score.set_xlabel('Frame Number', fontsize=11)
        self.ax_score.set_ylabel('Separation (yards)', fontsize=11)
        self.ax_score.set_title('Frame-by-Frame Separation', fontsize=12, fontweight='bold')
        self.ax_score.grid(True, alpha=0.3)
        self.ax_score.legend(fontsize=9)
        
        # Update running average
        self.ax_running.clear()
        running_avgs = []
        for i in range(continuous_frame_num):
            if i < len(frame_metrics_plot):
                avg = frame_metrics_plot.iloc[:i+1]["sep_nearest"].mean()
                running_avgs.append(avg)
        
        if running_avgs:
            self.ax_running.plot(range(1, len(running_avgs)+1), running_avgs,
                              'g-', linewidth=2.5, marker='o', markersize=5, label='Running Average')
            self.ax_running.axvline(x=continuous_frame_num, color='r', linestyle='--', linewidth=2)
            self.ax_running.axhline(y=running_avgs[-1], color='r', linestyle='--', linewidth=2, alpha=0.5)
        self.ax_running.set_xlabel('Frame Number', fontsize=11)
        self.ax_running.set_ylabel('Cumulative Avg', fontsize=11)
        self.ax_running.set_title(f'Running Average: {running_avg:.2f} yds', fontsize=12, fontweight='bold')
        self.ax_running.grid(True, alpha=0.3)
        self.ax_running.legend(fontsize=9)
        
        # Update info panel
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        if not frame_metric.empty:
            # Large separation box
            sep_box_text = f"SEPARATION: {current_sep:.2f} yds"
            if current_sep < 2.0:
                sep_box_color = 'red'
            elif current_sep < 5.0:
                sep_box_color = 'yellow'
            else:
                sep_box_color = 'lime'
            
            self.ax_info.text(0.5, 0.98, sep_box_text, transform=self.ax_info.transAxes,
                            fontsize=18, fontweight='bold', ha='center',
                            verticalalignment='top', family='monospace',
                            bbox=dict(boxstyle='round,pad=1.2', facecolor=sep_box_color, 
                                    alpha=0.9, edgecolor='black', linewidth=3))
            
            # Detailed metrics
            leverage_angle = frame_metric.iloc[0].get('leverage_angle', np.nan)
            leverage_str = f"{leverage_angle:.1f}°" if not np.isnan(leverage_angle) else "N/A"
            
            # Split metrics into two columns for better visibility
            info_text_left = f"""FRAME {continuous_frame_num}/{self.total_frames} ({frame_type.upper()})
{'='*60}
Current Separation: {current_sep:.2f} yds
Running Average: {running_avg:.2f} yds

FORMATION INFO
Offense: {self.offense_formation} {self.receiver_alignment}
Coverage: {self.coverage_type}
Down & Distance: {self.down} & {self.yards_to_go}
Pass Length: {self.pass_length} yds
Route: {self.route}

SEPARATION
Nearest Defender: {frame_metric.iloc[0]['sep_nearest']:.2f} yds
Defenders within 2 yds: {frame_metric.iloc[0]['num_def_within_2']}
Defenders within 3 yds: {frame_metric.iloc[0]['num_def_within_3']}
Defenders within 5 yds: {frame_metric.iloc[0]['num_def_within_5']}
"""
            
            info_text_right = f"""
LEVERAGE ANGLE
Angle: {leverage_str}
(Larger = Better: Defender in front)

MOTION
Speed: {frame_metric.iloc[0]['receiver_speed']:.2f} yds/s
Acceleration: {frame_metric.iloc[0]['receiver_accel']:.2f} yds/s²

BALL PROXIMITY
Distance to Ball: {frame_metric.iloc[0]['dist_to_ball']:.2f} yds

CONTROLS
Left/Right: Navigate frames
Up/Down: Jump to start/end
Q/Esc: Quit
"""
            
            # Display metrics in two columns below the dominance box
            self.ax_info.text(0.02, 0.70, info_text_left, transform=self.ax_info.transAxes,
                        fontsize=9, verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            self.ax_info.text(0.52, 0.70, info_text_right, transform=self.ax_info.transAxes,
                        fontsize=9, verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            info_text = f"Frame {continuous_frame_num}/{self.total_frames} ({frame_type})\nNo metrics available"
            self.ax_info.text(0.05, 0.70, info_text, transform=self.ax_info.transAxes,
                        fontsize=9, verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
        
        # Update figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():
    print("=" * 80)
    print("INTERACTIVE ROUTE DOMINANCE VIEWER")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    input_df = pd.read_csv("data/input_2023_w01.csv")
    output_df = pd.read_csv("data/output_2023_w01.csv")
    supp_df = pd.read_csv("data/Supplementary.csv")
    
    # Initialize scorer
    print("Initializing Route Dominance Scorer...")
    scorer = RouteDominanceScorer(input_df, output_df, supp_df)
    
    # Example play
    game_id = 2023090700
    play_id = 101
    
    # Get targeted receiver
    target_info = input_df[
        (input_df["game_id"] == game_id) &
        (input_df["play_id"] == play_id) &
        (input_df["player_role"] == "Targeted Receiver")
    ]
    
    if target_info.empty:
        print("No targeted receiver found. Using first available play...")
        target_info = input_df[
            input_df["player_role"] == "Targeted Receiver"
        ].iloc[0:1]
        game_id = target_info["game_id"].iloc[0]
        play_id = target_info["play_id"].iloc[0]
    
    target_nfl_id = target_info["nfl_id"].iloc[0]
    target_name = target_info["player_name"].iloc[0]
    
    print(f"\nGame: {game_id}, Play: {play_id}")
    print(f"Targeted Receiver: {target_name} (ID: {target_nfl_id})")
    
    # Create interactive viewer
    viewer = InteractiveRouteDominanceViewer(scorer, game_id, play_id, target_nfl_id)


if __name__ == "__main__":
    main()

