"""
Route Dominance Scoring System for NFL Plays

This module provides:
1. Frame-by-frame dominance scoring
2. Route-level aggregation
3. Interactive/animated visualizations
4. LSTM-based prediction approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
try:
    import seaborn as sns
except ImportError:
    sns = None  # Optional dependency
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Field constants
FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3
FRAME_RATE = 10  # frames per second (NFL tracking data)


class RouteDominanceScorer:
    """Calculate frame-by-frame and route-level dominance scores"""
    
    def __init__(self, input_df: pd.DataFrame, output_df: pd.DataFrame, 
                 supp_df: pd.DataFrame):
        """
        Initialize with dataframes
        
        Args:
            input_df: Pre-throw tracking data (all frames before throw)
            output_df: Post-throw tracking data (ball in flight)
            supp_df: Supplementary play context data
        """
        self.input_df = input_df.copy()
        self.output_df = output_df.copy()
        self.supp_df = supp_df.copy()
        
        # Standardize coordinates
        self._standardize_coordinates()
        self._standardize_reciever_side()
        
        # Combine all frames
        self.all_frames_df = self._combine_all_frames()
        
    def _standardize_coordinates(self):
        """Standardize coordinates so offense always drives right"""
        # Standardize input data
        right_mask = self.input_df["play_direction"].str.lower() == "right"
        self.input_df["x_std"] = self.input_df["x"]
        self.input_df["y_std"] = self.input_df["y"]
        self.input_df.loc[~right_mask, "x_std"] = FIELD_LENGTH - self.input_df.loc[~right_mask, "x"]
        self.input_df.loc[~right_mask, "y_std"] = FIELD_WIDTH - self.input_df.loc[~right_mask, "y"]
        
        # Standardize ball landing
        self.input_df["ball_land_x_std"] = self.input_df["ball_land_x"]
        self.input_df["ball_land_y_std"] = self.input_df["ball_land_y"]
        self.input_df.loc[~right_mask, "ball_land_x_std"] = FIELD_LENGTH - self.input_df.loc[~right_mask, "ball_land_x"]
        self.input_df.loc[~right_mask, "ball_land_y_std"] = FIELD_WIDTH - self.input_df.loc[~right_mask, "ball_land_y"]
        
        # Get play direction mapping for output
        play_dir_map = self.input_df[["game_id", "play_id", "play_direction"]].drop_duplicates()
        self.output_df = self.output_df.merge(play_dir_map, on=["game_id", "play_id"], how="left")
        
        # Standardize output data
        right_mask_out = self.output_df["play_direction"].str.lower() == "right"
        self.output_df["x_std"] = self.output_df["x"]
        self.output_df["y_std"] = self.output_df["y"]
        self.output_df.loc[~right_mask_out, "x_std"] = FIELD_LENGTH - self.output_df.loc[~right_mask_out, "x"]
        self.output_df.loc[~right_mask_out, "y_std"] = FIELD_WIDTH - self.output_df.loc[~right_mask_out, "y"]
        
        # Add velocity components
        self.input_df["vx"] = self.input_df["s"] * np.cos(np.deg2rad(self.input_df["dir"].fillna(0)))
        self.input_df["vy"] = self.input_df["s"] * np.sin(np.deg2rad(self.input_df["dir"].fillna(0)))

    def _standardize_reciever_side(self):
        """Flip coordinates of players so reciever always above qb (on the left side from qb's perspective)"""
        #Find what side the reciever is on
        qb_pos = self.input_df[(self.input_df["player_role"] == "Passer") & (self.input_df['frame_id'] == 1)][['y_std', 'play_id', 'game_id']]
        wr_pos = self.input_df[(self.input_df["player_role"] == "Targeted Receiver") & (self.input_df['frame_id'] == 1)][['y_std', 'play_id', 'game_id']]
        merged_pos = pd.merge(qb_pos, wr_pos, on=['play_id', 'game_id'], how='left')
        merged_pos['receiver_side'] = np.where(merged_pos['y_std_x'] < merged_pos['y_std_y'], 'left', 'right')
        
        # Merge receiver_side into input_df, output_df, and supp_df based on game_id and play_id
        receiver_side_df = merged_pos[['game_id', 'play_id', 'receiver_side']]
        self.supp_df = self.supp_df.merge(receiver_side_df, on=['game_id', 'play_id'], how='left')
        self.input_df = self.input_df.merge(receiver_side_df, on=['game_id', 'play_id'], how='left')
        self.output_df = self.output_df.merge(receiver_side_df, on=['game_id', 'play_id'], how='left')

        #Flip the field of inputs, outputs and ball landing when reciever aligns to right of qb
        self.input_df.loc[self.input_df['receiver_side'] == 'right', 'y_std'] = FIELD_WIDTH - self.input_df.loc[self.input_df['receiver_side'] == 'right', 'y_std']
        self.output_df.loc[self.output_df['receiver_side'] == 'right', 'y_std'] = FIELD_WIDTH - self.output_df.loc[self.output_df['receiver_side'] == 'right', 'y_std']
        self.input_df.loc[self.input_df['receiver_side'] == 'right', 'ball_land_y_std'] = FIELD_WIDTH - self.input_df.loc[self.input_df['receiver_side'] == 'right', 'ball_land_y_std']
    
    def _combine_all_frames(self) -> pd.DataFrame:
        """
        Combine input and output frames to get complete play sequence
        
        Returns:
            DataFrame with all frames for each play
        """
        # Get metadata from input (player info, ball landing, etc.)
        input_meta = self.input_df[[
            "game_id", "play_id", "nfl_id", "player_name", "player_position",
            "player_side", "player_role", "ball_land_x_std", "ball_land_y_std",
            "num_frames_output"
        ]].drop_duplicates()
        
        # Prepare input frames (pre-throw)
        input_frames = self.input_df[[
            "game_id", "play_id", "nfl_id", "frame_id", "x_std", "y_std",
            "s", "a", "dir", "vx", "vy"
        ]].copy()
        input_frames["frame_type"] = "input"
        
        # Prepare output frames (ball in flight)
        # Need to map frame_id: output frames start at 1, but they correspond to
        # frames after the throw. We'll offset them
        output_frames = self.output_df[[
            "game_id", "play_id", "nfl_id", "frame_id", "x_std", "y_std"
        ]].copy()
        
        # Merge with input to get speed/acceleration if available
        # For output frames, we'll need to estimate or use previous values
        output_frames = output_frames.merge(
            input_meta, on=["game_id", "play_id", "nfl_id"], how="left"
        )
        
        # Get the last input frame for each player to estimate velocity
        last_input = input_frames.sort_values("frame_id").groupby(
            ["game_id", "play_id", "nfl_id"]
        ).last().reset_index()
        
        # For output frames, estimate speed from position changes
        output_frames = output_frames.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
        
        # Initialize speed/acceleration columns
        output_frames["s"] = np.nan
        output_frames["a"] = np.nan
        output_frames["dir"] = np.nan
        output_frames["vx"] = np.nan
        output_frames["vy"] = np.nan
        
        # Calculate speed from position changes
        for (gid, pid, nid), group in output_frames.groupby(["game_id", "play_id", "nfl_id"]):
            group = group.sort_values("frame_id")
            
            # Get last input frame for this player to use as starting point
            last_input_row = last_input[
                (last_input["game_id"] == gid) &
                (last_input["play_id"] == pid) &
                (last_input["nfl_id"] == nid)
            ]
            
            # Calculate position changes
            group["prev_x"] = group["x_std"].shift(1)
            group["prev_y"] = group["y_std"].shift(1)
            
            # For first output frame, use last input position
            if not last_input_row.empty:
                first_idx = group.index[0]
                output_frames.loc[first_idx, "prev_x"] = last_input_row.iloc[0]["x_std"]
                output_frames.loc[first_idx, "prev_y"] = last_input_row.iloc[0]["y_std"]
            
            # Calculate displacement
            dx = group["x_std"] - group["prev_x"]
            dy = group["y_std"] - group["prev_y"]
            
            # Calculate speed (yards per second)
            # Frame rate is 10 fps, so multiply by 10 to get yards/second
            speed = np.sqrt(dx**2 + dy**2) * FRAME_RATE
            direction = np.rad2deg(np.arctan2(dy, dx))
            
            # Calculate acceleration (change in speed)
            prev_speed = speed.shift(1)
            if not last_input_row.empty:
                prev_speed.iloc[0] = last_input_row.iloc[0]["s"]
            acceleration = (speed - prev_speed) * FRAME_RATE
            
            # Calculate velocity components
            vx = speed * np.cos(np.deg2rad(direction))
            vy = speed * np.sin(np.deg2rad(direction))
            
            # Fill NaN values with last known values
            speed = speed.ffill().fillna(0)
            direction = direction.ffill().fillna(0)
            acceleration = acceleration.fillna(0)
            vx = vx.fillna(0)
            vy = vy.fillna(0)
            
            # Special handling for first output frame: use average of last input and second output
            if len(group) > 1 and not last_input_row.empty:
                first_idx = group.index[0]  # First output frame
                second_idx = group.index[1]  # Second output frame
                
                last_input_speed = last_input_row.iloc[0]["s"]
                second_output_speed = speed.loc[second_idx]
                
                # Average the speeds for the first output frame
                if not np.isnan(second_output_speed) and not np.isnan(last_input_speed):
                    avg_speed = (last_input_speed + second_output_speed) / 2.0
                    speed.loc[first_idx] = avg_speed
                    
                    # Recalculate direction and velocity for first frame using averaged speed
                    first_dx = group.loc[first_idx, "x_std"] - group.loc[first_idx, "prev_x"]
                    first_dy = group.loc[first_idx, "y_std"] - group.loc[first_idx, "prev_y"]
                    first_dir = np.rad2deg(np.arctan2(first_dy, first_dx))
                    direction.loc[first_idx] = first_dir
                    vx.loc[first_idx] = avg_speed * np.cos(np.deg2rad(first_dir))
                    vy.loc[first_idx] = avg_speed * np.sin(np.deg2rad(first_dir))
                    
                    # Recalculate acceleration for first frame
                    acceleration.loc[first_idx] = (avg_speed - last_input_speed) * FRAME_RATE
            
            # Update output_frames
            for idx in group.index:
                output_frames.loc[idx, "s"] = speed.loc[idx]
                output_frames.loc[idx, "dir"] = direction.loc[idx]
                output_frames.loc[idx, "a"] = acceleration.loc[idx]
                output_frames.loc[idx, "vx"] = vx.loc[idx]
                output_frames.loc[idx, "vy"] = vy.loc[idx]
        
        # Clean up temporary columns
        if "prev_x" in output_frames.columns:
            output_frames = output_frames.drop(columns=["prev_x", "prev_y"])
        output_frames["frame_type"] = "output"
        
        # Combine input and output
        input_cols = ["game_id", "play_id", "nfl_id", "frame_id", "x_std", "y_std", 
                     "s", "a", "dir", "vx", "vy", "frame_type"]
        output_cols = ["game_id", "play_id", "nfl_id", "frame_id", "x_std", "y_std",
                      "s", "a", "dir", "vx", "vy", "frame_type"]
        
        all_frames = pd.concat([
            input_frames[input_cols],
            output_frames[output_cols]
        ], ignore_index=True)
        
        # Add metadata
        all_frames = all_frames.merge(input_meta, on=["game_id", "play_id", "nfl_id"], how="left")
        
        # Sort by frame_id
        all_frames = all_frames.sort_values(["game_id", "play_id", "frame_id", "nfl_id"])
        
        return all_frames
    
    def calculate_frame_dominance(self, game_id: int, play_id: int, 
                                   target_nfl_id: int) -> pd.DataFrame:
        """
        Calculate dominance score for each frame of a play
        
        Args:
            game_id: Game identifier
            play_id: Play identifier
            target_nfl_id: NFL ID of targeted receiver
            
        Returns:
            DataFrame with frame-by-frame dominance metrics
        """
        # Get all frames for this play
        play_frames = self.all_frames_df[
            (self.all_frames_df["game_id"] == game_id) &
            (self.all_frames_df["play_id"] == play_id)
        ].copy()
        
        if play_frames.empty:
            raise ValueError(f"Play {game_id}-{play_id} not found")
        
        # Get targeted receiver frames
        target_frames = play_frames[play_frames["nfl_id"] == target_nfl_id].copy()
        
        if target_frames.empty:
            raise ValueError(f"Targeted receiver {target_nfl_id} not found in play")
        
        # Get ball landing coordinates
        ball_land_x = target_frames["ball_land_x_std"].iloc[0]
        ball_land_y = target_frames["ball_land_y_std"].iloc[0]
        num_frames_output = target_frames["num_frames_output"].iloc[0]
        
        # Get play context
        supp_row = self.supp_df[
            (self.supp_df["game_id"] == game_id) &
            (self.supp_df["play_id"] == play_id)
        ]
        
        if supp_row.empty:
            route = "UNKNOWN"
            pass_result = "UNKNOWN"
        else:
            route = supp_row.iloc[0].get("route_of_targeted_receiver", "UNKNOWN")
            pass_result = supp_row.iloc[0].get("pass_result", "UNKNOWN")
        
        # Calculate metrics for each frame
        # Need to iterate through both input and output frames separately
        # because they have overlapping frame_ids (both start at 1)
        frame_metrics = []
        
        # Get input and output frames separately
        input_target_frames = target_frames[target_frames["frame_type"] == "input"].sort_values("frame_id")
        output_target_frames = target_frames[target_frames["frame_type"] == "output"].sort_values("frame_id")
        
        # Helper function to process a single frame
        def process_frame(target_frame_row, frame_type):
            frame_id = target_frame_row["frame_id"]
            target_frame = target_frame_row
            
            # Get all players at this frame (must match both frame_id AND frame_type)
            # This ensures we get all players from the output file for output frames
            frame_players = play_frames[
                (play_frames["frame_id"] == frame_id) & 
                (play_frames["frame_type"] == frame_type)
            ]
            
            # Get defenders
            defenders = frame_players[frame_players["player_side"] == "Defense"]
            
            # 1. Separation from nearest defender
            if not defenders.empty:
                def_dists = np.sqrt(
                    (defenders["x_std"] - target_frame["x_std"])**2 +
                    (defenders["y_std"] - target_frame["y_std"])**2
                )
                sep_nearest = def_dists.min()
                sep_second = def_dists.nsmallest(2).iloc[-1] if len(def_dists) > 1 else np.nan
                num_def_within_2 = (def_dists <= 2.0).sum()
                num_def_within_3 = (def_dists <= 3.0).sum()
                num_def_within_5 = (def_dists <= 5.0).sum()
            else:
                sep_nearest = np.inf
                sep_second = np.nan
                num_def_within_2 = 0
                num_def_within_3 = 0
                num_def_within_5 = 0
            
            # 2. Receiver speed and acceleration
            receiver_speed = target_frame["s"]
            receiver_accel = target_frame["a"]
            
            # 3. Distance to ball landing spot
            dist_to_ball = np.sqrt(
                (target_frame["x_std"] - ball_land_x)**2 +
                (target_frame["y_std"] - ball_land_y)**2
            )
            
            # 4. Time to reach ball (estimated)
            # Simple estimate: distance / speed (if moving toward ball)
            # More sophisticated: account for direction
            receiver_vx = target_frame["vx"]
            receiver_vy = target_frame["vy"]
            
            # Vector from receiver to ball
            to_ball_x = ball_land_x - target_frame["x_std"]
            to_ball_y = ball_land_y - target_frame["y_std"]
            to_ball_dist = np.sqrt(to_ball_x**2 + to_ball_y**2)
            
            if to_ball_dist > 0:
                to_ball_unit_x = to_ball_x / to_ball_dist
                to_ball_unit_y = to_ball_y / to_ball_dist
                
                # Speed component toward ball
                speed_toward_ball = receiver_vx * to_ball_unit_x + receiver_vy * to_ball_unit_y
                
                if speed_toward_ball > 0:
                    time_to_ball = to_ball_dist / speed_toward_ball
                else:
                    time_to_ball = np.inf
            else:
                time_to_ball = 0
            
            # 5. Defender time to ball (for nearest defender)
            if not defenders.empty and sep_nearest < np.inf:
                nearest_def = defenders.loc[def_dists.idxmin()]
                def_to_ball_x = ball_land_x - nearest_def["x_std"]
                def_to_ball_y = ball_land_y - nearest_def["y_std"]
                def_to_ball_dist = np.sqrt(def_to_ball_x**2 + def_to_ball_y**2)
                
                if def_to_ball_dist > 0:
                    def_to_ball_unit_x = def_to_ball_x / def_to_ball_dist
                    def_to_ball_unit_y = def_to_ball_y / def_to_ball_dist
                    
                    # Get defender velocity (may need to calculate from position if not available)
                    def_vx = nearest_def.get("vx", 0)
                    def_vy = nearest_def.get("vy", 0)
                    
                    # If velocity not available, try to estimate from speed and direction
                    if def_vx == 0 and def_vy == 0:
                        def_s = nearest_def.get("s", 0)
                        def_dir = nearest_def.get("dir", 0)
                        if def_s > 0 and not np.isnan(def_dir):
                            def_vx = def_s * np.cos(np.deg2rad(def_dir))
                            def_vy = def_s * np.sin(np.deg2rad(def_dir))
                    
                    def_speed_toward_ball = def_vx * def_to_ball_unit_x + def_vy * def_to_ball_unit_y
                    
                    if def_speed_toward_ball > 0:
                        def_time_to_ball = def_to_ball_dist / def_speed_toward_ball
                    else:
                        def_time_to_ball = np.inf
                else:
                    def_time_to_ball = 0
                
                time_advantage = def_time_to_ball - time_to_ball
            else:
                def_time_to_ball = np.inf
                time_advantage = np.inf
            
            # 6. Leverage Angle - angle between defender-to-receiver and receiver-to-ball vectors
            leverage_angle = np.nan
            if not defenders.empty and sep_nearest < np.inf:
                nearest_def = defenders.loc[def_dists.idxmin()]
                
                # Vector from nearest defender to receiver
                def_to_rec_x = target_frame["x_std"] - nearest_def["x_std"]
                def_to_rec_y = target_frame["y_std"] - nearest_def["y_std"]
                
                # Vector from receiver to ball landing
                rec_to_ball_x = ball_land_x - target_frame["x_std"]
                rec_to_ball_y = ball_land_y - target_frame["y_std"]
                
                # Calculate angle between vectors using dot product
                # angle = arccos((v1 · v2) / (||v1|| * ||v2||))
                dot_product = def_to_rec_x * rec_to_ball_x + def_to_rec_y * rec_to_ball_y
                mag_def_to_rec = np.sqrt(def_to_rec_x**2 + def_to_rec_y**2)
                mag_rec_to_ball = np.sqrt(rec_to_ball_x**2 + rec_to_ball_y**2)
                
                if mag_def_to_rec > 0 and mag_rec_to_ball > 0:
                    cos_angle = dot_product / (mag_def_to_rec * mag_rec_to_ball)
                    # Clamp to [-1, 1] to avoid numerical errors
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_rad = np.arccos(cos_angle)
                    angle_deg = np.rad2deg(angle_rad)
                    
                    # Normalize to always be the smaller angle (≤ 180 degrees)
                    # arccos gives [0, π] which is [0, 180°], so we take the smaller of angle and 180-angle
                    leverage_angle = min(angle_deg, 180.0 - angle_deg)
                else:
                    leverage_angle = np.nan
            
            # 7. Initial Leverage (angle advantage) - calculated at start of route
            # Leverage: is the defender between the receiver and the ball?
            if frame_id == 1 and not defenders.empty:
                # Get initial positions
                target_start_x = target_frame["x_std"]
                target_start_y = target_frame["y_std"]
                
                # Vector from target to ball
                target_to_ball_x = ball_land_x - target_start_x
                target_to_ball_y = ball_land_y - target_start_y
                
                leverage_scores = []
                for _, def_row in defenders.iterrows():
                    # Vector from target to defender
                    target_to_def_x = def_row["x_std"] - target_start_x
                    target_to_def_y = def_row["y_std"] - target_start_y
                    
                    # Check if defender is "between" target and ball
                    # Using dot product to check alignment
                    dot_product = (target_to_def_x * target_to_ball_x + 
                                 target_to_def_y * target_to_ball_y)
                    
                    if dot_product > 0:  # Defender is in front (bad leverage)
                        leverage = -1.0
                    else:  # Defender is behind (good leverage)
                        leverage = 1.0
                    
                    leverage_scores.append(leverage)
                
                initial_leverage = np.mean(leverage_scores) if leverage_scores else 0.0
            else:
                initial_leverage = np.nan if frame_id == 1 else None
            
            # 8. Calculate composite dominance score
            # Normalize and combine metrics
            # Higher score = more dominant
            
            # Separation component (0-1, higher is better)
            sep_score = min(sep_nearest / 10.0, 1.0) if sep_nearest < np.inf else 0.5
            
            # Speed component (normalized, higher is better)
            speed_score = min(receiver_speed / 8.0, 1.0) if receiver_speed > 0 else 0.0
            
            # Acceleration component (normalized)
            accel_score = min(max(receiver_accel / 3.0, -1.0), 1.0) / 2.0 + 0.5
            
            # Time advantage (normalized, positive is better)
            if time_advantage < np.inf and time_advantage > -np.inf:
                time_score = min(max(time_advantage / 2.0, -1.0), 1.0) / 2.0 + 0.5
            else:
                time_score = 0.5
            
            # Defender pressure (inverse of defenders nearby)
            pressure_score = 1.0 - min(num_def_within_3 / 5.0, 1.0)
            
            # Leverage angle component (normalized, larger angle is better)
            # Leverage angle is the angle between defender-to-receiver and receiver-to-ball vectors
            # Larger angles mean defender is more in front of receiver relative to ball (better for receiver)
            # Convert angle to score: 180° = 1.0 (best), 0° = 0.0 (worst)
            if not np.isnan(leverage_angle):
                leverage_score = leverage_angle / 180.0  # Larger angle → higher score
            else:
                leverage_score = 0.5  # Neutral if no leverage data
            
            # Weighted combination
            # Adjusted weights to include leverage (reduced other weights slightly)
            dominance_score = (
                0.25 * sep_score +        # Reduced from 0.30
                0.18 * speed_score +      # Reduced from 0.20
                0.12 * accel_score +     # Reduced from 0.15
                0.18 * time_score +      # Reduced from 0.20
                0.12 * pressure_score +   # Reduced from 0.15
                0.15 * leverage_score     # NEW: Leverage component
            )
            
            return {
                "game_id": game_id,
                "play_id": play_id,
                "nfl_id": target_nfl_id,
                "frame_id": frame_id,
                "frame_type": frame_type,
                "x": target_frame["x_std"],
                "y": target_frame["y_std"],
                "sep_nearest": sep_nearest,
                "sep_second": sep_second,
                "num_def_within_2": num_def_within_2,
                "num_def_within_3": num_def_within_3,
                "num_def_within_5": num_def_within_5,
                "receiver_speed": receiver_speed,
                "receiver_accel": receiver_accel,
                "dist_to_ball": dist_to_ball,
                "time_to_ball": time_to_ball,
                "def_time_to_ball": def_time_to_ball,
                "time_advantage": time_advantage,
                "leverage_angle": leverage_angle,
                "initial_leverage": initial_leverage if frame_id == 1 and frame_type == "input" else None,
                "dominance_score": dominance_score,
                "route": route,
                "pass_result": pass_result
            }
        
        # Process all input frames
        for _, target_frame_row in input_target_frames.iterrows():
            metrics = process_frame(target_frame_row, "input")
            frame_metrics.append(metrics)
        
        # Process all output frames
        for _, target_frame_row in output_target_frames.iterrows():
            metrics = process_frame(target_frame_row, "output")
            frame_metrics.append(metrics)
        
        return pd.DataFrame(frame_metrics)
    
    def calculate_route_dominance(self, frame_metrics: pd.DataFrame,
                                   method: str = "weighted_average") -> float:
        """
        Aggregate frame-by-frame scores to route-level score
        
        Args:
            frame_metrics: DataFrame from calculate_frame_dominance
            method: Aggregation method ('average', 'weighted_average', 'max', 'min')
            
        Returns:
            Single route dominance score
        """
        if method == "average":
            return frame_metrics["dominance_score"].mean()
        
        elif method == "weighted_average":
            # Weight later frames more (ball in flight is more important)
            # Also weight frames with higher separation more
            weights = []
            for _, row in frame_metrics.iterrows():
                frame_weight = 1.0
                
                # Later frames (ball in flight) get higher weight
                if row["frame_type"] == "output":
                    frame_weight *= 1.5
                
                # Frames with good separation get higher weight
                if row["sep_nearest"] > 3.0:
                    frame_weight *= 1.2
                
                weights.append(frame_weight)
            
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            return np.average(frame_metrics["dominance_score"], weights=weights)
        
        elif method == "max":
            return frame_metrics["dominance_score"].max()
        
        elif method == "min":
            return frame_metrics["dominance_score"].min()
        
        else:
            raise ValueError(f"Unknown method: {method}")


class RouteDominanceVisualizer:
    """Create visualizations for route dominance"""
    
    def __init__(self, scorer: RouteDominanceScorer):
        self.scorer = scorer
    
    def visualize_play_dominance(self, game_id: int, play_id: int,
                                  target_nfl_id: int, save_path: Optional[str] = None,
                                  show_animation: bool = True):
        """
        Create animated visualization showing dominance evolution
        
        Args:
            game_id: Game identifier
            play_id: Play identifier
            target_nfl_id: NFL ID of targeted receiver
            save_path: Path to save animation (optional)
            show_animation: Whether to display animation
        """
        # Calculate frame metrics
        frame_metrics = self.scorer.calculate_frame_dominance(
            game_id, play_id, target_nfl_id
        )
        
        # Get all frames for visualization
        play_frames = self.scorer.all_frames_df[
            (self.scorer.all_frames_df["game_id"] == game_id) &
            (self.scorer.all_frames_df["play_id"] == play_id)
        ]
        
        # Get play context
        supp_row = self.scorer.supp_df[
            (self.scorer.supp_df["game_id"] == game_id) &
            (self.scorer.supp_df["play_id"] == play_id)
        ]
        
        route = frame_metrics["route"].iloc[0]
        pass_result = frame_metrics["pass_result"].iloc[0]
        
        # Get ball landing
        ball_land_x = play_frames["ball_land_x_std"].iloc[0]
        ball_land_y = play_frames["ball_land_y_std"].iloc[0]
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # Field view
        ax_field = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
        
        # Dominance score over time
        ax_score = plt.subplot2grid((3, 4), (0, 3))
        
        # Running average
        ax_running = plt.subplot2grid((3, 4), (1, 3))
        
        # Info panel
        ax_info = plt.subplot2grid((3, 4), (2, 3))
        ax_info.axis('off')
        
        # Draw field
        def draw_field():
            ax_field.clear()
            ax_field.set_xlim(-5, FIELD_LENGTH + 5)
            ax_field.set_ylim(-5, FIELD_WIDTH + 5)
            ax_field.set_aspect('equal')
            ax_field.set_facecolor('#0d5f20')
            
            # End zones
            endzone = Rectangle((0, 0), 10, FIELD_WIDTH, facecolor='navy', alpha=0.5)
            ax_field.add_patch(endzone)
            endzone2 = Rectangle((FIELD_LENGTH - 10, 0), 10, FIELD_WIDTH, facecolor='navy', alpha=0.5)
            ax_field.add_patch(endzone2)
            
            # Yard lines
            for yard in range(10, int(FIELD_LENGTH - 10) + 1, 5):
                ax_field.axvline(x=yard, color='white', linewidth=0.5, alpha=0.3)
                if yard % 10 == 0:
                    ax_field.text(yard, FIELD_WIDTH/2, str(yard), ha='center', va='center',
                                 color='white', fontsize=8, fontweight='bold',
                                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Animation function
        # Create continuous frame sequence: input frames first, then output frames
        # Since frame_id overlaps (both start at 1), we need to handle them separately
        input_frames_vis = play_frames[play_frames["frame_type"] == "input"].sort_values("frame_id")
        output_frames_vis = play_frames[play_frames["frame_type"] == "output"].sort_values("frame_id")
        
        # Create list of (frame_id, frame_type) tuples for proper sequencing
        frames_list = []
        for _, row in input_frames_vis.iterrows():
            frames_list.append((row["frame_id"], "input", row.name))
        for _, row in output_frames_vis.iterrows():
            frames_list.append((row["frame_id"], "output", row.name))
        
        total_frames = len(frames_list)
        
        def animate(frame_idx):
            frame_id, frame_type, row_idx = frames_list[frame_idx]
            
            # Draw field
            draw_field()
            
            # Get frame data - need to match both frame_id AND frame_type
            frame_data = play_frames[
                (play_frames["frame_id"] == frame_id) & 
                (play_frames["frame_type"] == frame_type)
            ]
            frame_metric = frame_metrics[
                (frame_metrics["frame_id"] == frame_id) & 
                (frame_metrics["frame_type"] == frame_type)
            ]
            
            if not frame_metric.empty:
                current_dominance = frame_metric.iloc[0]["dominance_score"]
                # Calculate running average up to current continuous frame
                running_avg = frame_metrics.iloc[:frame_idx+1]["dominance_score"].mean()
            else:
                current_dominance = 0.0
                running_avg = 0.0
            
            # Plot all players with color-coding based on dominance (for targeted receiver)
            for _, player in frame_data.iterrows():
                is_target = player["nfl_id"] == target_nfl_id
                
                if is_target:
                    # Color-code targeted receiver by dominance score
                    if not frame_metric.empty:
                        dom_score = frame_metric.iloc[0]["dominance_score"]
                        # Color scale: red (low) -> yellow (medium) -> green (high)
                        if dom_score < 0.33:
                            target_color = 'red'
                        elif dom_score < 0.67:
                            target_color = 'yellow'
                        else:
                            target_color = 'lime'
                    else:
                        target_color = 'yellow'
                        dom_score = 0.0
                    
                    # Highlight targeted receiver with dominance-based color
                    ax_field.scatter(player["x_std"], player["y_std"], 
                                   c=target_color, s=400, marker='*',
                                   edgecolors='black', linewidths=3, zorder=10)
                    
                    # Add dominance score text near receiver
                    ax_field.annotate(f'{dom_score:.2f}', 
                                    (player["x_std"], player["y_std"]),
                                    xytext=(0, 15), textcoords='offset points',
                                    fontsize=12, fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.5', 
                                            facecolor='white', alpha=0.9,
                                            edgecolor='black', linewidth=2),
                                    ha='center', zorder=11)
                else:
                    # Regular players
                    color = 'orange' if player["player_side"] == "Offense" else 'blue'
                    ax_field.scatter(player["x_std"], player["y_std"],
                                   c=color, s=150, alpha=0.7, zorder=5)
            
            # Plot ball landing
            ax_field.scatter(ball_land_x, ball_land_y, c='yellow', s=400,
                           marker='X', edgecolors='black', linewidths=2, zorder=9)
            
            # Draw separation circle for targeted receiver with dominance-based color
            if not frame_metric.empty:
                target_row = frame_data[frame_data["nfl_id"] == target_nfl_id]
                if not target_row.empty:
                    target_x = target_row.iloc[0]["x_std"]
                    target_y = target_row.iloc[0]["y_std"]
                    sep = frame_metric.iloc[0]["sep_nearest"]
                    dom_score = frame_metric.iloc[0]["dominance_score"]
                    
                    if sep < np.inf:
                        # Color circle based on dominance
                        if dom_score < 0.33:
                            circle_color = 'red'
                        elif dom_score < 0.67:
                            circle_color = 'orange'
                        else:
                            circle_color = 'green'
                        
                        circle = Circle((target_x, target_y), sep, fill=False,
                                      edgecolor=circle_color, linewidth=2.5, 
                                      linestyle='--', alpha=0.7)
                        ax_field.add_patch(circle)
                        
                        # Draw line to nearest defender
                        defenders = frame_data[frame_data["player_side"] == "Defense"]
                        if not defenders.empty:
                            def_dists = np.sqrt(
                                (defenders["x_std"] - target_x)**2 +
                                (defenders["y_std"] - target_y)**2
                            )
                            nearest_idx = def_dists.idxmin()
                            nearest_def = defenders.loc[nearest_idx]
                            ax_field.plot([target_x, nearest_def["x_std"]], 
                                         [target_y, nearest_def["y_std"]],
                                         color=circle_color, linewidth=2, 
                                         linestyle=':', alpha=0.6, zorder=1)
            
            # Title - show continuous frame number and frame type
            continuous_frame_num = frame_idx + 1
            ax_field.set_title(
                f"Game {game_id}, Play {play_id} | Frame {continuous_frame_num}/{total_frames} ({frame_type}) | "
                f"Route: {route} | Result: {pass_result}",
                fontsize=12, fontweight='bold', color='white', pad=10
            )
            
            # Update score plot - use continuous frame numbers
            ax_score.clear()
            # Create continuous frame numbers for plotting
            frame_metrics_plot = frame_metrics.copy()
            frame_metrics_plot['continuous_frame'] = range(1, len(frame_metrics_plot) + 1)
            
            ax_score.plot(frame_metrics_plot["continuous_frame"], frame_metrics_plot["dominance_score"],
                         'b-', linewidth=2, label='Frame Dominance')
            ax_score.axvline(x=continuous_frame_num, color='r', linestyle='--', linewidth=2)
            ax_score.set_xlabel('Frame Number (Continuous)')
            ax_score.set_ylabel('Dominance Score')
            ax_score.set_title('Frame-by-Frame Dominance')
            ax_score.set_ylim(0, 1)
            ax_score.grid(True, alpha=0.3)
            ax_score.legend()
            
            # Update running average
            ax_running.clear()
            running_avgs = []
            # Calculate running average up to current continuous frame
            for i in range(continuous_frame_num):
                if i < len(frame_metrics_plot):
                    avg = frame_metrics_plot.iloc[:i+1]["dominance_score"].mean()
                    running_avgs.append(avg)
            
            if running_avgs:
                ax_running.plot(range(1, len(running_avgs)+1), running_avgs,
                              'g-', linewidth=2, marker='o', markersize=4, label='Running Average')
                ax_running.axvline(x=continuous_frame_num, color='r', linestyle='--', linewidth=2)
                ax_running.axhline(y=running_avgs[-1], color='r', linestyle='--', linewidth=2, alpha=0.5)
            ax_running.set_xlabel('Frame Number (Continuous)')
            ax_running.set_ylabel('Cumulative Avg Dominance')
            ax_running.set_title(f'Running Average: {running_avg:.3f}')
            ax_running.set_ylim(0, 1)
            ax_running.grid(True, alpha=0.3)
            ax_running.legend()
            
            # Update info panel
            ax_info.clear()
            ax_info.axis('off')
            
            if not frame_metric.empty:
                info_text = f"""
FRAME {continuous_frame_num}/{total_frames} ({frame_type.upper()})
{'='*40}
Current Dominance: {current_dominance:.3f}
Running Average: {running_avg:.3f}

SEPARATION
Nearest Defender: {frame_metric.iloc[0]['sep_nearest']:.2f} yds
Defenders within 3 yds: {frame_metric.iloc[0]['num_def_within_3']}

MOTION
Speed: {frame_metric.iloc[0]['receiver_speed']:.2f} yds/s
Acceleration: {frame_metric.iloc[0]['receiver_accel']:.2f} yds/s²

BALL PROXIMITY
Distance to Ball: {frame_metric.iloc[0]['dist_to_ball']:.2f} yds
Time to Ball: {frame_metric.iloc[0]['time_to_ball']:.2f} s
Time Advantage: {frame_metric.iloc[0]['time_advantage']:.2f} s
"""
            else:
                info_text = f"Frame {continuous_frame_num}/{total_frames} ({frame_type})\nNo metrics available"
            
            # Add dominance score indicator box (large, prominent)
            dom_box_text = f"DOMINANCE: {current_dominance:.3f}"
            if current_dominance < 0.33:
                dom_box_color = 'red'
            elif current_dominance < 0.67:
                dom_box_color = 'yellow'
            else:
                dom_box_color = 'lime'
            
            ax_info.text(0.5, 0.98, dom_box_text, transform=ax_info.transAxes,
                        fontsize=16, fontweight='bold', ha='center',
                        verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round,pad=1', facecolor=dom_box_color, 
                                alpha=0.8, edgecolor='black', linewidth=3))
            
            ax_info.text(0.05, 0.85, info_text, transform=ax_info.transAxes,
                        fontsize=10, verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
        
        # Create animation
        # Use continuous frame numbering for display
        total_frames = len(frames_list)
        print(f"Creating animation with {total_frames} frames ({len(input_frames_vis)} input + {len(output_frames_vis)} output)")
        
        anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                      interval=200, repeat=True)
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            try:
                anim.save(save_path, writer='pillow', fps=5)
                print(f"✓ Animation saved successfully!")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print("Trying with imagemagick writer...")
                try:
                    anim.save(save_path.replace('.gif', '.mp4'), writer='ffmpeg', fps=5)
                    print(f"✓ Saved as MP4 instead")
                except:
                    print("Could not save animation. Displaying instead...")
        
        if show_animation:
            plt.tight_layout()
            plt.show()
        
        return anim, frame_metrics


def prepare_lstm_features(scorer: RouteDominanceScorer, 
                          game_ids: List[int], play_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features for LSTM model
    
    Args:
        scorer: RouteDominanceScorer instance
        game_ids: List of game IDs
        play_ids: List of play IDs
        
    Returns:
        X: Feature sequences (n_samples, n_frames, n_features)
        y: Route dominance scores (n_samples,)
    """
    sequences = []
    targets = []
    
    for game_id, play_id in zip(game_ids, play_ids):
        # Get targeted receiver
        play_input = scorer.input_df[
            (scorer.input_df["game_id"] == game_id) &
            (scorer.input_df["play_id"] == play_id) &
            (scorer.input_df["player_role"] == "Targeted Receiver")
        ]
        
        if play_input.empty:
            continue
        
        target_nfl_id = play_input["nfl_id"].iloc[0]
        
        # Calculate frame metrics
        try:
            frame_metrics = scorer.calculate_frame_dominance(game_id, play_id, target_nfl_id)
        except:
            continue
        
        # Extract features for each frame
        feature_cols = [
            'sep_nearest', 'sep_second', 'num_def_within_2', 'num_def_within_3',
            'num_def_within_5', 'receiver_speed', 'receiver_accel', 'dist_to_ball',
            'time_to_ball', 'time_advantage'
        ]
        
        # Fill NaN values
        frame_features = frame_metrics[feature_cols].fillna(0).values
        
        # Pad or truncate to fixed length (e.g., 30 frames)
        max_frames = 30
        if len(frame_features) < max_frames:
            padding = np.zeros((max_frames - len(frame_features), len(feature_cols)))
            frame_features = np.vstack([frame_features, padding])
        else:
            frame_features = frame_features[:max_frames]
        
        sequences.append(frame_features)
        
        # Calculate route-level dominance
        route_dom = scorer.calculate_route_dominance(frame_metrics, method="weighted_average")
        targets.append(route_dom)
    
    X = np.array(sequences)
    y = np.array(targets)
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    
    # Load data
    input_df = pd.read_csv("data/input_2023_w01.csv")
    output_df = pd.read_csv("data/output_2023_w01.csv")
    supp_df = pd.read_csv("data/Supplementary.csv")
    
    # Initialize scorer
    scorer = RouteDominanceScorer(input_df, output_df, supp_df)
    
    # Example: Calculate dominance for a specific play
    game_id = 2023090700
    play_id = 101
    
    # Get targeted receiver
    target_info = input_df[
        (input_df["game_id"] == game_id) &
        (input_df["play_id"] == play_id) &
        (input_df["player_role"] == "Targeted Receiver")
    ]
    
    if not target_info.empty:
        target_nfl_id = target_info["nfl_id"].iloc[0]
        
        print(f"Calculating dominance for Game {game_id}, Play {play_id}, Receiver {target_nfl_id}")
        
        # Calculate frame-by-frame metrics
        frame_metrics = scorer.calculate_frame_dominance(game_id, play_id, target_nfl_id)
        
        # Calculate route-level score
        route_dom = scorer.calculate_route_dominance(frame_metrics, method="weighted_average")
        
        print(f"\nRoute Dominance Score: {route_dom:.3f}")
        print(f"\nFrame-by-frame metrics:")
        print(frame_metrics[["frame_id", "dominance_score", "sep_nearest", 
                           "receiver_speed", "dist_to_ball"]].head(10))
        
        # Visualize
        visualizer = RouteDominanceVisualizer(scorer)
        print("\nCreating visualization...")
        anim, _ = visualizer.visualize_play_dominance(
            game_id, play_id, target_nfl_id, 
            save_path="route_dominance_animation.gif",
            show_animation=True
        )
        
    else:
        print("Targeted receiver not found for this play")

