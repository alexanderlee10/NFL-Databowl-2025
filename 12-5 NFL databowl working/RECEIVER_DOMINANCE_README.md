# Receiver Dominance (Continuous Receiver Pressure - CRP)

This module implements a receiver dominance metric similar to the **2023 NFL Data Bowl winner's Continuous Pocket Pressure (CPP)** approach, adapted for receiver analysis.

## Overview

The system calculates receiver dominance using:
- **Multivariate normal distributions** to model defender and receiver influence
- **Separation, leverage angle, speed, and defender pressure** as key factors
- **Visualizations** matching the CPP style (contour plots, color schemes, graphs)

## Files

1. **`receiver_dominance_cpp.py`** - Core calculation and field visualization
2. **`receiver_dominance_visualization.py`** - Analysis graphs (completion %, events)
3. **`receiver_dominance_example.py`** - Example usage with your dataframe

## Quick Start

### 1. Calculate Dominance for Your DataFrame

```python
import pandas as pd
from receiver_dominance_example import add_dominance_scores_to_dataframe, calculate_max_dominance_per_play

# Load your training dataframe
training_df = pd.read_csv('route_dominance_training_data.csv')

# Add dominance scores
df_with_dominance = add_dominance_scores_to_dataframe(training_df)

# Calculate max dominance per play
df_with_max = calculate_max_dominance_per_play(df_with_dominance)
```

### 2. Create Visualizations

```python
from receiver_dominance_visualization import plot_dual_dominance_graphs

# Create dual plot (completion % and events vs dominance)
fig = plot_dual_dominance_graphs(
    df_with_max,
    dominance_col='max_dominance',
    completion_col='is_complete'
)
plt.show()
```

### 3. Visualize a Single Play

```python
from receiver_dominance_cpp import calculate_receiver_dominance, visualize_receiver_dominance

# Get a single frame
frame_data = training_df[training_df['play_id'] == 101].iloc[0:1]

# Calculate dominance
dominance = calculate_receiver_dominance(
    frame_data,
    receiver_nfl_id=44930,
    ball_land_x=66.5,
    ball_land_y=41.6
)

# Visualize
fig = visualize_receiver_dominance(
    frame_data,
    receiver_nfl_id=44930,
    ball_land_x=66.5,
    ball_land_y=41.6,
    dominance_score=dominance
)
plt.show()
```

### 4. Full Analysis

```python
from receiver_dominance_example import create_dominance_analysis

# Run complete analysis
create_dominance_analysis(training_df, output_dir='./dominance_results')
```

## How It Works

### Calculation Method (Similar to CPP)

1. **Defender Influence PDF**: Each defender creates a multivariate normal distribution based on:
   - Position and projected movement
   - Speed and direction
   - Distance from receiver
   - Weighted by separation (closer = more influence)

2. **Receiver Advantage PDF**: Receiver creates an influence PDF based on:
   - Separation from nearest defender (30% weight)
   - Leverage angle (25% weight)
   - Speed toward ball (25% weight)
   - Number of defenders nearby (20% weight)

3. **Dominance Ratio**: 
   ```
   dominance = receiver_pdf / (defense_pdf + receiver_pdf)
   ```

4. **Final Score**: Normalized to 0-1 scale (similar to CPP normalization)

### Visualization Style

- **Field Background**: Dark green field with white yard lines (matching CPP)
- **Contour Plot**: Purple colormap showing dominance regions
- **Color Indicator**: 
  - Green: High dominance (≥80%)
  - Yellow: Medium-high (65-80%)
  - Orange: Medium (50-65%)
  - Red: Low dominance (<50%)

## Data Requirements

Your dataframe should have:
- `nfl_id` or `receiver_nfl_id`: Receiver NFL ID
- `receiver_x`, `receiver_y` or `x`, `y`: Receiver coordinates
- `ball_land_x_std`, `ball_land_y_std` or `ball_land_x`, `ball_land_y`: Ball landing position
- `sep_nearest`: Separation to nearest defender
- `leverage_angle`: Leverage angle (if available)
- `receiver_speed`: Receiver speed
- `num_def_within_3`: Number of defenders within 3 yards
- `player_side` or `team`: To identify defenders
- `is_complete`: Completion status (for analysis)

## Key Differences from CPP

| CPP (Pocket Pressure) | CRP (Receiver Dominance) |
|----------------------|-------------------------|
| Focus: QB pocket area | Focus: Receiver area |
| Defense: Pass rushers | Defense: All defenders |
| Offense: Pass blockers | Offense: Receiver advantage factors |
| Pressure = Bad | Dominance = Good |
| Lower score = better | Higher score = better |

## Example Outputs

1. **Field Visualization**: Shows receiver position, defenders, ball landing, and dominance heatmap
2. **Completion % Graph**: Completion rate vs maximum dominance (similar to CPP)
3. **Events Graph**: Number of incompletions/events vs maximum dominance
4. **Dual Plot**: Both graphs side-by-side (matching CPP style)

## References

Based on the 2023 NFL Data Bowl winning submission's Continuous Pocket Pressure (CPP) methodology:
- Uses multivariate normal distributions for player influence
- Similar normalization and visualization techniques
- Matching color schemes and plot styles

