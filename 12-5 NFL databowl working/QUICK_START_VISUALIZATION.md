# Quick Start: Running Receiver Dominance Visualizations

## Option 1: Run the Complete Script (Easiest)

```bash
python run_receiver_dominance_viz.py
```

This will:
- Load your training dataframe
- Calculate dominance scores
- Create all visualizations
- Save PNG files
- Display the plots

## Option 2: Run Step-by-Step in Python/Jupyter

### Step 1: Import and Load Data

```python
import pandas as pd
import matplotlib.pyplot as plt
from receiver_dominance_visualization import plot_dual_dominance_graphs
from receiver_dominance_example import add_dominance_scores_to_dataframe, calculate_max_dominance_per_play

# Load your dataframe (or use existing variable)
training_df = pd.read_csv('route_dominance_training_data.csv')
# OR if you already have it loaded:
# training_df = your_existing_dataframe
```

### Step 2: Calculate Dominance Scores

```python
# Add dominance scores to your dataframe
training_df = add_dominance_scores_to_dataframe(training_df)

# Calculate max dominance per play
training_df = calculate_max_dominance_per_play(training_df)

# Get play-level data
play_level_df = training_df.groupby(['game_id', 'play_id']).first().reset_index()
```

### Step 3: Create Visualizations

```python
# Create the dual plot (completion % and events vs dominance)
fig = plot_dual_dominance_graphs(
    play_level_df,
    dominance_col='max_dominance',
    completion_col='is_complete'
)

# Show the plot
plt.show()

# Save it
plt.savefig('my_dominance_graph.png', dpi=300, bbox_inches='tight')
```

### Step 4: View Individual Graphs

```python
from receiver_dominance_visualization import plot_completion_vs_dominance, plot_pressure_events_vs_dominance

# Just completion %
fig1 = plot_completion_vs_dominance(
    play_level_df,
    dominance_col='max_dominance',
    completion_col='is_complete'
)
plt.show()

# Just events
fig2 = plot_pressure_events_vs_dominance(
    play_level_df,
    dominance_col='max_dominance'
)
plt.show()
```

## Option 3: Quick Test with Sample Data

If you want to test quickly without calculating all dominance scores:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from receiver_dominance_visualization import plot_dual_dominance_graphs

# Create sample data for testing
sample_data = pd.DataFrame({
    'game_id': [1, 1, 2, 2, 3, 3],
    'play_id': [101, 102, 201, 202, 301, 302],
    'max_dominance': [0.2, 0.4, 0.6, 0.8, 0.3, 0.7],
    'is_complete': [0, 1, 1, 1, 0, 1]
})

# Create visualization
fig = plot_dual_dominance_graphs(
    sample_data,
    dominance_col='max_dominance',
    completion_col='is_complete'
)
plt.show()
```

## Troubleshooting

### If you get "receiver_dominance column not found":
- Run `add_dominance_scores_to_dataframe()` first
- This calculates dominance for all frames (may take a few minutes)

### If you get "is_complete column not found":
- The completion graphs won't work, but events graph will
- Or add a dummy column: `training_df['is_complete'] = 1` for testing

### If visualizations don't show:
- Make sure you have matplotlib installed: `pip install matplotlib`
- In Jupyter, use `%matplotlib inline` at the top
- Try `plt.show()` after creating the figure

## Expected Output

You should see:
1. **Completion % vs Dominance** - Line graph showing completion rate at different dominance levels
2. **Events vs Dominance** - Line graph showing number of events at different dominance levels
3. **Dual Plot** - Both graphs side-by-side (matching CPP style)

All graphs use the same styling as the 2023 Data Bowl winner's CPP visualizations!

