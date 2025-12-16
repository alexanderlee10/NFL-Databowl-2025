# How to Run the Interactive Play Viewer

## Quick Start

Simply run:
```bash
python launch_play_viewer.py
```

## What It Does

The interactive viewer lets you navigate through all plays in your training dataset using arrow keys. Each play shows:
- All players on the field (standardized coordinates)
- Field visualization with dominance contour
- Play information (Game ID, Play ID, Dominance score)

## Controls

Once the viewer window opens:
- **← Left Arrow** or **'a'**: Go to previous play
- **→ Right Arrow** or **'d'**: Go to next play
- **'g'**: Generate GIF for current play
- **'q'** or **Escape**: Quit

⚠️ **Important**: Click on the plot window first to give it focus, then use arrow keys!

## Alternative: Use in Your Code

If you want to customize it or use it in a notebook:

```python
import pandas as pd
from route_dominance_scoring import RouteDominanceScorer
from interactive_play_viewer import launch_interactive_viewer

# Load your data
input_df = pd.read_csv('../data/input_2023_w01.csv')
output_df = pd.read_csv('../data/output_2023_w01.csv')
supp_df = pd.read_csv('../data/Supplementary.csv')
training_df = pd.read_csv('route_dominance_training_data.csv')

# Initialize scorer
scorer = RouteDominanceScorer(input_df, output_df, supp_df)

# Launch viewer
viewer = launch_interactive_viewer(training_df, scorer)

# Optional: Start at a specific play
# viewer = launch_interactive_viewer(training_df, scorer, 
#                                    start_game_id=2023090700, 
#                                    start_play_id=101)
```

## Troubleshooting

**Arrow keys don't work?**
- Make sure you clicked on the plot window to give it focus
- Try clicking on the window again

**Window doesn't appear?**
- Make sure matplotlib backend supports interactive mode
- Try running: `matplotlib.use('TkAgg')` before importing

**Data not found?**
- Make sure you're in the correct directory
- Check that all data files exist:
  - `../data/input_2023_w01.csv`
  - `../data/output_2023_w01.csv`
  - `../data/Supplementary.csv`
  - `route_dominance_training_data.csv`

