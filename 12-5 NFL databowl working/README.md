# NFL Route Dominance Analysis - Project Structure

This project analyzes route dominance in NFL plays using tracking data. The codebase has been organized for clarity and maintainability.

## 📁 Folder Structure

```
12-5 NFL databowl working/
├── src/                          # Core Python modules (imported by notebook)
│   ├── __init__.py              # Package initialization
│   ├── route_dominance_scoring.py      # Main scoring module
│   ├── interactive_route_dominance.py  # Interactive viewer
│   └── create_dominance_gif.py         # GIF creation module
│
├── scripts/                      # Utility and helper scripts
│   ├── create_gif_with_all_players.py
│   ├── display_gif_notebook.py
│   ├── interactive_play_viewer.py
│   ├── launch_play_viewer.py
│   ├── run_create_gif.py
│   └── ... (other utility scripts)
│
├── outputs/                      # Generated files (created by notebook)
│   ├── dominance_gifs/          # Animated GIFs of plays
│   ├── route_dominance_training_data.csv  # Training dataset
│   └── *.png                    # Visualization images
│
├── data/                        # Input data (in parent directory)
│   ├── input_2023_w*.csv       # Pre-throw tracking data
│   ├── output_2023_w*.csv      # Post-throw tracking data
│   └── Supplementary.csv       # Play context data
│
├── 12-5 NFL databowl working notebook.ipynb  # Main analysis notebook
│
└── *.md                         # Documentation files
```

## 🚀 Getting Started

### Running the Notebook

1. **Open the notebook**: `12-5 NFL databowl working notebook.ipynb`

2. **The notebook automatically imports from `src/`**:
   ```python
   from src.route_dominance_scoring import RouteDominanceScorer
   from src.interactive_route_dominance import InteractiveRouteDominanceViewer
   from src.create_dominance_gif import create_gif_for_play
   ```

3. **Data paths**: The notebook expects data files in `../data/` (parent directory)

4. **Output files**: All generated files are saved to `outputs/`:
   - Training data: `outputs/route_dominance_training_data.csv`
   - GIFs: `outputs/dominance_gifs/`
   - Images: `outputs/*.png`

### Using the Core Modules

All core functionality is in the `src/` folder:

- **`RouteDominanceScorer`**: Calculate frame-by-frame and route-level dominance scores
- **`InteractiveRouteDominanceViewer`**: Interactive frame-by-frame visualization
- **`create_gif_for_play()`**: Generate animated GIFs of plays

### Running Utility Scripts

Scripts in the `scripts/` folder are standalone utilities. They automatically add the parent directory to the Python path to import from `src/`.

Example:
```bash
python scripts/run_create_gif.py
```

## 📝 Key Files

### Core Modules (`src/`)

- **`route_dominance_scoring.py`**: Main module containing the `RouteDominanceScorer` class
  - Calculates dominance metrics frame-by-frame
  - Aggregates route-level scores
  - Handles coordinate standardization

- **`interactive_route_dominance.py`**: Interactive visualization tool
  - Navigate frames with arrow keys
  - View dominance metrics in real-time

- **`create_dominance_gif.py`**: GIF generation
  - Creates animated visualizations of plays
  - Shows receiver dominance evolving over time

### Notebook

- **`12-5 NFL databowl working notebook.ipynb`**: Main analysis notebook
  - Loads and processes data
  - Calculates dominance metrics
  - Creates training datasets
  - Generates visualizations

## 🔧 Import Structure

### In the Notebook

```python
from src.route_dominance_scoring import RouteDominanceScorer
from src.interactive_route_dominance import InteractiveRouteDominanceViewer
from src.create_dominance_gif import create_gif_for_play
```

### In Scripts

Scripts automatically add the parent directory to the path:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.route_dominance_scoring import RouteDominanceScorer
```

## 📊 Data Flow

1. **Input**: Data files in `../data/` (input, output, supplementary CSVs)
2. **Processing**: Notebook uses modules from `src/` to calculate metrics
3. **Output**: Generated files saved to `outputs/`:
   - Training data CSV
   - Animated GIFs
   - Visualization images

## 🎯 Quick Reference

- **Core modules**: `src/` folder
- **Utilities**: `scripts/` folder  
- **Generated files**: `outputs/` folder
- **Input data**: `../data/` folder (parent directory)
- **Main notebook**: `12-5 NFL databowl working notebook.ipynb`

## 📚 Documentation

Additional documentation files:
- `DOMINANCE_EXPLANATION.md`: Explanation of dominance metrics
- `FORMULAS.md`: Mathematical formulas used
- `HOW_TO_RUN_VIEWER.md`: How to use the interactive viewer
- `QUICK_START_VISUALIZATION.md`: Quick start guide for visualizations
- `RECEIVER_DOMINANCE_README.md`: Detailed receiver dominance documentation
- `VIEW_GIF.md`: How to view generated GIFs
