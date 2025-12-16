# 🚀 Quick Start Guide - Everything You Need

## ✅ Required Files & Folders

### 1. **Core Project Structure** (Already organized!)
```
12-5 NFL databowl working/
├── src/                          # ✅ Core modules (REQUIRED)
│   ├── __init__.py
│   ├── route_dominance_scoring.py
│   ├── interactive_route_dominance.py
│   └── create_dominance_gif.py
│
├── scripts/                      # ⚙️ Utility scripts (optional)
│   └── (various helper scripts)
│
├── outputs/                      # 📊 Generated files (created automatically)
│   ├── dominance_gifs/
│   └── route_dominance_training_data.csv
│
└── 12-5 NFL databowl working notebook.ipynb  # 📓 Main notebook (REQUIRED)
```

### 2. **Data Files** (Required - in parent directory)
```
../data/
├── input_2023_w*.csv            # Pre-throw tracking data (weeks 1-18)
├── output_2023_w*.csv           # Post-throw tracking data (weeks 1-18)
└── Supplementary.csv            # Play context data
```

## 📦 Required Python Packages

Install these packages if you don't have them:

```bash
pip install pandas numpy matplotlib seaborn scipy imageio tqdm
```

**Core dependencies:**
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `scipy` - Statistical functions
- `imageio` - GIF creation
- `seaborn` - Enhanced plotting (optional but recommended)
- `tqdm` - Progress bars (optional but recommended)

## 🎯 What You Need to Run

### Minimum Requirements:
1. ✅ **Notebook**: `12-5 NFL databowl working notebook.ipynb`
2. ✅ **Core modules**: `src/` folder with all 3 Python files
3. ✅ **Data files**: `../data/` folder with CSV files
4. ✅ **Python packages**: pandas, numpy, matplotlib, scipy, imageio

### Optional (but useful):
- `scripts/` folder - Utility scripts for specific tasks
- `outputs/` folder - Will be created automatically when you run the notebook

## 📝 Step-by-Step: Getting Started

### Step 1: Verify Your Setup
```python
# Run this in a notebook cell to check everything is ready:
import os
import sys

# Check core modules exist
assert os.path.exists('src/route_dominance_scoring.py'), "Missing src/route_dominance_scoring.py"
assert os.path.exists('src/interactive_route_dominance.py'), "Missing src/interactive_route_dominance.py"
assert os.path.exists('src/create_dominance_gif.py'), "Missing src/create_dominance_gif.py"

# Check data folder exists
assert os.path.exists('../data/Supplementary.csv'), "Missing ../data/Supplementary.csv"

print("✅ All required files found!")
```

### Step 2: Import and Use
```python
# In your notebook, the first cell should have:
from src.route_dominance_scoring import RouteDominanceScorer
from src.interactive_route_dominance import InteractiveRouteDominanceViewer
from src.create_dominance_gif import create_gif_for_play
```

### Step 3: Load Data
```python
# Load your data (paths are already set in the notebook)
input_df = load_all_input_files(weeks=[1])  # or weeks=None for all weeks
output_df = load_all_output_files(weeks=[1])
supp_df = pd.read_csv('../data/Supplementary.csv')
```

### Step 4: Run Analysis
```python
# Initialize scorer
scorer = RouteDominanceScorer(input_df, output_df, supp_df)

# Calculate dominance metrics
frame_metrics = scorer.calculate_frame_dominance(game_id, play_id, receiver_nfl_id)

# Create training dataset
training_df = create_training_dataframe(...)  # See notebook for details
```

## 📂 File Locations Reference

| What You Need | Where It Is |
|--------------|------------|
| **Main notebook** | `12-5 NFL databowl working notebook.ipynb` |
| **Core scoring module** | `src/route_dominance_scoring.py` |
| **Interactive viewer** | `src/interactive_route_dominance.py` |
| **GIF creator** | `src/create_dominance_gif.py` |
| **Input data** | `../data/input_2023_w*.csv` |
| **Output data** | `../data/output_2023_w*.csv` |
| **Supplementary data** | `../data/Supplementary.csv` |
| **Training data (generated)** | `outputs/route_dominance_training_data.csv` |
| **GIFs (generated)** | `outputs/dominance_gifs/` |

## 🔍 Quick Checklist

Before running the notebook, make sure:

- [ ] `src/` folder exists with all 3 Python files
- [ ] `../data/` folder exists with CSV files
- [ ] Python packages installed (pandas, numpy, matplotlib, scipy, imageio)
- [ ] Notebook is in `12-5 NFL databowl working/` directory
- [ ] You're running the notebook from the correct directory

## 💡 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution**: Make sure you're running the notebook from the `12-5 NFL databowl working/` directory, not from a subdirectory.

### Issue: "FileNotFoundError: ../data/Supplementary.csv"
**Solution**: The data folder should be in the parent directory. Check that `../data/` exists relative to your notebook location.

### Issue: "ImportError: cannot import name 'RouteDominanceScorer'"
**Solution**: Verify `src/route_dominance_scoring.py` exists and contains the `RouteDominanceScorer` class.

## 🎓 What Each Component Does

1. **`route_dominance_scoring.py`**
   - Calculates dominance metrics frame-by-frame
   - Aggregates route-level scores
   - Main analysis engine

2. **`interactive_route_dominance.py`**
   - Interactive visualization tool
   - Navigate frames with arrow keys
   - View metrics in real-time

3. **`create_dominance_gif.py`**
   - Creates animated GIFs of plays
   - Shows dominance evolving over time
   - Saves to `outputs/dominance_gifs/`

4. **Notebook**
   - Orchestrates the analysis
   - Loads data and processes plays
   - Creates training datasets
   - Generates visualizations

## 📚 Next Steps

1. Open the notebook: `12-5 NFL databowl working notebook.ipynb`
2. Run the first cell to import libraries
3. Configure which weeks to process
4. Run cells sequentially to create your training dataset
5. Generate visualizations and GIFs as needed

That's everything you need! 🎉
