# ✅ Setup Checklist - Everything You Need

## 📋 Quick Checklist

### 1. Folder Structure ✅
- [ ] `src/` folder exists
  - [ ] `src/route_dominance_scoring.py` exists
  - [ ] `src/interactive_route_dominance.py` exists
  - [ ] `src/create_dominance_gif.py` exists
  - [ ] `src/__init__.py` exists
- [ ] `12-5 NFL databowl working notebook.ipynb` exists
- [ ] `../data/` folder exists (parent directory)
  - [ ] `../data/Supplementary.csv` exists
  - [ ] `../data/input_2023_w*.csv` files exist (at least one week)
  - [ ] `../data/output_2023_w*.csv` files exist (at least one week)

### 2. Python Packages 📦
Install with: `pip install -r requirements.txt`

- [ ] pandas
- [ ] numpy
- [ ] matplotlib
- [ ] scipy
- [ ] imageio
- [ ] seaborn (optional but recommended)
- [ ] tqdm (optional but recommended)

### 3. Ready to Run! 🚀

## 🎯 What You Actually Need (Minimal)

**Absolute minimum:**
1. ✅ `src/` folder with 3 Python files
2. ✅ `12-5 NFL databowl working notebook.ipynb`
3. ✅ `../data/` folder with CSV files
4. ✅ Python with pandas, numpy, matplotlib, scipy, imageio

**That's it!** Everything else is optional or auto-generated.

## 📂 File Structure Summary

```
Your Project
│
├── 12-5 NFL databowl working/          ← You are here
│   ├── src/                            ← REQUIRED: Core modules
│   │   ├── route_dominance_scoring.py
│   │   ├── interactive_route_dominance.py
│   │   └── create_dominance_gif.py
│   │
│   ├── scripts/                        ← OPTIONAL: Utilities
│   │   └── (helper scripts)
│   │
│   ├── outputs/                        ← AUTO-GENERATED: Results
│   │   ├── dominance_gifs/
│   │   └── route_dominance_training_data.csv
│   │
│   └── 12-5 NFL databowl working notebook.ipynb  ← REQUIRED: Main notebook
│
└── data/                               ← REQUIRED: Input data
    ├── input_2023_w*.csv
    ├── output_2023_w*.csv
    └── Supplementary.csv
```

## 🔍 Verify Setup

Run this in a Python cell to check everything:

```python
import os
import sys

print("Checking setup...\n")

# Check core modules
checks = {
    "src/route_dominance_scoring.py": os.path.exists("src/route_dominance_scoring.py"),
    "src/interactive_route_dominance.py": os.path.exists("src/interactive_route_dominance.py"),
    "src/create_dominance_gif.py": os.path.exists("src/create_dominance_gif.py"),
    "Notebook": os.path.exists("12-5 NFL databowl working notebook.ipynb"),
    "Data folder": os.path.exists("../data/Supplementary.csv"),
}

for item, exists in checks.items():
    status = "✅" if exists else "❌"
    print(f"{status} {item}")

# Check Python packages
print("\nChecking Python packages...")
try:
    import pandas
    import numpy
    import matplotlib
    import scipy
    import imageio
    print("✅ All required packages installed")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("   Run: pip install -r requirements.txt")
```

## 🚀 You're Ready When...

✅ All files in `src/` exist  
✅ Notebook exists  
✅ Data folder exists with CSV files  
✅ Python packages installed  
✅ No errors when running the verification script above

**Then just open the notebook and run it!** 🎉
