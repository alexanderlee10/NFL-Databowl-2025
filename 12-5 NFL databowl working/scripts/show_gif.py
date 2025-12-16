"""
Display the created GIF visualization
"""

import os
import subprocess
import sys

gif_path = "dominance_gifs/dominance_game2023090700_play101.gif"

if os.path.exists(gif_path):
    print("="*80)
    print("GIF FOUND!")
    print("="*80)
    print(f"\nLocation: {os.path.abspath(gif_path)}")
    print(f"Size: {os.path.getsize(gif_path) / 1024 / 1024:.2f} MB")
    print("\nOpening GIF...")
    
    # Try to open with default application
    try:
        if sys.platform == "win32":
            os.startfile(os.path.abspath(gif_path))
        elif sys.platform == "darwin":
            subprocess.run(["open", gif_path])
        else:
            subprocess.run(["xdg-open", gif_path])
        print("\nGIF opened in default viewer!")
    except Exception as e:
        print(f"\nCould not auto-open. Please manually open:")
        print(f"  {os.path.abspath(gif_path)}")
    
    # Also try to display in notebook if available
    try:
        from IPython.display import Image, display
        print("\nDisplaying in notebook...")
        display(Image(gif_path))
    except:
        print("\n(Not in Jupyter notebook - opened in external viewer)")
        
else:
    print(f"GIF not found at: {gif_path}")
    print("\nAvailable files in dominance_gifs/:")
    if os.path.exists("dominance_gifs"):
        for f in os.listdir("dominance_gifs"):
            print(f"  - {f}")
    else:
        print("  (dominance_gifs folder does not exist)")

