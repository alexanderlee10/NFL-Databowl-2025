"""
Quick script to open the GIF
"""

import os
import sys

gif_path = "dominance_gifs/dominance_game2023090700_play101.gif"

if os.path.exists(gif_path):
    abs_path = os.path.abspath(gif_path)
    print(f"Opening GIF: {abs_path}")
    
    if sys.platform == "win32":
        os.startfile(abs_path)
    elif sys.platform == "darwin":
        os.system(f"open {gif_path}")
    else:
        os.system(f"xdg-open {gif_path}")
    
    print("GIF opened!")
else:
    print(f"GIF not found: {gif_path}")

