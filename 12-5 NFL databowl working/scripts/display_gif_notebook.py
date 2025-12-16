"""
Display GIF in Jupyter Notebook

Copy this code into a notebook cell to display the GIF:
"""

# Display the GIF in Jupyter Notebook
from IPython.display import Image, display
import os

gif_path = "dominance_gifs/dominance_game2023090700_play101.gif"

if os.path.exists(gif_path):
    print("Displaying Receiver Dominance GIF:")
    print(f"Game 2023090700, Play 101 - 47 frames")
    print("\nThe GIF shows:")
    print("  - Receiver (green star) moving through the play")
    print("  - Defender (blue circle) tracking")
    print("  - Purple contour plot showing dominance regions")
    print("  - Dominance score updating frame-by-frame")
    print("  - Separation circle and stats")
    print()
    display(Image(gif_path))
else:
    print(f"GIF not found at: {gif_path}")
    print("\nTo create a GIF, run:")
    print("  from src.create_dominance_gif import create_gif_for_play")
    print("  gif_path = create_gif_for_play(training_df, 2023090700, 101)")

