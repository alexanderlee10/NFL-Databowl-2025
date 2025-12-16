# View Your Receiver Dominance GIF

## GIF Created Successfully!

**Location:** `dominance_gifs/dominance_game2023090700_play101.gif`  
**Size:** 1.81 MB  
**Frames:** 47 frames  
**FPS:** 5 (smooth animation)

## How to View

### Option 1: Already Opened
The GIF should have automatically opened in your default image viewer!

### Option 2: In Jupyter Notebook
Copy this into a notebook cell:

```python
from IPython.display import Image, display

gif_path = "dominance_gifs/dominance_game2023090700_play101.gif"
display(Image(gif_path))
```

### Option 3: Manual Open
Navigate to:
```
12-5 NFL databowl working/dominance_gifs/dominance_game2023090700_play101.gif
```
And double-click to open in your default viewer.

## What You'll See

The GIF shows the play frame-by-frame with:

1. **Dark Green Field** - Football field with white yard lines (CPP style)
2. **Green Star** - Target receiver moving through the play
3. **Blue Circle** - Nearest defender tracking the receiver
4. **Purple Contour Plot** - Dominance regions (updates each frame)
5. **Yellow X** - Ball landing position
6. **Dominance % Indicator** - Updates each frame (top-left corner)
7. **Separation Circle** - Cyan circle showing separation distance
8. **Stats Box** - Shows separation, speed, leverage angle

## Features Used from Your Dataframe

The visualization uses these features from your training dataframe:

- `receiver_x`, `receiver_y` - Receiver position
- `nearest_defender_x`, `nearest_defender_y` - Defender position
- `sep_nearest` - Separation distance
- `receiver_speed` - Receiver speed
- `leverage_angle` - Leverage angle
- `ball_land_x_std`, `ball_land_y_std` - Ball landing
- `receiver_dominance` - Dominance score (if calculated)
- `continuous_frame` - Frame number

## Create More GIFs

```python
from create_dominance_gif import create_gif_for_play

# Create GIF for any play
gif_path = create_gif_for_play(
    training_df,
    game_id=2023090700,
    play_id=101,
    fps=5  # 5 = smooth, 10 = faster
)
```

