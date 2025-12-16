"""
FIXED Field Visualization - Ensures everything shows up properly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import multivariate_normal
import math

def draw_field_with_players():
    """Draw field with players - ensuring everything is visible"""
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(10.66, 24))
    
    # Field background
    rect = patches.Rectangle((0, 0), 53.3, 120, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)
    ax.add_patch(rect)
    
    # Field lines
    plt.plot([0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             [10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             color='white', linewidth=2)
    
    # End zones
    home_endzone = patches.Rectangle((0, 0), 53.3, 10,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='purple',
                                    alpha=0.2,
                                    zorder=10)
    away_endzone = patches.Rectangle((0, 110), 53.3, 10,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='purple',
                                    alpha=0.2,
                                    zorder=10)
    ax.add_patch(home_endzone)
    ax.add_patch(away_endzone)
    
    # Yard markers
    for y in range(20, 110, 10):
        numb = y
        if y > 50:
            numb = 120 - y
        plt.text(5, y-1.5, str(numb - 10),
                 horizontalalignment='center',
                 fontsize=20,
                 color='white', rotation=270, fontweight='bold')
        plt.text(53.3 - 5, y - 0.95, str(numb - 10),
                 horizontalalignment='center',
                 fontsize=20,
                 color='white', rotation=90, fontweight='bold')
    
    # Hash lines
    for y in range(11, 110):
        ax.plot([0.7, 0.4], [y, y], color='white', linewidth=0.5)
        ax.plot([53.0, 52.5], [y, y], color='white', linewidth=0.5)
        ax.plot([22.91, 23.57], [y, y], color='white', linewidth=0.5)
        ax.plot([29.73, 30.39], [y, y], color='white', linewidth=0.5)
    
    # Sample player positions
    receiver_x, receiver_y = 41.0, 41.1
    defender_x, defender_y = 44.4, 41.6
    ball_x, ball_y = 66.5, 41.6
    
    # Create contour plot (dominance regions)
    x, y = np.mgrid[0:53.3:0.5, 0:120:0.5]
    
    # Create a simple dominance PDF centered on receiver
    receiver_pdf = multivariate_normal([receiver_y, receiver_x], [[8, 0], [0, 8]]).pdf(np.dstack((x, y)))
    
    # Create defender pressure PDF
    defender_pdf = multivariate_normal([defender_y, defender_x], [[6, 0], [0, 6]]).pdf(np.dstack((x, y)))
    
    # Calculate dominance (receiver advantage)
    total_pdf = receiver_pdf + defender_pdf + 1e-10
    dominance_pdf = receiver_pdf / total_pdf
    
    # Draw contour plot
    contour = ax.contourf(x, y, dominance_pdf, cmap='Purples', alpha=0.7, levels=15)
    
    # Plot receiver (green star - like QB in CPP)
    ax.scatter(receiver_y, receiver_x, color='limegreen', s=500, 
              marker='*', edgecolors='black', linewidths=3, zorder=10)
    ax.annotate('WR', (receiver_y, receiver_x), 
               xytext=(receiver_y-1, receiver_x-1),
               color='white', fontweight='bold', fontsize=16)
    
    # Plot defender (blue circle)
    ax.scatter(defender_y, defender_x, color='blue', s=400, 
              edgecolors='white', linewidths=2, zorder=9)
    ax.annotate('CB', (defender_y, defender_x),
               xytext=(defender_y-1, defender_x-1),
               color='white', fontsize=14)
    
    # Plot ball landing (yellow X)
    ax.scatter(ball_y, ball_x, color='yellow', s=600,
              marker='X', edgecolors='black', linewidths=3, zorder=8)
    
    # Add more players for realism
    # QB
    ax.scatter(26.6, 10.0, color='red', s=400, edgecolors='white', linewidths=2, zorder=9)
    ax.annotate('QB', (26.6, 10.0), xytext=(26.6-1, 10.0-1), color='white', fontsize=12)
    
    # Another defender
    ax.scatter(38.0, 45.0, color='blue', s=400, edgecolors='white', linewidths=2, zorder=9)
    ax.annotate('S', (38.0, 45.0), xytext=(38.0-1, 45.0-1), color='white', fontsize=12)
    
    # Dominance indicator
    dominance_score = 0.75
    dominance_percent = int(dominance_score * 100)
    
    if dominance_score >= 0.8:
        indicator_color = '#00FF00'
    elif dominance_score >= 0.65:
        indicator_color = '#FFFF00'
    elif dominance_score >= 0.5:
        indicator_color = '#FFA500'
    else:
        indicator_color = '#FF0000'
    
    ax.text(2, 112, f"Dominance: {dominance_percent}%",
           fontsize=25, fontweight='bold', color='white',
           bbox=dict(boxstyle='round,pad=1', facecolor=indicator_color, 
                    alpha=0.8, edgecolor='black', linewidth=2))
    
    # Set axis limits and turn off axis
    ax.set_xlim(0, 53.3)
    ax.set_ylim(0, 120)
    ax.set_aspect('equal')
    plt.axis('off')
    
    return fig

# Create and show
print("Creating field visualization...")
fig = draw_field_with_players()
plt.savefig('field_viz_fixed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: field_viz_fixed.png")
print("✓ Displaying...")
plt.show()
print("\nYou should see:")
print("  - Dark green field")
print("  - White yard lines")
print("  - Green star = Receiver")
print("  - Blue circles = Defenders")
print("  - Red circle = QB")
print("  - Yellow X = Ball")
print("  - Purple contour plot = Dominance regions")


