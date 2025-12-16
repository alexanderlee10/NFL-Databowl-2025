"""
MINIMAL EXAMPLE - Run this to see the visualization immediately!

This creates a simple example visualization you can run right now.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from receiver_dominance_visualization import plot_dual_dominance_graphs

print("="*80)
print("CREATING SAMPLE VISUALIZATION")
print("="*80)

# Create sample data (simulating what your real data would look like)
print("\n1. Creating sample data...")
sample_data = pd.DataFrame({
    'game_id': [1] * 50 + [2] * 50 + [3] * 50,
    'play_id': list(range(101, 151)) + list(range(201, 251)) + list(range(301, 351)),
    'max_dominance': np.random.uniform(0.0, 1.0, 150),  # Random dominance scores
    'is_complete': np.random.choice([0, 1], 150, p=[0.4, 0.6])  # 60% completion rate
})

# Bucket the dominance scores (like the CPP approach)
def bucket_score(val):
    if val < 0.20: return 0.0
    elif val < 0.40: return 0.2
    elif val < 0.6: return 0.4
    elif val < 0.8: return 0.6
    elif val < 1.0: return 0.8
    else: return 1.0

sample_data['dominance_bucket'] = sample_data['max_dominance'].apply(bucket_score)

print(f"   Created {len(sample_data)} sample plays")
print(f"   Dominance range: {sample_data['max_dominance'].min():.2f} - {sample_data['max_dominance'].max():.2f}")

# Create the visualization
print("\n2. Creating dual plot visualization...")
fig = plot_dual_dominance_graphs(
    sample_data,
    dominance_col='max_dominance',
    completion_col='is_complete',
    figsize=(20, 10)
)

print("   ✓ Visualization created!")

# Show it
print("\n3. Displaying plot...")
plt.show()

# Save it
print("\n4. Saving plot...")
plt.savefig('sample_dominance_visualization.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: sample_dominance_visualization.png")

print("\n" + "="*80)
print("DONE! Check the plot that appeared above.")
print("="*80)
print("\nTo use with YOUR data:")
print("  1. Load your training dataframe")
print("  2. Replace 'sample_data' with your dataframe")
print("  3. Make sure you have 'max_dominance' and 'is_complete' columns")
print("  4. Run the same plot_dual_dominance_graphs() function")

