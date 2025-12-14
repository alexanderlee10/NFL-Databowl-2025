"""
Receiver Dominance Visualization
Similar to cpp_graphs.py from the 2023 Data Bowl winner

Creates graphs showing:
- Completion % vs Maximum Dominance
- Pressure events vs Maximum Dominance
- Similar styling to CPP visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


def bucket_dominance_score(val: float) -> float:
    """
    Group dominance scores into bucket intervals (similar to takeFloor in CPP)
    
    Args:
        val: Dominance score (0-1)
    
    Returns:
        Bucketed value
    """
    if val < 0.20:
        return 0.0
    elif val < 0.40:
        return 0.2
    elif val < 0.6:
        return 0.4
    elif val < 0.8:
        return 0.6
    elif val < 1.0:
        return 0.8
    elif val == 1.0:
        return 1.0
    return 0.0


def plot_completion_vs_dominance(
    df: pd.DataFrame,
    dominance_col: str = 'max_dominance',
    completion_col: str = 'is_complete',
    figsize: Tuple[int, int] = (20, 10)
) -> plt.Figure:
    """
    Plot completion percentage vs maximum dominance during play
    
    Similar to CPP graph showing completion % vs maximum CPP
    
    Args:
        df: DataFrame with dominance scores and completion status
        dominance_col: Column name for maximum dominance score
        completion_col: Column name for completion (1 = complete, 0 = incomplete)
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure
    """
    # Create bucketed dominance scores
    df_plot = df.copy()
    df_plot['dominance_bucket'] = df_plot[dominance_col].apply(bucket_dominance_score)
    
    # Filter to passing plays only
    passing_plays = df_plot[
        (df_plot[completion_col] == 1) | (df_plot[completion_col] == 0)
    ].copy()
    
    # Calculate completion percentage by bucket
    completion_pct = passing_plays.groupby('dominance_bucket')[completion_col].mean()
    
    # Create plot with CPP-style
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    ax.plot(completion_pct.index, completion_pct.values, 
           linewidth=3, marker='o', markersize=10, color='#2E86AB')
    ax.set_title('Completion % vs Maximum Receiver Dominance During Play', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Maximum Dominance During Play', fontsize=14, fontweight='bold')
    ax.set_ylabel('Completion %', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    return fig


def plot_pressure_events_vs_dominance(
    df: pd.DataFrame,
    dominance_col: str = 'max_dominance',
    figsize: Tuple[int, int] = (20, 10)
) -> plt.Figure:
    """
    Plot number of pressure events vs maximum dominance
    
    Similar to CPP graph showing sacks, hits, hurries vs maximum CPP
    Adapted for receiver context: could show drops, incompletions, etc.
    
    Args:
        df: DataFrame with dominance scores
        dominance_col: Column name for maximum dominance score
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure
    """
    # Create bucketed dominance scores
    df_plot = df.copy()
    df_plot['dominance_bucket'] = df_plot[dominance_col].apply(bucket_dominance_score)
    
    # Count events by bucket
    # For receiver context, we might track:
    # - Drops (if available)
    # - Incompletions
    # - Pass breakups
    
    # Example: count incompletions
    if 'is_complete' in df_plot.columns:
        incompletions = df_plot[df_plot['is_complete'] == 0].groupby('dominance_bucket').size()
    else:
        incompletions = pd.Series(dtype=int)
    
    # Create plot
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    if len(incompletions) > 0:
        ax.plot(incompletions.index, incompletions.values,
               linewidth=3, marker='o', markersize=10, 
               label='Incompletions', color='#A23B72')
    
    ax.legend(title='Event Type', fontsize=12, title_fontsize=14)
    ax.set_title('Number of Events vs Maximum Receiver Dominance During Play',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Maximum Dominance During Play', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Events', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_dual_dominance_graphs(
    df: pd.DataFrame,
    dominance_col: str = 'max_dominance',
    completion_col: str = 'is_complete',
    figsize: Tuple[int, int] = (20, 10)
) -> plt.Figure:
    """
    Create dual plot similar to CPP graphs: completion % and events vs dominance
    
    Args:
        df: DataFrame with dominance scores and completion data
        dominance_col: Column name for maximum dominance score
        completion_col: Column name for completion status
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure with two subplots
    """
    # Create bucketed dominance scores
    df_plot = df.copy()
    df_plot['dominance_bucket'] = df_plot[dominance_col].apply(bucket_dominance_score)
    
    # Filter to passing plays
    passing_plays = df_plot[
        (df_plot[completion_col] == 1) | (df_plot[completion_col] == 0)
    ].copy()
    
    # Calculate metrics
    completion_pct = passing_plays.groupby('dominance_bucket')[completion_col].mean()
    
    if 'is_complete' in df_plot.columns:
        incompletions = df_plot[df_plot['is_complete'] == 0].groupby('dominance_bucket').size()
    else:
        incompletions = pd.Series(dtype=int)
    
    # Create dual plot (similar to CPP)
    plt.style.use('fivethirtyeight')
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('#FFFFFF')
    axs[0].set_facecolor('#FFFFFF')
    axs[1].set_facecolor('#FFFFFF')
    
    # Left plot: Completion %
    axs[0].plot(completion_pct.index, completion_pct.values,
               linewidth=3, marker='o', markersize=10, color='#2E86AB')
    axs[0].set_title('Completion % vs Maximum Dominance During Play',
                    fontsize=16, fontweight='bold', pad=15)
    axs[0].set_xlabel('Maximum Dominance During Play', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Completion %', fontsize=12, fontweight='bold')
    axs[0].grid(True, alpha=0.3, linestyle='--')
    axs[0].set_ylim([0, 1])
    axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Right plot: Events
    if len(incompletions) > 0:
        axs[1].plot(incompletions.index, incompletions.values,
                   linewidth=3, marker='o', markersize=10,
                   label='Incompletions', color='#A23B72')
    axs[1].legend(title='Event Type', fontsize=11, title_fontsize=12)
    axs[1].set_title('Number of Events vs Maximum Dominance During Play',
                    fontsize=16, fontweight='bold', pad=15)
    axs[1].set_xlabel('Maximum Dominance During Play', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Number of Events', fontsize=12, fontweight='bold')
    axs[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

