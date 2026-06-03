import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
from matplotlib.colors import ListedColormap


# --- Configuration ---
FILE_PATH = sys.argv[1]# '/Users/absera/Capstone/swarm_two/src/logs/camera_captures.txt'
ENV_W, ENV_H = 2500, 2500
CROP_W, CROP_H = 200, 200
HALF_W, HALF_H = CROP_W // 2, CROP_H // 2

# Paper-friendly palette (Matplotlib defaults)
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"

def apply_crop(grid, center_x, center_y):
    """Updates the occupancy grid with a 200x200 crop based on the center."""
    start_x = max(0, int(center_x) - HALF_W)
    end_x = min(ENV_W, int(center_x) + HALF_W)
    start_y = max(0, int(center_y) - HALF_H)
    end_y = min(ENV_H, int(center_y) + HALF_H)
    
    grid[start_y:end_y, start_x:end_x] = 1
    return grid

def analyze_swarm_data(filepath):
    # 1. Load the data
    # Assuming the format is: agent_id, x, y
    df = pd.read_csv(filepath, names=['agent_id', 'x', 'y'], header=None)
    
    agents = df['agent_id'].unique()
    num_events = len(df)
    
    # Initialize knowledge grids (0 = unseen, 1 = seen)
    # Communicating: One shared grid for the whole swarm
    shared_knowledge_grid = np.zeros((ENV_H, ENV_W), dtype=np.int8)
    
    # Non-Communicating: Individual grids for each agent
    individual_grids = {agent: np.zeros((ENV_H, ENV_W), dtype=np.int8) for agent in agents}
    
    # Tracking metrics over time (assuming each row is a time step/event)
    history = {
        'event_tick': [],
        'swarm_coverage_pct': [],
        'avg_individual_coverage_pct': []
    }
    
    total_pixels = ENV_W * ENV_H

    print(f"Processing {num_events} capture events...")

    # 2. Process data step-by-step to build the time-series
    for index, row in df.iterrows():
        agent = row['agent_id']
        cx, cy = row['x'], row['y']
        
        # Update shared knowledge (Communicating)
        apply_crop(shared_knowledge_grid, cx, cy)
        
        # Update individual knowledge (Non-Communicating)
        apply_crop(individual_grids[agent], cx, cy)
        
        # Calculate current percentages
        swarm_pct = (np.sum(shared_knowledge_grid) / total_pixels) * 100
        
        indiv_pcts = [(np.sum(grid) / total_pixels) * 100 for grid in individual_grids.values()]
        avg_indiv_pct = np.mean(indiv_pcts)
        
        history['event_tick'].append(index)
        history['swarm_coverage_pct'].append(swarm_pct)
        history['avg_individual_coverage_pct'].append(avg_indiv_pct)

    # 3. Final Statistics Printout
    print("-" * 30)
    print("FINAL COVERAGE STATISTICS")
    print("-" * 30)
    print(f"Communicating Swarm Knowledge: {history['swarm_coverage_pct'][-1]:.2f}% of the map.")
    
    print("\nNon-Communicating Individual Knowledge:")
    for agent, grid in individual_grids.items():
        pct = (np.sum(grid) / total_pixels) * 100
        print(f"  - Agent {agent}: {pct:.2f}%")
    print(f"  - Average Individual: {history['avg_individual_coverage_pct'][-1]:.2f}%")
    print("-" * 30)

    # 4. Generate Visualizations
    generate_plots(history, shared_knowledge_grid, individual_grids, agents)

def generate_plots(history, shared_grid, indiv_grids, agents):
    plt.style.use('default')
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    
    # --- Plot 1: Knowledge Accumulation Over Time ---
    fig1, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        history['event_tick'],
        history['swarm_coverage_pct'],
        label='Communicating Swarm (Shared Map)',
        color=BLUE,
        linewidth=2.2
    )
    ax.plot(
        history['event_tick'],
        history['avg_individual_coverage_pct'],
        label='Non-Communicating (Avg Individual Map)',
        color=ORANGE,
        linestyle='--',
        linewidth=2.2
    )
    ax.set_title('Map Knowledge Accumulation Over Time')
    ax.set_xlabel('Number of Photos Taken (Time)')
    ax.set_ylabel('% of Map Known')
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.legend(frameon=True)
    fig1.tight_layout()
    fig1.savefig('knowledge_over_time.pdf', bbox_inches='tight')
    fig1.savefig('knowledge_over_time.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # --- Plot 2: Heatmap Comparison ---
    # We will compare the Swarm's map vs. Agent 0's map
    target_agent = agents[0]
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.6))
    
    # Swarm Heatmap
    swarm_cmap = ListedColormap(["#f7f7f7", BLUE])
    individual_cmap = ListedColormap(["#f7f7f7", ORANGE])
    ax1.imshow(shared_grid, cmap=swarm_cmap, origin='lower', extent=[0, ENV_W, 0, ENV_H], interpolation='nearest', vmin=0, vmax=1)
    ax1.set_title('Communicating Swarm Knowledge')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_xlim(0, ENV_W)
    ax1.set_ylim(0, ENV_H)
    ax1.grid(False)
    
    # Individual Heatmap
    ax2.imshow(indiv_grids[target_agent], cmap=individual_cmap, origin='lower', extent=[0, ENV_W, 0, ENV_H], interpolation='nearest', vmin=0, vmax=1)
    ax2.set_title(f'Non-Communicating Agent {int(target_agent)} Knowledge')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_xlim(0, ENV_W)
    ax2.set_ylim(0, ENV_H)
    ax2.grid(False)
    
    fig2.tight_layout()
    fig2.savefig('knowledge_heatmaps.pdf', bbox_inches='tight')
    fig2.savefig('knowledge_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

if __name__ == "__main__":
    # Create a dummy file if it doesn't exist just for testing the script
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found. Please ensure your log file is in the same directory.")
    else:
        analyze_swarm_data(FILE_PATH)