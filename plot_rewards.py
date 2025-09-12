import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('rewards/osc1_rewards.csv')

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot rewards for both models
# Model 1: Qwen2.5-1.5B
ax.plot(df['train/global_step'], 
        df['Qwen/Qwen2.5-1.5B-Instruct-r8-g64-1756158958 - train/reward'], 
        label='Qwen2.5-1.5B', linewidth=2, marker='o', markersize=4)

# Model 2: Qwen2.5-0.5B (only plot up to where data exists)
mask_0_5b = df['Qwen/Qwen2.5-0.5B-Instruct-r8-g64-1755898769 - train/reward'].notna()
ax.plot(df.loc[mask_0_5b, 'train/global_step'], 
        df.loc[mask_0_5b, 'Qwen/Qwen2.5-0.5B-Instruct-r8-g64-1755898769 - train/reward'], 
        label='Qwen2.5-0.5B', linewidth=2, marker='s', markersize=4)

# Customize the plot
ax.set_xlabel('Global Step', fontsize=14)
ax.set_ylabel('Training Reward', fontsize=14)
ax.set_title('Training Rewards vs Global Steps', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Format x-axis to show steps in thousands
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))

# Add some styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout and display
plt.tight_layout()
plt.show()

# Optional: Save the plot
plt.savefig('training_rewards_vs_steps.png', dpi=300, bbox_inches='tight')
plt.close()
