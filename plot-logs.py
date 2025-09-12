import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator


import os
import json

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
# pip install SciencePlots

import scienceplots
import numpy as np
from scipy.stats import gaussian_kde
from adjustText import adjust_text

from dataclasses import dataclass, field
plt.style.use('science')

carbon = {
  "black": { "100": "#000000" },
  "white": { "0": "#ffffff" },
  "blue": { "10": "#edf5ff", "20": "#d0e2ff", "30": "#a6c8ff", "40": "#78a9ff", "50": "#4589ff", "60": "#0f62fe", "70": "#0043ce", "80": "#002d9c", "90": "#001d6c", "100": "#001141" },
  "coolGray": { "10": "#f2f4f8", "20": "#dde1e6", "30": "#c1c7cd", "40": "#a2a9b0", "50": "#878d96", "60": "#697077", "70": "#4d5358", "80": "#343a3f", "90": "#21272a", "100": "#121619" },
  "cyan": { "10": "#e5f6ff", "20": "#bae6ff", "30": "#82cfff", "40": "#33b1ff", "50": "#1192e8", "60": "#0072c3", "70": "#00539a", "80": "#003a6d", "90": "#012749", "100": "#061727" },
  "gray": { "10": "#f4f4f4", "20": "#e0e0e0", "30": "#c6c6c6", "40": "#a8a8a8", "50": "#8d8d8d", "60": "#6f6f6f", "70": "#525252", "80": "#393939", "90": "#262626", "100": "#161616" },
  "green": { "10": "#defbe6", "20": "#a7f0ba", "30": "#6fdc8c", "40": "#42be65", "50": "#24a148", "60": "#198038", "70": "#0e6027", "80": "#044317", "90": "#022d0d", "100": "#071908" },
  "magenta": { "10": "#fff0f7", "20": "#ffd6e8", "30": "#ffafd2", "40": "#ff7eb6", "50": "#ee5396", "60": "#d02670", "70": "#9f1853", "80": "#740937", "90": "#510224", "100": "#2a0a18" },
  "orange": { "10": "#fff2e8", "20": "#ffd9be", "30": "#ffb784", "40": "#ff832b", "50": "#eb6200", "60": "#ba4e00", "70": "#8a3800", "80": "#5e2900", "90": "#3e1a00", "100": "#231000" },
  "purple": { "10": "#f6f2ff", "20": "#e8daff", "30": "#d4bbff", "40": "#be95ff", "50": "#a56eff", "60": "#8a3ffc", "70": "#6929c4", "80": "#491d8b", "90": "#31135e", "100": "#1c0f30" },
  "red": { "10": "#fff1f1", "20": "#ffd7d9", "30": "#ffb3b8", "40": "#ff8389", "50": "#fa4d56", "60": "#da1e28", "70": "#a2191f", "80": "#750e13", "90": "#520408", "100": "#2d0709" },
  "teal": { "10": "#d9fbfb", "20": "#9ef0f0", "30": "#3ddbd9", "40": "#08bdba", "50": "#009d9a", "60": "#007d79", "70": "#005d5d", "80": "#004144", "90": "#022b30", "100": "#081a1c" },
  "warmGray": { "10": "#f7f3f2", "20": "#e5e0df", "30": "#cac5c4", "40": "#ada8a8", "50": "#8f8b8b", "60": "#726e6e", "70": "#565151", "80": "#3c3838", "90": "#272525", "100": "#171414" },
  "yellow": { "10": "#fcf4d6", "20": "#fddc69", "30": "#f1c21b", "40": "#d2a106", "50": "#b28600", "60": "#8e6a00", "70": "#684e00", "80": "#483700", "90": "#302400", "100": "#1c1500" },
  "rose": { "10": "#fcf2ed", "20": "#f9d9d1", "30": "#f0b8a7", "40": "#e99179", "50": "#d37258", "60": "#ab5638", "70": "#853c27", "80": "#622517", "90": "#46170c", "100": "#2a0d09" },
  "almond": { "10": "#f8f3ea", "20": "#efdcc8", "30": "#e1bd9f", "40": "#d1a080", "50": "#b88361", "60": "#97633f", "70": "#76472b", "80": "#55301e", "90": "#3a1e12", "100": "#27120a" }
}
matplotlib.style.use({
    'axes.edgecolor': carbon['gray']['90'],
    'xtick.color': carbon['gray']['90'],
    'ytick.color': carbon['gray']['90'],
    'axes.labelcolor': carbon['gray']['90'],
    'axes.titlecolor': carbon['gray']['90'],
})




def search_event_file(run_dir):
    event_dir = Path(run_dir)
    p = list(event_dir.glob("events.out.tfevents.*"))
    if len(p) > 0:
        return p[0]
    return None

log_ids = [
    "oscillator1_llmsr_Qwen2.5-0.5B-Instruct",
    # "oscillator1_llmsr_Qwen2.5-1.5B-Instruct",
    "oscillator1_llmsr_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp50",
    # "oscillator1_llmsr_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp100",
    "oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct",
    # "oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct",
    "oscillator1_llmsr_dec_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp50_v2",
    # "oscillator1_llmsr_dec_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp100",
]

model_variants = [
    "LLM-SR (Qwen2.5-0.5B)",
    # "LLM-SR (Qwen2.5-1.5B)",
    "LLM-SR + Adapt (Qwen2.5-0.5B)",
    # "LLM-SR + Adapt (Qwen2.5-1.5B)",
    "LLM-SR + Decomp (Qwen2.5-0.5B)",
    # "LLM-SR + Decomp (Qwen2.5-1.5B)",
    "LLM-SR + Adapt + Decomp (Qwen2.5-0.5B)",
    # "LLM-SR + Adapt + Decomp (Qwen2.5-1.5B)",
]

# Define colors - similar colors for same model variants with different backbones
# colors = [
#     '#1f77b4',  # Blue for LLM-SR (0.5B)
#     # '#1f77b4',  # Blue for LLM-SR (1.5B) - same variant
#     '#ff7f0e',  # Orange for LLM-SR + Adapt (0.5B)
#     # '#ff7f0e',  # Orange for LLM-SR + Adapt (1.5B) - same variant
#     '#2ca02c',  # Green for LLM-SR + Decomp (0.5B)
#     # '#2ca02c',  # Green for LLM-SR + Decomp (1.5B) - same variant
#     '#d62728',  # Red for LLM-SR + Adapt + Decomp (0.5B)
#     # '#d62728',  # Red for LLM-SR + Adapt + Decomp (1.5B) - same variant
# ]


colors = [
    'coolGray',
    'magenta',
    'cyan',
    'rose'
]

# Line styles to differentiate between 0.5B and 1.5B backbones
# line_styles = ['-', '--', '-', '--', '-', '--', '-', '--']
line_styles = ['-', '-', '-', '-']


event_paths = []
all_results = []

for log_id in log_ids:
    event_path = search_event_file(f"./logs/{log_id}")
    event_paths.append(event_path)

for event_file in event_paths:
    if event_file is not None:
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()
        key = 'Best Score of Function'
        values = [e.value for e in ea.Scalars(key)]
        values = np.array(values)
        # Limit to index 2000
        values = values[:1000]
        all_results.append(values)
    else:
        all_results.append(np.array([]))


import pandas as pd

data_path = f'./data/oscillator1/train.csv'
df = pd.read_csv(data_path)
data = df.values
y = data[:, -1].reshape(-1)
y_std = np.std(y)

# Create the plot
plt.figure(figsize=(12, 8))

for i, (model_variant, color, line_style, results) in enumerate(zip(model_variants, colors, line_styles, all_results)):
    if len(results) > 0:
        # Plot MSE (-value) vs number of samples (index)
        x_axis = np.arange(len(results))
        if y_std == 0:
            nmse = -results  # Avoid division by zero
        else:
            nmse = -results / (y_std ** 2)
        plt.plot(x_axis, nmse, color=carbon[color]["50"], linestyle=line_style, 
                label=model_variant, linewidth=2)

plt.xlabel('Number of Samples', fontsize=12)
plt.ylabel('NMSE', fontsize=12)
# plt.title('Discovery Curves for Different Model Variants', fontsize=14)
plt.yscale('log')  # Set y-axis to log scale
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1000)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('./figs/discovery_curves_oscillator1.png', dpi=300, bbox_inches='tight')

plt.show()