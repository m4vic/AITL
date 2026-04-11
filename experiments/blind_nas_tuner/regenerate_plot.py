# Quick script to regenerate the paper-quality plot from existing data
# Run: python regenerate_plot.py

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Data from our 42-iteration run (manually extracted from terminal output)
losses = [
    0.5241, 0.5312, 0.3694, 0.4509, 0.4641, 0.3862, 0.3890, 0.3847, 0.2778,  # 1-9
    0.3605, 0.3650, 0.3712, 0.3638, 0.3498, 0.6064, 0.3866, 0.3248, 0.3789,  # 10-18
    0.3980, 0.4006, 0.3704, 0.3732, 0.4637, 0.5094, 0.5108, 0.3449, 0.3648,  # 19-27
    0.3375, 0.3883, 0.3412, 0.3808, 0.3614, 0.3572, 0.4451, 0.3566, 0.3523,  # 28-36
    0.5236, 0.3941, 0.3941,                                                      # 37-39
]

from runner import plot_results
plot_results(losses)
