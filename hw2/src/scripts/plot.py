import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

hw2 = Path(__file__).resolve().parent.parent.parent
exp_dir = hw2/"exp"

# ======================= Pendulum =======================
runs = sorted(exp_dir.iterdir())
plt.figure(figsize=(8,6))


for run in runs:
    if "Pendulum" not in run.name:
        continue
    df = pd.read_csv(run / "log.csv")
    plt.plot(df["Train_EnvstepsSoFar"],
             df["Eval_AverageReturn"],
             label=run.name)

plt.legend()
plt.xlabel("Env Steps")
plt.ylabel("Eval Avg Return")
plt.title("Eval Avg Return vs Env Steps Across Runs")
plt.grid(True)
plt.show()
# ======================= LunarLander =======================
# runs = sorted(exp_dir.iterdir())
# plt.figure(figsize=(8,6))

# # Plot avg train return vs env steps for each run
# for run in runs:
#     if "LunarLander" not in run.name:
#         continue
#     df = pd.read_csv(run / "log.csv")
#     plt.plot(df["Train_EnvstepsSoFar"],
#              df["Eval_AverageReturn"],
#              label=run.name)

# plt.legend()
# plt.xlabel("Env Steps")
# plt.ylabel("Eval Avg Return")
# plt.title("Eval Avg Return vs Env Steps Across Runs")
# plt.grid(True)
# plt.show()
#  ============== HalfCheetah ================ 
# run = exp_dir/"HalfCheetah-v4_cheetah_baseline_steps_doubled_sd1_20260218_225001"
# df = pd.read_csv(run / "log.csv")
# plt.figure(figsize=(8,6))

# # Plot Baseline Loss
# plt.plot(df["Train_EnvstepsSoFar"],
#          df["Baseline Loss"],
#          label=run.name)

# plt.legend()
# plt.xlabel("Env Steps")
# plt.ylabel("Baseline Loss")
# plt.title("Baseline Loss vs Env Steps Across Runs")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8,6))

# # Plot Baseline Loss
# plt.plot(df["Train_EnvstepsSoFar"],
#          df["Eval_AverageReturn"],
#          label=run.name)

# plt.legend()
# plt.xlabel("Env Steps")
# plt.ylabel("Eval Average Return")
# plt.title("Eval Average Return vs Env Steps Across Runs")
# plt.grid(True)
# plt.show()



# ======================= CARTPOLE =======================
# runs = sorted(exp_dir.iterdir())
# # Plot avg train return vs env steps for each run
# for run in runs:
#     if "cartpole_lb" not in run.name:
#         continue
#     df = pd.read_csv(run / "log.csv")
#     plt.plot(df["Train_EnvstepsSoFar"],
#              df["Train_AverageReturn"],
#              label=run.name)

# plt.legend()
# plt.xlabel("Env Steps")
# plt.ylabel("Train Avg Return")
# plt.title("Train Avg Return vs Env Steps Across Runs")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8,6))

# # Plot avg train return vs env steps for each run
# for run in runs:
#     if "cartpole_lb" not in run.name:
#         continue
#     df = pd.read_csv(run / "log.csv")
#     plt.plot(df["Train_EnvstepsSoFar"],
#              df["Eval_AverageReturn"],
#              label=run.name)

# plt.legend()
# plt.xlabel("Env Steps")
# plt.ylabel("Eval Avg Return")
# plt.title("Eval Avg Return vs Env Steps Across Runs")
# plt.grid(True)
# plt.show()