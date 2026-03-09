import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

hw3 = Path(__file__).resolve().parent.parent.parent
exp_dir = hw3/"exp"

plt.figure(figsize=(8,6))

runs = sorted(exp_dir.iterdir())
# Plot avg train return vs env steps for each run
for run in runs:
    if "Hopper" not in run.name:
        continue
    df = pd.read_csv(run / "log.csv")
    eval_df = df[df["Eval_AverageReturn"].notna()]
    # plt.plot(eval_df["step"],
    #             eval_df["Eval_AverageReturn"],
    #             label=run.name)
    plt.plot(eval_df["step"],
             eval_df["q_values"],
             label=run.name)

plt.xlabel("Env Steps")
plt.ylabel("Q Values")
plt.title("Q Values vs Training")
plt.grid(True)
plt.legend()

plt.show()