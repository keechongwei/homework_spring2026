import matplotlib.pyplot as plt
from train import TrainConfig
import numpy as np
import re
from pathlib import Path
import wandb

# Fetch evaluation rewards from wandb
api = wandb.Api()
run = api.run("keechongwei-uc-berkeley-electrical-engineering-computer-/hw1-imitation/lg3w4nqk")

steps = []
rewards = []

for row in run.scan_history(keys=["eval/mean_reward", "_step"]):
    if row.get("eval/mean_reward") is not None:
        steps.append(row["_step"])
        rewards.append(row["eval/mean_reward"])

steps = np.array(steps)
rewards = np.array(rewards)


# Parse training loss from output.log
ROOT = Path(__file__).resolve().parent 
log_path = ROOT / "output.log"

with open(log_path, "r") as f:
    lines = f.readlines()

pattern = re.compile(
    r"Epoch \[(\d+)/\d+\] Step \[(\d+)\] Loss: ([0-9.]+)"
)

train_steps =  []
epochs = []
losses = []

for line in lines:
    match = pattern.search(line)
    if match:
        epoch = int(match.group(1))
        step = int(match.group(2))
        loss = float(match.group(3))
        epochs.append(epoch)
        train_steps.append(step)
        losses.append(loss)

train_steps = np.array(train_steps)
epochs = np.array(epochs)
losses = np.array(losses)

unique_epochs = np.unique(epochs)
unique_steps = np.unique(train_steps)

mean_loss_per_step = np.array([
    losses[train_steps == s].mean()
    for s in unique_steps
])

mean_loss_per_epoch = np.array([
    losses[epochs == e].mean()
    for e in unique_epochs
])

plt.figure()
plt.plot(unique_steps, mean_loss_per_step)
plt.xlabel("Step")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Step")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(unique_epochs, mean_loss_per_epoch)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(steps, rewards)
plt.xlabel("Training Step")
plt.ylabel("Mean Evaluation Reward")
plt.title("Evaluation Reward vs Training Step")
plt.grid(True)
plt.show()