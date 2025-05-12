import torch
from torchmetrics.segmentation import DiceScore


# Create an instance of DiceScore
dice_score = DiceScore(num_classes=3)

# Example predictions and targets simulating a batch with segmentation maps
# Adding a batch dimension and a channel dimension
preds = torch.tensor([[[0, 1, 2],
                        [1, 2, 2],
                        [0, 0, 1]]], dtype=torch.int32)  # shape: [1, 1, height, width]

targets = torch.tensor([[[0, 1, 2],
                          [1, 0, 2],
                          [0, 0, 1]]], dtype=torch.int32)  # shape: [1, 1, height, width]

# Setting ignore_index
ignore_index = 0

# Flatten the tensors and filter out the entries with the ignore_index
mask = (targets != ignore_index).view(-1)  # Flatten the mask
filtered_preds = preds.view(-1)[mask]       # Flatten preds and apply mask
filtered_targets = targets.view(-1)[mask]   # Flatten targets and apply mask

# Update the metric with the filtered predictions and targets
dice_score.update(filtered_preds, filtered_targets)

# Compute the score
score = dice_score.compute()
print(f"Dice Score: {score}")

# Optional - reset the metric for the next computation
dice_score.reset()
