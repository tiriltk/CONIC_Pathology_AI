import numpy as np

"""
Calculates and prints mean and standard deviation of performance metrics.
"""

def summarize(values, name="Metric"):
    mean = np.mean(values)
    std = np.std(values)
    print(f"{name}: Mean {mean:.4f} Std {std:.4f}")

mpq_values0 = [0.5215320905892238, 0.5248571056833443, 0.5208786321136809, 0.5218202227298262, 0.5241153148854243]
r2_values0 = [0.8057580697428574, 0.7842661638904255, 0.782116490502953, 0.7163622469812191, 0.746329603226415]
dice_values0 = [0.4846515, 0.48652583, 0.4827529, 0.49032784, 0.48513192]

print("Results for Split 0")
summarize(dice_values0, "Dice")
summarize(mpq_values0, "mPQ+")
summarize(r2_values0, "R2")

mpq_values1 = [0.5134014768764985, 0.5140324334349793, 0.505195884036125, 0.4922326827464486, 0.5077302562597316]
r2_values1 = [0.7689920581737846, 0.7291416703028413, 0.7430769071619929, 0.7774472140473063, 0.7391721540111978]
dice_values1 = [0.4756288, 0.47699246, 0.4643383, 0.46149358, 0.47136098]

print("Results for Split 1")
summarize(dice_values1, "Dice")
summarize(mpq_values1, "mPQ+")
summarize(r2_values1, "R2")

mpq_values2 = [0.5374861569862733, 0.5383395666112354, 0.5342286417588067, 0.5363919174478271, 0.5376094771994977]
r2_values2 = [0.816243308600928, 0.8147478321245435, 0.8215896266872196, 0.8218955851876352, 0.8188316707302332]
dice_values2 = [0.4986689, 0.49746197, 0.49034324, 0.49557227, 0.49596462]

print("Results for Split 2")
summarize(dice_values2, "Dice")
summarize(mpq_values2, "mPQ+")
summarize(r2_values2, "R2")