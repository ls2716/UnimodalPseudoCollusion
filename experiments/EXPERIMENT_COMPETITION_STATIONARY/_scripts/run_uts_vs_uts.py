"""Run competition between two epsilon-greedy agents and analyse the results."""

import numpy as np
import matplotlib.pyplot as plt
from tools.competition import run_multiple_episodes, analyse_aggregate
from agent_definition import get_agent


# PARAMETERS
N_AGENTS = 2
ITERATIONS = 2000
NO_SIMS = 100  # Number of episodes to average over

# Logistic function parameters
A = 1.0
c = 5.0
MU = 0.25
A_0 = -5

STD = 0.01
VARIANCE = STD**2
CORRELATION = 0.


agents = [get_agent("uts") for _ in range(N_AGENTS)]
results = run_multiple_episodes(
    no_sims=NO_SIMS,
    agents=agents,
    iterations=ITERATIONS,
    a=A,
    c=c,
    mu=MU,
    a_0=A_0,
    variance=VARIANCE,
    correlation=CORRELATION,
)


analysis = analyse_aggregate(results)

# Compute the average standard deviation of prices across all episodes
avg_std_prices = np.mean(analysis["std_prices"], axis=0)
print(f"Average standard deviation of prices across episodes: {avg_std_prices}")

# Compute the averge correlation of prices across all episodes
if "prices_correlation" in analysis:
    avg_correlation = np.mean(analysis["prices_correlation"], axis=0)
    print(f"Average correlation of prices across episodes:\n{avg_correlation}")


# Compute the average prices accoss all episodes and the standard deviation of the mean of average prices
all_avg_prices = np.array(analysis["average_prices"])
mean_avg_prices = np.mean(all_avg_prices, axis=0)
std_avg_prices = np.std(all_avg_prices, axis=0)/np.sqrt(NO_SIMS)
print("Nash price:", analysis["nash_price"])
print(f"Mean of average prices across episodes: {mean_avg_prices}")
print(f"Standard deviation of the mean of average prices across episodes: {std_avg_prices}")

# Print the average percenrage increase of the margin
nash_margin = analysis["nash_price"] - c
mean_margin = mean_avg_prices - c
perc_increase_margin = (mean_margin - nash_margin) / nash_margin * 100
print(f"Percentage increase of the margin over the Nash margin: {perc_increase_margin}")

# Compute the standard deviation in terms of percentage of the margin
std_margin_perc = std_avg_prices / mean_margin * 100
print(f"Standard deviation of the mean of average prices as percentage of the margin: {std_margin_perc}")