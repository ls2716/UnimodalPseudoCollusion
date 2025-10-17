"""Combine the results of algorithm evaluations into a single CSV file for analysis."""

import pandas as pd


agent_types = ["epsilon_greedy", "ucb_u", "uts"]
noises = ["0.0", "0.05", "0.1"]
T_values = [ "300", "400", "500", "1000"]
cases = ["stationary", "nonstationary", "average"]

# Load the data into a dictionary of DataFrames
data = {}
for agent in agent_types:
    # Load the CSV file for the agent
    filepath = f"results_{agent}.csv"
    df = pd.read_csv(filepath)
    # Iterate over rows and store in dictionary
    for _, row in df.iterrows():
        T = str(int(row["T"]))
        noise = str(row["noise_level"])
        key = (agent, noise, T)
        data[key] = {}
        data[key]["stationary_regret"] = row["stationary_regret"]
        data[key]["nonstationary_regret"] = row["nonstationary_regret"]
        data[key]["average_regret"] = row["average_regret"]
        print(key)

# Format a latex table
# & & nonstationary&& & average && \\
# T & noise & epsilon_greedy & ucb_u & uts & epsilon_greedy & ucb_u & uts \\
header = " & & " + " nonstationary&& average &&" + " \\\\"
subheader = "T & noise &  SW-$\\varepsilon$-U & SW-UCB-U & SW-UTS & SW-$\\varepsilon$-U & SW-UCB-U & SW-UTS \\\\"
# Create stationary rows for each noise level
rows = []
for noise in noises:
    row = f"stationary & {noise}  "
    row += "& - & - & - " # Empty cells for nonstationary case
    elems = []
    for agent in agent_types:
        elems.append(data[(agent, noise, '1000')]['stationary_regret'])
    for elem in elems:
        if elem == min(elems):
            row += f" & \\textbf{{{elem:.3f}}}"
        else: 
            row += f" & {elem:.3f}"
                
    row += " \\\\"
    rows.append(row)

for T in T_values:
    for noise in noises:
        row = f"{T} & {noise} "
        for case in ["nonstationary", "average"]:
            elems = []
            for agent in agent_types:
                elems.append(data[(agent, noise, T)][f"{case}_regret"])
            for elem in elems:
                if elem == min(elems):
                    row += f" & \\textbf{{{elem:.3f}}}"
                else:   
                    row += f" & {elem:.3f}"
        row += " \\\\"
        rows.append(row)

print(rows)

# Print latex table
print("\\begin{tabular}{llrrrrrr}")
print(header)
print(subheader) 
for row in rows:
    print(row)
print("\\end{tabular}")