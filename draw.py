import json
import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV 
from pymoo.indicators.igd import IGD


# Load data
with open("Illustration/Bi TSP 20/EoH/population/pop_20.json", "r") as f:
    data = json.load(f)

with open("Illustration/Bi TSP 20/MEoH/population/pop_20.json", "r") as f:
    data2 = json.load(f)

with open("Illustration/Bi TSP 20/LLM PFG/population/pop_20.json", "r") as f:
    data3 = json.load(f)

# Extract objectives
objectives = np.array([ind['score'] for ind in data])
objectives2 = np.array([ind['score'] for ind in data2])
objectives3 = np.array([ind['score'] for ind in data3])


# Combine all objectives to build reference front
all_objectives = np.vstack([objectives, objectives2, objectives3])

# Filter non-dominated front (Pareto front)
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
nds = NonDominatedSorting().do(all_objectives, only_non_dominated_front=True)
reference_front = all_objectives[nds]
igd = IGD(reference_front)
igd1 = igd(objectives)
igd2 = igd(objectives2)
igd3 = igd(objectives3)
igd_values = [igd1, igd2, igd3]
formatted_igd = [f"{v:.3f}" for v in igd_values]


# Compute HV
hv = HV(ref_point=[-175, 2.25])
hv1 = hv(objectives)
hv2 = hv(objectives2)
hv3 = hv(objectives3)

# Prepare data
method_names = ["EoH", "MEoH", "LLMPFG"]
hv_values = [hv1, hv2, hv3]
formatted_values = [f"{v:.3f}" for v in hv_values]

# Determine best (highest) HV
best_index = np.argmax(hv_values)

# Create figure
plt.figure(figsize=(10, 6))
plt.scatter(objectives[:, 0], objectives[:, 1], color='blue', label='EoH', marker='*')
plt.scatter(objectives2[:, 0], objectives2[:, 1], color='red', label='MEoH', marker='o')
plt.scatter(objectives3[:, 0], objectives3[:, 1], color='green', label='LLMPFG', marker='x')

plt.xlabel("Negative Hypervolume")
plt.ylabel("Time")
plt.title("Comparison - Bi-objective TSP (Size 20)")
plt.grid(True)
plt.legend()

column_labels = ["Method", "HV ↑", "IGD ↓"]
table_data = []
best_hv_index = np.argmax(hv_values)
best_igd_index = np.argmin(igd_values)

for i, (name, hv_val, igd_val) in enumerate(zip(method_names, formatted_values, formatted_igd)):
    hv_display = f"$\\bf{{{hv_val}}}$" if i == best_hv_index else hv_val
    igd_display = f"$\\bf{{{igd_val}}}$" if i == best_igd_index else igd_val
    name_display = f"$\\bf{{{name}}}$" if i == best_hv_index or i == best_igd_index else name
    table_data.append([name_display, hv_display, igd_display])


# Add table to plot
plt.table(cellText=table_data,
          colLabels=column_labels,
          loc='lower center',
          cellLoc='center',
          colLoc='center',
          bbox=[0.0, -0.57, 1, 0.3],
          edges='horizontal')

plt.subplots_adjust(bottom=0.35)
plt.show()