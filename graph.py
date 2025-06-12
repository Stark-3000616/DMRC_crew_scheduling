import ast

import matplotlib.pyplot as plt

# Read numbers from the file
with open("experiments/ls/objectives_46.txt") as f:
    line = f.read()
    numbers = ast.literal_eval(line)
    # numbers = [float(line.strip()) for line in f if line.strip()]
    # Read numbers from the second file
with open("experiments/ls/cg_objectives_46.txt") as f2:
    line2 = f2.read()
    numbers2 = ast.literal_eval(line2)

# Concatenate the two lists
all_numbers = numbers + numbers2

# Plot the first set
plt.plot(range(len(numbers)), numbers, marker='o', linestyle='-', color='r', markersize=2, label='Column Generation')

# Plot the second set, continuing from the end of the first
plt.plot(range(len(numbers), len(all_numbers)), numbers2, marker='o', linestyle='-', color='b', markersize=2, label='Heuristic')

plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.title('Graph of Objective values')
plt.grid(True)
plt.legend()


plt.show()
