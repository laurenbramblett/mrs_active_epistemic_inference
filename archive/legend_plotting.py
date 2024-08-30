import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_9

# plt.rcParams['figure.dpi'] = 400

# Create sample data
num_agents = 5
cr = Set1_9.mpl_colors

# Create a figure and axis
fig, ax = plt.subplots()

# Plot three lines
lines = []
for id in range(num_agents):
    line, = ax.plot([0], [0], label=f'Path {id}', color=cr[id], linewidth=4)
    lines.append(line)

# Plot three scatter points
scatters = []
for id in range(num_agents):
    scatter = ax.scatter([0], [0], label=f'Agent {id}', color=cr[id], marker='o', edgecolors='k')
    scatters.append(scatter)

# Add an empty element to the legend
empty_element = plt.Line2D([0], [0], color='w', label='Task Location')

# Create the legend
all_elements = lines + scatters + [empty_element]
# all_elements = lines + [empty_element]
ax.legend(handles=all_elements, loc='upper left', handlelength=1, markerscale=2, fontsize=15)

# Display the plot
ax.set_xlim(-1,0)
plt.show()
