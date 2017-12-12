import matplotlib.pyplot as plt
from math import pi


def plot_as_radar(projection_vec, memory_labels):
    """
    # radar plots not built-in to matplotlib
    # reference code uses pandas: https://python-graph-gallery.com/390-basic-radar-chart/
    """

    # number of variable
    p = len(memory_labels)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(p) * 2 * pi for n in range(p)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # Draw one ax per variable + add labels labels yet
    plt.xticks(angles[:-1], memory_labels, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
    plt.ylim(0, 40)

    # Plot data
    ax.plot(angles, projection_vec, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, projection_vec, 'b', alpha=0.1)
