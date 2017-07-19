import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D


def plot_simplex(N):
    normal = [1, 1, 1]
    intercepts = [(N, 0, 0), (0, N, 0), (0, 0, N)]

    # create surface
    x1range = np.linspace(0.0, N, 100)
    x2range = np.linspace(0.0, N, 100)
    xx, yy = np.meshgrid(x1range, x2range)
    z = (N - normal[0] * xx - normal[1] * yy) * 1. / normal[2]

    # plot surface
    cmap = colors.ListedColormap(['white', 'red'])
    bounds = [0, 5, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, z, alpha=0.4, cmap=cmap, color='blue')
    ax.scatter(intercepts[0], intercepts[1], intercepts[2], color=['red', 'green', 'blue'])

    ax.set_zlim(0.0, intercepts[2][2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # note = 'A1 = %.2f \ngamma = %.2f \nh1 = %.2f, h2 = %.2f, h3 = %.2f' % (A[0,0], W[0,0], H[0,0], H[0,1], H[0,2])
    # ax.text(intercepts[0][0]*0.55, intercepts[1][1]*0.6, intercepts[2][2]*0.6, note, fontsize=7)
    return fig