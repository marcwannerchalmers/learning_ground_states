from scipy.stats.qmc import Sobol
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

if __name__ == "__main__":
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)

from model.geometry import GridMap
from util.helper import get_n_terms

def plot_sequence():
    sampler = Sobol(d=2, scramble=True)
    points = sampler.random_base2(9)
    tau = lambda x: 2*x - 1
    phi = norm.ppf
    phi_pt = phi(points)
    phi_pt = (phi_pt - np.min(phi_pt, axis=0))/np.tile((np.max(phi_pt, axis=0) - np.min(phi_pt, axis=0)).reshape(1,2), (512,1))
    print(phi_pt)
    plt.scatter(tau(points[:128,0]), tau(points[:128,1]), label="x")
    plt.scatter(tau(points[256:384,0]), tau(points[256:384,1]), label="$\Phi^{-1}(x)$", marker="^")
    plt.scatter(tau(points[128:256,0]), tau(points[128:256,1]), label="x1")
    plt.scatter(tau(points[384:,0]), tau(points[384:,1]), label="$\Phi^{-1}(x)1$", marker="^")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Transformed LDS")
    plt.legend()
    plt.savefig("example_norm.pdf")
    plt.show()


def generate_sequence(d: int, log2_n_data: int, path_save: str):
    sampler = Sobol(d=d, scramble=True)
    points = sampler.random_base2(log2_n_data)
    #filename = "lds.txt"
    #file = os.path.join(path_save, filename)
    np.savetxt(path_save, points, delimiter="\t")

def main():
    log2_n_data = 14
    base_path = "lds_sequences/"
    for Lx in range(4, 10):
        shape = (Lx, 5)
        gm = GridMap(shape)
        n_params = get_n_terms("edges", gm)
        path = os.path.join(base_path, "lds_{}x{}.txt".format(*shape))
        generate_sequence(d=n_params, log2_n_data=log2_n_data, path_save=path)

if __name__ == "__main__":   
    main()
