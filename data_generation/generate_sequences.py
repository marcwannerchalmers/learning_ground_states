from scipy.stats.qmc import Sobol
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def main():
    sampler = Sobol(d=2, scramble=True)
    points = sampler.random_base2(9)
    tau = lambda x: 2*x - 1
    phi = norm.ppf
    phi_pt = phi(points)
    phi_pt = (phi_pt - np.min(phi_pt, axis=0))/np.tile((np.max(phi_pt, axis=0) - np.min(phi_pt, axis=0)).reshape(1,2), (512,1))
    print(phi_pt)
    plt.scatter(tau(points[:,0]), tau(points[:,1]), label="x")
    plt.scatter(tau(phi_pt[:,0]), tau(phi_pt[:,1]), label="$\Phi^{-1}(x)$", marker="^")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Transformed LDS")
    plt.legend()
    plt.savefig("example_norm.pdf")
    plt.show()

if __name__ == "__main__":
    main()
