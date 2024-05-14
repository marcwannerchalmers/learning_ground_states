import seaborn as sns
import pandas as pd
from parse import parse
import matplotlib.pyplot as plt

# Mostly adapted from lllewis

def read_prediction_errors(filename: str) -> list:
    pat = "({:g}, {:g})"
    errors = []
    with open(filename) as f:
        for i, line in enumerate(f):
            # skip names of the tuples
            if i % 2 == 0:
                continue
            _, err = parse(pat, line.strip())
            errors.append(err)
    return errors

# hard coded for demonstration purposes
def initial_visualization():
    shapes = [(4, 5), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5)]
    data = {"shape": [],
            "error": [],
            "method": []}

    for shape in shapes:
        path_dl = "results/init_results_{}x{}_rand_data.txt".format(*shape)
        path_reg = "clean_results_llewis/new_algorithm_500data_sklearn_maxiter=10000_tol=0.001_seed=42/test_size=0.5_shadow_size=500_qubits_d=1/results_{}x{}_new_data.txt".format(*shape)

        errors_dl = read_prediction_errors(path_dl)
        errors_reg = read_prediction_errors(path_reg)

        for err in errors_dl:
            data["shape"].append("{}x{}".format(*shape))
            data["error"].append(err)
            data["method"].append("DL_algorithm")

        for err in errors_reg:
            data["shape"].append("{}x{}".format(*shape))
            data["error"].append(err)
            data["method"].append("reg_algorithm")

    data = pd.DataFrame(data=data)
    print(data)
    pal = sns.color_palette("pastel").as_hex()
    sns.lineplot(data, x="shape", y="error", hue="method", palette=[pal[3], pal[0]], style="method", markers=["s", "^"])
    plt.xlabel("System size (n)")
    plt.ylabel("Average prediction error")
    plt.show()

def main():
    initial_visualization()

if __name__ == "__main__":
    main()