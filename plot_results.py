import seaborn as sns
import pandas as pd
from parse import parse
import matplotlib.pyplot as plt
import os
import itertools
import copy

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
    data1 = {"shape": [],
            "error": [],
            "Algorithm": [],
            "Training set size (N)": [],
            "$\delta_1$": []}
    data2 = copy.deepcopy(data1)
    data3 = copy.deepcopy(data1)
    
    splits = [0.1, 0.3, 0.5, 0.7, 0.9]
    delta1s = [0, 1, 2, 3, 4, 5]
    delta1 = 0
    split = 0.9
    seqs = ["lds", "rand"]
    seq = "lds"
    pattern_dl_path = "../results/dl_{}/res_{}x{}/"
    pattern_dl_file = "dl_results_{}_{}x{}_split_{}_delta1_{}.txt"
    pattern_rg_path = "../results2/rg_{}/"
    pattern_rg_file = "rg_results_{}_{}x{}_split_{}_delta1_{}.txt"
    shape = (9, 5)
    name_dict = {"rand": "Random", "lds": "LDS","nonlocal": "Non-local"}
    seq = "lds"
    for seq in seqs:
        for shape in shapes:
            
            folder_dl = pattern_dl_path.format(seq, *shape)
            folder_reg = pattern_rg_path.format(seq, *shape)
            file_dl = pattern_dl_file.format(seq, *shape, split, delta1)
            file_reg = pattern_rg_file.format(seq, *shape, split, delta1)

            path_dl = os.path.join(folder_dl, file_dl)
            path_reg = os.path.join(folder_reg, file_reg)

            errors_dl = read_prediction_errors(path_dl)
            errors_reg = read_prediction_errors(path_reg)
            for err in errors_dl:
                
                data1["shape"].append("{}x{}".format(*shape))
                data1["error"].append(err)
                data1["Algorithm"].append("Deep Learning, {}".format(name_dict[seq]))
                data1["Training set size (N)"].append(int(split*4096))
                data1["$\delta_1$"].append(delta1)

            for err in errors_reg:
                data1["shape"].append("{}x{}".format(*shape))
                data1["error"].append(err)
                data1["Algorithm"].append("Regression, {}".format(name_dict[seq]))
                data1["Training set size (N)"].append(str(split*4096))
                data1["$\delta_1$"].append(delta1)
            
    for delta1 in delta1s:
        for split in splits:
            folder_dl = pattern_dl_path.format(seq, *shape)
            folder_reg = pattern_rg_path.format(seq, *shape)
            file_dl = pattern_dl_file.format(seq, *shape, split, delta1)
            file_reg = pattern_rg_file.format(seq, *shape, split, delta1)

            path_dl = os.path.join(folder_dl, file_dl)
            path_reg = os.path.join(folder_reg, file_reg)

            errors_dl = read_prediction_errors(path_dl)

            for err in errors_dl:
                data2["shape"].append("{}x{}".format(*shape))
                data2["error"].append(err)
                data2["Algorithm"].append("Deep Learning, {}".format(name_dict[seq]))
                data2["Training set size (N)"].append(int(split*4096))
                data2["$\delta_1$"].append(delta1)

            """for err in errors_reg:
                data["shape"].append("{}x{}".format(*shape))
                data["error"].append(err)
                data["Algorithm"].append("Regression, {}".format(name_dict[seq]))
                data["Training set size (N)"].append(str(split*4096))
                data["$\delta_1$"].append(delta1)"""
    delta1=0
    split=0.9
    seq = "nonlocal"
    for split in splits:
        for shape in shapes[:3]:
            folder_dl = pattern_dl_path.format(seq, *shape)
            folder_reg = pattern_rg_path.format(seq, *shape)
            file_dl = pattern_dl_file.format(seq, *shape, split, delta1)
            file_reg = pattern_rg_file.format(seq, *shape, split, delta1)

            path_dl = os.path.join(folder_dl, file_dl)
            path_reg = os.path.join(folder_reg, file_reg)

            errors_dl = read_prediction_errors(path_dl)
            # errors_reg = read_prediction_errors(path_reg)

            for err in errors_dl:
                data3["shape"].append("{}x{}".format(*shape))
                data3["error"].append(err)
                data3["Algorithm"].append("Deep Learning, {}".format(name_dict[seq]))
                data3["Training set size (N)"].append(int(split*4096))
                data3["$\delta_1$"].append(delta1)

            """for err in errors_reg:
                data["shape"].append("{}x{}".format(*shape))
                data["error"].append(err)
                data["Algorithm"].append("Regression, {}".format(name_dict[seq]))
                data["Training set size (N)"].append(str(split*4096))
                data["$\delta_1$"].append(delta1)"""

    #print(data1)
    data1 = pd.DataFrame(data=data1)
    data2 = pd.DataFrame(data=data2)
    data3 = pd.DataFrame(data=data3)

    #print(data)
    #plt.rc('font', size=17, family='serif')
    font={"family": "serif", "size": 25}
    rc = {"font.family": "serif", "font.size": 25}
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(ncols=3, figsize=(30, 8), sharey=False)
        #fig.tight_layout()
    #plt.margins(0.01,0.01)
    pal = sns.color_palette("pastel").as_hex()
    sns.set_theme(font="serif")
    sns.lineplot(data1, ax=axes[0],x="shape", y="error", hue="Algorithm", 
                 palette=[pal[3], pal[0], pal[1], pal[2], pal[4], pal[5]], 
                 style="Algorithm", markers=True, linewidth=5, markersize=15)
    
    sns.lineplot(data2, ax=axes[1],x="Training set size (N)", y="error", hue="$\delta_1$", 
                 palette=[pal[3], pal[0], pal[1], pal[2], pal[4], pal[5]], 
                 style="$\delta_1$", markers=True, linewidth=5, markersize=15)

    sns.lineplot(data3, ax=axes[2],x="shape", y="error", hue="Training set size (N)", 
                 palette=[pal[3], pal[0], pal[1], pal[2], pal[4], pal[5]], 
                 style="Training set size (N)", markers=True, linewidth=5, markersize=15)
    
    axes[2].set_xlabel("System size (n)", fontdict=font)
    axes[0].set_xlabel("System size (n)", fontdict=font)
    axes[1].set_xlabel("Training set size (N)", fontdict=font)
    # plt.xlabel("$\delta_1$")
    axes[0].set_ylabel("Average prediction error", fontdict=font)
    axes[1].set_ylabel("Average prediction error", fontdict=font)
    axes[2].set_ylabel("Average prediction error", fontdict=font)

    axes[0].set_xticklabels(["{}x{}".format(*shape) for shape in shapes], fontsize=20, font="serif")
    axes[1].set_xticks(ticks=[int(split*4096) for split in splits])
    axes[1].set_xticklabels([int(split*4096) for split in splits], size=20, font="serif")
    axes[2].set_xticklabels(["{}x{}".format(*shape) for shape in shapes], size=20, font="serif")

    titles = ["Algorithm", "$\\bf{\delta_1}$", "Training set size (N)"]
    
    for i, ax in enumerate(axes):
        plt.setp(axes[i].get_legend().get_texts(), fontsize='25')
        axes[i].get_legend().set_title(titles[i])
        plt.setp(axes[i].get_legend().get_title(), fontsize='25', fontweight="bold")
        #axes[i].set_yticklabels([round(elem, 2) for elem in axes[0].get_yticks()], fontsize=20, font="serif")
        axes[i].tick_params(labelsize=25, labelfontfamily="serif")
    #plt.legend([],[], frameon=False)
    #plt.ylim([0.2, 0.3])
    #sns.set_theme(rc=rc)
    plt.savefig("../all.pdf")
    plt.show()

def main():
    initial_visualization()

if __name__ == "__main__":
    main()