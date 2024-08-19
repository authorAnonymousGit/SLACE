# creating table 2 from paper, need to run prediction_summaries.csv before
import os
from collections import defaultdict
from utils import *

# %%
count = 0
metrics = ["accuracy", "mse", "mae", "cem", "kendall_tau", "qwk", "macro_acc", "amae"]
datasets = ["ailerons5", "bank5", "elevators5", "pumadyn5", "census5", "california5", "abalone5", "ailerons10",
            "bank10", "elevators10", "pumadyn10", "census10", "california10", "abalone10", "stock5", "stock10",
            "boston5", "boston10", "ailerons7", "bank7", "elevators7", "pumadyn7", "census7", "california7", "abalone7",
            "stock7", "boston7"]
# datasets=["ailerons10","bank10","elevators10","pumadyn10","census10","california10","abalone10","stock10","boston10"]
# datasets=["ailerons5","bank5","elevators5","pumadyn5","census5","california5","abalone5","stock5","boston5"]
# datasets=["tae","thyroid","eucalyptus","wine_full","ERA","ESL","car","balance-scale","SWD","LEV"]

losses = ["Accumulating_SORD_prox_max"]
final_folder = "regression_summaries"
# %%
better_than = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
how_much_better_than = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
# %%
for data in datasets:
    data_folder = "outputs/results/" + data
    for filename in os.listdir(data_folder):
        if filename.startswith('bin') and filename.endswith('metrics.csv'):
            count += 1
            file_path = os.path.join(data_folder, filename)

            df = pd.read_csv(file_path, index_col=0)
            df = df.applymap(lambda x: float(str(x).replace('*', '')))
            print(data)
            for metric in metrics:
                for index1 in losses:
                    for index2 in ["Cross_Entropy", "SORD", "OLL"]:
                        how_much_better_than[metric][index1][index2] += df.loc[index1, metric] - df.loc[index2, metric]
                        if metric == "mse" or metric == "mae" or metric == "amae":
                            if df.loc[index1, metric] <= df.loc[index2, metric]:
                                better_than[metric][index1][index2] += 1
                        else:
                            if df.loc[index1, metric] >= df.loc[index2, metric]:
                                better_than[metric][index1][index2] += 1

# %%
string_better_than = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
# %%
max = count
count = 0
for data in datasets:
    data_folder = "outputs/results/" + data
    for filename in os.listdir(data_folder):
        if filename.startswith('bin') and filename.endswith('metrics.csv'):
            count += 1
            if count == max:
                for metric in metrics:
                    for index1 in losses:
                        for index2 in ["Cross_Entropy", "SORD", "OLL"]:
                            string_better_than[metric][index1][index2] = str(
                                round(better_than[metric][index1][index2] / max * 100, 2)) + "% (" + str(
                                round(how_much_better_than[metric][index1][index2] / max, 3)) + ")"

# %%
for metric in metrics:
    string_better_than[metric]
    df = pd.DataFrame(string_better_than[metric]).T
    df.to_csv(final_folder + "/" + metric + "_summary.csv")