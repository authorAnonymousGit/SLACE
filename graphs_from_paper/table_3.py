#create table 3 from paper
import sklearn.metrics
import os
from collections import defaultdict
from scipy.stats import ttest_rel
from utils import *
from sklearn.metrics import cohen_kappa_score
from scipy.stats import kendalltau
from sklearn.metrics import balanced_accuracy_score
import math
from sklearn.metrics import mean_absolute_error
#%%
num_seeds = 10
cross_entropy_path = ''
alpha_ttest = 0.05
#%%

# data_sets=["ailerons5","bank5","elevators5","pumadyn5","census5","california5","abalone5","ailerons10","bank10","elevators10","pumadyn10","census10","california10","abalone10","stock5","stock10","boston5","boston10"]

data_sets=["tae","thyroid","eucalyptus","wine_full","ERA","ESL","car","balance-scale","SWD","LEV"]
#
# losses=["Cross_Entropy", "SORD", "OLL","Accumulating","Accumulating_SORD_prox_max", "SORD_max"]
#
# final_path="final_summaries/ordinal_best_alphas.csv"

# data_sets=["ailerons10","bank10","elevators10","pumadyn10","census10","california10","abalone10","stock10","boston10","ailerons7","bank7","elevators7","pumadyn7","census7","california7","abalone7","stock7","boston7"]
losses=["Cross_Entropy", "SORD","SORD_max", "OLL","OLL_max","Accumulating_SORD","Accumulating_SORD_prox_max"]

final_path= "../final_summaries/ordinal.csv"

#%%
def average_mae(true_values, predicted_values):
    classes = np.unique(true_values)
    maes = []

    for cls in classes:
        cls_true_values = true_values[true_values == cls]
        cls_predicted_values = predicted_values[true_values == cls]
        mae = mean_absolute_error(cls_true_values, cls_predicted_values)
        maes.append(mae)

    amae = np.mean(maes)
    return amae
#%%
def calculate_metrics(predictions, labels):
    accuracy = (predictions == labels).mean()
    mse = sklearn.metrics.mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    cem = float(cem2(predictions, np.array(labels)))
    kendall_tau, _ = kendalltau(labels, predictions)
    qwk = cohen_kappa_score(labels, predictions, weights='quadratic')
    macro_acc = balanced_accuracy_score(labels, predictions)
    amae = average_mae(labels, predictions)

    return accuracy, mse, mae, cem, kendall_tau, qwk, macro_acc, amae
#%%
def load_csv(loss_dir, filename):
    filepath = os.path.join(loss_dir, filename)
    return pd.read_csv(filepath)
#%%
def get_seeds_metrics(loss_dir):
    metric_dict = {}
    for metric_name in metrics:
        metric_dict[metric_name] = []

    for seed in range(num_seeds):
        csv_filename = f"seed_{seed}.csv"
        data = load_csv(loss_dir, csv_filename)
        accuracy, mse, mae, cem, kendall_tau, qwk, macro_acc, amae = calculate_metrics(data['Predictions'],
                                                                                       data['Labels'])
        for metric_name, metric in zip(metrics, [accuracy, mse, mae, cem, kendall_tau, qwk, macro_acc, amae]):
            metric_dict[metric_name].append(metric)
    return metric_dict
#%%
metrics = ["accuracy", "mse", "mae", "cem", "kendall_tau", "qwk", "macro_acc", "amae"]
#%%
def process_data_folder(data_folder, best_results, best_results_csv, all_results, best_alphas):
    all_losses = set(())

    for bin_folder in os.listdir(data_folder):
        bin_path = os.path.join(data_folder, bin_folder)

        for loss_folder in os.listdir(bin_path):
            label_kind = "max"
            # label_kinds = ["max"]
            if loss_folder in losses:
                loss_path = os.path.join(bin_path, loss_folder)
                all_losses.add(loss_folder)
                best_metric_for_loss = {}
                for metric_name in metrics:
                    best_metric_for_loss[metric_name] = 0
                best_metric_for_loss["mse"] = float("inf")
                best_metric_for_loss["mae"] = float("inf")
                best_metric_for_loss["amae"] = float("inf")

                current_preds = {}

                for alpha_folder in os.listdir(loss_path):
                    alpha_path = os.path.join(loss_path, alpha_folder)
                    metric_for_alpha = {}
                    for metric_name in metrics:
                        metric_for_alpha[metric_name] = []

                    for seed_file in os.listdir(alpha_path):
                        seed_path = os.path.join(alpha_path, seed_file)
                        try:
                            df = pd.read_csv(seed_path)
                        except:
                            print(seed_path)

                        if label_kind == "max":
                            accuracy, mse, mae, cem, kendall_tau, qwk, macro_acc, amae = calculate_metrics(
                                df["Predictions"], df["Labels"])

                        for metric_name, metric in zip(metrics,
                                                       [accuracy, mse, mae, cem, kendall_tau, qwk, macro_acc, amae]):
                            metric_for_alpha[metric_name].append(metric)

                    alpha = float(alpha_folder.split("_")[1])

                    for metric in metrics:
                        if (metric != "mse" and metric != "mae" and metric != "amae" and np.mean(
                                metric_for_alpha[metric]) > best_metric_for_loss[metric]) \
                                or ((metric == "mse" or metric == "mae" or metric == "amae") and np.mean(
                            metric_for_alpha[metric]) < best_metric_for_loss[metric]):
                            current_preds[metric] = alpha_path

                            best_metric_for_loss[metric] = np.mean(metric_for_alpha[metric])
                            best_results[data_folder][bin_folder][loss_folder][metric] = [alpha_path, alpha,
                                                                                          best_metric_for_loss[metric],
                                                                                          None]
                            best_results_csv[data_folder][bin_folder][loss_folder][metric] = round(
                                best_metric_for_loss[metric], 3)
                            best_alphas[data_folder][bin_folder][loss_folder][metric] = alpha


                for metric in metrics:
                    for seed_file in os.listdir(current_preds[metric]):
                        seed_path = os.path.join(current_preds[metric], seed_file)
                        try:
                            df = pd.read_csv(seed_path)
                        except:
                            print(seed_path)

                        accuracy, mse, mae, cem, kendall_tau, qwk, macro_acc, amae = calculate_metrics(
                                df["Predictions"], df["Labels"])

                        for metric_name, metric_value in zip(metrics,
                                                       [accuracy, mse, mae, cem, kendall_tau, qwk, macro_acc, amae]):
                            if metric_name==metric:
                                all_results[loss_folder][metric_name].append(metric_value)

    return all_results
#%%
best_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
best_results_csv = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
all_results = defaultdict(lambda: defaultdict(list))
best_alphas = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
#%%
for data_folder in data_sets:
    print(data_folder)
    all_results = process_data_folder(
        os.path.join("results", data_folder), best_results, best_results_csv, all_results, best_alphas)
#%%
def calculate_means(d):
    new_dict = {}
    for key, sub_dict in d.items():
        new_dict[key] = {sub_key: round(np.mean(sub_list),3) for sub_key, sub_list in sub_dict.items()}
    return new_dict

all_results_mean = calculate_means(all_results)

#%%
losses_for_stat=["Cross_Entropy", "SORD", "OLL"]
our_loss="Accumulating_SORD_prox_max"
#%%
stat_results = defaultdict(dict)

#%%
def stat_data_all(all_results,all_results_mean):
    for loss in losses_for_stat:
        for metric_name in metrics:

            t_stat, p_value = ttest_rel(all_results[our_loss][metric_name], all_results[loss][metric_name])
            if (not math.isnan(p_value)) and ((
                                                      metric_name != "mse" and metric_name != "mae" and metric_name != "amae" and p_value < alpha_ttest and
                                                      all_results_mean[our_loss][metric_name] >  all_results_mean[loss][metric_name])
                                              or ((
                                                          metric_name == "mse" or metric_name == "mae" or metric_name == "amae") and p_value < alpha_ttest and
                                                all_results_mean[our_loss][metric_name] < all_results_mean[loss][metric_name])):

                stat_results[loss][metric_name] = round(p_value, 3)
            else:
                stat_results[loss][metric_name] = None

    return stat_results
#%%
# print(stat_data_all(all_results,all_results_mean))
#%%
df_all_results = pd.DataFrame(all_results)
df_all_results_mean = pd.DataFrame(all_results_mean)
stat_data = stat_data_all(all_results,all_results_mean)


df_all_results_mean = df_all_results_mean.astype(str)
# add statistical significant
for metric in metrics:
    if stat_data["Cross_Entropy"][metric] != None and stat_data["Cross_Entropy"][metric] != np.nan:
        df_all_results_mean[our_loss][metric] = df_all_results_mean[our_loss][metric] + '*'

    if stat_data["SORD"][metric] != None and stat_data["SORD"][metric] != np.nan:
        df_all_results_mean[our_loss][metric] = df_all_results_mean[our_loss][metric] + '^'

    if stat_data["OLL"][metric] != None and stat_data["OLL"][metric] != np.nan:
        df_all_results_mean[our_loss][metric] = df_all_results_mean[our_loss][metric] + '`'


if not os.path.exists("../final_summaries"):
    os.makedirs("../final_summaries")
df_all_results_mean.transpose().to_csv(final_path)