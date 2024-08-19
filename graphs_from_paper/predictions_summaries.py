# save for each loss best alphas and best results for each metric on every dataset
import sklearn.metrics
import os
from collections import defaultdict
from utils import *
from sklearn.metrics import cohen_kappa_score
from scipy.stats import kendalltau
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_absolute_error

# %%
num_seeds = 10
# %%

# data_sets=["ailerons5","bank5","elevators5","pumadyn5","census5","california5","abalone5","ailerons10","bank10","elevators10","pumadyn10","census10","california10","abalone10","stock5","stock10","boston5","boston10"]
# data_sets=["tae","thyroid","eucalyptus","wine_full","ERA","ESL","car","balance-scale","SWD","LEV"]
data_sets = ["ailerons7", "bank7", "elevators7", "pumadyn7", "census7", "california7", "abalone7", "stock7", "boston7"]


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


# %%
def load_csv(loss_dir, filename):
    filepath = os.path.join(loss_dir, filename)
    return pd.read_csv(filepath)


# %%
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


metrics = ["accuracy", "mse", "mae", "cem", "kendall_tau", "qwk", "macro_acc", "amae"]


# %%
# upadte dicts with best results (metrics and alpha per loss) for data
def process_data_folder(data_folder, best_results, best_results_csv, best_alphas):
    # for binning in data
    for bin_folder in os.listdir(data_folder):
        bin_path = os.path.join(data_folder, bin_folder)
        # for loss
        for loss_folder in os.listdir(bin_path):
            loss_path = os.path.join(bin_path, loss_folder)
            best_metric_for_loss = {}

            # initializing max metric values
            for metric_name in metrics:
                best_metric_for_loss[metric_name] = 0
            best_metric_for_loss["mse"] = float("inf")
            best_metric_for_loss["mae"] = float("inf")
            best_metric_for_loss["amae"] = float("inf")

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
                        best_metric_for_loss[metric] = np.mean(metric_for_alpha[metric])
                        best_results[data_folder][bin_folder][loss_folder][metric] = [alpha_path, alpha,
                                                                                      best_metric_for_loss[metric],
                                                                                      None]
                        best_results_csv[data_folder][bin_folder][loss_folder][metric] = round(
                            best_metric_for_loss[metric], 3)
                        best_alphas[data_folder][bin_folder][loss_folder][metric] = alpha

    return best_results_csv, best_alphas


# %%
best_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
best_results_csv = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
best_alphas = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

# %%
for data_folder in data_sets:
    print(data_folder)
    best_results_csv, best_alphas = process_data_folder(
        os.path.join("results", data_folder), best_results, best_results_csv, best_alphas)
# %%
# save for each loss best alphas and best results for metrics on every dataset
for data_folder in data_sets:
    data_folder = os.path.join("results", data_folder)

    for bin_folder in os.listdir(data_folder):
        bin_path = os.path.join(data_folder, bin_folder)

        df_best_results = pd.DataFrame(best_results_csv[data_folder][bin_folder])
        df_best_alphas = pd.DataFrame(best_alphas[data_folder][bin_folder])
        df_best_results = df_best_results.astype(str)

        directory_path = f"outputs/{data_folder}/"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        df_best_results.transpose().to_csv(directory_path + f"{bin_folder}_metrics.csv")
        df_best_alphas.transpose().to_csv(directory_path + f"{bin_folder}_alphas.csv")
# %%
