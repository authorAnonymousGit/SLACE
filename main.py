from result_csvs import *
from utils import *
from process_data import *
from losses import *
from config import *
import csv
import numpy as np
import sklearn.metrics



def pred_xgb(loss, alpha, x_train, y_train, x_val, y_val, x_test, y_test, seed):
    params = PARAMS

    loss_xgb = xgb.XGBClassifier(objective=call_loss(loss, alpha), seed=seed, **params)

    loss_xgb.fit(x_train, y_train, verbose=False, eval_metric=cem,
                 eval_set=[(x_val, y_val)])

    preds = loss_xgb.predict(x_test)
    pred_probabilities = loss_xgb.predict_proba(x_test)

    return preds, pred_probabilities


def calculate_metrics(preds, y_test):
    cem_val = float(cem2(preds, np.array(y_test)))
    accuracy = sum(np.array(y_test) == preds) / len(preds)
    mse = sklearn.metrics.mean_squared_error(y_test, preds)
    return cem_val, accuracy, mse


def process_data_and_metrics(data, bins, loss, alpha, seed):
    full_data = Process_data(data[0], data[1])
    x_train, y_train, x_val, y_val, x_test, y_test = full_data.get_binned_data(bins)

    preds, probabilities = pred_xgb(loss, alpha, x_train, y_train, x_val, y_val, x_test, y_test, seed)
    cem_val, accuracy, mse = calculate_metrics(preds, y_test)

    save_results(data, bins, loss, alpha, preds, probabilities, y_test, seed)

    return cem_val, accuracy, mse, alpha


def save_results(data, bins, loss, alpha, preds, probabilities, y_test, seed):
    # Create directories

    data_dir = os.path.join("results", data[1])
    bin_dir = os.path.join(data_dir, f"bin_{str(bins)}")
    loss_dir = os.path.join(bin_dir, loss)
    alpha_dir = os.path.join(loss_dir, f"alpha_{alpha}")
    os.makedirs(alpha_dir, exist_ok=True)

    # Save CSV files
    csv_filename = f"seed_{seed}.csv"
    csv_path = os.path.join(alpha_dir, csv_filename)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Predictions", "Probabilities", "Labels"])
        for pred, prob, label in zip(preds, probabilities, y_test):
            writer.writerow([pred, prob, label])


def find_best_metrics(data, bins, LOSS, ALPHA):
    best_metrics_per_loss = {loss: [0, 0, ""] for loss in LOSS}

    for loss in LOSS:
        best_cem = [0, 0, ""]
        best_accuracy = [0, 0, ""]
        best_mse = [10000, 0, ""]

        if loss == "Cross_Entropy":
            alpha = 1
            for seed in range(RANGE_OF_SEEDS[0], RANGE_OF_SEEDS[1]):
                cem_val, accuracy, mse, alpha = process_data_and_metrics(data, bins, loss, alpha, seed)

                if cem_val > best_cem[0]:
                    best_cem = [cem_val, alpha, loss]

                if accuracy > best_accuracy[0]:
                    best_accuracy = [accuracy, alpha, loss]

                if mse < best_mse[0]:
                    best_mse = [mse, alpha, loss]



        else:
            for alpha in ALPHA:
                for seed in range(RANGE_OF_SEEDS[0], RANGE_OF_SEEDS[1]):
                    cem_val, accuracy, mse, alpha = process_data_and_metrics(data, bins, loss, alpha, seed)

                    if cem_val > best_cem[0]:
                        best_cem = [cem_val, alpha, loss]

                    if accuracy > best_accuracy[0]:
                        best_accuracy = [accuracy, alpha, loss]

                    if mse < best_mse[0]:
                        best_mse = [mse, alpha, loss]

        best_metrics_per_loss[loss] = {
            "Best CEM": best_cem,
            "Best Accuracy": best_accuracy,
            "Best MSE": best_mse
        }

    return best_metrics_per_loss


if __name__ == '__main__':
    for data in DATA:
        data_name = data[1]

        for bins in data[2]:
            print("Processing data:", data_name, "with bins:", bins)

            best_metrics_per_loss = find_best_metrics(data, bins, LOSS, ALPHA)

