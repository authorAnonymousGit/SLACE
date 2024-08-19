# creating figures 4 and 5 from paper, need to run prediction_summaries.csv before
import os
from collections import defaultdict
from utils import *
import matplotlib.pyplot as plt

#%%
# create figure 5 from paper
metrics=["accuracy", "mse", "mae", "cem", "kendall_tau", "qwk", "macro_acc", "amae"]
datasets=[(["ailerons5","bank5","elevators5","pumadyn5","census5","california5","abalone5","stock5","boston5"],5),(["ailerons7","bank7","elevators7","pumadyn7","census7","california7","abalone7","stock7","boston7"],7),(["ailerons10","bank10","elevators10","pumadyn10","census10","california10","abalone10","stock10","boston10"],10)]

losses=["Cross_Entropy", "SORD", "OLL","Accumulating_SORD_prox_max"]
graph=defaultdict(lambda: defaultdict(float))
#%%
for amount in datasets:
    accuracies= {}
    for loss in losses:
        accuracies[loss]=[]
    for data in amount[0]:
        data_folder="outputs/results/"+data
        for filename in os.listdir(data_folder):
            if filename.startswith('bin') and filename.endswith('metrics.csv'):
                file_path = os.path.join(data_folder, filename)

                df = pd.read_csv(file_path, index_col=0)
                df_no_stat = df.applymap(lambda x: float(str(x).replace('*', '')))

                for loss in losses:
                    accuracies[loss].append(df_no_stat.loc[loss, "cem"])
        for loss in losses:
            graph[loss][amount[1]]=np.mean(accuracies[loss])


#%%
colors = ['#D55E00', '#0072B2', '#009E73', '#CC79A7']  # Orange, Blue, Green, Pink
# line_styles = ['--', '-.', ':','-']
markers = ['o', 's', '^', 'D']

# Plot each line
for i, (name, values) in enumerate(graph.items()):
    x_values = list(values.keys())
    y_values = list(values.values())
    plt.plot(x_values, y_values, label=name, color=colors[i], marker=markers[i])

# Add labels and title
plt.xticks([5, 7, 10])
plt.xlabel('number of classes', fontsize=20)
plt.ylabel('CEM', fontsize=20)
plt.legend(["CE","SORD","OLL","SLACE_prox"],fontsize=15)

# Show plot
plt.show()
plt.close()

#%%
# create figure 4 from paper
datasets=["tae","thyroid","eucalyptus","wine_full","ERA","ESL","car","balance-scale","SWD","LEV"]
#%%
summaries_folder="final_summaries/ordinal.csv"
#%%
df = pd.read_csv(summaries_folder, index_col=0)
df = df.applymap(lambda x: str(x).replace('*', ''))
df = df.applymap(lambda x: str(x).replace('`', ''))
df = df.applymap(lambda x: str(x).replace('^', ''))
#%%
types= [("with prox",["OLL_max","SORD_max","Accumulating_SORD_prox_max"]),("without prox",["OLL","SORD","Accumulating_SORD"])]
losses=["OLL","SORD","SLACE"]
graph_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
#%%
for type in types:
    for i,loss in enumerate(type[1]):
        for metric in metrics:
            graph_metrics[metric][type[0]][losses[i]]=float(df.loc[loss, metric])



def plot_metrics(graph_metrics):
    for metric in ["accuracy", "mae","cem"]:
        graph = graph_metrics[metric]
        x_values = list(next(iter(graph.values())).keys())
        x_indices = np.arange(len(x_values))  # Array of indices for the x positions

        width = 0.3  # Reduce width to bring bars closer together

        fig, ax = plt.subplots(figsize=(12, 8))  # Set figure size

        # Plot each name's values as a bar
        for i, (name, values) in enumerate(graph.items()):
            y_values = list(values.values())
            bars = ax.bar(x_indices + i * width, y_values, width, label=name)

            # Add value labels above the bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.3f}',
                        ha='center', va='bottom', fontsize=30)

        if metric=="mae" or metric=="cem":
            ax.set_ylabel(metric.upper(), fontsize=50)
        if metric=="accuracy":
            ax.set_ylabel("Acc", fontsize=50)
        ax.set_xticks(x_indices + width * (len(graph) - 1) / 2)
        ax.set_xticklabels(x_values, fontsize=50)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=40)
        plt.yticks(fontsize=30)

        ax.set_ylim(0.35, max(max(values.values()) for values in graph.values()) + 0.1)

        # Show plot
        plt.tight_layout()
        plt.show()
        # plt.savefig("prox_vs_no_prox/" + metric + '.png')
        plt.close()

# Example usage
plot_metrics(graph_metrics)