import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    Ts = ['T1_pre', 'T2']
    features = ['average', 'lower_quartile', 'median', 'upper_quartile']

    plt.rcParams['font.size'] = 17
    plt.figure(figsize=(10, 8), dpi=1000)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    labels = []

    for T in Ts:
        for feature in features:
            fpr_file = f"./experiments/threshold_approach/{T}/fpr_list_{T}_{feature}.npy"
            tpr_file = f"./experiments/threshold_approach/{T}/tpr_list_{T}_{feature}.npy"

            fpr = np.load(fpr_file)
            tpr = np.load(tpr_file)

            metrics_file = f"./experiments/threshold_approach/{T}/metrics_{T}_{feature}.txt"
            with open(metrics_file, 'r') as file:
                line = file.readline()
                auc = round(float(line.strip().split(" ")[1])*100, 1)

            plt.plot(fpr, tpr)
            f = {'average': 'A', 'median': 'M', 'lower_quartile': 'LQ', 'upper_quartile': 'UQ'}.get(feature, None)
            
            labels.append(f"{T[:2]} {f} (AUC = {auc})")
    
    plt.legend(labels)

    save_path = r"./experiments/threshold_approach/figures"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "ROC_all.png"), bbox_inches='tight', dpi=1000)
    

