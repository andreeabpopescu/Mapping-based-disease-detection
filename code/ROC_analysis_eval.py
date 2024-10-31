import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, auc

from utils import create_data_arrays, draw_confusion_metrics

if __name__ == "__main__":
    data_path = r"./datalists"

    with open(os.path.join(data_path, "averaged_myo_statistics_model.json"), "r") as f:
        myo_statistics = json.load(f)

    with open(os.path.join(data_path, f"test_list.json"), "r") as f:
        test_patients = json.load(f)

    category = 'T1_pre' # T1_pre, T2
    feature = 'upper_quartile' #average, lower_quartile, median, upper_quartile

    X_test, y_test, pt_list = create_data_arrays(myo_statistics,
                                        test_patients,
                                        categories=[category],
                                        features=[feature],
                                        classification='bin')

    metrics_file = f"./experiments/threshold_approach/{category}/metrics_{category}_{feature}.txt"
    with open(metrics_file, 'r') as file:
        line = file.readlines()[2]
        threshold = round(float(line.strip().split(" ")[2]), 1)

    y_real, y_pred = [], []
    for x, y in zip(X_test, y_test):
        y_real.append(y)
        y_pred.append(int(x[0] > threshold))

    #Save all preds
    with open(os.path.join(f"./experiments/threshold_approach/{category}", f"test_predictions_{feature}.json"), 'w') as f:
        json.dump(y_pred, f)

    f1 = f1_score(y_real, y_pred)
    acc = accuracy_score(y_real, y_pred)
    precision = precision_score(y_real, y_pred)
    recall = recall_score(y_real, y_pred)  
    cf_matrix = confusion_matrix(y_real, y_pred)

    exp_path = f"./experiments/threshold_approach/{category}"
    exp_name = f'{category}_{feature}'

    draw_confusion_metrics(cf_matrix, exp_path, exp_name)
    with open(metrics_file, 'a') as f:
        f.write("\nTest metrics: \n")
        f.write(f"F1 score: {f1}\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")

    print()

