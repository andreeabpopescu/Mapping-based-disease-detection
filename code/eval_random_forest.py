import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pickle
from pickle import dump, load
import json
from sklearn.metrics import f1_score 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, auc

from utils import create_data_arrays, draw_confusion_metrics

  
if __name__ == "__main__":
  
    save_dir = r"./experiments/RF/RF_T1_A_LQ_M_UQ_T2_A_LQ_M_UQ"
    torch.manual_seed(0)

    #Category in [T1_pre, T1_post, T2]; Feature in [average, lower_quartile, median, upper_quartile]
    categories = ['T1_pre', 'T1_pre', 'T1_pre', 'T1_pre', 'T2', 'T2', 'T2', 'T2']
    features = ['average', 'lower_quartile', 'median', 'upper_quartile', 'average', 'lower_quartile', 'median', 'upper_quartile']

    trial, ntrees = 7, 70 # optimal number of trees!!!
    exp_name = f"{trial}_experiment_RF_{ntrees}"
    exp_dir = os.path.join(save_dir, exp_name)

    data_path = r"./datalists"

    with open(os.path.join(data_path, "averaged_myo_statistics_model.json"), "r") as f:
        myo_statistics = json.load(f)

    with open(os.path.join(data_path, f"test_list.json"), "r") as f:
        test_patients = json.load(f)


    print("Starting testing")

    X_test, y_test, _ = create_data_arrays(myo_statistics, test_patients, categories=categories, features=features, classification='bin')

    print("Number of testing samples: ", len(y_test))

    regressor = pickle.load(open(os.path.join(exp_dir, 'model.sav'), 'rb'))
    
    y_pred = regressor.predict(X_test)
    y_pred = np.array([int(p) for p in y_pred])

    #Save all preds
    with open(os.path.join(save_dir, "test_predictions.json"), 'w') as f:
        json.dump(y_pred.tolist(), f)

    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cf_matrix = confusion_matrix(y_test, y_pred)

    draw_confusion_metrics(cf_matrix, save_dir, exp_name)
    with open(os.path.join(save_dir, "metrics.txt"), 'a') as f:
        f.write(f"\n\nResults on test - num of trees {ntrees}\n")
        f.write(f"F1 score {f1}\n")
        f.write(f"Accuracy {accuracy}\n")
        f.write(f"Precision {precision}\n")
        f.write(f"Recall {recall}\n")

    print("Finished testing.")