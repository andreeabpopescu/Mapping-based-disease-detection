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

from utils import create_data_arrays

def save_scaler(scaler_file, X_train, X_val, scaler_type):
    X = np.concatenate((X_train, X_val))

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    scaler.fit(X)
    dump(scaler, open(scaler_file, 'wb'))

def apply_scaling(scaler_file, X_train, X_val):
    scaler = load(open(scaler_file, 'rb'))
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val
  
if __name__ == "__main__":
  
    save_dir = r"./experiments/RF/RF_T1_A_LQ_M_UQ_T2_A_LQ_M_UQ"
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(0)

    #Category in [T1_pre, T1_post, T2]; Feature in [average, lower_quartile, median, upper_quartile]
    categories = ['T1_pre', 'T1_pre', 'T1_pre', 'T1_pre', 'T2', 'T2', 'T2', 'T2']
    features = ['average', 'lower_quartile', 'median', 'upper_quartile', 'average', 'lower_quartile', 'median', 'upper_quartile']

    data_path = r"./datalists"

    #Load_data
    with open(os.path.join(data_path, "averaged_myo_statistics_model.json"), "r") as f:
        myo_statistics = json.load(f)

    with open(os.path.join(data_path, f"train_list.json"), "r") as f:
        train_patients = json.load(f)

    with open(os.path.join(data_path, f"val_list.json"), "r") as f:
        val_patients = json.load(f)

    print("Starting training")
    num_of_trees = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
    trial = 1

    X_train, y_train, _ = create_data_arrays(myo_statistics, train_patients, categories=categories, features=features, classification='bin')
    X_val, y_val, _ = create_data_arrays(myo_statistics, val_patients, categories=categories, features=features, classification='bin')


    print("Number of training samples: ", len(y_train))
    print("Number of validation samples: ", len(y_val))

    f1_list = []
    for ntrees in num_of_trees:
        print(f"\nNumber of trees: {ntrees}")

        exp_name = f"{trial}_experiment_RF_{ntrees}"
        exp_dir = os.path.join(save_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        regressor = RandomForestClassifier(n_estimators=ntrees, random_state=0)
        
        regressor.fit(X_train, y_train)

        filename = f'model.sav'
        pickle.dump(regressor, open(os.path.join(exp_dir, filename), 'wb'))

        y_pred = regressor.predict(X_val)
        y_pred = np.array([int(p) for p in y_pred])

        f1 = f1_score(y_val, y_pred)
        f1_list.append(f1)
        print("F1 score on validation: ", f1)

        trial += 1

    best_index = np.argmax(f1_list)
    optimal_nr_trees = num_of_trees[best_index]
    with open(os.path.join(save_dir, "metrics.txt"), 'w') as f:
        f.write(f"Optimal number of trees: {optimal_nr_trees}; F1 score: {f1_list[best_index]}")

    print("Finished training.")