import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import json
from pickle import dump, load

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, auc
from sklearn.preprocessing import StandardScaler

from utils import create_data_arrays, draw_confusion_metrics
from fully_connected_net import dataset, Net

def apply_scaling(scaler_file, X_test):
    scaler = load(open(scaler_file, 'rb'))
    X_test = scaler.transform(X_test)

    return X_test
  
if __name__ == "__main__":
 
    save_dir = r"./experiments/NN/NN_T1_A_LQ_M_UQ_T2_A_LQ_M_UQ"
    torch.manual_seed(0)

    #Category in [T1_pre, T1_post, T2]; Feature in [average, lower_quartile, median, upper_quartile]
    categories = ['T1_pre', 'T1_pre', 'T1_pre', 'T1_pre', 'T2', 'T2', 'T2', 'T2']
    features = ['average', 'lower_quartile', 'median', 'upper_quartile', 'average', 'lower_quartile', 'median', 'upper_quartile']

    trial, lr, bs = 19, 0.1, 8    # optimal parameters!!!
    exp_name = f"{trial}_experiments_NN_lr_{lr}_bs_{bs}"
    exp_dir = os.path.join(save_dir, exp_name)

    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    print("Working on deivce: ", device)

    data_path = r"./datalists"

    #Load_data
    with open(os.path.join(data_path, "averaged_myo_statistics_model.json"), "r") as f:
        myo_statistics = json.load(f)

    with open(os.path.join(data_path, f"test_list.json"), "r") as f:
        test_patients = json.load(f)

    print("Starting training")

    X_test, y_test, _ = create_data_arrays(myo_statistics, test_patients, categories=categories, features=features, classification='bin')

    print("Number of testing samples: ", len(y_test))

    ### Normalization
    scaler_file = os.path.join(save_dir, 'scaler.pkl')
    if os.path.exists(scaler_file):
        X_test = apply_scaling(scaler_file, X_test)
    ###
    
    testset = dataset(X_test, y_test)

    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    model = Net() 
    model.to(device)
    model_path = os.path.join(exp_dir, f"model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    all_preds, all_trues = [], []

    with torch.no_grad():
        for i,(input,label) in enumerate(testloader):

            input, label = input.to(device, dtype=torch.float), label.to(device, dtype=torch.float)

            output = model(input)

            all_preds.append(int(output>0.5))
            all_trues.append(label.item())

    with open(os.path.join(save_dir, "test_predictions.json"), 'w') as f:
        json.dump(all_preds, f)

    f1 = f1_score(all_trues, all_preds)
    accuracy = accuracy_score(all_trues, all_preds)
    precision = precision_score(all_trues, all_preds)
    recall = recall_score(all_trues, all_preds)
    cf_matrix = confusion_matrix(all_trues, all_preds)

    draw_confusion_metrics(cf_matrix, save_dir, exp_name)
    with open(os.path.join(save_dir, "hyperparameter_tuning_results.txt"), 'a') as f:
        f.write(f"\n\nResults on test - LR {lr}, BS {bs}\n")
        f.write(f"F1 score {f1}\n")
        f.write(f"Accuracy {accuracy}\n")
        f.write(f"Precision {precision}\n")
        f.write(f"Recall {recall}\n")

    print("Finished testing.")

   

             