import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import json
from pickle import dump, load

from sklearn.metrics import f1_score 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import create_data_arrays
from fully_connected_net import dataset, Net

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
  
    save_dir = r"./experiments/NN/NN_T1_A_LQ_M_UQ_T2_A_LQ_M_UQ"
    torch.manual_seed(0)

    #Category in [T1_pre, T1_post, T2]; Feature in [average, lower_quartile, median, upper_quartile]
    categories = ['T1_pre', 'T1_pre', 'T1_pre', 'T1_pre', 'T2', 'T2', 'T2', 'T2']
    features = ['average', 'lower_quartile', 'median', 'upper_quartile', 'average', 'lower_quartile', 'median', 'upper_quartile']

    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    print("Working on deivce: ", device)

    data_path = r",/datalists"

    #Load_data
    with open(os.path.join(data_path, "averaged_myo_statistics_model.json"), "r") as f:
        myo_statistics = json.load(f)

    with open(os.path.join(data_path, f"train_list.json"), "r") as f:
        train_patients = json.load(f)

    with open(os.path.join(data_path, f"val_list.json"), "r") as f:
        val_patients = json.load(f)

    print("Starting training")
    trial = 1

    #hyper parameters
    learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1]
    epochs = 200
    batch_size = [1, 4, 8, 16]

    X_train, y_train, _ = create_data_arrays(myo_statistics, train_patients, categories=categories, features=features, classification='bin')
    X_val, y_val, _ = create_data_arrays(myo_statistics, val_patients, categories=categories, features=features, classification='bin')

    ### Normalization
    scaler_file = os.path.join(save_dir, 'scaler.pkl')
    if not os.path.exists(scaler_file):
        save_scaler(scaler_file, X_train, X_val, scaler_type='standard')
    X_train, X_val = apply_scaling(scaler_file, X_train, X_val)
    ###

    trainset = dataset(X_train, y_train)
    valset = dataset(X_val, y_val)


    print("Number of training samples: ", len(y_train))
    print("Number of validation samples: ", len(y_val))

    loss_fn = nn.BCELoss()

    exp_losses_list = []
    param_combinations = []
    for lr in learning_rate:
        for bs in batch_size:
            # Model 
            model = Net() 
            model.to(device)

            #DataLoader
            trainloader = DataLoader(trainset, batch_size=bs, shuffle=True)
            valloader = DataLoader(valset, batch_size=1, shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(),lr=lr)

            exp_name = f"{trial}_experiments_NN_lr_{lr}_bs_{bs}"
            exp_dir = os.path.join(save_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            
            train_losses = []
            eval_losses = []
            best_eval_loss = 10000

            for epoch in range(epochs):
                train_loss, eval_loss = [], []

                model.train()
                for i,(input,label) in enumerate(trainloader):
                    
                    input, label = input.to(device, dtype=torch.float), label.to(device, dtype=torch.float)

                    output = model(input)
                
                    loss = loss_fn(output,label.reshape(-1,1))
                    train_loss.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_loss = np.sum(train_loss)/ len(train_loss)
                train_losses.append(train_loss)
            
                #Validation
                all_preds, all_trues = [], []

                model.eval()
                with torch.no_grad():
                    for i,(input,label) in enumerate(valloader):

                        input, label = input.to(device, dtype=torch.float), label.to(device, dtype=torch.float)

                        output = model(input)

                        loss = loss_fn(output,label.reshape(-1,1))
                        eval_loss.append(loss.item())

                        all_preds.append(int(output>0.5))
                        all_trues.append(label.item())

                eval_loss = np.sum(eval_loss)/ len(eval_loss)
                eval_losses.append(eval_loss)

                if eval_loss <= best_eval_loss:
                    #save new best model
                    torch.save(model.state_dict(), os.path.join(exp_dir, f"model.pth"))
                    best_eval_loss = eval_loss

            plt.rcParams.update({'font.size': 13})
            plt.plot(train_losses)
            plt.plot(eval_losses, 'r')
            plt.legend(['Train', 'Evaluation'])
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.grid('on')
            plt.savefig(os.path.join(exp_dir, f'Loss.png'), dpi=300)
            plt.clf()

            with open(os.path.join(save_dir, 'hyperparameter_tuning_results.txt'), 'a') as ff:
                best_epoch = np.argmin(eval_losses)
                new_info = f"\nLearning rate: {lr}; Batch size: {bs}; Best epoch: {best_epoch}; Loss on validation: {eval_losses[best_epoch]}"
                ff.write(new_info)
            print(new_info)

            exp_losses_list.append(eval_losses[best_epoch])
            param_combinations.append((lr, bs))
            trial += 1


    best_index = np.argmin(exp_losses_list)
    optimal_lr, optimal_bs = param_combinations[best_index]
    
    with open(os.path.join(save_dir, 'hyperparameter_tuning_results.txt'), 'a') as ff:
       new_info = f"\n\nBest results achieved for [lr, bs] = [{optimal_lr}, {optimal_bs}]. Loss on  validation subset: {exp_losses_list[best_index]}"
       ff.write(new_info)

    print("Finished training")
    print(save_dir)