import os
import json
import numpy as np

from utils import create_data_arrays, get_n_roc_coordinates_custom

if __name__ == "__main__":
    data_path = r"./datalists"

    with open(os.path.join(data_path, "averaged_myo_statistics_model.json"), "r") as f:
        myo_statistics = json.load(f)

    with open(os.path.join(data_path, f"train_list.json"), "r") as f:
        train_patients = json.load(f)

    with open(os.path.join(data_path, f"val_list.json"), "r") as f:
        train_patients.extend(json.load(f))

    category = 'T1_pre' # T1_pre, T2
    feature = 'upper_quartile'  #average, lower_quartile, median, upper_quartile

    X_train, y_train, _ = create_data_arrays(myo_statistics,
                                        train_patients,
                                        categories=[category],
                                        features=[feature],
                                        classification='bin')

    #Do ROC analysis and find optimal cut off value based on training data
    limits = [round(np.min(X_train)), round(np.max(X_train))]
    
    exp_name = f'{category}_{feature}'

    tpr_list, fpr_list, youden_j_list, auc_val, metrics = get_n_roc_coordinates_custom(y_train, X_train, limits[0], limits[1], 10)

    save_path = f"./experiments/threshold_approach/{category}"
    os.makedirs(save_path, exist_ok = True)

    np.save(os.path.join(save_path, f"tpr_list_{exp_name}.npy"), tpr_list)
    np.save(os.path.join(save_path, f"fpr_list_{exp_name}.npy"), fpr_list)

    thresholds_list, f1_list, acc_list, prec_list, rec_list = metrics   

    optimal_idx = np.argmax(youden_j_list) #The optimal threshold is the one that maximizes the Youden's J statistic (TPR−FPR)
    optimal_threshold = thresholds_list[optimal_idx]

    with open(os.path.join(save_path, f'metrics_{exp_name}.txt'), 'w') as f:
        f.write(f"AUC: {auc_val}\n\n")

        f.write(f"Best threshold: {optimal_threshold}\n")
        f.write("Train metrics: \n")
        f.write(f"F1 score: {f1_list[optimal_idx]}\n")
        f.write(f"Accuracy: {acc_list[optimal_idx]}\n")
        f.write(f"Precision: {prec_list[optimal_idx]}\n")
        f.write(f"Recall: {rec_list[optimal_idx]}\n")



    