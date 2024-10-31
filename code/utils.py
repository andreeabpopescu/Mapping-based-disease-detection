import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, auc
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ROC_AUC(tpr, fpr, auc, save_path):
    plt.rcParams['font.size'] = 17
    plt.figure(figsize=(10, 8), dpi=300)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    #plt.title("AUC & ROC Curve")
    plt.plot(fpr, tpr, 'b')
    plt.fill_between(fpr, tpr, facecolor='lightblue', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=20, weight='bold', color='black')
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    plt.savefig(save_path, bbox_inches='tight')

def assigne_class(actual_class):
  #return class 0 for healthy patient and 1 for any disaesed patient
  if actual_class == 'normal':
    return 0
  else:
    return 1

def assigne_class_multi(actual_class):
  if actual_class == 'normal':
    return 0
  elif actual_class == 'myokarditis':
    return 1
  elif actual_class == 'sarkoidose':
    return 2
  elif actual_class == 'systemerkrankung':
    return 3

def assigne_class_myo(actual_class):
  if actual_class == 'myokarditis':
    return 1
  else:
    return 0

def assigne_class_sar(actual_class):
  if actual_class == 'sarkoidose':
    return 1
  else:
    return 0

def assigne_class_sys(actual_class):
  if actual_class == 'systemerkrankung':
    return 1
  else:
    return 0

def create_data_arrays(myo_statistics, patients, categories, features, classification='bin'):
 
  '''
  classification - how will be the classes assigned
                'bin' - binary classification 1-diseased, 0-normal 
                'multi' - multi-class classification - each label - one class
                'myo' - binary classification: 1-myokarditis, 0-rest
                'sar' - binary classification: 1-sarkoidose, 0-rest
                'sys' - binary classification: 1-systemerkrankung, 0-rest
  '''
  X, y, pt_list = [], [], []

  for pt in patients:
      row = []
      for category, feature in zip(categories, features):
        with open(f'./datalists/patients_with_{category}.json', 'r') as f:
          pations_with_req_data = json.load(f)

          if pt in pations_with_req_data:
              row.append(myo_statistics[pt][category][feature])
              pt_list.append(pt)
          
      if len(row) != len(categories):
        continue

      X.append(row)

      if classification=='bin':
        y.append(assigne_class(myo_statistics[pt]['disease']))
      elif classification=='multi':
        y.append(assigne_class_multi(myo_statistics[pt]['disease']))
      elif classification=='myo':
        y.append(assigne_class_myo(myo_statistics[pt]['disease']))
      elif classification=='sar':
        y.append(assigne_class_sar(myo_statistics[pt]['disease']))
      elif classification=='sys':
        y.append(assigne_class_sys(myo_statistics[pt]['disease']))

  X = np.array(X)
  y = np.array(y)

  return X, y, pt_list

def calculate_tpr_fpr(y_real, y_pred):
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr

def get_n_roc_coordinates_custom(y_real, feature, minim, maxim, step):
    tpr_list = []
    fpr_list = []
    youden_j_list = []

    thresholds_list, f1_list, acc_list, prec_list, rec_list = [], [], [], [], []
    minim = minim - step
    maxim = maxim + step
    n = round((maxim - minim)/step)

    for i in range(n+1):
        threshold = minim + i*step
        y_pred = feature > threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        youden_j_list.append(tpr - fpr)

        thresholds_list.append(threshold)
        f1_list.append(f1_score(y_real, y_pred))
        acc_list.append(accuracy_score(y_real, y_pred))
        prec_list.append(precision_score(y_real, y_pred))
        rec_list.append(recall_score(y_real, y_pred))

    auc_val = auc(fpr_list, tpr_list)

    return tpr_list, fpr_list, youden_j_list, auc_val, (thresholds_list, f1_list, acc_list, prec_list, rec_list)

def draw_confusion_metrics(cf_matrix, exp_path, exp_name):
  plt.rcParams["font.size"] = 14
  
  group_names = ["True Negative","False Positive","False Negative","True Positive"]
  group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
  group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
  labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)

  plt.figure(figsize = (6,6))
  sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='binary', cbar = False)
  plt.savefig(os.path.join(exp_path, f'confusion_matrix_{exp_name}.png'), dpi=300, bbox_inches='tight')
