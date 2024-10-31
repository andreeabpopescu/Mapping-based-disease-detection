import json
import os
import numpy as np

if __name__ == "__main__":
    data_path = r"./datalists"
    with open(os.path.join(data_path, "myo_statistics_model.json"), "r") as f:
        myo_statistics = json.load(f)

    averaged_statistics = {}
    for pt in myo_statistics.keys():
        T1_A, T1_LQ, T1_M, T1_UQ = [], [], [], []
        T2_A, T2_LQ, T2_M, T2_UQ = [], [], [], []

        averaged_statistics[pt] = {}
        for key in myo_statistics[pt].keys():
            if key=='disease':
                averaged_statistics[pt]['disease'] = myo_statistics[pt]['disease']
            elif key == 'T1_pre':
                for sopuid in myo_statistics[pt][key]:
                    for feature in myo_statistics[pt][key][sopuid]:
                        if feature == 'average':
                            T1_A.append(myo_statistics[pt][key][sopuid][feature])
                        elif feature == 'lower_quartile':
                            T1_LQ.append(myo_statistics[pt][key][sopuid][feature])
                        elif feature == 'median':
                            T1_M.append(myo_statistics[pt][key][sopuid][feature])
                        elif feature == 'upper_quartile':
                            T1_UQ.append(myo_statistics[pt][key][sopuid][feature])
            elif key == 'T2':
                for sopuid in myo_statistics[pt][key]:
                    for feature in myo_statistics[pt][key][sopuid]:
                        if feature == 'average':
                            T2_A.append(myo_statistics[pt][key][sopuid][feature])
                        elif feature == 'lower_quartile':
                            T2_LQ.append(myo_statistics[pt][key][sopuid][feature])
                        elif feature == 'median':
                            T2_M.append(myo_statistics[pt][key][sopuid][feature])
                        elif feature == 'upper_quartile':
                            T2_UQ.append(myo_statistics[pt][key][sopuid][feature])
        
        if len(T1_A) > 0:
            averaged_statistics[pt]['T1_pre'] = {'average': np.mean(T1_A),
                                                'lower_quartile': np.mean(T1_LQ),
                                                'median': np.mean(T1_M),
                                                'upper_quartile': np.mean(T1_UQ)}
        if len(T2_A) > 0:
            averaged_statistics[pt]['T2'] = {'average': np.mean(T2_A),
                                                'lower_quartile': np.mean(T2_LQ),
                                                'median': np.mean(T2_M),
                                                'upper_quartile': np.mean(T2_UQ)}

        with open(os.path.join(data_path, "averaged_myo_statistics_model.json"), "w") as f:
            json.dump(averaged_statistics, f, indent=3)

