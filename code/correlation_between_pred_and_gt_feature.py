from sklearn.metrics import r2_score
import os
import json
import matplotlib.pyplot as plt
import scipy

if __name__ == "__main__":
    #Compute correlation between statistical features extracted from the predicted mask and those extracted from the annotated masks

    data_path = r"./datalists"

    with open(os.path.join(data_path, "averaged_myo_statistics_model.json"), "r") as f:
        myo_statistics = json.load(f)

    with open(os.path.join(data_path, "averaged_myo_statistics_testGT_obs1.json"), "r") as f:
        myo_statistics_testGT_obs1 = json.load(f)

    with open(os.path.join(data_path, "averaged_myo_statistics_testGT_obs2.json"), "r") as f:
        myo_statistics_testGT_obs2 = json.load(f)

    with open(os.path.join(data_path, f"test_list.json"), "r") as f:
        test_patients = json.load(f)

    feature_lists_pred = {'T1_pre': {'average': [], 'lower_quartile': [], 'median': [], 'upper_quartile': []}, 'T2': {'average': [], 'lower_quartile': [], 'median': [], 'upper_quartile': []}}
    feature_lists_obs1 = {'T1_pre': {'average': [], 'lower_quartile': [], 'median': [], 'upper_quartile': []}, 'T2': {'average': [], 'lower_quartile': [], 'median': [], 'upper_quartile': []}}
    feature_lists_obs2 = {'T1_pre': {'average': [], 'lower_quartile': [], 'median': [], 'upper_quartile': []}, 'T2': {'average': [], 'lower_quartile': [], 'median': [], 'upper_quartile': []}}
    
    for pt in myo_statistics:
        if pt in test_patients:
            for T in feature_lists_pred.keys():
                for feature in feature_lists_pred[T].keys():
                    feature_lists_pred[T][feature].append(myo_statistics[pt][T][feature])
                    feature_lists_obs1[T][feature].append(myo_statistics_testGT_obs1[pt][T][feature])
                    feature_lists_obs2[T][feature].append(myo_statistics_testGT_obs2[pt][T][feature])

    plots_path = "./experiments/correlation"
    os.makedirs(plots_path, exist_ok = True)

    with open(os.path.join(plots_path, "correlation_results.txt"), 'w') as f:
        for T in feature_lists_pred.keys():
            for feature in feature_lists_pred[T].keys():
                #Observer 1 vs Observer 2
                r2 =  r2_score(feature_lists_obs1[T][feature], feature_lists_obs2[T][feature])
                #f.write(f"Obs1 vs Obs2 {T} {feature} R2: {r2} \n\n")

                #pearson correlation
                coef, p_value = scipy.stats.pearsonr(feature_lists_obs1[T][feature], feature_lists_obs2[T][feature])
                f.write(f"Obs1 vs Obs2 {T} {feature} Pearson correlation coefficient: {coef}, p_value: {p_value} \n")

                plt.clf()
                plt.scatter(list(feature_lists_obs1[T][feature]), feature_lists_obs2[T][feature])
                plt.title(f"$R^{2}$ = {r2:.2f}")
                plt.xlabel("Observer 1")
                plt.ylabel("Observer 2")
                plt.grid("on")
                #plt.show()
                plt.savefig(os.path.join(plots_path, f"Correlation_obs1_vs_obs2_{T}_{feature}.png"), bbox_inches='tight')

                #Observer 1 vs Prediction
                r2 =  r2_score(feature_lists_obs1[T][feature], feature_lists_pred[T][feature])
                #f.write(f"Obs1 vs Pred {T} {feature} R2: {r2} \n")

                #pearson correlation
                coef, p_value = scipy.stats.pearsonr(feature_lists_obs1[T][feature], feature_lists_pred[T][feature])
                f.write(f"Obs1 vs Pred {T} {feature} Pearson correlation coefficient: {coef}, p_value: {p_value} \n")

                plt.clf()
                plt.scatter(list(feature_lists_obs1[T][feature]), feature_lists_pred[T][feature])
                plt.title(f"$R^{2}$ = {r2:.2f}")
                plt.xlabel("Observer 1")
                plt.ylabel("Model")
                plt.grid("on")
                #plt.show()
                plt.savefig(os.path.join(plots_path, f"Correlation_obs1_vs_pred_{T}_{feature}.png"), bbox_inches='tight')

                #Observer 2 vs Prediction
                r2 =  r2_score(feature_lists_obs2[T][feature], feature_lists_pred[T][feature])
                #f.write(f"Obs2 vs Pred {T} {feature} R2: {r2} \n")

                #pearson correlation
                coef, p_value = scipy.stats.pearsonr(feature_lists_obs2[T][feature], feature_lists_pred[T][feature])
                f.write(f"Obs2 vs Pred {T} {feature} Pearson correlation coefficient: {coef}, p_value: {p_value} \n\n")

                plt.clf()
                plt.scatter(list(feature_lists_obs2[T][feature]), feature_lists_pred[T][feature])
                plt.title(f"$R^{2}$ = {r2:.2f} \n")
                plt.xlabel("Observer 2")
                plt.ylabel("Model")
                plt.grid("on")
                #plt.show()
                plt.savefig(os.path.join(plots_path, f"Correlation_obs2_vs_pred_{T}_{feature}.png"), bbox_inches='tight')

    print("Finished")