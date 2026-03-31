import os
import pickle
import sys

import numpy as np
import pandas as pd
from skExSTraCS import ExSTraCS
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def train_lcs(
    features_dir="/content/drive/MyDrive/cnn-lcs/features",
    output_dir="/content/drive/MyDrive/cnn-lcs/lcs",
):
    os.makedirs(output_dir, exist_ok=True)

    X_train = np.load(f"{features_dir}/train_features.npy")
    y_train = np.load(f"{features_dir}/train_labels.npy")
    X_test = np.load(f"{features_dir}/test_features.npy")
    y_test = np.load(f"{features_dir}/test_labels.npy")

    print(
        f"Loaded features: "
        f"train={X_train.shape}, test={X_test.shape}, "
        f"train_normal={(y_train == 0).sum()}, train_anomalous={(y_train == 1).sum()}"
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    with open(f"{output_dir}/scaler.pkl", "wb") as file_obj:
        pickle.dump(scaler, file_obj)

    selector = SelectKBest(f_classif, k=50)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    selected_indices = selector.get_support(indices=True)
    print(f"Selected features: k={len(selected_indices)}")

    with open(f"{output_dir}/selector.pkl", "wb") as file_obj:
        pickle.dump(selector, file_obj)

    np.save(f"{output_dir}/selected_feature_indices.npy", selected_indices)

    print("Training ExSTraCS...")
    model = ExSTraCS(learning_iterations=50000, N=2000, nu=10)
    model.fit(X_train, y_train)

    with open(f"{output_dir}/lcs_model.pkl", "wb") as file_obj:
        pickle.dump(model, file_obj)
    print(f"Saved LCS model to {output_dir}/lcs_model.pkl")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("LCS test evaluation")
    print(f"Accuracy:  {(y_pred == y_test).mean():.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"AUC:       {roc_auc_score(y_test, y_prob):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("(rows=actual, cols=predicted | 0=normal, 1=anomalous)")

    try:
        pop = model.population.popSet
        print(f"Total rules in population: {len(pop)}")

        rules = sorted(pop, key=lambda rule: rule.accuracy * rule.numerosity, reverse=True)

        print("Top 10 rules:")
        for i, rule in enumerate(rules[:10]):
            mapped_attributes = [int(selected_indices[attr]) for attr in rule.specifiedAttList]
            print(f"Rule {i + 1}:")
            print(f"  Condition:               {rule.condition}")
            print(f"  Prediction:              {'Anomalous' if rule.phenotype == 1 else 'Normal'}")
            print(f"  Accuracy:                {rule.accuracy:.4f}")
            print(f"  Numerosity:              {rule.numerosity}")
            print(f"  Fitness:                 {rule.fitness:.4f}")
            print(f"  Selected feature index:  {rule.specifiedAttList}")
            print(f"  Original feature index:  {mapped_attributes}")

        rules_data = []
        for rule in rules:
            mapped_attributes = [int(selected_indices[attr]) for attr in rule.specifiedAttList]
            rules_data.append(
                {
                    "condition": str(rule.condition),
                    "prediction": "Anomalous" if rule.phenotype == 1 else "Normal",
                    "accuracy": rule.accuracy,
                    "numerosity": rule.numerosity,
                    "fitness": rule.fitness,
                    "selected_feature_indices": str(rule.specifiedAttList),
                    "original_feature_indices": str(mapped_attributes),
                    "correct_count": rule.correctCount,
                    "match_count": rule.matchCount,
                }
            )

        rules_df = pd.DataFrame(rules_data)
        rules_df.to_csv(f"{output_dir}/rules.csv", index=False)
        print(f"Saved {len(rules_df)} rules to {output_dir}/rules.csv")

    except Exception as error:
        print(f"Could not extract rules: {error}")


if __name__ == "__main__":
    train_lcs()
