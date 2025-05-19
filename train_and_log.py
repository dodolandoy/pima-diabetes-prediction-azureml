# === Imports ===
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from azureml.core import Run

run = Run.get_context()
mlflow.start_run(run_id=run.id)



# === Training and model logging===
def evaluate_models(models_dict, X_train, y_train, X_test, y_test, cv_splitter):
    results = []

    # Plots
    plt.figure(figsize=(8, 6))  # ROC
    fig_roc = plt.gcf()

    plt.figure(figsize=(8, 6))  # Learning curve
    fig_lc = plt.gcf()
    train_sizes_vals = np.linspace(0.1, 1.0, 5)

    all_roc_data = pd.DataFrame()
    all_learning_data = pd.DataFrame()


    for name, model in models_dict.items():
        print(f"Start of treatment for {name}")

        # Cross-validation
        acc_train = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv_splitter)
        rec_train = cross_val_score(model, X_train, y_train, scoring='recall', cv=cv_splitter)
        prec_train = cross_val_score(model, X_train, y_train, scoring='precision', cv=cv_splitter)
        f1_train = cross_val_score(model, X_train, y_train, scoring='f1', cv=cv_splitter)

        # Fit
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]
        y_proba_train = model.predict_proba(X_train)[:, 1]

        # Metrics Test
        acc_test = accuracy_score(y_test, y_pred)
        rec_test = recall_score(y_test, y_pred)
        prec_test = precision_score(y_test, y_pred)
        f1_test = f1_score(y_test, y_pred)

        # AUCs
        fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test)
        fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train)
        auc_test = auc(fpr_test, tpr_test)
        auc_train = auc(fpr_train, tpr_train)

        roc_df = pd.DataFrame({
        "Model": [name] * (len(fpr_train) + len(fpr_test)),
        "Set": ["Train"] * len(fpr_train) + ["Test"] * len(fpr_test),
        "False Positive Rate": list(fpr_train) + list(fpr_test),
        "True Positive Rate": list(tpr_train) + list(tpr_test)
        })

        all_roc_data = pd.concat([all_roc_data, roc_df], ignore_index=True)


        metrics = {
            "Train Accuracy": np.mean(acc_train),
            "Test Accuracy": acc_test,
            "Train Recall": np.mean(rec_train),
            "Test Recall": rec_test,
            "Train Precision": np.mean(prec_train),
            "Test Precision": prec_test,
            "Train F1 Score": np.mean(f1_train),
            "Test F1 Score": f1_test,
            "Train AUC": auc_train,
            "Test AUC": auc_test
        }

        run.log_table(name=f"{name}_metrics", value=metrics)

        # === Courbe ROC
        plt.figure(fig_roc.number)
        plt.plot(fpr_test, tpr_test, label=f"{name} - Test (AUC = {auc_test:.2f})")
        plt.plot(fpr_train, tpr_train, linestyle='--', label=f"{name} - Train (AUC = {auc_train:.2f})")

        # === Learning curve 
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=cv_splitter, scoring='f1',
                train_sizes=train_sizes_vals, n_jobs=-1
            )
            lc_df = pd.DataFrame({
            "Model": [name] * len(train_sizes),
            "Training Size": train_sizes,
            "Train Score": train_scores.mean(axis=1),
            "Validation Score": val_scores.mean(axis=1)
            })

            all_learning_data = pd.concat([all_learning_data, lc_df], ignore_index=True)

            plt.figure(fig_lc.number)
            plt.plot(train_sizes, train_scores.mean(axis=1), '--', label=f"{name} - Train")
            plt.plot(train_sizes, val_scores.mean(axis=1), '-', label=f"{name} - Validation")
        except Exception as e:
            print(f"Erreur learning curve for {name}: {e}")

        # === Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix - {name}")
        plt.grid(False)
        mlflow.log_figure(plt.gcf(), f"{name.replace(' ', '_')}_confusion_matrix.png")
        plt.close()

        # === Export model
        model_path = f"pimadiabetes_{name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        results.append({ "Model": name, **metrics })

    # === Saving figures
    plt.figure(fig_roc.number)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False positive rate")
    plt.ylabel("Rate of true positives")
    plt.title("Courbe ROC - Test vs Train")
    plt.legend()
    plt.grid()
    mlflow.log_figure(fig_roc, "roc_curve_comparison.png")
    plt.close()

    plt.figure(fig_lc.number)
    plt.xlabel("Training sample size")
    plt.ylabel("F1 Score")
    plt.title("Learning curves - Train vs Validation")
    plt.legend()
    plt.grid()
    mlflow.log_figure(fig_lc, "learning_curves_comparison.png")
    plt.close()

    # === Export results to CSV for viewing in Outputs + logs
    results_df = pd.DataFrame(results)
    results_df.to_csv("metrics_summary.csv", index=False)
    mlflow.log_artifact("metrics_summary.csv")

    # === Export ROC and Learning Curve data for Power BI
    all_roc_data.to_csv("roc_combined.csv", index=False)
    mlflow.log_artifact("roc_combined.csv")

    all_learning_data.to_csv("learning_curves_combined.csv", index=False)
    mlflow.log_artifact("learning_curves_combined.csv")

    return results_df


# === Principal pipeline ===
def main(input_path):
    try:
        data = pd.read_csv(input_path, engine='python')

        X = data.drop(columns="HasDiabetes")
        y = data["HasDiabetes"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        models = {
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced')),
            "SVM": make_pipeline(StandardScaler(), SVC(class_weight='balanced', probability=True)),
        }

        cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        results_df = evaluate_models(models, X_train, y_train, X_test, y_test, cv_splitter)
        print(results_df)

    except FileNotFoundError:
        print(f"Error: file not found in location {input_path}")
    except Exception as e:
        print(f"Unexpected error : {e}")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to the cleaned CSV file")
    args = parser.parse_args()
    main(args.input_path)
