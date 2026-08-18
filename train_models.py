import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_excel("Dry_Bean_Dataset.xlsx")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ==========================================
# 5. TRAIN MODELS AND CALCULATE METRICS
# ==========================================

results = {}

os.makedirs("model", exist_ok=True)

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Probability predictions for AUC
    y_proba = model.predict_proba(X_test)

    # Classes
    classes = model.classes_

    # Convert labels to binary matrix for multiclass AUC
    y_test_binary = label_binarize(
        y_test,
        classes=classes
    )

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test_binary,
        y_proba,
        multi_class="ovr",
        average="weighted"
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    results[name] = {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"MCC      : {mcc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    filename = name.lower().replace(" ", "_") + ".pkl"

    with open(os.path.join("model", filename), "wb") as file:
        pickle.dump(model, file)


# ==========================================
# 6. CREATE COMPARISON TABLE
# ==========================================

results_df = pd.DataFrame(results).T

print("\n")
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(results_df.round(4))


# Save results
results_df.round(4).to_csv(
    "model_comparison.csv"
)


# ==========================================
# 7. SAVE TEST DATA
# ==========================================

test_data = X_test.copy()
test_data["Class"] = y_test.values

test_data.to_csv(
    "test_data.csv",
    index=False
)


# ==========================================
# 8. SAVE TEST LABELS
# ==========================================

with open("model/test_labels.pkl", "wb") as file:
    pickle.dump(y_test, file)


print("\nFiles created successfully:")
print("- test_data.csv")
print("- model_comparison.csv")
print("- model/*.pkl")