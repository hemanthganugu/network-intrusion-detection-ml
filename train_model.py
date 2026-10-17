"""Train and evaluate the demo intrusion detection model."""
import argparse
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from src.model_utils import FEATURES, load_dataset, build_pipeline, train_and_save

def main():
    parser = argparse.ArgumentParser(description="Train a network intrusion detection model.")
    parser.add_argument("--data", default="data/sample_network_traffic.csv",
                        help="CSV path containing the documented feature columns and label")
    args = parser.parse_args()

    data = load_dataset(args.data)
    X, y = data[FEATURES], data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\\n=== Evaluation on held-out split ===")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print("\\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix (labels in sorted order):")
    print(confusion_matrix(y_test, pred, labels=sorted(y.unique())))

    # Save a final model trained on all rows for the app.
    _, _, _ = train_and_save(args.data)
    print("\\nSaved final model to models/intrusion_detector.joblib")

if __name__ == "__main__":
    main()
