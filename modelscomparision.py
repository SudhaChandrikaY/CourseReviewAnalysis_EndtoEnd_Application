import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("reviews.csv")  # Ensure this file contains 'Review' and 'Label' columns
data = data[['Review', 'Label']]

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['Review'], data['Label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Use 5000 most important words
X_train = tfidf_vectorizer.fit_transform(train_texts)
X_val = tfidf_vectorizer.transform(val_texts)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "SGD Classifier": SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

# Store results
results = []
predictions = {}

# Example reviews for predictions
example_reviews = [
    "This course was excellent! The instructor was very knowledgeable and helpful.",
    "The course was okay, but the assignments were too difficult to complete on time.",
    "Absolutely terrible course. The lectures were confusing and unstructured.",
    "The content was good, but the pacing of the course was way too fast.",
    "An amazing learning experience! The projects were challenging yet rewarding."
]
example_tfidf = tfidf_vectorizer.transform(example_reviews)

# Evaluate each model
for name, model in models.items():
    model.fit(X_train, train_labels)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    example_preds = model.predict(example_tfidf)

    # Calculate metrics
    train_loss = np.mean((train_preds - train_labels) ** 2)
    val_loss = np.mean((val_preds - val_labels) ** 2)
    accuracy = accuracy_score(val_labels, val_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_labels, val_preds, average="weighted"
    )

    # Append results
    results.append({
        "Model": name,
        "Training Loss": train_loss,
        "Validation Loss": val_loss,
        "Validation Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
    })

    # Store numeric predictions
    predictions[name] = example_preds

# Convert results to DataFrame for display
results_df = pd.DataFrame(results)

# Prepare prediction results
prediction_df = pd.DataFrame(predictions, index=example_reviews)
prediction_df.index.name = "Example Review"

# Add Actual Sentiments Column with Numeric Labels as the first column
prediction_df.insert(0, "Actual Labels", [5, 3, 1, 3, 5])

# Display results
print("Model Performance Comparison:")
print(results_df)

print("\nExample Predictions by Each Model:")
print(prediction_df)
