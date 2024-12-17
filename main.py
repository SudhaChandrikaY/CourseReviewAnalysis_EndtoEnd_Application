
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset
data = pd.read_csv("reviews.csv")  # Replace with your dataset file
data = data[['Review', 'Label']]  # Ensure these columns exist in your dataset

# Visualization 1: Distribution of Labels (Bar Plot)
def plot_label_distribution(data):
    label_counts = data['Label'].value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.title("Distribution of Labels")
    plt.xlabel("Labels (1-5)")
    plt.ylabel("Count")
    plt.xticks([0, 1, 2, 3, 4], labels=["1", "2", "3", "4", "5"])
    plt.show()

# Visualization 2: Distribution of Labels (Pie Chart)
def plot_label_pie_chart(data):
    label_counts = data['Label'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(label_counts)))
    plt.title("Proportion of Labels")
    plt.axis('equal')
    plt.show()

# Visualization 3: Word Cloud of Reviews
def plot_word_cloud(data):
    text = " ".join(review for review in data['Review'])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Reviews")
    plt.show()

# Visualization 4: Length of Reviews by Label
def plot_review_length_distribution(data):
    data['Review_Length'] = data['Review'].apply(len)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Label', y='Review_Length', data=data, palette="coolwarm")
    plt.title("Review Length Distribution by Label")
    plt.xlabel("Labels")
    plt.ylabel("Review Length (Characters)")
    plt.show()

# Visualization 5: Average Review Length by Label
def plot_avg_review_length(data):
    data['Review_Length'] = data['Review'].apply(len)
    avg_lengths = data.groupby('Label')['Review_Length'].mean()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=avg_lengths.index, y=avg_lengths.values, palette="mako")
    plt.title("Average Review Length by Label")
    plt.xlabel("Labels")
    plt.ylabel("Average Review Length (Characters)")
    plt.show()

# Call all visualizations
plot_label_distribution(data)
plot_label_pie_chart(data)
plot_word_cloud(data)
plot_review_length_distribution(data)
plot_avg_review_length(data)

# Main Model

# Disable Weights & Biases logging
os.environ["WANDB_DISABLED"] = "true"

# Load dataset
data = pd.read_csv("reviews.csv")  # Replace with your dataset file
data = data[['Review', 'Label']]  # Ensure these columns exist in your dataset

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['Review'], data['Label'] - 1, test_size=0.2, random_state=42
)

# Tokenizer setup
model_name = "microsoft/deberta-v3-base"  # Use DeBERTa
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# PyTorch Dataset
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentimentDataset(train_encodings, list(train_labels))
val_dataset = SentimentDataset(val_encodings, list(val_labels))

# Load pretrained model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Training arguments with wandb disabled
training_args = TrainingArguments(
    output_dir="./DeBERT_results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    report_to="none"  # Ensure no logging to wandb or other trackers
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./DeBert_model")
tokenizer.save_pretrained("./DeBert_model")

# Evaluate the model
trainer.evaluate()

# Set the device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model and tokenizer for inference
model = AutoModelForSequenceClassification.from_pretrained("./DeBert_model")  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained("./DeBert_model")  # Path to your saved tokenizer

# Move the model to the correct device
model.to(device)

# Prediction function
def DeBERT_predict_sentiment(review):
    # Tokenize the input
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Move inputs to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction + 1  # Convert 0-indexed label to 1-indexed


# Example reviews for prediction
example_reviews = [
    "This course was excellent! The instructor was very knowledgeable and helpful.",
    "The course was okay, but the assignments were too difficult to complete on time.",
    "Absolutely terrible course. The lectures were confusing and unstructured.",
    "The content was good, but the pacing of the course was way too fast.",
    "An amazing learning experience! The projects were challenging yet rewarding."
]

# Actual labels for example reviews
actual_labels = [5, 3, 1, 3, 5]  # Provided actual labels for the example reviews

# Predict sentiments for example reviews
predicted_labels = [DeBERT_predict_sentiment(review) for review in example_reviews]

# Create a DataFrame for actual vs predicted labels
results_df = pd.DataFrame({
    "Example Review": example_reviews,
    "Actual Label": actual_labels,
    "Predicted Label": predicted_labels
})

# Use tabulate to display the table neatly
print("\nDeBERTa Actual vs Predicted Labels")
print(tabulate(results_df, headers="keys", tablefmt="fancy_grid", showindex=False))


# Confusion Matrix for BERT
def plot_confusion_matrix(trainer, val_dataset):
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    cm = confusion_matrix(val_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix DeBERTA")
    plt.show()

plot_confusion_matrix(trainer, val_dataset)

import numpy as np
# Extract per-epoch logs from the training process
def extract_epoch_logs(trainer_state):
    logs = {
        "epoch": [],
        "training_loss": [],
        "validation_loss": [],
        "training_accuracy": [],
        "validation_accuracy": []
    }
    epoch_training_losses = []
    epoch_training_preds = []
    epoch_training_labels = []
    current_epoch = None

    for log in trainer_state.log_history:
        # Collect batch-wise training losses and predictions
        if "loss" in log and "epoch" in log:
            if current_epoch is None or current_epoch == log["epoch"]:
                epoch_training_losses.append(log["loss"])
                if "training_preds" in log and "training_labels" in log:
                    epoch_training_preds.extend(log["training_preds"])
                    epoch_training_labels.extend(log["training_labels"])
            else:
                # Average the training losses and calculate accuracy for the completed epoch
                logs["training_loss"].append(np.mean(epoch_training_losses))
                if epoch_training_preds and epoch_training_labels:
                    train_accuracy = accuracy_score(epoch_training_labels, epoch_training_preds)
                    logs["training_accuracy"].append(train_accuracy)
                epoch_training_losses = [log["loss"]]  # Reset for the next epoch
                epoch_training_preds = []
                epoch_training_labels = []

            current_epoch = log["epoch"]

        # Collect validation metrics at epoch-end
        if "eval_loss" in log and "epoch" in log:
            logs["epoch"].append(int(log["epoch"]))
            logs["validation_loss"].append(log["eval_loss"])
        if "eval_accuracy" in log and "epoch" in log:
            logs["validation_accuracy"].append(log["eval_accuracy"])

    # Add the last epoch's training loss and accuracy
    if epoch_training_losses:
        logs["training_loss"].append(np.mean(epoch_training_losses))
    if epoch_training_preds and epoch_training_labels:
        train_accuracy = accuracy_score(epoch_training_labels, epoch_training_preds)
        logs["training_accuracy"].append(train_accuracy)

    # Ensure alignment of logs
    max_epochs = len(logs["epoch"])
    logs["training_loss"] = logs["training_loss"][:max_epochs]
    logs["validation_loss"] = logs["validation_loss"][:max_epochs]
    logs["training_accuracy"] = logs["training_accuracy"][:max_epochs]
    logs["validation_accuracy"] = logs["validation_accuracy"][:max_epochs]

    return logs
# Retrieve logs
logs = extract_epoch_logs(trainer.state)


# Align epochs
epochs = range(1, len(logs["epoch"]) + 1)