# final_hate_speech_project.py

import pandas as pd
import re
import pickle
from typing import Dict
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler

# ----------------------
# Setup Stopwords
# ----------------------
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))


# ----------------------
# Task 1: Load & Inspect
# ----------------------
def load_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} tweets from {file_path}")
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nClass distribution:")
    print(df['class'].value_counts())


# ----------------------
# Task 2: Cleaning
# ----------------------
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)
    text = text.replace("#", "").lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)


def clean_and_balance(df: pd.DataFrame) -> pd.DataFrame:
    df['cleaned_tweet'] = df['tweet'].apply(clean_text)
    df_hate = df[df['class'] == 0]
    df_offensive = df[df['class'] == 1].sample(n=3500, random_state=42)
    df_neutral = df[df['class'] == 2]
    balanced_df = pd.concat([df_hate, df_offensive, df_neutral])
    balanced_df = balanced_df[['cleaned_tweet', 'class']]
    print(f"Balanced dataset size: {len(balanced_df)}")
    return balanced_df


# ----------------------
# Task 3: Train Model
# ----------------------
def train_model(df: pd.DataFrame, output_file: str) -> None:
    X = df['cleaned_tweet']
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Handle imbalance
    ros = RandomOverSampler(random_state=42)
    X_train_vec_res, y_train_res = ros.fit_resample(X_train_vec, y_train)

    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=42,
        verbose=True
    )

    print("\nTraining model...")
    model.fit(X_train_vec_res, y_train_res)

    predictions = model.predict(X_test_vec)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print(f"Test Accuracy: {accuracy_score(y_test, predictions):.4f}")

    # Save artifacts
    with open(output_file, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'model': model}, f)
    print(f"Model and vectorizer saved to {output_file}")


# ----------------------
# Task 4: Prediction Example
# ----------------------
def predict_example(model_file: str) -> None:
    try:
        with open(model_file, 'rb') as f:
            artifacts = pickle.load(f)
        vectorizer = artifacts['vectorizer']
        model = artifacts['model']

        sample_tweets = [
            "I hate you so much! You're the worst person ever.",
            "Check out my new blog post at http://example.com",
            "Had a great time at the concert last night!"
        ]
        for tweet in sample_tweets:
            cleaned = clean_text(tweet)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            print(f"\nTweet: {tweet}\nPredicted Class: {pred}")
    except FileNotFoundError:
        print(f"Model file '{model_file}' not found. Train the model first.")


# ----------------------
# Main Execution
# ----------------------
if __name__ == "__main__":
    INPUT_FILE = 'hate_speech.csv'
    CLEANED_FILE = 'cleaned_hate_dataset.csv'
    MODEL_FILE = 'hate_speech_model.pkl'

    df = load_dataset(INPUT_FILE)
    inspect_dataset(df)

    cleaned_df = clean_and_balance(df)
    cleaned_df.to_csv(CLEANED_FILE, index=False)

    train_model(cleaned_df, MODEL_FILE)

    predict_example(MODEL_FILE)
