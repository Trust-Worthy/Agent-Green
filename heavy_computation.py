# heavy_ai_computation.py
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics.pairwise import cosine_similarity
import random

# --- Configuration ---
NUM_SAMPLES = 50000        # Number of data samples
NUM_FEATURES = 200         # Number of features per sample
NUM_TRAINING_EPOCHS = 100  # Simulate model training epochs
INFERENCE_BATCH_SIZE = 1000 # Number of samples to process per inference batch
NUM_INFERENCE_BATCHES = 50 # Total batches for inference
VECTOR_DB_DIMENSION = 512 # Dimension of vectors in our 'database'
NUM_VECTORS_IN_DB = 10000  # Number of vectors in our simulated DB
NUM_QUERY_VECTORS = 100    # Number of query vectors for similarity search

# --- Helper Functions ---

def generate_synthetic_data(num_samples, num_features):
    """Generates a synthetic dataset for classification/regression."""
    print(f"Generating synthetic data: {num_samples} samples, {num_features} features...")
    X = np.random.rand(num_samples, num_features) * 10
    # Create a simple dependent variable based on some features
    y = np.sum(X[:, :5], axis=1) + np.random.randn(num_samples) * 0.5
    print("Data generation complete.")
    return X, y

def preprocess_data(X):
    """Scales data and adds polynomial features."""
    print("Preprocessing data (scaling and polynomial features)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)
    print(f"Preprocessing complete. New feature count: {X_poly.shape[1]}")
    return X_poly

def simulate_model_training(X, y, epochs):
    """Simulates a computationally heavy model training process."""
    print(f"Simulating model training for {epochs} epochs on {X.shape[0]} samples...")
    # This loop is a placeholder for actual deep learning training.
    # Each iteration simulates a batch or epoch processing.
    dummy_model_weights = np.random.rand(X.shape[1], 1)
    learning_rate = 0.001

    for epoch in range(epochs):
        # Simulate forward pass and backward pass
        predictions = np.dot(X, dummy_model_weights)
        errors = predictions - y.reshape(-1, 1)
        gradient = np.dot(X.T, errors) / X.shape[0]
        dummy_model_weights -= learning_rate * gradient

        if (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0:
            loss = np.mean(errors**2)
            print(f"  Epoch {epoch+1}/{epochs}, Simulated Loss: {loss:.4f}", end='\r')
    print("\nSimulated model training complete.")
    return dummy_model_weights # Return dummy weights to simulate a trained model

def simulate_model_inference(X_new, model_weights, batch_size):
    """Simulates inference on new data in batches."""
    print(f"Simulating model inference on {X_new.shape[0]} new samples in batches...")
    all_predictions = []
    num_batches = int(np.ceil(X_new.shape[0] / batch_size))

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X_new.shape[0])
        batch_X = X_new[start_idx:end_idx]

        # Simulate prediction
        batch_predictions = np.dot(batch_X, model_weights)
        all_predictions.extend(batch_predictions.flatten())

        if (i + 1) % (num_batches // 10 if num_batches >= 10 else 1) == 0:
            print(f"  Processed inference batch {i+1}/{num_batches}", end='\r')
    print("\nSimulated model inference complete.")
    return np.array(all_predictions)

def simulate_vector_database_search(db_vectors, query_vectors):
    """Simulates a vector database similarity search."""
    print(f"Simulating vector database search: {query_vectors.shape[0]} queries against {db_vectors.shape[0]} vectors...")

    # Normalize vectors for cosine similarity
    db_vectors_norm = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)
    query_vectors_norm = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

    # Calculate cosine similarity between query vectors and all DB vectors
    # This is often the most computationally intensive part in vector DBs
    # as it involves many dot products.
    similarities = cosine_similarity(query_vectors_norm, db_vectors_norm)

    # For a real vector DB, you'd find top-k similar vectors
    # Here, we just do the full similarity matrix computation
    # and maybe a dummy argmax to simulate finding the best match.
    top_matches = np.argmax(similarities, axis=1)
    print("Vector database search complete.")
    return top_matches

def run_heavy_ai_computations():
    print("Starting computationally heavy AI/ML script...")
    start_time = time.time()

    # --- 1. Data Generation & Preprocessing ---
    X_train, y_train = generate_synthetic_data(NUM_SAMPLES, NUM_FEATURES)
    X_train_processed = preprocess_data(X_train)

    # --- 2. Model Training ---
    trained_weights = simulate_model_training(X_train_processed, y_train, NUM_TRAINING_EPOCHS)

    # --- 3. Model Inference ---
    X_inference, _ = generate_synthetic_data(INFERENCE_BATCH_SIZE * NUM_INFERENCE_BATCHES, NUM_FEATURES)
    X_inference_processed = preprocess_data(X_inference) # Apply same preprocessing
    inference_predictions = simulate_model_inference(X_inference_processed, trained_weights, INFERENCE_BATCH_SIZE)

    # --- 4. Vector Database Operations (Similarity Search) ---
    # Create dummy vector database
    vector_db = np.random.rand(NUM_VECTORS_IN_DB, VECTOR_DB_DIMENSION)
    # Create dummy query vectors
    query_vectors = np.random.rand(NUM_QUERY_VECTORS, VECTOR_DB_DIMENSION)

    # Perform similarity search
    best_matches = simulate_vector_database_search(vector_db, query_vectors)
    print(f"Found best matches for {len(best_matches)} queries (sample match: {best_matches[0]})")


    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- Heavy AI/ML computations finished in {duration:.2f} seconds ---")

if __name__ == "__main__":
    run_heavy_ai_computations()