# heavy_ai_computation_with_tracker.py
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics.pairwise import cosine_similarity
import random

# Assume foss_energy_tracker is installed and available
from energy_tracker import EnergyTracker, CarbonEstimator # Your library!

# --- Configuration ---
NUM_SAMPLES = 50000
NUM_FEATURES = 200
NUM_TRAINING_EPOCHS = 100
INFERENCE_BATCH_SIZE = 1000
NUM_INFERENCE_BATCHES = 50
VECTOR_DB_DIMENSION = 512
NUM_VECTORS_IN_DB = 10000
NUM_QUERY_VECTORS = 100

# --- Helper Functions (same as above) ---
def generate_synthetic_data(num_samples, num_features):
    print(f"Generating synthetic data: {num_samples} samples, {num_features} features...")
    X = np.random.rand(num_samples, num_features) * 10
    y = np.sum(X[:, :5], axis=1) + np.random.randn(num_samples) * 0.5
    print("Data generation complete.")
    return X, y

def preprocess_data(X):
    print("Preprocessing data (scaling and polynomial features)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)
    print(f"Preprocessing complete. New feature count: {X_poly.shape[1]}")
    return X_poly

def simulate_model_training(X, y, epochs):
    print(f"Simulating model training for {epochs} epochs on {X.shape[0]} samples...")
    dummy_model_weights = np.random.rand(X.shape[1], 1)
    learning_rate = 0.001

    for epoch in range(epochs):
        predictions = np.dot(X, dummy_model_weights)
        errors = predictions - y.reshape(-1, 1)
        gradient = np.dot(X.T, errors) / X.shape[0]
        dummy_model_weights -= learning_rate * gradient

        if (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0:
            loss = np.mean(errors**2)
            print(f"  Epoch {epoch+1}/{epochs}, Simulated Loss: {loss:.4f}", end='\r')
    print("\nSimulated model training complete.")
    return dummy_model_weights

def simulate_model_inference(X_new, model_weights, batch_size):
    print(f"Simulating model inference on {X_new.shape[0]} new samples in batches...")
    all_predictions = []
    num_batches = int(np.ceil(X_new.shape[0] / batch_size))

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X_new.shape[0])
        batch_X = X_new[start_idx:end_idx]

        batch_predictions = np.dot(batch_X, model_weights)
        all_predictions.extend(batch_predictions.flatten())

        if (i + 1) % (num_batches // 10 if num_batches >= 10 else 1) == 0:
            print(f"  Processed inference batch {i+1}/{num_batches}", end='\r')
    print("\nSimulated model inference complete.")
    return np.array(all_predictions)

def simulate_vector_database_search(db_vectors, query_vectors):
    print(f"Simulating vector database search: {query_vectors.shape[0]} queries against {db_vectors.shape[0]} vectors...")

    db_vectors_norm = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)
    query_vectors_norm = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

    similarities = cosine_similarity(query_vectors_norm, db_vectors_norm)
    top_matches = np.argmax(similarities, axis=1)
    print("Vector database search complete.")
    return top_matches

def run_heavy_ai_computations():
    print("Starting computationally heavy AI/ML script...")

    # --- Initialize Energy Tracker ---
    tracker = EnergyTracker(project_name="AIFlowCarbonFootprint", region="US") # Set your region
    tracker.start()

    # --- 1. Data Generation & Preprocessing ---
    tracker.track_step("data_preprocessing")
    X_train, y_train = generate_synthetic_data(NUM_SAMPLES, NUM_FEATURES)
    X_train_processed = preprocess_data(X_train)
    tracker.end_step("data_preprocessing")

    # --- 2. Model Training ---
    tracker.track_step("model_training")
    trained_weights = simulate_model_training(X_train_processed, y_train, NUM_TRAINING_EPOCHS)
    tracker.end_step("model_training")

    # --- 3. Model Inference ---
    tracker.track_step("model_inference")
    X_inference, _ = generate_synthetic_data(INFERENCE_BATCH_SIZE * NUM_INFERENCE_BATCHES, NUM_FEATURES)
    X_inference_processed = preprocess_data(X_inference)
    inference_predictions = simulate_model_inference(X_inference_processed, trained_weights, INFERENCE_BATCH_SIZE)
    tracker.end_step("model_inference")

    # --- 4. Vector Database Operations (Similarity Search) ---
    tracker.track_step("vector_db_search")
    vector_db = np.random.rand(NUM_VECTORS_IN_DB, VECTOR_DB_DIMENSION)
    query_vectors = np.random.rand(NUM_QUERY_VECTORS, VECTOR_DB_DIMENSION)
    best_matches = simulate_vector_database_search(vector_db, query_vectors)
    print(f"Found best matches for {len(best_matches)} queries (sample match: {best_matches[0]})")
    tracker.end_step("vector_db_search")

    # --- Finalize tracking and get results ---
    tracker.stop()
    results = tracker.get_results()

    print(f"\n--- AI/ML computations finished ---")
    print("\n--- Energy Tracking Results ---")
    for step, data in results["steps"].items():
        print(f"Step: {step}")
        print(f"  Duration: {data['duration']:.2f} seconds")
        if 'cpu_energy_joules' in data:
            print(f"  CPU Energy: {data['cpu_energy_joules'] / 3600 / 1000:.6f} kWh")
        if 'gpu_energy_joules' in data:
            print(f"  GPU Energy: {data['gpu_energy_joules'] / 3600 / 1000:.6f} kWh")
        if 'carbon_emissions_gco2' in data:
            print(f"  Carbon Emissions: {data['carbon_emissions_gco2']:.4f} gCO2")
    print(f"\nTotal Energy (kWh): {results['total_energy_kwh']:.6f}")
    print(f"Total Carbon Emissions (gCO2): {results['total_carbon_gco2']:.4f}")

if __name__ == "__main__":
    run_heavy_ai_computations()