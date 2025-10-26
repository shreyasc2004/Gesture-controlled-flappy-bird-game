import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import tensorflow as tf
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
MODEL_PATH = 'gesture_model.keras'
ENCODER_PATH = 'label_encoder.pkl'
HISTORY_PATH = 'training_history.pkl'
DATA_PATH = 'gestures.csv'
CLASSES = ['NEUTRAL', 'THUMBS_UP', 'THUMBS_DOWN'] # Ensure this order matches your training

# --- 1. PLOT TRAINING & VALIDATION CURVES ---
def plot_history(history):
    """Plots accuracy and loss curves from the training history."""
    print("Generating training/validation plots...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot Accuracy
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='lower right')
    
    # Plot Loss
    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_plots.png')
    print("Saved 'training_plots.png'")
    plt.close()

# --- 2. PLOT CONFUSION MATRIX ---
def plot_confusion_matrix(model, X_test, y_test_indices, label_encoder):
    """Plots a confusion matrix for the test data."""
    print("Generating confusion matrix...")
    
    # Get model predictions
    y_pred_probs = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test_indices, y_pred_indices)
    
    # Get class names from label encoder
    class_names = label_encoder.classes_
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Saved 'confusion_matrix.png'")
    plt.close()

# --- 3. GENERATE SHAP (XAI) PLOT ---
def plot_shap_summary(model, X_train, X_test, label_encoder):
    """Generates and saves a SHAP summary plot."""
    print("Generating SHAP feature importance plot...")
    
    # Create a feature names list (e.g., 'landmark_0_x', 'landmark_0_y', ...)
    feature_names = []
    for i in range(21):
        feature_names.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
        
    # SHAP requires a "background" dataset to compute expected values.
    # We'll use a sample of the training data (e.g., 100 random samples)
    if len(X_train) > 100:
        background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    else:
        background = X_train
        
    # Create the SHAP Explainer
    explainer = shap.KernelExplainer(model.predict, background)
    
    # Calculate SHAP values for a sample of the test data
    if len(X_test) > 50:
        test_sample = X_test[np.random.choice(X_test.shape[0], 50, replace=False)]
    else:
        test_sample = X_test
        
    shap_values = explainer.shap_values(test_sample)
    
    # Get class names
    class_names = label_encoder.classes_
    
    # Plot and save the summary plot
    # We plot for all classes, but you can choose one (e.g., shap_values[1] for THUMBS_UP)
    shap.summary_plot(shap_values, test_sample, feature_names=feature_names, 
                      class_names=class_names, show=False, plot_type='bar')
    
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png')
    print("Saved 'shap_summary_plot.png'")
    plt.close()

# --- MAIN FUNCTION ---
def main():
    print("Starting Model Analysis...")
    
    # Check for necessary files
    if not all(os.path.exists(p) for p in [MODEL_PATH, ENCODER_PATH, HISTORY_PATH, DATA_PATH]):
        print(f"Error: Missing one or more files.")
        print(f"Make sure '{MODEL_PATH}', '{ENCODER_PATH}', '{HISTORY_PATH}', and '{DATA_PATH}' are present.")
        return

    # --- Load Data and Model ---
    print("Loading model, data, and history...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        with open(HISTORY_PATH, 'rb') as f:
            history = pickle.load(f)
        
        # Load and preprocess data (must match ModelTrainer.py)
        df = pd.read_csv(DATA_PATH)
        X = df.iloc[:, 1:].values  # Landmark data
        y = df.iloc[:, 0].values   # Labels
        
        # Scale data (same as in training)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Encode labels
        y_indices = label_encoder.transform(y)
        y_categorical = tf.keras.utils.to_categorical(y_indices, num_classes=len(CLASSES))
        
        # Split data (use the same random_state as trainer for consistency)
        _, X_test, _, y_test_one_hot = train_test_split(
            X_scaled, y_categorical, test_size=0.2, random_state=42
        )
        
        # We need y_test as indices for confusion matrix
        y_test_indices = np.argmax(y_test_one_hot, axis=1)
        
        # Get X_train for SHAP (needed for background)
        X_train, _, _, _ = train_test_split(
            X_scaled, y_categorical, test_size=0.2, random_state=42
        )

    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # --- Run Analyses ---
    plot_history(history)
    plot_confusion_matrix(model, X_test, y_test_indices, label_encoder)
    plot_shap_summary(model, X_train, X_test, label_encoder)
    
    print("\nAnalysis complete. All 3 plots have been saved as .png files.")

if __name__ == "__main__":
    main()

