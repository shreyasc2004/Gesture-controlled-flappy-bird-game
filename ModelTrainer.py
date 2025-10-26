import pandas as pd
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- CONFIGURATION ---
DATA_PATH = 'gestures.csv'
MODEL_PATH = 'gesture_model.keras'
ENCODER_PATH = 'label_encoder.pkl'
HISTORY_PATH = 'training_history.pkl'
SCALER_PATH = 'scaler.pkl'  # <-- NEW: Path to save the scaler

CLASSES = ['NEUTRAL', 'THUMBS_UP', 'THUMBS_DOWN']

# --- 1. Load and Preprocess Data ---
print("Loading and preprocessing data...")
df = pd.read_csv(DATA_PATH)

# Separate features (landmarks) and labels (gestures)
X = df.iloc[:, 1:].values  # All rows, all columns *except* the first
y = df.iloc[:, 0].values   # All rows, *only* the first column

# Encode labels (e.g., 'THUMBS_UP' -> 1)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(CLASSES))

# Scale features (landmarks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # We fit AND transform the training data

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42
)

# --- 2. Build the MLP Model ---
print("Building the MLP model...")
model = Sequential([
    # Input layer: 63 features (21 landmarks * 3 coords)
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # Dropout layer to prevent overfitting
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    # Output layer: 3 classes (Neutral, Up, Down)
    Dense(len(CLASSES), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 3. Train the Model ---
print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=50,  # 50 passes through the data
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- 4. Save Everything ---
print("Saving model, encoder, history, and scaler...")

# Save the trained model
model.save(MODEL_PATH)

# Save the label encoder
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(encoder, f)

# Save the training history
with open(HISTORY_PATH, 'wb') as f:
    pickle.dump(history.history, f)

# ** HERE IS THE NEW CODE: Save the scaler **
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print("\nTraining complete!")
print(f"Model saved to: {MODEL_PATH}")
print(f"Encoder saved to: {ENCODER_PATH}")
print(f"History saved to: {HISTORY_PATH}")
print(f"Scaler saved to: {SCALER_PATH}") # <-- NEW

