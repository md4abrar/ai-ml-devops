"""
California Housing Price Prediction with TensorFlow & Keras
-----------------------------------------------------------
This script demonstrates a full ML workflow using the California housing dataset:
- Data loading & preprocessing
- Neural network model building with Keras
- Training with TensorBoard logging
- Evaluation on test set
- Visualization of training curves

You can run this file directly (Python >=3.8 with TensorFlow and scikit-learn installed).
"""

# -------------------------------
# 1. Import required libraries
# -------------------------------
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

# -------------------------------
# 2. Load and split the dataset
# -------------------------------
# California housing dataset (features = house stats, target = median house price)
housing = fetch_california_housing()

# Split into training+validation and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42
)

# Split again to create a validation set
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42
)

# -------------------------------
# 3. Standardize the data
# -------------------------------
# Scaling ensures features have mean=0 and std=1, which helps NN training
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Fit on training data
X_valid = scaler.transform(X_valid)       # Use same stats for validation
X_test = scaler.transform(X_test)         # Use same stats for test

# -------------------------------
# 4. Build the model
# -------------------------------
# Sequential NN with two hidden layers (30 neurons each, ReLU [Rectified Linear Unit] activation)
# Final layer has 1 unit (linear) for regression output
model = tf.keras.Sequential([
    layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    layers.Dense(30, activation="relu"),
    layers.Dense(1)
])

# -------------------------------
# 5. Compile the model
# -------------------------------
# Loss: Mean Squared Error (MSE) since this is regression
# Optimizer: SGD (simple, but slower; Adam is often better)
model.compile(loss="mean_squared_error", optimizer="sgd")

# -------------------------------
# 6. Set up TensorBoard logging
# -------------------------------
log_dir = os.path.join("logs", "fit", "04_02")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# -------------------------------
# 7. Train the model
# -------------------------------
# Train for 20 epochs, using validation set to monitor overfitting
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_valid, y_valid),
    callbacks=[tensorboard_callback]
)

# -------------------------------
# 8. Evaluate on test set
# -------------------------------
# Model performance on unseen test data
mse_test = model.evaluate(X_test, y_test)
print(f"Mean Squared Error on Test Set: {mse_test:.4f}")

# -------------------------------
# 9. Plot training & validation loss
# -------------------------------
# Create output directory if it doesn’t exist
os.makedirs("output", exist_ok=True)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig('output/04_02_loss_plot.png')

print("Loss curve saved to output/04_02_loss_plot.png")

"""
How to Explore Further:
-----------------------
1. TensorBoard:
   Run `tensorboard --logdir logs/fit` in your terminal and open the given link in browser.

2. Improving the model:
   - Try optimizer="adam" instead of "sgd"
   - Add metrics=['mae'] in compile to monitor Mean Absolute Error
   - Use EarlyStopping to prevent overfitting

3. Saving the trained model:
   model.save("output/my_model.keras")
"""
