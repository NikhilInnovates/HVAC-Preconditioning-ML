import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, LayerNormalization, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Load synthetic data
data = pd.read_csv('synthetic_data.csv')
data.dropna(inplace=True)

# Feature Engineering
# Cyclical Encoding for Timestamp
data['sin_time'] = np.sin(2 * np.pi * data['Timestamp'] / 1440)
data['cos_time'] = np.cos(2 * np.pi * data['Timestamp'] / 1440)

# Lag Features
data['Heating_Cooling_Time_Lag1'] = data['Heating_Cooling_Time'].shift(1).fillna(0)
data['Heating_Cooling_Time_Lag2'] = data['Heating_Cooling_Time'].shift(2).fillna(0)

# Rolling Statistics
data['Inside_Temperature_RollingMean'] = data['Inside_Temperature'].rolling(window=3).mean().fillna(0)

# Interaction Features
data['Temp_Interaction'] = data['Inside_Temperature'] * data['Outside_Temperature']

# Features and target
features = ['Inside_Temperature', 'Outside_Temperature', 'Set_Point_Temperature',
            'Heat_Flow', 'sin_time', 'cos_time', 'Heating_Cooling_Time_Lag1',
            'Heating_Cooling_Time_Lag2', 'Inside_Temperature_RollingMean', 'Temp_Interaction']
X = data[features].values
y = data['Heating_Cooling_Time'].values

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Reshape for LSTM
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_scaled, test_size=0.2, random_state=42)

# -------------------------------
# Improved LSTM Model
# -------------------------------
def build_lstm():
    model = Sequential([
        Bidirectional(GRU(512, return_sequences=True, dropout=0.3, input_shape=(X_train.shape[1], X_train.shape[2]))),
        LayerNormalization(),
        Bidirectional(GRU(256, return_sequences=True, dropout=0.3)),
        LayerNormalization(),
        GRU(128, return_sequences=False, dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0), loss=tf.keras.losses.Huber())
    return model

lstm_model = build_lstm()

# Training
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)

lstm_model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_split=0.2,
               verbose=1, callbacks=[early_stopping, reduce_lr])

# Evaluate LSTM
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = y_scaler.inverse_transform(lstm_predictions)
y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics for LSTM
lstm_mae = mean_absolute_error(y_test_original, lstm_predictions) / 60
lstm_rmse = np.sqrt(mean_squared_error(y_test_original, lstm_predictions)) / 60
lstm_mse = mean_squared_error(y_test_original, lstm_predictions)
lstm_r2 = r2_score(y_test_original, lstm_predictions)


# -------------------------------
# Improved ESRNN Model
# -------------------------------
class ESRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, dropout=0.3):
        super(ESRNN, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.layer_norm(out)
        out = self.fc(self.dropout(out[:, -1, :]))
        return out

hidden_size = 512
num_layers = 5
esrnn_model = ESRNN(input_size=X_train.shape[2], hidden_size=hidden_size, output_size=1, num_layers=num_layers)

criterion = nn.HuberLoss()
optimizer = optim.AdamW(esrnn_model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

# Training Loop
for epoch in range(1000):
    esrnn_model.train()
    optimizer.zero_grad()
    outputs = esrnn_model(torch.tensor(X_train.squeeze(), dtype=torch.float32).unsqueeze(1))
    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(esrnn_model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

# Evaluate ESRNN
esrnn_model.eval()
with torch.no_grad():
    predictions = esrnn_model(torch.tensor(X_test.squeeze(), dtype=torch.float32).unsqueeze(1)).numpy()

predictions = y_scaler.inverse_transform(predictions)
# Calculate metrics for ESRNN
esrnn_mae = mean_absolute_error(y_test_original, predictions) / 60
esrnn_rmse = np.sqrt(mean_squared_error(y_test_original, predictions)) / 60
esrnn_mse = mean_squared_error(y_test_original, predictions)
esrnn_r2 = r2_score(y_test_original, predictions)

# -------------------------------
# Compare Results
# -------------------------------
# Print results
print("\n--- Final Results ---")
print(f"Improved LSTM - MAE: {lstm_mae:.2f} mins, RMSE: {lstm_rmse:.2f} mins, MSE: {lstm_mse:.2f}, R²: {lstm_r2:.2f}")
print(f"Improved ESRNN - MAE: {esrnn_mae:.2f} mins, RMSE: {esrnn_rmse:.2f} mins, MSE: {esrnn_mse:.2f}, R²: {esrnn_r2:.2f}")