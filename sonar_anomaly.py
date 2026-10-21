import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

print("--- Step 1: Loading and Preparing Data ---")
# 1. Load dataset directly from repository URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
df = pd.read_csv(url, header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y_binary = np.where(y == 'R', 0, 1)

# Split normal (Rock = 0) and anomaly (Mine = 1)
X_normal = X[y_binary == 0]
X_anomaly = X[y_binary == 1]

X_train_normal, X_val_normal = train_test_split(X_normal, test_size=0.2, random_state=42)
X_test_normal, _ = train_test_split(X_normal, test_size=0.2, random_state=42)
X_test = np.vstack([X_test_normal, X_anomaly])
y_test = np.array([0] * len(X_test_normal) + [1] * len(X_anomaly))

# Scale features to mean=0, variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_val_scaled = scaler.transform(X_val_normal)
X_test_scaled = scaler.transform(X_test)

print(f"Training normal samples shape: {X_train_scaled.shape}")
print(f"Test samples shape (Normal + Anomaly): {X_test_scaled.shape}\n")


print("--- Step 2: Defining Autoencoder Architecture ---")
class SonarAutoencoder(nn.Module):
    def __init__(self, input_dim=60):
        super(SonarAutoencoder, self).__init__()
        
        # Encoder: 60 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Decoder: 8 -> 16 -> 32 -> 60
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = SonarAutoencoder(input_dim=60)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


print("--- Step 3: Training the Model (Only on Normal Data) ---")
train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(train_tensor), batch_size=16, shuffle=True)

epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs = batch[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")


print("\n--- Step 4: Evaluation & Thresholding ---")
model.eval()
with torch.no_grad():
    # Set threshold using validation normal data (95th percentile of error)
    val_preds = model(val_tensor)
    val_mse = torch.mean((val_tensor - val_preds) ** 2, dim=1).numpy()
    threshold = np.percentile(val_mse, 95)
    print(f"Anomaly Detection Threshold Set At: {threshold:.4f}\n")

    # Evaluate on test set
    test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    test_preds = model(test_tensor)
    test_mse = torch.mean((test_tensor - test_preds) ** 2, dim=1).numpy()
    
    y_pred = (test_mse > threshold).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (Rock)', 'Anomaly (Mine)']))

auc_score = roc_auc_score(y_test, test_mse)
print(f"ROC-AUC Score: {auc_score:.4f}")


print("\n--- Step 5: Generating Visualization Plot ---")
# Separate test MSE errors by their true class label
normal_errors = test_mse[y_test == 0]
anomaly_errors = test_mse[y_test == 1]

plt.figure(figsize=(10, 6))
plt.hist(normal_errors, bins=25, alpha=0.7, label='Normal (Rock)', color='skyblue', edgecolor='black')
plt.hist(anomaly_errors, bins=25, alpha=0.7, label='Anomaly (Mine)', color='salmon', edgecolor='black')
plt.axvline(threshold, color='green', linestyle='--', linewidth=2.5, label=f'Threshold ({threshold:.4f})')

plt.xlabel('Reconstruction Error (Mean Squared Error)', fontsize=12)
plt.ylabel('Sample Count', fontsize=12)
plt.title('Sonar Anomaly Detection: Reconstruction Error Distribution', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)

# Save the plot as an image file you can drop straight into your report/presentation
plt.savefig('sonar_anomaly_distribution.png', dpi=300, bbox_inches='tight')
print("Graph saved successfully as 'sonar_anomaly_distribution.png' in your project folder!")

# Show the plot pop-up window
plt.show()