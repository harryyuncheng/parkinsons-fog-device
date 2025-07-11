# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import glob
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set default constants
LEARNING_RATE = 0.004
BATCH_SIZE = 32
NUM_EPOCHS = 100
SEQUENCE_LENGTH = 256
DROPOUT_RATE = 0.3
NUM_CLASSES = 3
INPUT_CHANNELS = 6
OVERLAP = 0.5

# Classes
class IMUDataset(Dataset):
    """Dataset for IMU time series data with sliding windows"""

    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class FOGClassifier(nn.Module):
    """CNN-LSTM model for FOG classification"""

    def __init__(self, input_channels=INPUT_CHANNELS, sequence_length=SEQUENCE_LENGTH, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super(FOGClassifier, self).__init__()

        # 1D CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2) # Output length: sequence_length / 2

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2) # Output length: (sequence_length / 2) / 2 = sequence_length / 4

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2) # Output length: (sequence_length / 4) / 2 = sequence_length / 8

        # Calculate the input size for the LSTM based on the sequence_length
        lstm_input_size = 256 # Number of features from the last CNN layer
        lstm_sequence_length = sequence_length // 8 # Length after 3 pooling layers

        self.lstm = nn.LSTM(lstm_input_size, 128, batch_first=True, dropout=dropout_rate if dropout_rate > 0 else 0)

        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input: [batch_size, sequence_length, channels]
        # Transpose for Conv1d: [batch_size, channels, sequence_length]
        x = x.transpose(1, 2)

        # CNN feature extraction
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Transpose back for LSTM: [batch_size, sequence_length_after_pooling, features]
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last output for classification
        x = lstm_out[:, -1, :]  # [batch_size, hidden_size]

        # Classification
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
# Functions
def load_csv_data(csv_files):
    """Load all CSV files and combine them"""
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
            print(f"Loaded {csv_file}: {len(df)} samples")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    if not all_data:
        return None

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total combined data: {len(combined_df)} samples")

    return combined_df

def create_sequences(data, sequence_length=SEQUENCE_LENGTH, overlap=OVERLAP):
    """Create sliding window sequences from IMU data"""

    # Extract features and labels
    feature_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    X = data[feature_cols].values
    y = data['label'].values

    # Create label mapping
    label_map = {'walking': 0, 'standing': 1, 'freezing': 2}
    y_encoded = np.array([label_map.get(label, 1) for label in y])  # Default to standing if unknown

    # Create sequences
    step_size = int(sequence_length * (1 - overlap))
    sequences = []
    labels = []

    for i in range(0, len(X) - sequence_length + 1, step_size):
        sequence = X[i:i + sequence_length]
        # Use the most common label in the window
        window_labels = y_encoded[i:i + sequence_length]
        most_common_label = np.bincount(window_labels).argmax()

        sequences.append(sequence)
        labels.append(most_common_label)

    return np.array(sequences), np.array(labels), label_map

def normalize_data(X_train, X_val, X_test):
    """Normalize the data using training set statistics"""
    # Calculate mean and std across all features and time steps for training set
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1, std)  # Avoid division by zero

    # Normalize all sets
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_val_norm, X_test_norm, {'mean': mean, 'std': std}

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, device='cpu'):
    """Train the model"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Load best model
    model.load_state_dict(best_model_state)
    print(f'Training completed! Best validation accuracy: {best_val_acc:.2f}%')

    return history

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model on test data"""
    model.eval()
    model.to(device)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    return all_predictions, all_labels

def plot_results(history, predictions, labels, label_map):
    """Plot training history and confusion matrix"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Training history
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    class_names = list(label_map.keys())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax3)
    ax3.set_title('Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')

    # Accuracy by class
    accuracy_by_class = []
    for i in range(len(class_names)):
        if cm[i].sum() > 0:
            accuracy_by_class.append(cm[i][i] / cm[i].sum())
        else:
            accuracy_by_class.append(0)

    ax4.bar(class_names, accuracy_by_class)
    ax4.set_title('Accuracy by Class')
    ax4.set_ylabel('Accuracy')
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

def main(csv_files, learning_rate=None, batch_size=None, num_epochs=None, sequence_length=None, dropout_rate=None, overlap=None, input_channels=None, num_classes=None):
    """Main training function"""
    print("1dCNN-LSTM Classification Training Started")
    print("="*50)

    # Use provided parameters or fall back to defaults
    learning_rate = learning_rate if learning_rate is not None else LEARNING_RATE
    batch_size = batch_size if batch_size is not None else BATCH_SIZE
    num_epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    sequence_length = sequence_length if sequence_length is not None else SEQUENCE_LENGTH
    dropout_rate = dropout_rate if dropout_rate is not None else DROPOUT_RATE
    overlap = overlap if overlap is not None else OVERLAP
    input_channels = input_channels if input_channels is not None else INPUT_CHANNELS
    num_classes = num_classes if num_classes is not None else NUM_CLASSES

    # Load data
    data = load_csv_data(csv_files)
    if data is None:
        print("No data found. Make sure you have training sessions (.csv) selected.")
        return

    # Print data statistics
    print(f"\nData Statistics:")
    print(f"Total samples: {len(data)}")
    print(f"Label distribution:")
    print(data['label'].value_counts())

    # Create sequences
    print(f"\nCreating sequences with length {sequence_length} and overlap {overlap}...")
    sequences, labels, label_map = create_sequences(data, sequence_length=sequence_length, overlap=overlap)
    print(f"Created {len(sequences)} sequences of length {sequence_length}")
    print(f"Label mapping: {label_map}")

    # Split data
    train_size = int(0.7 * len(sequences))
    val_size = int(0.15 * len(sequences))
    test_size = len(sequences) - train_size - val_size
    X_train = sequences[:train_size]
    y_train = labels[:train_size]
    X_val = sequences[train_size:train_size + val_size]
    y_val = labels[train_size:train_size + val_size]
    X_test = sequences[train_size + val_size:]
    y_test = labels[train_size + val_size:]

    print(f"\nDataset split:")
    print(f"Training: {len(X_train)} sequences")
    print(f"Validation: {len(X_val)} sequences")
    print(f"Test: {len(X_test)} sequences")

    # Normalize data
    X_train_norm, X_val_norm, X_test_norm, norm_params = normalize_data(X_train, X_val, X_test)

    # Create datasets and dataloaders
    train_dataset = IMUDataset(X_train_norm, y_train)
    val_dataset = IMUDataset(X_val_norm, y_val)
    test_dataset = IMUDataset(X_test_norm, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = FOGClassifier(
        input_channels=input_channels,
        sequence_length=sequence_length,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )

    print(f"\nModel Architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train model
    print(f"\nStarting training...")
    history = train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)

    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    predictions, test_labels = evaluate_model(model, test_loader, device)

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\nTest Results:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"\nDetailed Classification Report:")
    print(classification_report(test_labels, predictions, target_names=list(label_map.keys())))

    # Plot results
    plot_results(history, predictions, test_labels, label_map)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'fog_classifier_{timestamp}.pth'

    torch.save({
        'model_state_dict': model.state_dict(),
        'label_map': label_map,
        'normalization_params': norm_params,
        'model_config': {
            'input_channels': input_channels,
            'sequence_length': sequence_length,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate
        }
    }, model_path)

    print(f"\nModel saved as: {model_path}")
    print(f"Training completed")
    return model

if __name__ == "__main__":
    model = main([], learning_rate=None, batch_size=None, num_epochs=None, sequence_length=None, dropout_rate=None, overlap=None, input_channels=None, num_classes=None)