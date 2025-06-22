import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import threading
import time

# Fix OMP threading issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Force PyTorch to use single thread to prevent resource exhaustion
torch.set_num_threads(1)

# Disable multiprocessing for PyTorch to prevent resource leaks
torch.multiprocessing.set_sharing_strategy('file_system')

class FOGClassifier(nn.Module):
    """CNN-LSTM model for FOG classification - same as training"""
    
    def __init__(self, input_channels=6, sequence_length=256, num_classes=3, dropout_rate=0.3):
        super(FOGClassifier, self).__init__()
        
        # 1D CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(256, 128, batch_first=True, dropout=dropout_rate if dropout_rate > 0 else 0)
        
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
        
        # Transpose back for LSTM: [batch_size, sequence_length, features]
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

class FOGPredictor:
    """Real-time FOG prediction using sliding window approach"""
    
    def __init__(self, model_path='models/fog_classifier_20250622_060902.pth', sequence_length=256):
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model, self.label_map, self.norm_params = self.load_model(model_path)
        self.model.eval()
        
        # Reverse label mapping for predictions
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        
        # Sliding window buffer for real-time data
        self.data_buffer = deque(maxlen=256)  # Original buffer size
        self.buffer_lock = threading.Lock()

        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)  # Original smoothing window
        
        print(f"✅ FOG Predictor initialized on {self.device}")
        print(f"📊 Label mapping: {self.label_map}")
        
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            # Use weights_only=False for compatibility with models containing numpy data
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract model configuration
            model_config = checkpoint['model_config']
            label_map = checkpoint['label_map']
            norm_params = checkpoint['normalization_params']
            
            # Initialize model with same config as training
            model = FOGClassifier(**model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            print(f"✅ Model loaded successfully")
            print(f"🏗️ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model, label_map, norm_params
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def normalize_data(self, data):
        """Normalize data using training set statistics"""
        mean = self.norm_params['mean']
        std = self.norm_params['std']
        
        # Convert to numpy if it's a list
        if isinstance(data, list):
            data = np.array(data)
        
        # Ensure data has the right shape
        if data.ndim == 2:  # [sequence_length, features]
            normalized = (data - mean.squeeze()) / std.squeeze()
        else:  # Single sample [features]
            normalized = (data - mean.squeeze()) / std.squeeze()
            
        return normalized
    
    def add_data_point(self, imu_data):
        """Add new IMU data point to the sliding window buffer"""
        try:
            # Extract IMU features in the same order as training
            features = [
                imu_data['acc_x'],
                imu_data['acc_y'], 
                imu_data['acc_z'],
                imu_data['gyro_x'],
                imu_data['gyro_y'],
                imu_data['gyro_z']
            ]
            
            with self.buffer_lock:
                self.data_buffer.append(features)
                
        except KeyError as e:
            print(f"❌ Missing IMU data field: {e}")
            return False
        except Exception as e:
            print(f"❌ Error adding data point: {e}")
            return False
            
        return True
    
    def predict(self):
        """Make prediction on current buffer contents"""
        with self.buffer_lock:
            if len(self.data_buffer) < self.sequence_length:
                return {
                    'prediction': 'standing',  # Default prediction
                    'confidence': 0.0,
                    'probabilities': {'walking': 0.33, 'standing': 0.34, 'freezing': 0.33},
                    'buffer_size': len(self.data_buffer),
                    'status': 'insufficient_data'
                }
            
            # Convert buffer to numpy array
            sequence = np.array(list(self.data_buffer))
        
        try:
            # Normalize the sequence
            normalized_sequence = self.normalize_data(sequence)
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(normalized_sequence).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Convert to label
            predicted_label = self.id_to_label[predicted_class]
            
            # Get all probabilities
            prob_dict = {}
            for label_id, prob in enumerate(probabilities[0]):
                label_name = self.id_to_label[label_id]
                prob_dict[label_name] = prob.item()
            
            # Add to prediction history for smoothing
            self.prediction_history.append(predicted_label)
            
            # Smooth prediction (majority vote from recent predictions)
            if len(self.prediction_history) >= 3:
                from collections import Counter
                vote_counts = Counter(self.prediction_history)
                smoothed_prediction = vote_counts.most_common(1)[0][0]
            else:
                smoothed_prediction = predicted_label
            
            return {
                'prediction': smoothed_prediction,
                'raw_prediction': predicted_label,
                'confidence': confidence,
                'probabilities': prob_dict,
                'buffer_size': len(self.data_buffer),
                'status': 'success'
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'prediction': 'standing',
                'confidence': 0.0,
                'probabilities': {'walking': 0.33, 'standing': 0.34, 'freezing': 0.33},
                'buffer_size': len(self.data_buffer),
                'status': 'error',
                'error': str(e)
            }
    
    def get_buffer_status(self):
        """Get current buffer status"""
        with self.buffer_lock:
            return {
                'current_size': len(self.data_buffer),
                'required_size': self.sequence_length,
                'fill_percentage': (len(self.data_buffer) / self.sequence_length) * 100,
                'ready_for_prediction': len(self.data_buffer) >= self.sequence_length
            }
    
    def reset_buffer(self):
        """Clear the data buffer"""
        with self.buffer_lock:
            self.data_buffer.clear()
            self.prediction_history.clear()
        print("🔄 Prediction buffer reset")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            with self.buffer_lock:
                self.data_buffer.clear()
                self.prediction_history.clear()
            
            # Clear model from GPU memory if using CUDA
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print("🧹 FOG Predictor resources cleaned up")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")

# Global predictor instance
fog_predictor = None

def initialize_predictor(model_path='models/fog_classifier_20250622_060902.pth'):
    """Initialize the global FOG predictor"""
    global fog_predictor
    try:
        # Clean up existing predictor if any
        if fog_predictor is not None:
            fog_predictor.cleanup()
        
        fog_predictor = FOGPredictor(model_path)
        return True
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return False

def get_predictor():
    """Get the global predictor instance"""
    return fog_predictor

def cleanup_predictor():
    """Clean up the global predictor"""
    global fog_predictor
    if fog_predictor is not None:
        fog_predictor.cleanup()
        fog_predictor = None