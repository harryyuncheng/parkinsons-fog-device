import torch
import torch.nn as nn
import numpy as np
from collections import deque
import threading
import time

class HAR_CNN(nn.Module):
    """A 1D-CNN model for Human Activity Recognition - must match training architecture"""
    def __init__(self, n_features, n_classes, sequence_length=150):
        super(HAR_CNN, self).__init__()
        
        self.n_features = n_features
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.flatten = nn.Flatten()
        
        # Calculate the flattened size after convolution and pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_features, self.sequence_length)
            dummy_output = self.conv_block2(self.conv_block1(dummy_input))
            flattened_size = dummy_output.shape[1] * dummy_output.shape[2]

        self.fc_block = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # Input x has shape (batch_size, sequence_length, n_features)
        # Conv1d expects (batch_size, n_features, sequence_length)
        x = x.permute(0, 2, 1)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        x = self.flatten(x)
        x = self.fc_block(x)
        return x

class FOGPredictor:
    """Real-time FOG/HAR prediction using sliding window approach"""
    
    def __init__(self, model_path='models/har_cnn_model_20250622_031639.pth', sequence_length=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and configuration
        print(f"Loading model from {model_path}...")
        self.model, self.label_map, self.norm_params, self.config = self.load_model(model_path)
        self.model.eval()
        
        # Use sequence length from model config if not specified
        self.sequence_length = sequence_length if sequence_length else self.config['sequence_length']
        
        # Reverse label mapping for predictions
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        
        # Sliding window buffer for real-time data
        self.data_buffer = deque(maxlen=self.sequence_length)
        self.buffer_lock = threading.Lock()
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)  # Last 5 predictions for smoothing
        
        print(f"✅ FOG/HAR Predictor initialized on {self.device}")
        print(f"📊 Label mapping: {self.label_map}")
        print(f"🔢 Sequence length: {self.sequence_length}")
        
    def load_model(self, model_path):
        """Load the trained HAR_CNN model"""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract model configuration
            model_config = checkpoint['model_config']
            label_map = checkpoint['label_map']
            norm_params = checkpoint['normalization_params']
            
            # Store additional config
            config = {
                'sequence_length': checkpoint.get('sequence_length', 150),
                'sampling_rate': checkpoint.get('sampling_rate', 100),
                'window_seconds': checkpoint.get('window_seconds', 1.5),
                'step_seconds': checkpoint.get('step_seconds', 0.5)
            }
            
            # Initialize model with saved config
            model = HAR_CNN(**model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            # Set dropout to eval mode
            model.eval()
            
            print(f"✅ Model loaded successfully")
            print(f"🏗️ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model, label_map, norm_params, config
            
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
            normalized = (data - mean.squeeze()) / (std.squeeze() + 1e-7)  # Add epsilon to avoid division by zero
        else:  # Single sample [features]
            normalized = (data - mean.squeeze()) / (std.squeeze() + 1e-7)
            
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
                # Return default prediction with equal probabilities
                default_probs = {label: 1.0/len(self.label_map) for label in self.label_map.keys()}
                return {
                    'prediction': list(self.label_map.keys())[0],  # First label as default
                    'confidence': 0.0,
                    'probabilities': default_probs,
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
                if label_id in self.id_to_label:
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
            import traceback
            traceback.print_exc()
            
            # Return default prediction on error
            default_probs = {label: 1.0/len(self.label_map) for label in self.label_map.keys()}
            return {
                'prediction': list(self.label_map.keys())[0],
                'confidence': 0.0,
                'probabilities': default_probs,
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

# Global predictor instance
fog_predictor = None

def initialize_predictor(model_path='models/har_cnn_model_20250622_031639.pth'):
    """Initialize the global FOG/HAR predictor"""
    global fog_predictor
    try:
        fog_predictor = FOGPredictor(model_path)
        return True
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return False

def get_predictor():
    """Get the global predictor instance"""
    return fog_predictor

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    if initialize_predictor('models/har_cnn_model_20250622_031639.pth'):
        predictor = get_predictor()
        
        # Simulate adding data points
        print("\n📊 Simulating real-time prediction...")
        for i in range(200):
            # Simulate IMU data
            simulated_data = {
                'acc_x': np.random.randn() * 2,
                'acc_y': np.random.randn() * 2,
                'acc_z': np.random.randn() * 2 + 9.8,  # Gravity
                'gyro_x': np.random.randn() * 0.5,
                'gyro_y': np.random.randn() * 0.5,
                'gyro_z': np.random.randn() * 0.5
            }
            
            # Add data point
            predictor.add_data_point(simulated_data)
            
            # Make prediction every 10 samples
            if i % 10 == 0:
                result = predictor.predict()
                if result['status'] == 'success':
                    print(f"Sample {i}: {result['prediction']} (confidence: {result['confidence']:.2f})")
                elif result['status'] == 'insufficient_data':
                    print(f"Sample {i}: Buffer filling... {result['buffer_size']}/{predictor.sequence_length}")