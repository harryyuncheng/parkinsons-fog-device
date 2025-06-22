import serial
import threading
import time
from datetime import datetime

class SerialController:
    """
    Controls serial device based on FOG prediction results.
    Sends 'p' signal when freezing is detected, 's' for all other states.
    """
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, timeout=1):
        """
        Initialize serial controller
        
        Args:
            port (str): Serial port (e.g., '/dev/ttyUSB0' on Linux/Mac, 'COM3' on Windows)
            baudrate (int): Baud rate for serial communication
            timeout (float): Serial timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self.is_connected = False
        
        # State tracking
        self.last_prediction = None
        self.current_signal = 's'  # Default signal
        self.lock = threading.Lock()
        
        # Connect to serial device
        self.connect()
    
    def connect(self):
        """Establish serial connection"""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            self.is_connected = True
            print(f"✅ Serial connection established on {self.port} at {self.baudrate} baud")
            
            # Send initial default signal
            self.send_signal('s')
            
        except serial.SerialException as e:
            print(f"❌ Failed to connect to serial port {self.port}: {e}")
            self.is_connected = False
        except Exception as e:
            print(f"❌ Unexpected error connecting to serial: {e}")
            self.is_connected = False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.is_connected = False
            print("🔌 Serial connection closed")
    
    def send_signal(self, signal):
        """
        Send signal to serial device
        
        Args:
            signal (str): Signal to send ('p' or 's')
        """
        if not self.is_connected or not self.serial_connection:
            print(f"⚠️ Cannot send signal '{signal}' - no serial connection")
            return False
        
        try:
            with self.lock:
                # Send signal as bytes
                self.serial_connection.write(signal.encode('utf-8'))
                self.serial_connection.flush()  # Ensure data is sent immediately
                
            print(f"📡 Sent signal: '{signal}' at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            return True
            
        except serial.SerialException as e:
            print(f"❌ Serial write error: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"❌ Error sending signal: {e}")
            return False
    
    def process_prediction(self, prediction_result):
        """
        Process prediction result and send appropriate signal
        
        Args:
            prediction_result (dict): Prediction result from FOG predictor
                Expected format: {'prediction': 'walking'/'standing'/'freezing', ...}
        """
        if not prediction_result or 'prediction' not in prediction_result:
            print("⚠️ Invalid prediction result received")
            return
        
        current_prediction = prediction_result['prediction']
        confidence = prediction_result.get('confidence', 0.0)
        
        # Check if prediction has changed
        if current_prediction != self.last_prediction:
            print(f"🔄 Prediction changed: {self.last_prediction} → {current_prediction} (confidence: {confidence:.2f})")
            
            # Determine signal to send
            if current_prediction == 'freezing':
                new_signal = 'p'
                print("🚨 FREEZING DETECTED - Sending 'p' signal")
            else:
                new_signal = 's'
                print(f"✅ Normal state ({current_prediction}) - Sending 's' signal")
            
            # Send signal only if it's different from current
            if new_signal != self.current_signal:
                if self.send_signal(new_signal):
                    self.current_signal = new_signal
            
            # Update last prediction
            self.last_prediction = current_prediction
        
        # Optional: Log periodic status (every 10th call to avoid spam)
        # Uncomment the lines below if you want periodic status updates
        # else:
        #     if hasattr(self, '_status_counter'):
        #         self._status_counter += 1
        #     else:
        #         self._status_counter = 1
        #     
        #     if self._status_counter % 10 == 0:
        #         print(f"📊 Status: {current_prediction} (signal: {self.current_signal})")
    
    def get_status(self):
        """Get current controller status"""
        return {
            'connected': self.is_connected,
            'port': self.port,
            'baudrate': self.baudrate,
            'last_prediction': self.last_prediction,
            'current_signal': self.current_signal,
            'timestamp': datetime.now().isoformat()
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.disconnect()


# Global serial controller instance
serial_controller = None

def initialize_serial_controller(port='/dev/ttyUSB0', baudrate=9600):
    """
    Initialize the global serial controller
    
    Args:
        port (str): Serial port to connect to
        baudrate (int): Baud rate for communication
        
    Returns:
        bool: True if successful, False otherwise
    """
    global serial_controller
    try:
        serial_controller = SerialController(port=port, baudrate=baudrate)
        return serial_controller.is_connected
    except Exception as e:
        print(f"❌ Failed to initialize serial controller: {e}")
        return False

def get_serial_controller():
    """Get the global serial controller instance"""
    return serial_controller

def process_prediction_for_serial(prediction_result):
    """
    Convenience function to process prediction and send serial signal
    
    Args:
        prediction_result (dict): Prediction result from FOG predictor
    """
    if serial_controller:
        serial_controller.process_prediction(prediction_result)
    else:
        print("⚠️ Serial controller not initialized")

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Serial Controller...")
    
    # Initialize controller (adjust port as needed)
    # Common ports:
    # - Linux/Mac: '/dev/ttyUSB0', '/dev/ttyACM0', '/dev/cu.usbserial-*'
    # - Windows: 'COM1', 'COM3', etc.
    
    if initialize_serial_controller('/dev/ttyUSB0', 9600):
        controller = get_serial_controller()
        
        # Simulate prediction changes
        test_predictions = [
            {'prediction': 'standing', 'confidence': 0.95},
            {'prediction': 'walking', 'confidence': 0.88},
            {'prediction': 'freezing', 'confidence': 0.92},  # Should send 'p'
            {'prediction': 'standing', 'confidence': 0.89},  # Should send 's'
            {'prediction': 'freezing', 'confidence': 0.94},  # Should send 'p'
            {'prediction': 'walking', 'confidence': 0.91},   # Should send 's'
        ]
        
        print("\n📊 Simulating prediction sequence...")
        for i, pred in enumerate(test_predictions):
            print(f"\n--- Test {i+1} ---")
            process_prediction_for_serial(pred)
            time.sleep(2)  # Wait between tests
        
        # Show final status
        print(f"\n📋 Final status: {controller.get_status()}")
        
    else:
        print("❌ Could not initialize serial controller - check port and connection")
