import time
import serial
from datetime import datetime

class SerialController:
    """
    Serial controller that sends 'p' for freezing state and 's' for other states
    """
    
    def __init__(self, port='/dev/cu.usbserial-0001', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.is_connected = False
        self.last_state = None
        
        self.connect()
    
    def connect(self):
        """Establish serial connection"""
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate)
            time.sleep(2)  # Wait for device to reset
            self.is_connected = True
            print(f"✅ Serial connected to {self.port}")
        except Exception as e:
            print(f"❌ Serial connection failed: {e}")
            self.is_connected = False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.is_connected = False
            print("🔌 Serial disconnected")
    
    def send_command(self, command):
        """Send command to device"""
        if not self.is_connected or not self.serial_connection:
            return False
        
        try:
            if command == 'p':
                self.serial_connection.write(b'p\n')
            else:
                self.serial_connection.write(b's\n')
            print(f"📡 Sent: {command}")
            return True
        except Exception as e:
            print(f"❌ Send error: {e}")
            self.is_connected = False
            return False
    
    def process_state(self, state):
        """Process state and send appropriate command"""
        if state != self.last_state:
            if state == 'freezing':
                print("🚨 FREEZING - Sending 'p'")
                self.send_command('p')
            else:
                print(f"✅ {state} - Sending 's'")
                self.send_command('s')
            self.last_state = state
    
    def get_status(self):
        """Get controller status"""
        return {
            'connected': self.is_connected,
            'port': self.port,
            'baudrate': self.baudrate,
            'last_state': self.last_state
        }

# Global controller instance
serial_controller = None

def initialize_serial_controller(port='/dev/cu.usbserial-0001', baudrate=115200):
    """Initialize the global serial controller"""
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
    """Process prediction and send serial command"""
    if serial_controller and 'prediction' in prediction_result:
        serial_controller.process_state(prediction_result['prediction'])

def send_state_to_serial(state):
    """Send state directly to serial device"""
    if serial_controller:
        serial_controller.process_state(state)

# Test script
if __name__ == "__main__":
    print("🧪 Testing Serial Controller...")
    
    if initialize_serial_controller('/dev/cu.usbserial-0001', 115200):
        # Test different states
        test_states = ['standing', 'walking', 'freezing', 'standing', 'freezing']
        
        for state in test_states:
            print(f"\n--- Testing: {state} ---")
            send_state_to_serial(state)
            time.sleep(2)
        
        print(f"\n📋 Status: {serial_controller.get_status()}")
    else:
        print("❌ Could not initialize serial controller")