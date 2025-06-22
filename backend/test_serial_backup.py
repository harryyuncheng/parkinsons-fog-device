#!/usr/bin/env python3
"""
Test script for serial controller functionality
Run this to test serial communication before integrating with the main app
"""

import time
from serial_controller import initialize_serial_controller, process_prediction_for_serial, get_serial_controller

def test_serial_controller():
    """Test the serial controller with various prediction scenarios"""
    
    print("🧪 Serial Controller Test Script")
    print("=" * 50)
    
    # Get port from user
    port = input("Enter serial port (default: /dev/ttyUSB0): ").strip()
    if not port:
        port = '/dev/ttyUSB0'
    
    # Get baudrate from user
    baudrate_input = input("Enter baud rate (default: 9600): ").strip()
    if not baudrate_input:
        baudrate = 9600
    else:
        try:
            baudrate = int(baudrate_input)
        except ValueError:
            print("Invalid baud rate, using 9600")
            baudrate = 9600
    
    print(f"\n🔌 Attempting to connect to {port} at {baudrate} baud...")
    
    # Initialize serial controller
    if not initialize_serial_controller(port, baudrate):
        print("❌ Failed to initialize serial controller")
        print("💡 Make sure:")
        print("   - Device is connected")
        print("   - Port is correct (/dev/ttyUSB0, /dev/ttyACM0, COM3, etc.)")
        print("   - You have permission to access the port")
        print("   - No other program is using the port")
        return False
    
    controller = get_serial_controller()
    print(f"✅ Serial controller initialized successfully!")
    print(f"📋 Status: {controller.get_status()}")
    
    # Test scenarios
    test_scenarios = [
        {'name': 'Initial Standing', 'prediction': 'standing', 'confidence': 0.95, 'expected_signal': 's'},
        {'name': 'Walking', 'prediction': 'walking', 'confidence': 0.88, 'expected_signal': 's'},
        {'name': 'FOG Episode', 'prediction': 'freezing', 'confidence': 0.92, 'expected_signal': 'p'},
        {'name': 'Recovery to Standing', 'prediction': 'standing', 'confidence': 0.89, 'expected_signal': 's'},
        {'name': 'Another FOG Episode', 'prediction': 'freezing', 'confidence': 0.94, 'expected_signal': 'p'},
        {'name': 'Walking Again', 'prediction': 'walking', 'confidence': 0.91, 'expected_signal': 's'},
    ]
    
    print(f"\n🎯 Running {len(test_scenarios)} test scenarios...")
    print("Watch your serial device for signals:")
    print("  's' = Normal state (standing/walking)")
    print("  'p' = FOG detected (freezing)")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Test {i}: {scenario['name']} ---")
        prediction_result = {
            'prediction': scenario['prediction'],
            'confidence': scenario['confidence'],
            'status': 'success'
        }
        
        print(f"📊 Prediction: {scenario['prediction']} (confidence: {scenario['confidence']:.2f})")
        print(f"🎯 Expected signal: '{scenario['expected_signal']}'")
        
        # Process prediction
        process_prediction_for_serial(prediction_result)
        
        # Wait for user confirmation
        input("Press Enter to continue to next test...")
    
    print(f"\n📋 Final status: {controller.get_status()}")
    print("✅ Test complete!")
    
    return True

def interactive_mode():
    """Interactive mode for manual testing"""
    print("\n🎮 Interactive Mode")
    print("Enter predictions manually (or 'quit' to exit)")
    
    controller = get_serial_controller()
    if not controller:
        print("❌ No active serial controller")
        return
    
    while True:
        try:
            prediction = input("\nEnter prediction (standing/walking/freezing) or 'quit': ").strip().lower()
            
            if prediction == 'quit':
                break
            
            if prediction not in ['standing', 'walking', 'freezing']:
                print("⚠️ Invalid prediction. Use: standing, walking, or freezing")
                continue
            
            confidence = input("Enter confidence (0.0-1.0, default 0.9): ").strip()
            try:
                confidence = float(confidence) if confidence else 0.9
            except ValueError:
                confidence = 0.9
            
            prediction_result = {
                'prediction': prediction,
                'confidence': confidence,
                'status': 'success'
            }
            
            process_prediction_for_serial(prediction_result)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Exiting interactive mode")

if __name__ == "__main__":
    try:
        # Run main test
        if test_serial_controller():
            # Offer interactive mode
            response = input("\n🎮 Run interactive mode? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        # Cleanup
        controller = get_serial_controller()
        if controller:
            controller.disconnect()
        print("🧹 Cleanup complete")