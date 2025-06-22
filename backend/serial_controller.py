import time

# Direct import of pyserial
try:
    import serial
    Serial = serial.Serial
    print("✅ Successfully imported pyserial")
except (ImportError, AttributeError) as e:
    print(f"❌ Error importing pyserial: {e}")
    print("Install with: pip install pyserial")
    exit(1)

# === Configuration ===
port = '/dev/cu.usbserial-0001'         # Change this to your ESP32's serial port
baud_rate = 115200

# === Open serial connection ===
try:
    ser = Serial(port, baud_rate)
    time.sleep(2)  # Wait for ESP32 to finish resetting

    # === Send a character ===
    ser.write(b'p\n')  # Send the character 'p' followed by newline
    ser.write(b'p\n')  # Send the character 'p' followed by newline
    ser.write(b'p\n')  # Send the character 'p' followed by newline
    ser.write(b'p\n')  # Send the character 'p' followed by newline

    print("✅ Character sent!")
    ser.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure your ESP32 is connected to the correct port")