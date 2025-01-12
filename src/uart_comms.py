import serial

def configure_uart(port='/dev/ttyS0', baudrate=115200):
    try:
        data = serial.Serial(port, baudrate)
        print(f"Opened port {port} at {baudrate} baud.")
        return data
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return None

def send_message(data, message):
    if data and data.is_open:
        data.write(message.encode())
        print(f"Sent: {message.strip()}")

def close_uart(data):
    if data and data.is_open:
        data.close()
        print("Serial port closed.")