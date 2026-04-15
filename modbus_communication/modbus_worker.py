import time
import struct
from pyModbusTCP.client import ModbusClient

def float_to_registers(val):
    """Convert a float32 to two 16-bit registers."""
    # 'f' is float32, '>' is big-endian
    packed = struct.pack('>f', val)
    # '>HH' converts 4 bytes to 2 unsigned 16-bit integers
    return struct.unpack('>HH', packed)

def run_modbus_client(shared_data, config):
    """
    Active Push Subroutine: Connects to PLC and sends data.
    Includes auto-reconnect/retry logic.
    """
    print("[MODBUS] Client thread started.")
    heartbeat = 0
    
    while True:
        # 1. Initialize Client based on current config
        plc_ip = config.get("plc_ip", "192.168.1.50")
        plc_port = config.get("plc_port", 502)
        
        c = ModbusClient(host=plc_ip, port=plc_port, auto_open=True, auto_close=True)
        
        print(f"[MODBUS] Attempting connection to PLC at {plc_ip}:{plc_port}...")
        
        while True:
            try:
                # 2. Check if we are connected
                if c.open():
                    shared_data["plc_online"] = True
                    
                    # 3. Pull latest values from Shared Dictionary
                    # NOTE: With new naming convention:
                    # - top_dist represents Left side (L1-L2 pair: LT1,LT2,LB1,LB2 markers)
                    # - bot_dist represents Right side (R1-R2 pair: RT1,RT2,RB1,RB2 markers)
                    top_dist = shared_data.get("top_dist", 0.0)    # Left side distance (L1-L2)
                    bot_dist = shared_data.get("bot_dist", 0.0)    # Right side distance (R1-R2)
                    error_code = shared_data.get("error_code", 0)

                    # 4. Prepare Registers
                    # Adr 0-1: Left side distance (L1-L2 pair, Float32)
                    # Adr 2-3: Right side distance (R1-R2 pair, Float32)
                    # Adr 4: Error Code (Int)
                    # Adr 5: Heartbeat (Counter)
                    
                    regs = []
                    regs.extend(float_to_registers(top_dist))
                    regs.extend(float_to_registers(bot_dist))
                    regs.append(int(error_code))
                    
                    heartbeat = (heartbeat + 1) % 65535
                    regs.append(heartbeat)
                    
                    # 5. Push to PLC
                    success = c.write_multiple_registers(0, regs)
                    
                    if not success:
                        print("[MODBUS] Push failed. PLC might be offline.")
                        shared_data["plc_online"] = False
                        break # Break inner loop to trigger retry logic
                        
                else:
                    print(f"[MODBUS] Link Down. Retrying in 5s...")
                    shared_data["plc_online"] = False
                    time.sleep(5)
                    break # Break inner loop to refresh config/reconnect
                
                # Update frequency (Industrial standard 100ms)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[MODBUS] Critical Error: {e}")
                shared_data["plc_online"] = False
                time.sleep(5)
                break
