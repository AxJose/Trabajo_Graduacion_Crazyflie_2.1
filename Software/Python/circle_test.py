import json
import time
import threading
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander
import cflib.crtp
from datetime import datetime, timezone
from crazyflie_python_commands_mod import *
import numpy as np
import msvcrt

## Vuelo del dron usando Mocap y Flowdeck

# -------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------
URI = "radio://0/80/2M/E7E7E7E7EA"
MQTT_TOPIC = 'mocap/drone2'
MQTT_BROKER = '192.168.50.200'
PORT = 1880

# -------------------------------------------------------
# VARIABLES GLOBALES
# -------------------------------------------------------
mocap_pose = {'x': 0.0, 'y': 0.0, 'z': 0.0}
real_trajectory = []
theoretical_trajectory = []
scf_global = None
last_ts = None

# Event que se activa con 'q' — cualquier bucle lo revisa
stop_event = threading.Event()

# -------------------------------------------------------
# PARO DE EMERGENCIA
# -------------------------------------------------------
def emergency_listener():
    print("Presiona 'q' en cualquier momento para paro de emergencia")
    while not stop_event.is_set():
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if key == 'q':
                print("\n🚨 PARO DE EMERGENCIA ACTIVADO")
                stop_event.set()       # detiene todos los bucles
                _cortar_motores()      # apaga motores inmediatamente
                break
        time.sleep(0.05)

def _cortar_motores():
    global scf_global
    if scf_global is not None:
        try:
            scf_global.cf.commander.send_stop_setpoint()
            print("✓ Motores apagados")
        except Exception as e:
            print(f"Error apagando motores: {e}")

# -------------------------------------------------------
# CALLBACK MQTT — ignorado si stop_event está activo
# -------------------------------------------------------
def on_message(client, userdata, msg):
    global scf_global, last_ts

    if stop_event.is_set():
        return  # no hace nada tras el paro

    try:
        data = json.loads(msg.payload.decode())
        pos = data['payload']['pose']['position']
        ts_str = data.get('ts', None)

        if ts_str is not None:
            msg_time = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            if last_ts is not None and msg_time <= last_ts:
                return
            last_ts = msg_time

        mocap_pose['x'] = float(pos['x'])
        mocap_pose['y'] = float(pos['y'])
        mocap_pose['z'] = float(pos['z'])

        if scf_global is not None:
            set_position(scf_global, mocap_pose['x'], mocap_pose['y'], mocap_pose['z'])

        real_trajectory.append([mocap_pose['x'], mocap_pose['y'], mocap_pose['z']])

    except Exception as e:
        print("Error en MQTT:", e)

def start_mqtt():
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, PORT, 60)
    client.subscribe(MQTT_TOPIC)
    client.loop_forever()

# -------------------------------------------------------
# SLEEP INTERRUPTIBLE
# -------------------------------------------------------
def interruptible_sleep(seconds, interval=0.05):
    """Igual que time.sleep pero se corta si stop_event se activa."""
    elapsed = 0.0
    while elapsed < seconds:
        if stop_event.is_set():
            return False
        time.sleep(interval)
        elapsed += interval
    return True

# -------------------------------------------------------
# TRAYECTORIA CIRCULAR
# -------------------------------------------------------
def fly_circle(scf, radius=0.5, hover_height=0.5, velocity=0.1, num_points=20):
    global theoretical_trajectory

    print("Esperando posición inicial del MoCap...")
    while mocap_pose['x'] == 0 and mocap_pose['y'] == 0 and mocap_pose['z'] == 0:
        if stop_event.is_set():
            return
        time.sleep(0.1)

    print(f"MoCap recibido: x={mocap_pose['x']:.4f}, y={mocap_pose['y']:.4f}, z={mocap_pose['z']:.4f}")

    x0, y0 = mocap_pose['x'], mocap_pose['y']

    print("Despegando...")
    takeoff(scf, height=hover_height, duration=3.0)
    if not interruptible_sleep(2.5):
        return  # paro durante despegue

    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    circle_points = [
        (x0 + radius * np.cos(theta),
         y0 + radius * np.sin(theta),
         hover_height)
        for theta in angles
    ]

    print("Ejecutando trayectoria circular...")
    for i, (x, y, z) in enumerate(circle_points):

        if stop_event.is_set():
            return  # motores ya cortados por emergency_listener

        print(f"  → Punto {i+1}/{num_points}: ({x:.2f}, {y:.2f}, {z:.2f})")
        move_to_position(scf, x, y, z, velocity=velocity)
        theoretical_trajectory.append([x, y, z])

        if not interruptible_sleep(0.2):
            return

    if not stop_event.is_set():
        move_to_position(scf, x0 + radius, y0, hover_height, velocity=velocity)
        theoretical_trajectory.append([x0 + radius, y0, hover_height])

    if not stop_event.is_set():
        print("Aterrizando...")
        land(scf, height=0.0, duration=2.0)
        interruptible_sleep(2.0)

# -------------------------------------------------------
# GRAFICAR
# -------------------------------------------------------
def plot_trajectories():
    if len(real_trajectory) == 0:
        print("Sin datos para graficar.")
        return

    real = np.array(real_trajectory)
    theo = np.array(theoretical_trajectory)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if len(theo) > 0:
        ax.plot(theo[:, 0], theo[:, 1], theo[:, 2], 'r--', label='Teórica')
    ax.plot(real[:, 0], real[:, 1], real[:, 2], 'b', label='Real')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Trayectoria del Crazyflie')
    plt.show()

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    global scf_global

    cflib.crtp.init_drivers()

    mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
    mqtt_thread.start()

    stop_thread = threading.Thread(target=emergency_listener, daemon=True)
    stop_thread.start()

    print("Conectando al dron...")
    scf_global = connect(URI)

    # --- AGREGAR ESTO ---
    print("Configurando estimador...")
    scf_global.cf.param.set_value('commander.enHighLevel', '1')
    scf_global.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf_global.cf.param.set_value('kalman.resetEstimation', '0')

    time.sleep(3.0)  # esperar que el EKF converja con datos del MoCap

    try:
        fly_circle(scf_global)

    except Exception as e:
        print(f"Error general: {str(e)}")
        _cortar_motores()

    finally:
        if not stop_event.is_set():
            land(scf_global)
        disconnect(scf_global)
        plot_trajectories()

if __name__ == '__main__':
    main()