"""
lab_hover_test.py — Prueba básica: despegar, ir a un punto, hover estable.

Controles:
  'q' → paro de emergencia inmediato
"""

import json
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import msvcrt
from datetime import datetime
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
import cflib.crtp

from crazyflie_python_commands_mod import (
    connect, disconnect, takeoff, land,
    move_to_position, set_position, get_pose
)

# -------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------
URI         = "radio://0/80/2M/E7E7E7E7EA"
MQTT_TOPIC  = 'mocap/drone2'
MQTT_BROKER = '192.168.50.200'
PORT        = 1880

# -------------------------------------------------------
# PARÁMETROS — solo estos dos importan para esta prueba
# -------------------------------------------------------
HOVER_HEIGHT  = 0.4    # [m] altura de despegue y hover
TARGET_OFFSET = 0.3   # [m] cuánto se mueve desde su posición inicial
                       # en dirección X. Cambiar a 0.0 para solo hover fijo.
HOVER_TIME    = 7.0   # [s] cuánto tiempo se queda estable en el punto

# -------------------------------------------------------
# VARIABLES GLOBALES
# -------------------------------------------------------
mocap_pose      = {'x': 0.0, 'y': 0.0, 'z': 0.0}
real_trajectory = []
timestamps      = []
scf_global      = None
last_ts         = None
stop_event      = threading.Event()

# -------------------------------------------------------
# PARO DE EMERGENCIA
# -------------------------------------------------------
def emergency_listener():
    print("Presiona 'q' para paro de emergencia")
    while not stop_event.is_set():
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if key == 'q':
                print("\n PARO DE EMERGENCIA")
                stop_event.set()
                _cortar_motores()
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
# MQTT
# -------------------------------------------------------
def on_message(client, userdata, msg):
    global scf_global, last_ts
    if stop_event.is_set():
        return
    try:
        data   = json.loads(msg.payload.decode())
        pos    = data['payload']['pose']['position']
        ts_str = data.get('ts', None)

        if ts_str is not None:
            msg_time = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            if last_ts is not None and msg_time <= last_ts:
                return
            last_ts = msg_time

        mocap_pose['x'] = float(pos['x'])
        mocap_pose['y'] = float(pos['y'])
        mocap_pose['z'] = float(pos['z'])

        # Mandar directo sin sleep ni print — más rápido y no bloquea el hilo
        if scf_global is not None:
            scf_global.cf.extpos.send_extpos(
                mocap_pose['x'],
                mocap_pose['y'],
                mocap_pose['z']
            )

        real_trajectory.append([mocap_pose['x'],
                                 mocap_pose['y'],
                                 mocap_pose['z']])
        timestamps.append(time.time())

    except Exception as e:
        print(f"Error MQTT: {e}")

def start_mqtt():
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, PORT, 60)
    client.subscribe(MQTT_TOPIC)
    client.loop_forever()

def interruptible_sleep(seconds, interval=0.05):
    elapsed = 0.0
    while elapsed < seconds:
        if stop_event.is_set():
            return False
        time.sleep(interval)
        elapsed += interval
    return True

# -------------------------------------------------------
# VUELO — despegar, ir a un punto, hover, aterrizar
# -------------------------------------------------------
def fly_hover_test(scf):

    # -- Esperar MoCap --
    print("Esperando posición inicial del MoCap...")
    timeout = 10.0
    t0      = time.time()
    while (abs(mocap_pose['x']) < 0.001 and
           abs(mocap_pose['y']) < 0.001 and
           abs(mocap_pose['z']) < 0.001):
        if stop_event.is_set():
            return
        if time.time() - t0 > timeout:
            print("Timeout esperando MoCap")
            return
        time.sleep(0.1)

    x0 = mocap_pose['x']
    y0 = mocap_pose['y']
    print(f"Posición inicial: x={x0:.3f}, y={y0:.3f}, z={mocap_pose['z']:.3f}")

    # Punto objetivo — solo se mueve TARGET_OFFSET en X
    # Si TARGET_OFFSET=0.0, solo hace hover en el mismo lugar
    target_x = x0 + TARGET_OFFSET
    target_y = y0
    target_z = HOVER_HEIGHT

    print(f"\nPlan de vuelo:")
    print(f"  1. Despegar a z={HOVER_HEIGHT}m")
    print(f"  2. Ir a ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")
    print(f"  3. Hover estable por {HOVER_TIME}s")
    print(f"  4. Aterrizar\n")

    # -- Paso 1: Despegue --
    print("Paso 1: Despegando...")
    takeoff(scf, height=HOVER_HEIGHT, duration=4.0)
    if not interruptible_sleep(4.0):   # 3s de hover estable post-despegue
        return
    print(f"  Altura actual: {mocap_pose['z']:.3f}m")

    # -- Paso 2: Ir al punto objetivo (solo si hay offset) --
    if TARGET_OFFSET > 0.0:
        print(f"Paso 2: Moviéndose a ({target_x:.3f}, {target_y:.3f})...")
        move_to_position(scf, target_x, target_y, target_z, velocity=0.1)
        if not interruptible_sleep(2.0):   # esperar estabilización
            return
        print(f"  Posición actual: x={mocap_pose['x']:.3f}, "
              f"y={mocap_pose['y']:.3f}, z={mocap_pose['z']:.3f}")
    else:
        print("Paso 2: Sin desplazamiento (TARGET_OFFSET=0.0)")

    # -- Paso 3: Hover estable --
    print(f"Paso 3: Hover estable por {HOVER_TIME}s...")
    print("  Monitoreando posición:")

    t_hover = time.time()
    while time.time() - t_hover < HOVER_TIME:
        if stop_event.is_set():
            return
        # Imprimir posición cada segundo para ver si se mantiene estable
        print(f"    t={time.time()-t_hover:.1f}s — "
              f"x={mocap_pose['x']:.3f}, "
              f"y={mocap_pose['y']:.3f}, "
              f"z={mocap_pose['z']:.3f}")
        time.sleep(1.0)

    # -- Paso 4: Aterrizaje suave --
    print("Paso 4: Aterrizando suavemente...")
    commander = scf.cf.high_level_commander

    # Bajar en pasos de 10cm cada 2 segundos
    current_z = mocap_pose['z']
    step = 0.1   # [m] bajar de a 10cm

    while current_z > 0.1:
        if stop_event.is_set():
            return
        current_z = max(current_z - step, 0.1)
        commander.go_to(mocap_pose['x'], mocap_pose['y'],
                        current_z, 0.0, duration_s=2.0)
        if not interruptible_sleep(2.0):
            return
        print(f"  Bajando... z actual={mocap_pose['z']:.3f}m")

    # Aterrizaje final desde 10cm
    commander.land(absolute_height_m=0.0, duration_s=2.0)
    interruptible_sleep(2.0)
    # NO llamar commander.stop() — dejar que el firmware maneje el final
    print("✓ Aterrizaje completado")

# -------------------------------------------------------
# GRÁFICA
# -------------------------------------------------------
def plot_trajectories():
    if len(real_trajectory) == 0:
        print("Sin datos para graficar.")
        return

    real = np.array(real_trajectory)

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Prueba hover — Despegue y punto estable",
                 fontsize=13, fontweight='bold')

    # -- Vista XY --
    ax1 = fig.add_subplot(131)
    ax1.plot(real[:, 0], real[:, 1], 'b', lw=1.5)
    ax1.scatter(*real[0, :2],  color='green', s=80, zorder=5, label='Inicio')
    ax1.scatter(*real[-1, :2], color='red',   s=80, zorder=5, label='Fin')
    ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]')
    ax1.set_title('Vista superior (XY)')
    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.4)
    ax1.legend(fontsize=9)

    # -- Altura vs tiempo --
    ax2 = fig.add_subplot(132)
    t = np.array(timestamps) - timestamps[0]
    ax2.plot(t, real[:, 2], 'b', lw=1.5)
    ax2.axhline(HOVER_HEIGHT, color='r', linestyle='--',
                lw=1.5, label=f'Objetivo: {HOVER_HEIGHT}m')
    ax2.set_xlabel('Tiempo [s]'); ax2.set_ylabel('Z [m]')
    ax2.set_title('Altura vs tiempo')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.4)

    # -- Posición XY vs tiempo --
    ax3 = fig.add_subplot(133)
    ax3.plot(t, real[:, 0], 'b', lw=1.5, label='X real')
    ax3.plot(t, real[:, 1], 'r', lw=1.5, label='Y real')
    ax3.axhline(real[0, 0] + TARGET_OFFSET, color='b',
                linestyle='--', lw=1, label=f'X objetivo')
    ax3.axhline(real[0, 1], color='r',
                linestyle='--', lw=1, label=f'Y objetivo')
    ax3.set_xlabel('Tiempo [s]'); ax3.set_ylabel('Posición [m]')
    ax3.set_title('X e Y vs tiempo')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("resultado_hover.png", dpi=150, bbox_inches='tight')
    print("Gráfica guardada: resultado_hover.png")
    plt.show()

    # -- Métricas --
    print("\n" + "=" * 40)
    print("  MÉTRICAS DE HOVER")
    print("=" * 40)
    print(f"  Altura objetivo:        {HOVER_HEIGHT:.3f} m")
    print(f"  Altura promedio real:   {real[:, 2].mean():.3f} m")
    print(f"  Error altura promedio:  {abs(real[:,2].mean()-HOVER_HEIGHT):.4f} m")
    print(f"  Std altura:             {real[:, 2].std():.4f} m")
    print(f"  Std X:                  {real[:, 0].std():.4f} m")
    print(f"  Std Y:                  {real[:, 1].std():.4f} m")
    print(f"  Puntos MoCap grabados:  {len(real_trajectory)}")
    print("=" * 40)

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    global scf_global

    cflib.crtp.init_drivers()
    threading.Thread(target=start_mqtt,         daemon=True).start()
    threading.Thread(target=emergency_listener, daemon=True).start()

    print("Conectando al drone...")
    scf_global = connect(URI)
    if scf_global is None:
        print("ERROR: No se pudo conectar.")
        return

    print("Configurando estimador para MoCap sin Flowdeck...")
    scf_global.cf.param.set_value('commander.enHighLevel', '1')
    scf_global.cf.param.set_value('stabilizer.controller', '1') #PID

    # Reset del Kalman después de cambiar el controlador
    scf_global.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf_global.cf.param.set_value('kalman.resetEstimation', '0')

    print("Esperando estabilización del EKF (5s)...")
    time.sleep(5.0)

    print("Esperando estabilización del EKF (5s)...")
    time.sleep(5.0)

    try:
        fly_hover_test(scf_global)
    except Exception as e:
        print(f"Error durante el vuelo: {e}")
        _cortar_motores()
    finally:
        if not stop_event.is_set():
            try:
                land(scf_global, height=0.0, duration=4)
            except:
                pass
        disconnect(scf_global)
        plot_trajectories()


if __name__ == '__main__':
    main()
