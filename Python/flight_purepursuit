"""
lab_trajectory_flight.py — Vuelo en laboratorio con Pure Pursuit + Catmull-Rom.

Flujo:
  1. Crea waypoints con create_waypoints.py
  2. Previsualiza con preview_trajectory.py
  3. Vuela con este archivo

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
from trajectory import TrajectoryGenerator
from pure_pursuit import PurePursuit

# -------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------
URI         = "radio://0/80/2M/E7E7E7E7EA"
MQTT_TOPIC  = 'mocap/drone2'
MQTT_BROKER = '192.168.50.200'
PORT        = 1880

# -------------------------------------------------------
# PARÁMETROS DE TRAYECTORIA
# -------------------------------------------------------

SPLINE_POINTS = 60 
METHOD        = 'catmull_rom'
TENSION       = 0.6

# -------------------------------------------------------
# PARÁMETROS PURE PURSUIT
# -------------------------------------------------------
LOOKAHEAD_DISTANCE = 0.21    # [m] distancia de anticipación
                             # Subir si oscila, bajar si corta curvas
CRUISE_SPEED       = 0.06    # [m/s] velocidad de crucero
TAKEOFF_SPEED      = 0.3    # [m/s] velocidad al ir al inicio
CONTROL_RATE       = 0.05   # [s] período del loop de control (20 Hz)

# -------------------------------------------------------
# VARIABLES GLOBALES
# -------------------------------------------------------
mocap_pose      = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
real_trajectory = []
theo_trajectory = []
timestamps      = []
scf_global      = None
last_ts         = None
stop_event      = threading.Event()

mqtt_client = None
mqtt_started = False

emergency_started = False
# -------------------------------------------------------
# PARO DE EMERGENCIA
# -------------------------------------------------------
def emergency_listener():
    print("Presiona 'q' para paro de emergencia")
    while not stop_event.is_set():
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if key == 'q':
                print("\n🚨 PARO DE EMERGENCIA")
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
# MQTT — recibe MoCap y alimenta el EKF
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

        # -------------------------------------------------------
        # DIAGNÓSTICO DEL CUATERNIÓN — sin afectar el vuelo
        # Gira el drone físicamente y observa si el yaw_deg cambia
        # de forma consistente con el movimiento real.
        # Cuando confirmes que es correcto, puedes usar send_extpose.
        # -------------------------------------------------------
        if 'rotation' in data['payload']['pose']:
            rot = data['payload']['pose']['rotation']
            qx  = float(rot['qx'])
            qy  = float(rot['qy'])
            qz  = float(rot['qz'])
            qw  = float(rot['qw'])

            # Calcular yaw desde el cuaternión
            yaw_deg = np.degrees(np.arctan2(
                2*(qw*qz + qx*qy),
                1 - 2*(qy*qy + qz*qz)
            ))

            # Guardar en mocap_pose para Pure Pursuit
            mocap_pose['yaw'] = np.radians(yaw_deg)

            # Imprimir cada ~1s para no saturar consola
            # (el MoCap manda ~1Hz según tus logs anteriores)
            # print(f"  yaw={yaw_deg:+.1f}°  "
              #    f"qx={qx:.3f} qy={qy:.3f} "
               #   f"qz={qz:.3f} qw={qw:.3f}")

        # -------------------------------------------------------
        # SIGUE USANDO send_extpos — sin cambios al vuelo
        # -------------------------------------------------------
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

    global mqtt_client, mqtt_started

    if mqtt_started:
        return

    mqtt_client = mqtt.Client()
    mqtt_client.on_message = on_message

    mqtt_client.connect(MQTT_BROKER, PORT, 60)
    mqtt_client.subscribe(MQTT_TOPIC)

    mqtt_client.loop_start()

    mqtt_started = True

    print("MQTT iniciado correctamente.")


def start_emergency_listener():

    global emergency_started

    if emergency_started:
        return

    threading.Thread(
        target=emergency_listener,
        daemon=True
    ).start()

    emergency_started = True

# -------------------------------------------------------
# SLEEP INTERRUPTIBLE
# -------------------------------------------------------
def interruptible_sleep(seconds, interval=0.05):
    elapsed = 0.0
    while elapsed < seconds:
        if stop_event.is_set():
            return False
        time.sleep(interval)
        elapsed += interval
    return True


# -------------------------------------------------------
# VELOCIDAD ADAPTATIVA
# -------------------------------------------------------

def compute_adaptive_speeds(path, v_max, v_min, curve_gain=10.0):
    speeds = np.full(len(path), v_max)

    for i in range(1, len(path) - 1):
        p_prev = path[i - 1]
        p_curr = path[i]
        p_next = path[i + 1]

        d1 = p_next - p_prev
        d2 = p_next - 2 * p_curr + p_prev

        norm_d1 = np.linalg.norm(d1)
        if norm_d1 < 1e-6:
            continue

        curvature = np.linalg.norm(np.cross(d1, d2)) / (norm_d1 ** 3)

        factor = 1.0 / (1.0 + curve_gain * curvature)
        speeds[i] = v_min + (v_max - v_min) * factor

    # suavizado
    smooth = speeds.copy()
    for i in range(1, len(speeds) - 1):
        smooth[i] = (speeds[i-1] + speeds[i] + speeds[i+1]) / 3.0

    return smooth

# -------------------------------------------------------
# VUELO CON PURE PURSUIT
# -------------------------------------------------------
def fly_trajectory(scf):
    global theo_trajectory

    # -- Cargar waypoints --
    try:
        import os

        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "waypoints.json")

        with open(file_path) as f:
            waypoints = json.load(f)
    except FileNotFoundError:
        print("ERROR: No se encontró waypoints.json")
        print("Primero crea los waypoints con create_waypoints.py")
        return

    print(f"\nWaypoints: {len(waypoints)}")
    for i, wp in enumerate(waypoints):
        print(f"  WP{i}: x={wp[0]:.2f}, y={wp[1]:.2f}, z={wp[2]:.2f}")

    # -- Generar spline Catmull-Rom --
    traj = TrajectoryGenerator(waypoints)
    path = traj.generate_spline(SPLINE_POINTS, method=METHOD, tension=TENSION)
    theo_trajectory = path.tolist()

    print(f"\nSpline: {len(path)} puntos | "
          f"longitud: {traj.total_length():.2f}m | "
          f"tiempo estimado: {traj.total_length()/CRUISE_SPEED:.1f}s")
    print(f"Pure Pursuit: lookahead={LOOKAHEAD_DISTANCE}m | "
          f"velocidad={CRUISE_SPEED}m/s\n")

    # -- Inicializar Pure Pursuit --
    pp = PurePursuit(
        lookahead_distance=LOOKAHEAD_DISTANCE,
        min_lookahead=0.15,
        max_lookahead=0.6
    )
    pp.set_path(path)

    # -- Esperar MoCap --
    print("Esperando posición inicial del MoCap...")
    timeout = 10.0
    t0      = time.time()
    while (mocap_pose['x'] == 0.0 and
           mocap_pose['y'] == 0.0 and
           mocap_pose['z'] == 0.0):
        if stop_event.is_set():
            return
        if time.time() - t0 > timeout:
            print("⚠ Timeout esperando MoCap")
            return
        time.sleep(0.1)

    print(f"Posición inicial: "
          f"x={mocap_pose['x']:.3f}, "
          f"y={mocap_pose['y']:.3f}, "
          f"z={mocap_pose['z']:.3f}")

    # -- Despegue --
    first_z = float(path[0][2])
    print(f"\nDespegando a z={first_z:.2f}m...")
    takeoff(scf, height=first_z, duration=3)
    if not interruptible_sleep(2.0):
        return

    # -- Ir al inicio con move_to_position (solo esta vez) --
    print("Moviéndose al inicio de la trayectoria...")
    move_to_position(scf,
                     float(path[0][0]),
                     float(path[0][1]),
                     float(path[0][2]),
                     velocity=TAKEOFF_SPEED)
    if not interruptible_sleep(1.0):
        return
    

    print("Cambiando a Low Level...")
    scf_global.cf.param.set_value('commander.enHighLevel', '0')
    time.sleep(0.5)


    # ----------------------------------------------------------
    # LOOP PURE PURSUIT
    #
    # Diferencia vs seguidor simple:
    #   Seguidor simple:  go_to(WP1)→espera→go_to(WP2)→espera...
    #                     El drone PARA en cada punto
    #
    #   Pure Pursuit:     Pure Pursuit: cada CONTROL_RATE segundos calcula
    #                      el punto de anticipación y genera comandos
    #                      continuos de velocidad para seguir la trayectoria.
    # ----------------------------------------------------------
    print(f"\nIniciando Pure Pursuit...")
    t_inicio   = time.time()
    last_print = time.time()
    


    adaptive_speeds = compute_adaptive_speeds(
        path,
        v_max=CRUISE_SPEED,
        v_min=0.06
    )
    

    z_cmd = path[0][2]

    speed_cmd = CRUISE_SPEED

    while not stop_event.is_set() and not pp.finished:

        x = mocap_pose['x']
        y = mocap_pose['y']
        yaw = mocap_pose['yaw']

        # velocidad adaptativa
        idx = min(pp.index, len(adaptive_speeds) - 1)
        target_speed = adaptive_speeds[idx]

        # suavizado de velocidad
        alpha_v = 0.2

        speed_cmd = alpha_v * target_speed + (1 - alpha_v) * speed_cmd
        target_speed = speed_cmd

        # lookahead adaptativo
        pp.adaptive_lookahead(target_speed)

        # comando base
        vx, vy, yaw_desired, lookahead = pp.get_velocity_command(
            x, y, yaw, target_speed
        )

        if lookahead is None:
            break

        # altura suave
        z_target = pp.get_desired_height(x, y)
        if z_target is None:
            z_target = z_cmd

        alpha_z = 0.2
        z_cmd = alpha_z * z_target + (1 - alpha_z) * z_cmd
        z_desired = z_cmd 

        # -----------------------------
        # YAW CONTROL
        # -----------------------------

        #yaw_error = yaw_desired - yaw
        #yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

        # controlador proporcional
        #yaw_rate = 2.0 * yaw_error

        # limitar velocidad angular
        #yaw_rate = np.clip(yaw_rate, -1.0, 1.0)

        # -----------------------------
        # YAW — versión estable
        # -----------------------------
        # yaw_cmd = 0.0   # yaw fijo en 0 — estable 

        # -----------------------------
        # LIMITAR VELOCIDAD EN CURVAS (AHORA SÍ)
        # -----------------------------
      #  max_speed_turn = 0.12

       # if abs(yaw_error) > 0.3:
      #      target_speed = min(target_speed, max_speed_turn)

        # recalcular comando con nueva velocidad
       # vx, vy, yaw_desired, lookahead = pp.get_velocity_command(
       #     x, y, yaw, target_speed
       # )
        # --------------------------------------------------
        # ENVIAR COMANDO
        # --------------------------------------------------
        # control vertical
        vz = 0.5 * (z_desired - mocap_pose['z'])
        vz = np.clip(vz, -0.09, 0.09)

        # yaw fijo por ahora
        yaw_rate = 0.0

        # comando low level
        scf.cf.commander.send_velocity_world_setpoint(
            float(vx),
            float(vy),
            float(vz),
            float(yaw_rate)
        )

        # Log cada 2 segundos
        if time.time() - last_print > 2.0:
            dist_fin = np.linalg.norm([path[-1][0]-x, path[-1][1]-y])
            print(f"  [{pp.get_progress():.0f}%] "
                  f"pos=({x:.2f},{y:.2f}) "
                  f"→ look=({lookahead[0]:.2f},{lookahead[1]:.2f}) "
                  f"dist_fin={dist_fin:.2f}m")
            last_print = time.time()

        time.sleep(CONTROL_RATE)

    print(f"\n✓ Pure Pursuit completado en {time.time()-t_inicio:.1f}s")

    # -- Hover activo (2s) — sigue mandando velocidad cero --
    print("Hover final (2s)...")
    t_hover = time.time()
    while time.time() - t_hover < 2.0:
        if stop_event.is_set():
            return
        scf.cf.commander.send_velocity_world_setpoint(0.0, 0.0, 0.0, 0.0)
        time.sleep(CONTROL_RATE)

    # -- Descenso controlado en low-level --
    print("Aterrizando suavemente...")
    descent_rate = 0.12   # m/s hacia abajo
    dt = 0.05             # 20 Hz

    while mocap_pose['z'] > 0.10:
        if stop_event.is_set():
            return
        # mantener XY con un P suave, bajar Z a tasa fija
        x_err = 0.0  # no hay setpoint XY nuevo, solo frenar deriva
        y_err = 0.0
        scf.cf.commander.send_velocity_world_setpoint(
            0.0, 0.0, -descent_rate, 0.0
        )
        time.sleep(dt)

    # -- Apagado limpio --
    # restaurar high level
    scf_global.cf.param.set_value('commander.enHighLevel', '1')
    time.sleep(0.2)

    # reset EKF
    scf_global.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf_global.cf.param.set_value('kalman.resetEstimation', '0')

    # detener commander
    scf.cf.commander.send_stop_setpoint()
    time.sleep(0.5)

# -------------------------------------------------------
# GRÁFICA Y MÉTRICAS
# -------------------------------------------------------
def plot_trajectories():
    if len(real_trajectory) == 0:
        print("Sin datos para graficar.")
        return

    real   = np.array(real_trajectory)
    theo   = np.array(theo_trajectory) if theo_trajectory else None
    wp_arr = None
    # -- Cargar waypoints --
    try:
        import os

        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "waypoints.json")

        with open(file_path) as f:
            waypoints = json.load(f)
    except FileNotFoundError:
        print("ERROR: No se encontró waypoints.json")
        print("Primero crea los waypoints con create_waypoints.py")
        return

    # -- Ventana 1: Trayectorias --
    fig1 = plt.figure(figsize=(14, 6),
                      num="Pure Pursuit — Trayectorias")
    fig1.suptitle("Resultado — Pure Pursuit + Catmull-Rom",
                  fontsize=13, fontweight='bold')

    ax1 = fig1.add_subplot(121, projection='3d')
    ax2 = fig1.add_subplot(122)

    for ax, is_3d in [(ax1, True), (ax2, False)]:
        if theo is not None:
            if is_3d:
                ax.plot(theo[:, 0], theo[:, 1], theo[:, 2],
                        'r--', lw=2, label='Spline (Catmull-Rom)')
            else:
                ax.plot(theo[:, 0], theo[:, 1],
                        'r--', lw=2, label='Spline (Catmull-Rom)')

        if wp_arr is not None:
            if is_3d:
                ax.plot(wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2],
                        'o--', color='orange', lw=1.5, markersize=8,
                        markeredgecolor='black', label='Waypoints')
            else:
                ax.plot(wp_arr[:, 0], wp_arr[:, 1],
                        'o--', color='orange', lw=1.5, markersize=8,
                        markeredgecolor='black', label='Waypoints')

        if is_3d:
            ax.plot(real[:, 0], real[:, 1], real[:, 2],
                    'b', lw=1.5, label='Real (MoCap)')
            ax.scatter(*real[0],  color='green', s=80, zorder=5)
            ax.scatter(*real[-1], color='red',   s=80, zorder=5)
            ax.set_zlabel('Z [m]')
            ax.set_title('Vista 3D')
            ax.view_init(elev=30, azim=135)
        else:
            ax.plot(real[:, 0], real[:, 1],
                    'b', lw=1.5, label='Real (MoCap)')
            ax.scatter(*real[0, :2],  color='green', s=80,
                       zorder=5, label='Inicio')
            ax.scatter(*real[-1, :2], color='red',   s=80,
                       zorder=5, label='Fin')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.4)
            ax.set_title('Vista superior (XY)')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("resultado_pure_pursuit.png", dpi=150, bbox_inches='tight')

    # -- Ventana 2: Métricas --
    fig2 = plt.figure(figsize=(14, 5),
                      num="Pure Pursuit — Métricas")
    fig2.suptitle("Métricas — Pure Pursuit + Catmull-Rom",
                  fontsize=13, fontweight='bold')

    ax3 = fig2.add_subplot(121)
    ax4 = fig2.add_subplot(122)

    ax3.plot(real[:, 2], 'b', lw=1.5, label='Real (MoCap)')
    if theo is not None:
        t_theo = np.linspace(0, len(real), len(theo))
        ax3.plot(t_theo, theo[:, 2], 'r--', lw=1.5, label='Deseada')
    ax3.set_xlabel('Muestra'); ax3.set_ylabel('Z [m]')
    ax3.set_title('Perfil de altura')
    ax3.legend(); ax3.grid(True, alpha=0.4)

    if theo is not None:
        n   = min(len(real), len(theo))
        err = np.linalg.norm(real[:n, :2] - theo[:n, :2], axis=1)
        ax4.plot(err, 'purple', lw=1.5)
        ax4.fill_between(range(len(err)), err, alpha=0.2, color='purple')
        ax4.axhline(err.mean(), color='red', linestyle='--', lw=1.5,
                    label=f'Promedio: {err.mean():.4f}m')
        ax4.axhline(np.percentile(err, 95), color='orange',
                    linestyle=':', lw=1.5,
                    label=f'P95: {np.percentile(err, 95):.4f}m')
        ax4.set_xlabel('Muestra'); ax4.set_ylabel('Error [m]')
        ax4.set_title('Cross-Track Error')
        ax4.legend(); ax4.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("metricas_pure_pursuit.png", dpi=150, bbox_inches='tight')
    plt.show()

    # -- Métricas consola --
    print("\n" + "=" * 50)
    print("  MÉTRICAS — PURE PURSUIT + CATMULL-ROM")
    print("=" * 50)
    if theo is not None:
            n   = min(len(real), len(theo))
            err = np.linalg.norm(real[:n, :2] - theo[:n, :2], axis=1)
            print(f"  Error XY promedio:    {err.mean():.4f} m")
            print(f"  Error XY máximo:      {err.max():.4f} m")
            print(f"  Error XY std:         {err.std():.4f} m")
            print(f"  Percentil 95:         {np.percentile(err, 95):.4f} m")

    print(f"  Altura promedio:      {real[:, 2].mean():.3f} m")
    print(f"  Std altura:              {real[:, 2].std():.4f} m")

        
    if len(timestamps) > 1 and len(real) > 1:

            # Alinear tamaños (clave)
            n = min(len(real), len(timestamps))
            real_sync = real[:n]
            time_sync = np.array(timestamps[:n])

            dists = np.linalg.norm(np.diff(real_sync, axis=0), axis=1)
            dts   = np.diff(time_sync)

            # evitar divisiones por 0
            dts   = np.where(dts > 0, dts, 1e-6)

            vels  = dists / dts

            # filtrar outliers
            vels  = vels[vels < 3.0]

            if len(vels) > 0:
                print(f"  Velocidad promedio:   {vels.mean():.3f} m/s")
                print(f"  Velocidad máxima:     {vels.max():.3f} m/s")

                print(f"  Velocidad promedio:   {vels.mean():.3f} m/s")
                print(f"  Velocidad máxima:     {vels.max():.3f} m/s")


    print(f"  Lookahead:                  {LOOKAHEAD_DISTANCE} m")
    print(f"  Velocidad crucero:          {CRUISE_SPEED} m/s")
    print(f"  Puntos MoCap grabados:      {len(real_trajectory)}")
    print("=" * 50)

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():

    global scf_global

    reset_state()

    cflib.crtp.init_drivers()

    start_mqtt()
    start_emergency_listener()

    print("Conectando al drone...")
    scf_global = connect(URI)
    if scf_global is None:
        print("ERROR: No se pudo conectar.")
        return

    print("Configurando estimador para MoCap sin Flowdeck...")
    scf_global.cf.param.set_value('commander.enHighLevel', '1')
    scf_global.cf.param.set_value('stabilizer.controller', '1')
    scf_global.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf_global.cf.param.set_value('kalman.resetEstimation', '0')
    # time.sleep(5.0)

    print("Esperando estabilización del EKF (5s)...")
    time.sleep(5.0)

    try:
        fly_trajectory(scf_global)
    except Exception as e:
        print(f"Error durante el vuelo: {e}")
        _cortar_motores()
    finally:
        if not stop_event.is_set():
            try:
                land(scf_global, height=0.0, duration=2.5)
            except:
                pass
        disconnect(scf_global)
        plot_trajectories()

        scf_global.cf.commander.send_stop_setpoint()

        time.sleep(0.2)

        disconnect(scf_global)

        scf_global = None


def reset_state():
    global scf_global, last_ts, mocap_pose

    # Limpiar estado de vuelo anterior
    stop_event.clear()
    real_trajectory.clear()
    theo_trajectory.clear()
    timestamps.clear()
    last_ts = None

    # Resetear pose a cero para forzar espera de MoCap
    mocap_pose['x'] = 0.0
    mocap_pose['y'] = 0.0
    mocap_pose['z'] = 0.0
    mocap_pose['yaw'] = 0.0

    scf_global = None

if __name__ == '__main__':
    main()
