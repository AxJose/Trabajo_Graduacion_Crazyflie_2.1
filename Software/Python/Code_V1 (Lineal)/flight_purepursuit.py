"""
flight_purepursuit.py — Vuelo en laboratorio (Robotat / OptiTrack) con
Pure Pursuit sobre un Crazyflie.

Flujo:
  1. Crea waypoints con create_waypoints.py (o el script de MATLAB)
  2. Previsualiza con preview_trajectory.py
  3. Vuela con este archivo

Controles:
  'espacio' → paro de emergencia inmediato
"""

import os
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
    connect, disconnect, land,
    set_position, get_pose
)
from trajectory import TrajectoryGenerator, compute_speed_profile
from pure_pursuit import PurePursuit

# -------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------
URI         = "radio://0/80/2M/E7E7E7E7EA"
MQTT_TOPIC  = 'mocap/drone2'
MQTT_BROKER = '192.168.50.200'
PORT        = 1880

# -------------------------------------------------------
# TRAYECTORIA
# -------------------------------------------------------
PATH_RESOLUTION = 0.02      # [m] separación entre puntos de la trayectoria
METHOD          = 'linear'  # 'catmull_rom' | 'linear' | 'cubic' | 'quadratic'
TENSION         = 0.5       # solo aplica a catmull_rom
FIXED_HEIGHT    = None      # [m] si no es None, aplana la Z de los waypoints

# -------------------------------------------------------
# PERFIL DE VELOCIDAD
# -------------------------------------------------------
V_MAX       = 1.15    # [m/s] rapidez máxima en recta
V_MIN       = 0.12    # [m/s] rapidez mínima en curva cerrada
A_LAT_MAX   = 0.35    # [m/s^2] aceleración lateral máxima en curva
A_TAN_MAX   = 0.80    # [m/s^2] aceleración y frenado a lo largo de la trayectoria
VZ_MAX_PATH = 0.30    # [m/s] velocidad vertical máxima que exige la trayectoria
V_START     = 0.12    # [m/s] rapidez al inicio
V_END       = 0.10    # [m/s] rapidez al final

# -------------------------------------------------------
# PURE PURSUIT
# -------------------------------------------------------
LOOKAHEAD_TIME    = 0.40   # [s] L = LOOKAHEAD_TIME * v
LOOKAHEAD_MIN     = 0.10   # [m]
LOOKAHEAD_MAX     = 1.00   # [m]
LOOKAHEAD_K_CURVE = 0.0    # acorta L en curva; 0 = desactivado
GOAL_TOLERANCE    = 0.10   # [m] arco restante para dar por terminada la ruta

# -------------------------------------------------------
# CONTROL
# -------------------------------------------------------
CONTROL_HZ   = 50.0
CONTROL_RATE = 1.0 / CONTROL_HZ

KP_LAT   = 1.8    # [1/s] ganancia del error lateral

# Corrección lateral adicional de magnitud saturada (0 = desactivada)
K_LAT_SAT     = 0.0    # [m/s] magnitud máxima
LAT_BOUNDARY  = 0.03   # [m] ancho de la capa límite

KP_Z     = 1.3    # [1/s] ganancia del error de altura

V_XY_MAX_CMD = 2.00   # [m/s] tope del comando horizontal
VZ_MAX_CMD   = 0.45   # [m/s] tope del comando vertical
ALPHA_V      = 0.30   # filtro de la rapidez objetivo
ALPHA_VZ     = 0.27   # filtro del comando vertical

FLIGHT_TIMEOUT_FACTOR = 2.5   # aborta si tarda más de X veces lo estimado

# -------------------------------------------------------
# YAW
# -------------------------------------------------------
# 'hold' mantiene la orientación fija; 'path' apunta en la dirección de la trayectoria.
YAW_MODE       = 'hold'
YAW_OFFSET_DEG = 180.0   # desfase entre el eje X del MoCap y el frente del dron
KP_YAW         = 0.6
YAW_RATE_MAX   = 0.5     # [rad/s]
YAW_ALPHA_REF  = 0.08    # filtro de la referencia de yaw
YAW_ALPHA_OUT  = 0.12    # filtro de la salida de yaw
YAW_DEADBAND   = 5.0     # [deg]

# -------------------------------------------------------
# MOCAP → EKF
# -------------------------------------------------------
EXTPOS_RATE_HZ = 50.0   # [Hz] frecuencia máxima de reenvío de la posición al EKF
USE_EXTPOSE    = False  # True → envía también la orientación (send_extpose)

# -------------------------------------------------------
# DESPEGUE / APROXIMACIÓN / ATERRIZAJE (todo low-level)
# -------------------------------------------------------
LOWLEVEL_VERTICAL_SPEED   = 0.10
GOTO_START_MAX_SPEED      = 0.20
GOTO_START_KP             = 0.6
GOTO_START_POS_TOLERANCE  = 0.04
GOTO_START_TIMEOUT        = 15.0
SETTLE_TIME               = 1.0    # [s] estabilización en el punto inicial
DESCENT_RATE              = 0.15   # [m/s]
DESCENT_TIMEOUT           = 15.0   # [s]

# -------------------------------------------------------
# REGISTRO DE VUELOS
# -------------------------------------------------------
# Cada vuelo guarda sus archivos en CARPETA_VUELOS/ con el prefijo <MODO>_<fecha>_<hora>
MODO_ETIQUETA  = 'Lineal_rapido'
CARPETA_VUELOS = 'vuelos_obstaculos'

# Umbrales de advertencia en el .txt (no descartan el vuelo)
VBAT_MIN_AVISO  = 3.10   # [V] voltaje mínimo bajo carga
HUECO_MAX_AVISO = 0.50   # [s] hueco máximo sin datos del MoCap

# -------------------------------------------------------
# VARIABLES GLOBALES
# -------------------------------------------------------
mocap_pose      = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
mocap_quat      = {'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0}
real_trajectory = []
theo_trajectory = []
timestamps      = []
ctrl_log        = []      # registro del lazo de control
flight_info     = {}      # resumen del vuelo para el .txt
scf_global      = None
last_ts         = None
stop_event      = threading.Event()

_mqtt_client     = None
_threads_started = False
_mqtt_msg_count  = 0
_extpos_count    = 0
_last_extpos_t   = 0.0
_last_mocap_t    = 0.0    # instante del último mensaje del MoCap

MOCAP_STALE_S = 0.25   # [s] antigüedad máxima del dato del MoCap en despegue y acercamiento


def mocap_is_fresh(max_age=MOCAP_STALE_S):
    return (time.time() - _last_mocap_t) < max_age


# -------------------------------------------------------
# PARO DE EMERGENCIA
# -------------------------------------------------------
def emergency_listener():
    print("Presiona 'espacio' para paro de emergencia")
    while not stop_event.is_set():
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if key == ' ':
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
    global scf_global, last_ts, _mqtt_msg_count, _last_extpos_t, _extpos_count
    global _last_mocap_t

    if stop_event.is_set():
        return

    _mqtt_msg_count += 1
    try:
        data   = json.loads(msg.payload.decode())
        pos    = data['payload']['pose']['position']
        ts_str = data.get('ts', None)

        # Cada mensaje actualiza la posición, sin descartar por timestamp
        if ts_str is not None:
            last_ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

        mocap_pose['x'] = float(pos['x'])
        mocap_pose['y'] = float(pos['y'])
        mocap_pose['z'] = float(pos['z'])
        _last_mocap_t   = time.time()

        if 'rotation' in data['payload']['pose']:
            rot = data['payload']['pose']['rotation']
            qx  = float(rot['qx'])
            qy  = float(rot['qy'])
            qz  = float(rot['qz'])
            qw  = float(rot['qw'])

            mocap_quat['qx'] = qx
            mocap_quat['qy'] = qy
            mocap_quat['qz'] = qz
            mocap_quat['qw'] = qw

            mocap_pose['yaw'] = float(np.arctan2(
                2 * (qw * qz + qx * qy),
                1 - 2 * (qy * qy + qz * qz)
            ))

        # Reenvío al EKF, limitado a EXTPOS_RATE_HZ
        now = time.time()
        if scf_global is not None and (now - _last_extpos_t) >= 1.0 / EXTPOS_RATE_HZ:
            _last_extpos_t = now
            _extpos_count += 1
            if USE_EXTPOSE:
                qx2, qy2, qz2, qw2 = _quat_con_offset()
                scf_global.cf.extpos.send_extpose(
                    mocap_pose['x'], mocap_pose['y'], mocap_pose['z'],
                    qx2, qy2, qz2, qw2
                )
            else:
                scf_global.cf.extpos.send_extpos(
                    mocap_pose['x'], mocap_pose['y'], mocap_pose['z']
                )

        real_trajectory.append([mocap_pose['x'], mocap_pose['y'], mocap_pose['z']])
        timestamps.append(now)

    except Exception as e:
        print(f"Error MQTT: {e}")


def _quat_con_offset():
    """
    Corrige el desfase de yaw entre el cuerpo rígido definido en OptiTrack
    y el frente físico del dron, componiendo con una rotación en Z.
        q_dron = q_mocap ⊗ q_z(-offset)
    """
    a  = np.radians(-YAW_OFFSET_DEG) * 0.5
    oz, ow = np.sin(a), np.cos(a)

    qx, qy = mocap_quat['qx'], mocap_quat['qy']
    qz, qw = mocap_quat['qz'], mocap_quat['qw']

    return (qx * ow + qy * oz,
            qy * ow - qx * oz,
            qz * ow + qw * oz,
            qw * ow - qz * oz)


def start_mqtt():
    global _mqtt_client
    _mqtt_client = mqtt.Client()
    _mqtt_client.on_message = on_message
    _mqtt_client.connect(MQTT_BROKER, PORT, 60)
    _mqtt_client.subscribe(MQTT_TOPIC)
    _mqtt_client.loop_forever()


def stop_mqtt():
    global _mqtt_client
    if _mqtt_client is not None:
        try:
            _mqtt_client.loop_stop()
            _mqtt_client.disconnect()
        except Exception:
            pass
        _mqtt_client = None


# -------------------------------------------------------
# UTILIDADES
# -------------------------------------------------------
def interruptible_sleep(seconds, interval=0.05):
    elapsed = 0.0
    while elapsed < seconds:
        if stop_event.is_set():
            return False
        time.sleep(interval)
        elapsed += interval
    return True


def _send(scf, vx, vy, vz, yaw_rate_degs=0.0):
    scf.cf.commander.send_velocity_world_setpoint(
        float(vx), float(vy), float(vz), float(yaw_rate_degs)
    )


# -------------------------------------------------------
# DESPEGUE MANUAL EN LOW-LEVEL
# -------------------------------------------------------
def lowlevel_takeoff(scf, target_z, vz=LOWLEVEL_VERTICAL_SPEED,
                     tolerance=0.05, timeout=10.0):
    print(f"Despegando (low-level) hasta z={target_z:.2f}m...")
    t0 = time.time()

    while not stop_event.is_set():
        if mocap_pose['z'] >= target_z - tolerance:
            break
        if time.time() - t0 > timeout:
            print("⚠ Timeout en despegue low-level — continuando.")
            break
        # Sin dato fresco del MoCap no se sigue empujando
        _send(scf, 0.0, 0.0, vz if mocap_is_fresh() else 0.0)
        time.sleep(0.02)

    _send(scf, 0.0, 0.0, 0.0)
    time.sleep(0.1)
    print(f"  Altura alcanzada: z={mocap_pose['z']:.3f}m")


# -------------------------------------------------------
# IR AL PUNTO INICIAL EN LOW-LEVEL
# -------------------------------------------------------
def lowlevel_goto(scf, target_x, target_y, target_z,
                  max_speed=GOTO_START_MAX_SPEED,
                  kp=GOTO_START_KP,
                  tolerance=GOTO_START_POS_TOLERANCE,
                  timeout=GOTO_START_TIMEOUT):
    print(f"Moviéndose (low-level) al inicio: "
          f"({target_x:.2f}, {target_y:.2f}, {target_z:.2f})...")
    t0   = time.time()
    kp_z = 0.8

    while not stop_event.is_set():
        ex = target_x - mocap_pose['x']
        ey = target_y - mocap_pose['y']
        ez = target_z - mocap_pose['z']

        if np.hypot(ex, ey) < tolerance and abs(ez) < 0.06:
            break
        if time.time() - t0 > timeout:
            print("⚠ Timeout yendo al punto inicial — continuando.")
            break

        vx, vy = kp * ex, kp * ey
        horiz  = np.hypot(vx, vy)
        if horiz > max_speed:
            vx *= max_speed / horiz
            vy *= max_speed / horiz

        if not mocap_is_fresh():
            vx = vy = 0.0

        _send(scf, vx, vy, np.clip(kp_z * ez, -0.15, 0.15) if mocap_is_fresh() else 0.0)
        time.sleep(0.02)

    _send(scf, 0.0, 0.0, 0.0)
    time.sleep(0.1)
    print(f"  Posición alcanzada: ({mocap_pose['x']:.3f}, "
          f"{mocap_pose['y']:.3f}, {mocap_pose['z']:.3f})")


# -------------------------------------------------------
# MANTENER POSICIÓN (hover activo sobre un punto fijo)
# -------------------------------------------------------
def hold_position(scf, target, duration, kp=1.5, kp_z=2.0, v_max=0.35):
    """Hover con realimentación de posición sobre un punto fijo."""
    t0 = time.time()
    while time.time() - t0 < duration and not stop_event.is_set():
        vx = kp   * (target[0] - mocap_pose['x'])
        vy = kp   * (target[1] - mocap_pose['y'])
        vz = kp_z * (target[2] - mocap_pose['z'])

        horiz = np.hypot(vx, vy)
        if horiz > v_max:
            vx *= v_max / horiz
            vy *= v_max / horiz

        _send(scf, vx, vy, np.clip(vz, -0.3, 0.3))
        time.sleep(CONTROL_RATE)


# -------------------------------------------------------
# DESCENSO CONTROLADO MANTENIENDO XY
# -------------------------------------------------------
def controlled_descent(scf, target_xy, rate=DESCENT_RATE,
                       z_stop=0.10, timeout=DESCENT_TIMEOUT, kp=1.5):
    print("Aterrizando suavemente...")
    t0 = time.time()

    while mocap_pose['z'] > z_stop and not stop_event.is_set():
        if time.time() - t0 > timeout:
            print("⚠ Timeout en el descenso.")
            break

        vx = kp * (target_xy[0] - mocap_pose['x'])
        vy = kp * (target_xy[1] - mocap_pose['y'])
        horiz = np.hypot(vx, vy)
        if horiz > 0.3:
            vx *= 0.3 / horiz
            vy *= 0.3 / horiz

        _send(scf, vx, vy, -rate)
        time.sleep(CONTROL_RATE)


# -------------------------------------------------------
# CONSTRUCCIÓN DE LA TRAYECTORIA
# -------------------------------------------------------
def build_path(waypoints):
    """Genera la trayectoria con paso de arco constante (PATH_RESOLUTION)."""
    traj = TrajectoryGenerator(waypoints, fixed_height=FIXED_HEIGHT)
    path = traj.generate_uniform(spacing=PATH_RESOLUTION,
                                 method=METHOD, tension=TENSION)
    return traj, path


# -------------------------------------------------------
# VUELO
# -------------------------------------------------------
def fly_trajectory(scf):
    global theo_trajectory, ctrl_log

    flight_info['estado'] = 'abortado antes del seguimiento'

    # -- Cargar waypoints --
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "waypoints.json")
    try:
        with open(file_path) as f:
            waypoints = json.load(f)
    except FileNotFoundError:
        print("ERROR: No se encontró waypoints.json")
        print("Primero crea los waypoints con create_waypoints.py")
        return

    print(f"\nWaypoints: {len(waypoints)}")
    for i, wp in enumerate(waypoints):
        print(f"  WP{i}: x={wp[0]:.2f}, y={wp[1]:.2f}, z={wp[2]:.2f}")

    # -- Trayectoria con paso de arco constante --
    traj, path = build_path(waypoints)
    theo_trajectory = path.tolist()

    # -- Pure Pursuit --
    pp = PurePursuit(
        lookahead_distance=LOOKAHEAD_TIME * V_MAX,
        min_lookahead=LOOKAHEAD_MIN,
        max_lookahead=LOOKAHEAD_MAX,
        lookahead_time=LOOKAHEAD_TIME,
        k_curve=LOOKAHEAD_K_CURVE,
        goal_tolerance=GOAL_TOLERANCE,
    )
    pp.set_path(path)

    # -- Perfil de velocidad --
    v_prof = compute_speed_profile(
        pp.s, pp.kappa, pp.T,
        v_max=V_MAX, v_min=V_MIN,
        a_lat=A_LAT_MAX, a_tan=A_TAN_MAX,
        vz_max=VZ_MAX_PATH,
        v_start=V_START, v_end=V_END,
    )

    # Tiempo estimado: integral de ds/v
    ds_arr    = np.diff(pp.s)
    t_est     = float(np.sum(ds_arr / np.maximum(v_prof[:-1], 1e-6)))
    r_min     = 1.0 / max(pp.kappa.max(), 1e-6)

    print(f"\nTrayectoria: {len(path)} puntos | longitud {pp.length:.2f} m | "
          f"radio mínimo {r_min:.2f} m")
    print(f"Perfil de velocidad: {v_prof.min():.2f} – {v_prof.max():.2f} m/s "
          f"(media {v_prof.mean():.2f}) | tiempo estimado {t_est:.1f} s")
    print(f"Pure Pursuit: L = {LOOKAHEAD_TIME:.2f}s * v "
          f"∈ [{LOOKAHEAD_MIN:.2f}, {LOOKAHEAD_MAX:.2f}] m | "
          f"KP_LAT={KP_LAT} | KP_Z={KP_Z} | {CONTROL_HZ:.0f} Hz\n")

    flight_info.update(longitud=pp.length, r_min=r_min, t_est=t_est,
                       v_perfil_max=float(v_prof.max()))

    # -- Esperar MoCap --
    print("Esperando posición inicial del MoCap...")
    t0 = time.time()
    while (mocap_pose['x'] == 0.0 and mocap_pose['y'] == 0.0
           and mocap_pose['z'] == 0.0):
        if stop_event.is_set():
            return
        if time.time() - t0 > 10.0:
            print("⚠ Timeout esperando MoCap")
            return
        time.sleep(0.1)

    print(f"Posición inicial: x={mocap_pose['x']:.3f}, "
          f"y={mocap_pose['y']:.3f}, z={mocap_pose['z']:.3f}")

    # -- Despegue --
    lowlevel_takeoff(scf, target_z=float(path[0][2]))
    if stop_event.is_set() or not interruptible_sleep(0.5):
        return

    # -- Ir al inicio de la trayectoria --
    lowlevel_goto(scf, float(path[0][0]), float(path[0][1]), float(path[0][2]))
    if stop_event.is_set():
        return

    # -- Estabilizar sobre el punto inicial antes de arrancar --
    # Se guardan los índices del hover para medir la distancia de paso del
    # primer y último waypoint
    i_hover = len(real_trajectory)
    hold_position(scf, path[0], SETTLE_TIME)
    flight_info['idx_hover_ini'] = (i_hover, len(real_trajectory))
    if stop_event.is_set():
        return

    pp.reset()
    pp.step(mocap_pose['x'], mocap_pose['y'])

    # -- Diagnóstico --
    pose_actual  = get_pose(scf)
    altura_ekf   = pose_actual[2] if pose_actual else 0.0
    yaw_ekf0     = pose_actual[5] if pose_actual else 0.0
    vbat0        = float(scf.pose_data.get('vbat', 0.0))
    vbat_min     = vbat0
    flight_info['vbat0'] = vbat0
    print(f"  Altura EKF={altura_ekf:.3f}m | MoCap={mocap_pose['z']:.3f}m | "
          f"MQTT={_mqtt_msg_count} msgs | extpos={_extpos_count} | "
          f"Batería={vbat0:.2f}V")
    if max(altura_ekf, mocap_pose['z']) < 0.15:
        print("⚠ ADVERTENCIA: altura baja — verificar que el dron despegó.")
    if vbat0 < 3.7:
        print(f"⚠ Batería ya arranca baja ({vbat0:.2f}V) — considera cambiarla "
              f"antes de este vuelo, va a sesgar la comparación con otros.")

    # ----------------------------------------------------------
    # LAZO DE CONTROL
    # ----------------------------------------------------------
    print(f"\nIniciando seguimiento...")

    speed_cmd    = float(v_prof[0])
    vz_cmd       = 0.0
    yaw_rate_cmd = 0.0
    yaw_ref_filt = mocap_pose['yaw']
    yaw_offset   = np.radians(YAW_OFFSET_DEG)

    ctrl_log   = []
    t_inicio   = time.perf_counter()
    last_print = t_inicio
    t_next     = time.perf_counter()
    timeout    = FLIGHT_TIMEOUT_FACTOR * t_est + 5.0

    while not stop_event.is_set():

        t_now = time.perf_counter()
        if t_now - t_inicio > timeout:
            print("⚠ Timeout de vuelo — abortando seguimiento.")
            flight_info['estado'] = 'timeout de vuelo'
            break

        x, y, z = mocap_pose['x'], mocap_pose['y'], mocap_pose['z']
        vbat_min = min(vbat_min, float(scf.pose_data.get('vbat', vbat_min)))

        # 1. Proyección sobre la trayectoria (avance acotado por ciclo)
        pp.step(x, y, max_advance=0.06 + 2.5 * speed_cmd * CONTROL_RATE)
        if pp.finished:
            print(f"  [PurePursuit -> hold_position] velocidad comandada = "
                  f"{speed_cmd:.2f} m/s al llegar a la meta "
                  f"(objetivo V_END={V_END:.2f} m/s) -- residual que "
                  f"hold_position tiene que absorber sin anticiparla.")
            break

        # 2. Rapidez objetivo del perfil, en función del arco recorrido
        v_target  = float(np.interp(pp.s_proj, pp.s, v_prof))
        speed_cmd = ALPHA_V * v_target + (1.0 - ALPHA_V) * speed_cmd

        # 3. Punto de anticipación
        kappa = pp.curvature_ahead(0.0)
        pp.adaptive_lookahead(speed_cmd, kappa)
        _, T_look, _ = pp.lookahead_state()

        # 4a. Horizontal: feedforward tangencial + error lateral + capa límite
        e_lat  = pp.e_lat
        e_norm = np.linalg.norm(e_lat)
        corr   = K_LAT_SAT * e_lat / max(e_norm, LAT_BOUNDARY)

        vx = speed_cmd * T_look[0] + KP_LAT * e_lat[0] + corr[0]
        vy = speed_cmd * T_look[1] + KP_LAT * e_lat[1] + corr[1]

        v_norm = np.hypot(vx, vy)
        if v_norm > V_XY_MAX_CMD:
            vx *= V_XY_MAX_CMD / v_norm
            vy *= V_XY_MAX_CMD / v_norm

        # 4b. Vertical: feedforward de la pendiente + error de altura
        z_ref  = float(pp.proj[2])
        vz_raw = speed_cmd * T_look[2] + KP_Z * (z_ref - z)
        vz_cmd = ALPHA_VZ * vz_raw + (1.0 - ALPHA_VZ) * vz_cmd
        vz     = float(np.clip(vz_cmd, -VZ_MAX_CMD, VZ_MAX_CMD))

        # 5. Yaw
        if YAW_MODE == 'path':
            yaw_ref_raw = np.arctan2(T_look[1], T_look[0]) + yaw_offset
            diff = (yaw_ref_raw - yaw_ref_filt + np.pi) % (2 * np.pi) - np.pi
            yaw_ref_filt += YAW_ALPHA_REF * diff

            yaw_err = (yaw_ref_filt - mocap_pose['yaw'] + np.pi) % (2 * np.pi) - np.pi
            if abs(yaw_err) < np.radians(YAW_DEADBAND):
                yaw_err = 0.0

            yaw_rate_raw = np.clip(KP_YAW * yaw_err, -YAW_RATE_MAX, YAW_RATE_MAX)
            yaw_rate_cmd = (YAW_ALPHA_OUT * yaw_rate_raw
                            + (1.0 - YAW_ALPHA_OUT) * yaw_rate_cmd)
        else:
            yaw_rate_cmd = 0.0

        _send(scf, vx, vy, vz, np.degrees(yaw_rate_cmd))

        # 6. Registro (sin print por ciclo, para no degradar la frecuencia del lazo)
        ctrl_log.append((t_now - t_inicio, x, y, z,
                         pp.s_proj, pp.e_lat_signed, z_ref, z_ref - z,
                         v_target, speed_cmd, pp.L, kappa))

        if t_now - last_print > 1.5:
            print(f"  [{pp.get_progress():5.1f}%] "
                  f"v={speed_cmd:.2f} L={pp.L:.2f} k={kappa:.2f} | "
                  f"e_lat={pp.e_lat_signed*100:+6.1f} cm  "
                  f"e_z={(z_ref - z)*100:+6.1f} cm")
            last_print = t_now

        # 7. Cadencia fija
        t_next += CONTROL_RATE
        sleep = t_next - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            t_next = time.perf_counter()

    t_vuelo = time.perf_counter() - t_inicio
    hz_real = len(ctrl_log) / t_vuelo if t_vuelo > 0 else 0.0
    flight_info.update(t_vuelo=t_vuelo, hz_real=hz_real,
                       progreso=pp.get_progress(), vbat_min=vbat_min)
    if stop_event.is_set():
        flight_info['estado'] = 'paro de emergencia durante el seguimiento'
    elif flight_info['estado'] != 'timeout de vuelo':
        flight_info['estado'] = 'completado'
    print(f"\n✓ Seguimiento completado en {t_vuelo:.1f}s "
          f"({hz_real:.0f} Hz reales, {pp.get_progress():.0f}% de la ruta)")

    if stop_event.is_set():
        return

    # -- Hover sobre el punto final --
    print("Hover final (2s)...")
    i_hover = len(real_trajectory)
    hold_position(scf, path[-1], 2.0)
    flight_info['idx_hover_fin'] = (i_hover, len(real_trajectory))
    if stop_event.is_set():
        return

    # -- Descenso manteniendo XY --
    controlled_descent(scf, (float(path[-1][0]), float(path[-1][1])))
    time.sleep(0.3)

    # -- Diagnóstico de batería --
    vbat_fin = float(scf.pose_data.get('vbat', vbat_min))
    flight_info.update(vbat_fin=vbat_fin, vbat_min=vbat_min)
    print(f"  Batería: {vbat0:.2f}V → {vbat_fin:.2f}V "
          f"(mínimo durante el vuelo: {vbat_min:.2f}V)")
    if vbat_min < 3.5:
        print(f"  ⚠ La batería bajó de 3.5V durante el vuelo — a este nivel el "
              f"empuje disponible cae y el dron pierde autoridad de control. "
              f"Compara este vuelo con cuidado, o mejor descártalo y repite "
              f"con batería nueva.")

    # -- Diagnóstico de deriva de yaw del EKF --
    pose_fin = get_pose(scf)
    if pose_fin is not None:
        yaw_ekf_fin  = pose_fin[5]
        deriva_ekf   = ((yaw_ekf_fin - yaw_ekf0 + 180) % 360) - 180
        yaw_mocap_dg = np.degrees(mocap_pose['yaw'])
        flight_info['deriva_yaw_ekf'] = deriva_ekf
        print(f"  Yaw EKF: {yaw_ekf0:.1f}° → {yaw_ekf_fin:.1f}° "
              f"(cambio {deriva_ekf:+.1f}°) | Yaw MoCap: {yaw_mocap_dg:.1f}°")
        if YAW_MODE == 'hold' and abs(deriva_ekf) > 8.0:
            print("  ⚠ El yaw del EKF derivó con el dron sin girar. Los comandos "
                  "de velocidad de mundo se están aplicando rotados.")
            print("    Considera activar USE_EXTPOSE = True.")


# -------------------------------------------------------
# GRÁFICA Y MÉTRICAS
# -------------------------------------------------------
def _nombre_metodo():
    return {'catmull_rom': 'Catmull-Rom', 'linear': 'Lineal'}.get(METHOD, METHOD)


def _distancia_paso(puntos, wp):
    """Distancia mínima 3D a un waypoint y sus componentes horizontal y vertical."""
    d = np.linalg.norm(puntos - wp[:3], axis=1)
    k = int(np.argmin(d))
    return (float(d[k]),
            float(np.hypot(*(puntos[k, :2] - wp[:2]))),
            float(abs(puntos[k, 2] - wp[2])))


def _lineas_metricas(real, t_mocap, log, wp_arr, stamp):
    L    = []
    info = flight_info

    L.append("=" * 60)
    L.append(f"  MÉTRICAS — PURE PURSUIT + {_nombre_metodo().upper()}")
    L.append(f"  Modo: {MODO_ETIQUETA} | {stamp}")
    L.append("=" * 60)

    # -- Resultado del vuelo --
    estado = info.get('estado', 'desconocido')
    if stop_event.is_set() and estado == 'completado':
        estado += ' (paro de emergencia después del seguimiento)'
    L.append("  -- Resultado del vuelo --")
    L.append(f"    Estado:              {estado}")
    if 'longitud' in info:
        L.append(f"    Trayectoria:         {info['longitud']:.2f} m | radio mínimo "
                 f"{info['r_min']:.2f} m")
        L.append(f"    Perfil:              máx {info['v_perfil_max']:.2f} m/s | "
                 f"tiempo estimado {info['t_est']:.1f} s")
    if 't_vuelo' in info:
        L.append(f"    Seguimiento:         {info['t_vuelo']:.1f} s | "
                 f"{info['progreso']:.0f}% de la ruta | "
                 f"{info['hz_real']:.0f} Hz reales")

    # -- Waypoints (para verificar que todos los vuelos usan el mismo circuito) --
    if wp_arr is not None:
        L.append("  -- Waypoints (waypoints.json) --")
        for i, wp in enumerate(wp_arr):
            L.append(f"    WP{i}: x={wp[0]:+.3f}  y={wp[1]:+.3f}  z={wp[2]:.3f}")

    # -- Distancia de paso por cada waypoint --
    # Intermedios: lazo de seguimiento. Primero y último: incluyen el hover.
    if wp_arr is not None and log is not None and len(log) > 1:
        seg   = log[:, 1:4]
        n_wp  = len(wp_arr)
        L.append("  -- Distancia de paso por cada waypoint [cm] --")
        L.append("             3D   horiz.  vert.")
        for i, wp in enumerate(wp_arr):
            puntos = seg
            nota   = ''
            clave  = ('idx_hover_ini' if i == 0 else
                      'idx_hover_fin' if i == n_wp - 1 else None)
            if clave is not None and clave in info:
                a, b = info[clave]
                b = min(b, len(real))
                if b > a:
                    puntos = np.vstack([seg, real[a:b]])
                    nota   = '  (incluye hover)'
            d3, dh, dv = _distancia_paso(puntos, wp)
            L.append(f"    WP{i}:  {d3*100:6.1f} {dh*100:7.1f} {dv*100:6.1f}{nota}")

    # -- Errores de seguimiento --
    if log is not None and len(log) > 1:
        e_lat = np.abs(log[:, 5])
        e_z   = np.abs(log[:, 7])
        t_log = log[:, 0]
        xyz   = log[:, 1:4]

        L.append("  -- Error lateral (perpendicular a la trayectoria) [cm] --")
        L.append(f"    Promedio:            {e_lat.mean()*100:.2f}")
        L.append(f"    Máximo:              {e_lat.max()*100:.2f}")
        L.append(f"    Desv. estándar:      {e_lat.std()*100:.2f}")
        L.append(f"    Percentil 95:        {np.percentile(e_lat, 95)*100:.2f}")
        L.append("  -- Error de altura [cm] --")
        L.append(f"    Promedio:            {e_z.mean()*100:.2f}")
        L.append(f"    Máximo:              {e_z.max()*100:.2f}")
        L.append(f"    Percentil 95:        {np.percentile(e_z, 95)*100:.2f}")

        # Rapidez media = arco avanzado / tiempo; la real se mide en ventanas de ~0.1 s
        dur    = max(t_log[-1] - t_log[0], 1e-6)
        v_arco = (log[-1, 4] - log[0, 4]) / dur
        w      = max(1, int(round(0.1 * CONTROL_HZ)))
        L.append("  -- Rapidez [m/s] --")
        L.append(f"    Media sobre la ruta: {v_arco:.3f}")
        if len(log) > w:
            v_real = (np.linalg.norm(xyz[w:] - xyz[:-w], axis=1)
                      / np.maximum(t_log[w:] - t_log[:-w], 1e-6))
            L.append(f"    Real máx. (MoCap):   {v_real.max():.3f}")
    else:
        L.append("  (sin registro del lazo de control)")

    # -- Batería --
    L.append("  -- Batería [V] --")
    L.append(f"    Reposo:              {info.get('vbat_reposo', float('nan')):.2f}")
    L.append(f"    Inicio seguimiento:  {info.get('vbat0', float('nan')):.2f}")
    L.append(f"    Mínimo en vuelo:     {info.get('vbat_min', float('nan')):.2f}")
    L.append(f"    Final:               {info.get('vbat_fin', float('nan')):.2f}")

    # -- MoCap --
    L.append("  -- MoCap --")
    L.append(f"    Puntos MoCap:        {len(real)}")
    gap_max = 0.0
    if len(t_mocap) > 2:
        gaps     = np.diff(t_mocap)
        gap_max  = float(gaps.max())
        n_huecos = int(np.sum(gaps > 0.10))
        L.append(f"    Hueco máximo:        {gap_max*1000:.0f} ms "
                 f"({n_huecos} huecos > 100 ms)")
    if 'deriva_yaw_ekf' in info:
        L.append(f"    Deriva yaw EKF:      {info['deriva_yaw_ekf']:+.1f}°")

    # -- Configuración --
    L.append("  -- Configuración --")
    L.append(f"    Método:              {METHOD} (tensión {TENSION})")
    L.append(f"    V_MAX / V_MIN:       {V_MAX} / {V_MIN} m/s")
    L.append(f"    A_LAT / A_TAN:       {A_LAT_MAX} / {A_TAN_MAX} m/s²")
    L.append(f"    VZ_MAX_PATH:         {VZ_MAX_PATH} m/s")
    L.append(f"    V_START / V_END:     {V_START} / {V_END} m/s")
    L.append(f"    Lookahead:           {LOOKAHEAD_TIME}s * v "
             f"∈ [{LOOKAHEAD_MIN}, {LOOKAHEAD_MAX}] m")
    L.append(f"    GOAL_TOLERANCE:      {GOAL_TOLERANCE} m")
    L.append(f"    KP_LAT / KP_Z:       {KP_LAT} / {KP_Z}")
    L.append(f"    K_LAT_SAT / capa:    {K_LAT_SAT} m/s / {LAT_BOUNDARY} m")
    L.append(f"    V_XY / VZ máx cmd:   {V_XY_MAX_CMD} / {VZ_MAX_CMD} m/s")
    L.append(f"    ALPHA_V / ALPHA_VZ:  {ALPHA_V} / {ALPHA_VZ}")
    L.append(f"    Control:             {CONTROL_HZ:.0f} Hz | extpos "
             f"{EXTPOS_RATE_HZ:.0f} Hz")
    L.append(f"    Yaw:                 {YAW_MODE}")
    L.append(f"    Altura fija:         {FIXED_HEIGHT}")

    # -- Advertencias (no descartan el vuelo) --
    avisos = []
    if estado != 'completado':
        avisos.append(f"el vuelo no terminó normalmente ({estado})")
    if info.get('vbat_min', 99.0) < VBAT_MIN_AVISO:
        avisos.append(f"batería bajo {VBAT_MIN_AVISO:.2f} V "
                      f"(mínimo {info['vbat_min']:.2f} V)")
    if gap_max > HUECO_MAX_AVISO:
        avisos.append(f"hueco de MoCap de {gap_max*1000:.0f} ms")
    if YAW_MODE == 'hold' and abs(info.get('deriva_yaw_ekf', 0.0)) > 8.0:
        avisos.append(f"deriva de yaw del EKF de "
                      f"{info['deriva_yaw_ekf']:+.1f}°")
    L.append("  -- Advertencias --")
    if avisos:
        for a in avisos:
            L.append(f"    ⚠ {a}")
    else:
        L.append("    ninguna")
    L.append("=" * 60)
    return L


def plot_trajectories():
    # Copia de los datos: el hilo MQTT sigue agregando posiciones.
    n_m     = min(len(real_trajectory), len(timestamps))
    real    = np.array(real_trajectory[:n_m])
    t_mocap = np.array(timestamps[:n_m])
    theo    = np.array(theo_trajectory) if theo_trajectory else None
    log     = np.array(ctrl_log) if ctrl_log else None

    if len(real) == 0:
        print("Sin datos para graficar.")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(base_dir, "waypoints.json")) as f:
            wp_arr = np.array(json.load(f))
    except FileNotFoundError:
        wp_arr = None

    stamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    carpeta = os.path.join(base_dir, CARPETA_VUELOS)
    os.makedirs(carpeta, exist_ok=True)
    prefijo = os.path.join(carpeta, f"{MODO_ETIQUETA}_{stamp}")

    # -- Métricas: consola + .txt (se guardan antes de abrir las gráficas) --
    lineas = _lineas_metricas(real, t_mocap, log, wp_arr, stamp)
    with open(prefijo + "_metricas.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    print("\n" + "\n".join(lineas))

    # -- Datos crudos --
    if log is not None and len(log) > 0:
        np.savetxt(prefijo + "_control.csv", log, delimiter=",", fmt="%.5f",
                   header="t,x,y,z,s_proj,e_lat,z_ref,e_z,v_perfil,v_cmd,"
                          "lookahead,kappa", comments="")
    t0 = t_mocap[0]
    np.savetxt(prefijo + "_mocap.csv",
               np.column_stack([t_mocap - t0, real]),
               delimiter=",", fmt="%.5f", header="t,x,y,z", comments="")

    nombre = _nombre_metodo()

    # -- Ventana 1: Trayectorias --
    fig1 = plt.figure(figsize=(14, 6), num="Pure Pursuit — Trayectorias")
    fig1.suptitle(f"Resultado — Pure Pursuit + {nombre}  ({MODO_ETIQUETA})",
                  fontsize=13, fontweight='bold')

    ax1 = fig1.add_subplot(121, projection='3d')
    ax2 = fig1.add_subplot(122)

    # Trayectoria seguida durante el lazo de control (no todo el vuelo)
    flown = log[:, 1:4] if log is not None and len(log) > 1 else real

    for ax, is_3d in [(ax1, True), (ax2, False)]:
        if theo is not None:
            if is_3d:
                ax.plot(theo[:, 0], theo[:, 1], theo[:, 2],
                        'r--', lw=2, label=f'Referencia ({nombre})')
            else:
                ax.plot(theo[:, 0], theo[:, 1],
                        'r--', lw=2, label=f'Referencia ({nombre})')

        if wp_arr is not None:
            if is_3d:
                ax.plot(wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2],
                        'o', color='orange', markersize=8,
                        markeredgecolor='black', label='Waypoints')
            else:
                ax.plot(wp_arr[:, 0], wp_arr[:, 1],
                        'o', color='orange', markersize=8,
                        markeredgecolor='black', label='Waypoints')

        if is_3d:
            ax.plot(real[:, 0], real[:, 1], real[:, 2],
                    color='0.7', lw=1.0, label='Vuelo completo')
            ax.plot(flown[:, 0], flown[:, 1], flown[:, 2],
                    'b', lw=1.8, label='Seguimiento (MoCap)')
            ax.set_zlabel('Z [m]')
            ax.set_title('Vista 3D')
            ax.view_init(elev=30, azim=135)
        else:
            ax.plot(real[:, 0], real[:, 1], color='0.7', lw=1.0,
                    label='Vuelo completo')
            ax.plot(flown[:, 0], flown[:, 1], 'b', lw=1.8,
                    label='Seguimiento (MoCap)')
            ax.scatter(*flown[0, :2],  color='green', s=80, zorder=5, label='Inicio')
            ax.scatter(*flown[-1, :2], color='red',   s=80, zorder=5, label='Fin')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.4)
            ax.set_title('Vista superior (XY)')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig1.savefig(prefijo + "_trayectoria.pdf", bbox_inches='tight')

    # -- Ventana 2: Métricas (contra el arco recorrido, no el índice) --
    if log is not None and len(log) > 1:
        s_log   = log[:, 4]
        e_lat   = log[:, 5]
        z_real  = log[:, 3]
        z_ref   = log[:, 6]
        e_z     = log[:, 7]
        v_tgt   = log[:, 8]
        v_cmd   = log[:, 9]
        L_log   = log[:, 10]
        k_log   = log[:, 11]

        fig2 = plt.figure(figsize=(15, 8), num="Pure Pursuit — Métricas")
        fig2.suptitle(f"Métricas de seguimiento  ({MODO_ETIQUETA})",
                      fontsize=13, fontweight='bold')

        ax3 = fig2.add_subplot(221)
        ax3.plot(s_log, e_lat * 100, color='purple', lw=1.4)
        ax3.axhline(0, color='k', lw=0.8)
        ax3.fill_between(s_log, e_lat * 100, alpha=0.2, color='purple')
        ax3.set_xlabel('Arco recorrido [m]')
        ax3.set_ylabel('Error lateral [cm]')
        ax3.set_title(f'Cross-track error  '
                      f'(|e| medio {np.abs(e_lat).mean()*100:.1f} cm)')
        ax3.grid(True, alpha=0.4)

        ax4 = fig2.add_subplot(222)
        ax4.plot(s_log, z_ref, 'r--', lw=1.5, label='Z deseada')
        ax4.plot(s_log, z_real, 'b', lw=1.5, label='Z real')
        ax4.set_xlabel('Arco recorrido [m]')
        ax4.set_ylabel('Z [m]')
        ax4.set_title(f'Altura  (|e_z| medio {np.abs(e_z).mean()*100:.1f} cm)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.4)

        ax5 = fig2.add_subplot(223)
        ax5.plot(s_log, v_tgt, 'g--', lw=1.5, label='Perfil')
        ax5.plot(s_log, v_cmd, 'g',   lw=1.5, label='Comandada')
        ax5.set_xlabel('Arco recorrido [m]')
        ax5.set_ylabel('Rapidez [m/s]')
        ax5.set_title('Perfil de velocidad')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.4)

        ax6 = fig2.add_subplot(224)
        ax6.plot(s_log, k_log, color='darkorange', lw=1.4, label='Curvatura [1/m]')
        ax6.set_xlabel('Arco recorrido [m]')
        ax6.set_ylabel('Curvatura [1/m]', color='darkorange')
        ax6.grid(True, alpha=0.4)
        ax6b = ax6.twinx()
        ax6b.plot(s_log, L_log, color='teal', lw=1.4, label='Lookahead [m]')
        ax6b.set_ylabel('Lookahead [m]', color='teal')
        ax6.set_title('Curvatura y lookahead')

        plt.tight_layout()
        fig2.savefig(prefijo + "_metricas.pdf", bbox_inches='tight')

    print(f"\n✓ Vuelo guardado en {CARPETA_VUELOS}/ como "
          f"{MODO_ETIQUETA}_{stamp}_*")

    plt.show()
    plt.close('all')


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    global scf_global, _threads_started

    reset_state()
    cflib.crtp.init_drivers()

    if not _threads_started:
        threading.Thread(target=start_mqtt, daemon=True).start()
        _threads_started = True

    threading.Thread(target=emergency_listener, daemon=True).start()

    print("Conectando al drone...")
    scf_global = connect(URI)
    if scf_global is None:
        print("ERROR: No se pudo conectar.")
        return

    vbat_reposo = float(scf_global.pose_data.get('vbat', 0.0))
    flight_info['vbat_reposo'] = vbat_reposo
    print(f"Batería en reposo (motores apagados): {vbat_reposo:.2f}V")
    if vbat_reposo < 3.9:
        print("⚠ Ya en reposo está baja — cámbiala antes de seguir.")

    print("Configurando estimador para MoCap sin Flowdeck...")
    scf_global.cf.param.set_value('commander.enHighLevel', '0')
    scf_global.cf.param.set_value('stabilizer.controller', '1')
    scf_global.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf_global.cf.param.set_value('kalman.resetEstimation', '0')

    print("Esperando estabilización del EKF (5s)...")
    time.sleep(5.0)

    print("Verificando EKF...")
    t_ekf  = time.time()
    ekf_ok = False
    while time.time() - t_ekf < 10.0:
        pose = get_pose(scf_global)
        if pose is not None and abs(pose[2]) < 2.0:
            print(f"  ✓ EKF listo: z={pose[2]:.3f}m")
            ekf_ok = True
            break
        time.sleep(0.2)

    if not ekf_ok:
        print("  ⚠ EKF no convergió — abortando")
        disconnect(scf_global)
        scf_global = None
        return

    try:
        fly_trajectory(scf_global)
    except Exception as e:
        import traceback
        print(f"Error durante el vuelo: {e}")
        flight_info['estado'] = f'error: {e}'
        traceback.print_exc()
        _cortar_motores()
    finally:
        try:
            scf_global.cf.commander.send_stop_setpoint()
            time.sleep(0.1)
            scf_global.cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.2)
            scf_global.cf.param.set_value('kalman.resetEstimation', '0')
            time.sleep(0.1)
            print("✓ EKF reseteado para próximo vuelo")
        except Exception:
            pass

        plot_trajectories()


def reset_state():
    global scf_global, last_ts, _mqtt_msg_count, _extpos_count, _last_extpos_t

    stop_event.clear()
    real_trajectory.clear()
    theo_trajectory.clear()
    timestamps.clear()
    ctrl_log.clear()
    flight_info.clear()

    last_ts         = None
    _mqtt_msg_count = 0
    _extpos_count   = 0
    _last_extpos_t  = 0.0

    mocap_pose['x'] = 0.0
    mocap_pose['y'] = 0.0
    mocap_pose['z'] = 0.0
    mocap_pose['yaw'] = 0.0

    scf_global = None


if __name__ == '__main__':
    while True:
        main()
        print("\n" + "=" * 50)
        respuesta = input("¿Volver a volar? (s/n): ").strip().lower()
        if respuesta != 's':
            print("Saliendo.")
            break
