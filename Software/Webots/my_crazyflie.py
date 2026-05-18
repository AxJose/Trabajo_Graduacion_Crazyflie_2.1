from controller import Robot
from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt
import json

from pid_controller import pid_velocity_fixed_height_controller
from trajectory import TrajectoryGenerator

real_trajectory = []

# --------------------------------------------------
# INIT
# --------------------------------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

m1 = robot.getDevice("m1_motor")
m2 = robot.getDevice("m2_motor")
m3 = robot.getDevice("m3_motor")
m4 = robot.getDevice("m4_motor")

for m, sign in zip([m1, m2, m3, m4], [-1, 1, -1, 1]):
    m.setPosition(float('inf'))
    m.setVelocity(sign)

imu  = robot.getDevice("inertial_unit"); imu.enable(timestep)
gps  = robot.getDevice("gps");           gps.enable(timestep)
gyro = robot.getDevice("gyro");          gyro.enable(timestep)

PID = pid_velocity_fixed_height_controller()

# --------------------------------------------------
# PARÁMETROS
# --------------------------------------------------

# Velocidad máxima en tramos rectos [m/s]
#   0.3 = conservador y preciso     
#   0.6 = moderado para carreras
#   1.0 = agresivo (ajustar radios)
#   1.5 = máximo del Crazyflie real

MAX_VELOCITY = 0.5  

# Velocidad mínima en curvas cerradas [m/s]
MIN_VELOCITY = 0.1

# Qué tanto frena en curvas (0.0 = no frena, 1.0 = frena al máximo)
CURVE_SLOWDOWN = 0.4 

# Ganancia proporcional posición → velocidad deseada
K_POS = 2.0

# Puntos de la spline
SPLINE_POINTS = 50


ACCEPT_RADIUS_BASE = 0.20

# Timeout por waypoint: se calcula como distancia/velocidad * TIMEOUT_FACTOR
# Así a mayor velocidad el timeout es proporcionalmente menor
TIMEOUT_FACTOR = 3.0   # margen sobre el tiempo teórico mínimo
TIMEOUT_MIN    = 1.0   # [s] nunca menos de esto
TIMEOUT_MAX    = 8.0   # [s] nunca más de esto

# Rampa de altura — SOLO para el despegue inicial
HEIGHT_RAMP_RATE = 0.2   # [m/s]
takeoff_done     = False

# --------------------------------------------------
# TRAYECTORIA
# --------------------------------------------------
with open("waypoints.json") as f:
    waypoints = json.load(f)

print(f"Waypoints cargados: {len(waypoints)}")
for i, wp in enumerate(waypoints):
    print(f"  WP{i}: x={wp[0]:.2f}, y={wp[1]:.2f}, z={wp[2]:.2f}")

traj = TrajectoryGenerator(waypoints)
path = traj.generate_spline(SPLINE_POINTS, method='catmull_rom', tension=0.5)

print(f"Spline: {len(path)} puntos, longitud ≈ {traj.total_length():.2f} m")

# --------------------------------------------------
# VELOCIDAD ADAPTATIVA — precalculada para toda la spline
#
# Idea: en cada punto de la spline, calcular la curvatura local.
# A mayor curvatura (curva cerrada) → menor velocidad.
# A menor curvatura (tramo recto)   → mayor velocidad.
#
# Curvatura κ = |v × a| / |v|³   (producto cruzado de velocidad y aceleración)
# --------------------------------------------------

def compute_adaptive_speeds(path, v_max, v_min, curve_slowdown):
    """
    Calcula la velocidad objetivo en cada punto de la spline
    basándose en la curvatura local.

    Retorna array de velocidades [m/s] de longitud len(path).
    """
    n      = len(path)
    speeds = np.full(n, v_max)

    for i in range(1, n - 1):
        # Vectores de primera y segunda derivada (diferencias finitas)
        d1 = path[i + 1] - path[i - 1]          # velocidad aproximada
        d2 = path[i + 1] - 2 * path[i] + path[i - 1]  # aceleración aproximada

        norm_d1 = np.linalg.norm(d1)
        if norm_d1 < 1e-6:
            continue

        # Curvatura = |d1 × d2| / |d1|³
        cross     = np.cross(d1, d2)
        curvature = np.linalg.norm(cross) / (norm_d1 ** 3)

        # Mapear curvatura a velocidad:
        # curvatura=0 (recto) → v_max
        # curvatura alta      → v_min
        # El factor 10 es un escalado empírico; ajustar si hace falta
        curve_factor = 1.0 / (1.0 + curve_slowdown * 10 * curvature)
        speeds[i]    = v_min + (v_max - v_min) * curve_factor

    # Suavizar el perfil de velocidad para evitar cambios bruscos
    # (media móvil de ventana 3)
    smoothed = speeds.copy()
    for i in range(1, n - 1):
        smoothed[i] = (speeds[i - 1] + speeds[i] + speeds[i + 1]) / 3.0

    return smoothed

adaptive_speeds = compute_adaptive_speeds(
    path, MAX_VELOCITY, MIN_VELOCITY, CURVE_SLOWDOWN
)

print(f"Velocidades adaptativas:")
print(f"  Max: {adaptive_speeds.max():.2f} m/s")
print(f"  Min: {adaptive_speeds.min():.2f} m/s")
print(f"  Promedio: {adaptive_speeds.mean():.2f} m/s\n")

# --------------------------------------------------
# TIMEOUT ADAPTATIVO — proporcional a distancia/velocidad
# --------------------------------------------------

def compute_timeout(index, path, speeds):
    """
    Calcula el timeout para el waypoint `index` basándose en
    la distancia al siguiente punto y la velocidad en ese tramo.
    """
    if index >= len(path) - 1:
        return TIMEOUT_MAX
    dist  = np.linalg.norm(path[index + 1][:2] - path[index][:2])
    v     = max(speeds[index], MIN_VELOCITY)
    t     = (dist / v) * TIMEOUT_FACTOR
    return float(np.clip(t, TIMEOUT_MIN, TIMEOUT_MAX))

# --------------------------------------------------
# ESTADO INICIAL
# --------------------------------------------------
index               = 0
flight_finished     = False
vx_smooth           = 0.0
vy_smooth           = 0.0

robot.step(timestep)

def read_gps():
    v = gps.getValues()
    return v[0], v[1], v[2]

past_x, past_y, z_init = read_gps()
past_time               = robot.getTime()
waypoint_start_time     = past_time
current_timeout         = compute_timeout(0, path, adaptive_speeds)
HEIGHT_DESIRED          = z_init
desired_yaw_rate        = 0.0   # inicializar antes del loop

print("Iniciando simulacion...")
print(f"Altura inicial: {z_init:.3f}m → primer WP: {path[0][2]:.2f}m")

# --------------------------------------------------
# LOOP PRINCIPAL
# --------------------------------------------------
while robot.step(timestep) != -1:

    dt = robot.getTime() - past_time
    if dt <= 0:
        continue

    roll, pitch, yaw = imu.getRollPitchYaw()
    yaw_rate         = gyro.getValues()[2]
    x, y, z          = read_gps()
    real_trajectory.append([x, y, z])

    vx_global = (x - past_x) / dt
    vy_global = (y - past_y) / dt
    v_x =  vx_global * cos(yaw) + vy_global * sin(yaw)
    v_y = -vx_global * sin(yaw) + vy_global * cos(yaw)

    if not flight_finished:

        target       = path[index]
        target_speed = adaptive_speeds[index]   # velocidad para este punto

        ex_world = target[0] - x
        ey_world = target[1] - y
        dist     = np.linalg.norm([ex_world, ey_world])

        # -- Rampa solo en despegue --
        height_target = target[2]
        if not takeoff_done:
            diff = height_target - HEIGHT_DESIRED
            step = HEIGHT_RAMP_RATE * dt
            if abs(diff) <= step:
                HEIGHT_DESIRED = height_target
                takeoff_done   = True
                print(f"[TAKEOFF] Altura alcanzada: {height_target:.2f}m")
            else:
                HEIGHT_DESIRED += step if diff > 0 else -step
        else:
            HEIGHT_DESIRED = height_target

        # -- Radio de aceptación adaptativo --
        # Escalar con la velocidad actual del tramo:
        # más velocidad → radio más grande (reacción más anticipada)
        speed_ratio   = target_speed / max(MAX_VELOCITY, 1e-6)
        accept_radius = ACCEPT_RADIUS_BASE * max(speed_ratio, 0.3)

        # -- Avance de waypoint --
        time_on_wp = robot.getTime() - waypoint_start_time
        if dist < accept_radius or time_on_wp > current_timeout:
            if time_on_wp > current_timeout:
                print(f"  [TIMEOUT] WP {index} dist={dist:.2f}m "
                      f"(timeout={current_timeout:.1f}s)")
            if index < len(path) - 1:
                index          += 1
                waypoint_start_time = robot.getTime()
                current_timeout     = compute_timeout(index, path, adaptive_speeds)
                pct = 100 * index / (len(path) - 1)
                print(f"  WP {index:2d}/{len(path)-1} ({pct:.0f}%)"
                      f" v={adaptive_speeds[index]:.2f}m/s"
                      f" → ({path[index][0]:.2f}, {path[index][1]:.2f})")
            else:
                print("Trayectoria completada.")
                flight_finished = True

        # -- Yaw deseado: apuntar hacia el siguiente waypoint --
        # En vez de yaw=0 siempre, el drone rota para mirar
        # en la dirección de movimiento (comportamiento de carrera)
        if dist > 0.01:
            desired_yaw = np.arctan2(ey_world, ex_world)
        else:
            desired_yaw = yaw  # si ya llegó, mantener yaw actual

        # -- Velocidad deseada adaptativa --
        if dist > 0.01:
            scale    = min(K_POS, target_speed / dist)
            vx_w_des = scale * ex_world
            vy_w_des = scale * ey_world
        else:
            vx_w_des = 0.0
            vy_w_des = 0.0

        # Rotar al frame del drone
        vx_des =  vx_w_des * cos(yaw) + vy_w_des * sin(yaw)
        vy_des = -vx_w_des * sin(yaw) + vy_w_des * cos(yaw)

        alpha     = 0.4
        vx_smooth = alpha * vx_des + (1 - alpha) * vx_smooth
        vy_smooth = alpha * vy_des + (1 - alpha) * vy_smooth

        vx = vx_smooth
        vy = vy_smooth

        # Yaw rate deseado: error de yaw * ganancia
        # Normalizar a [-pi, pi] para evitar rotaciones largas
        yaw_error = desired_yaw - yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        desired_yaw_rate = 2.0 * yaw_error   # ganancia de yaw

    else:
        vx = 0.0
        vy = 0.0
        desired_yaw_rate = 0.0
        HEIGHT_DESIRED -= HEIGHT_RAMP_RATE * dt
        HEIGHT_DESIRED  = max(HEIGHT_DESIRED, 0.0)
        if z < 0.05 or HEIGHT_DESIRED <= 0.0:
            print(f"Aterrizaje completado. (z={z:.3f}m)")
            break

    motor_power = PID.pid(
        dt, vx, vy, desired_yaw_rate,
        HEIGHT_DESIRED,
        roll, pitch, yaw_rate,
        z, v_x, v_y
    )

    m1.setVelocity(-motor_power[0])
    m2.setVelocity( motor_power[1])
    m3.setVelocity(-motor_power[2])
    m4.setVelocity( motor_power[3])

    past_time = robot.getTime()
    past_x    = x
    past_y    = y

# --------------------------------------------------
# GRÁFICA
# --------------------------------------------------
real    = np.array(real_trajectory)
desired = np.array(path)
wp_arr  = np.array(waypoints)

fig = plt.figure(figsize=(12, 5))

# proporción: trayectoria grande / velocidad pequeña
gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1])

# -- Gráfica 3D (más grande) --
ax = fig.add_subplot(gs[0], projection='3d')
ax.plot(desired[:, 0], desired[:, 1], desired[:, 2],
        'r--', lw=2, label='Spline deseada')
ax.plot(real[:, 0], real[:, 1], real[:, 2],
        'b', lw=1, label='Real')
ax.plot(wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2],
        'o--', color='orange', lw=1.5, markersize=7,
        markeredgecolor='black', markeredgewidth=0.8,
        zorder=6, label='Waypoints')
ax.scatter(*real[0],  color='green', s=60, zorder=7, label='Inicio')
ax.scatter(*real[-1], color='red',   s=60, zorder=7, label='Fin')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Trayectorias')
ax.legend(fontsize=7)
ax.view_init(elev=30, azim=135)

# -- Gráfica de velocidad (más pequeña) --
ax2 = fig.add_subplot(gs[1])
ax2.plot(adaptive_speeds, 'g-', lw=2, label='Velocidad objetivo')
ax2.axhline(MAX_VELOCITY, color='r', linestyle='--', lw=1)
ax2.axhline(MIN_VELOCITY, color='b', linestyle='--', lw=1)
ax2.fill_between(range(len(adaptive_speeds)), adaptive_speeds,
                 alpha=0.2, color='green')
ax2.set_xlabel('Índice')
ax2.set_ylabel('Velocidad [m/s]')
ax2.set_title('Velocidad adaptativa')
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("trayectoria.png", dpi=150, bbox_inches='tight')
plt.show()

n   = min(len(real), len(desired))
err = np.linalg.norm(real[:n, :2] - desired[:n, :2], axis=1)
print(f"\nError promedio XY: {err.mean():.4f} m")
print(f"Error maximo XY:   {err.max():.4f} m")
print(f"Velocidad promedio usada: {adaptive_speeds.mean():.2f} m/s")