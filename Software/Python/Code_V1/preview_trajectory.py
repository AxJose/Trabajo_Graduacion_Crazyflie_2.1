"""
preview_trajectory.py — Previsualiza la trayectoria antes de volar.

Usa EXACTAMENTE el mismo planificador y la misma ley de control que
flight_purepursuit.py, así que la simulación predice el vuelo real:

  waypoints → Catmull-Rom → paso de arco constante → perfil de velocidad
            → lazo: v = v_ref*T(s+L) + KP_LAT*e_lat + capa límite

El modelo del dron es un retardo de primer orden (tau ≈ 0.3 s), que es lo
que aproxima el lazo de velocidad del Crazyflie con setpoint de mundo.

Ventana 1: Vista superior (XY) + Perfil de altura
Ventana 2: Vista 3D + Perfil de velocidad y curvatura
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import json
import os
import sys

try:
    from trajectory import TrajectoryGenerator, compute_speed_profile
    from pure_pursuit import PurePursuit
except ImportError:
    print("ERROR: No se encontró trajectory.py / pure_pursuit.py en la carpeta.")
    sys.exit(1)

# --------------------------------------------------
# PARÁMETROS — mantener sincronizados con flight_purepursuit.py
# --------------------------------------------------
PATH_RESOLUTION = 0.02
METHOD          = 'catmull_rom'
TENSION         = 0.5

V_MAX       = 0.30
V_MIN       = 0.12
A_LAT_MAX   = 0.35
A_TAN_MAX   = 0.40
VZ_MAX_PATH = 0.30
V_START     = 0.12
V_END       = 0.10

LOOKAHEAD_TIME = 0.40
LOOKAHEAD_MIN  = 0.10
LOOKAHEAD_MAX  = 0.45
GOAL_TOLERANCE = 0.10

CONTROL_RATE = 0.02
KP_LAT       = 2.2
K_LAT_SAT    = 0.12
LAT_BOUNDARY = 0.03
KP_Z         = 2.0
V_XY_MAX_CMD = 0.85
VZ_MAX_CMD   = 0.45
ALPHA_V      = 0.30

SIM_TAU     = 0.30    # [s] retardo del lazo de velocidad del dron
SIM_LAG     = 0.05    # [s] retardo de transporte (radio + EKF)
PP_MAX_TIME = 300.0

# --------------------------------------------------
# CARGAR WAYPOINTS
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
wp_path    = os.path.join(script_dir, "waypoints.json")

if not os.path.exists(wp_path):
    print(f"ERROR: No se encontró {wp_path}")
    print("Primero crea los waypoints con create_waypoints.py")
    sys.exit(1)

with open(wp_path) as f:
    waypoints = json.load(f)

print(f"Waypoints cargados: {len(waypoints)}")
for i, wp in enumerate(waypoints):
    print(f"  WP{i}: x={wp[0]:.3f}, y={wp[1]:.3f}, z={wp[2]:.3f}")

# --------------------------------------------------
# TRAYECTORIA Y PERFIL DE VELOCIDAD
# --------------------------------------------------
traj   = TrajectoryGenerator(waypoints)
path   = traj.generate_uniform(spacing=PATH_RESOLUTION,
                               method=METHOD, tension=TENSION)
wp_arr = np.array(waypoints)

pp = PurePursuit(lookahead_distance=LOOKAHEAD_TIME * V_MAX,
                 min_lookahead=LOOKAHEAD_MIN,
                 max_lookahead=LOOKAHEAD_MAX,
                 lookahead_time=LOOKAHEAD_TIME,
                 k_curve=0.0,
                 goal_tolerance=GOAL_TOLERANCE)
pp.set_path(path)

speeds = compute_speed_profile(pp.s, pp.kappa, pp.T,
                               v_max=V_MAX, v_min=V_MIN,
                               a_lat=A_LAT_MAX, a_tan=A_TAN_MAX,
                               vz_max=VZ_MAX_PATH,
                               v_start=V_START, v_end=V_END)

s_arr      = pp.s
total_time = float(np.sum(np.diff(s_arr) / np.maximum(speeds[:-1], 1e-9)))
r_min      = 1.0 / max(pp.kappa.max(), 1e-6)

print(f"\nTrayectoria: {len(path)} puntos | longitud {pp.length:.3f} m")
print(f"Radio mínimo: {r_min:.3f} m  (limita v a "
      f"{np.sqrt(A_LAT_MAX * r_min):.2f} m/s por A_LAT_MAX)")
print(f"Velocidad: {speeds.min():.3f} – {speeds.max():.3f} m/s "
      f"(media {speeds.mean():.3f})")
print(f"Tiempo estimado (perfil): {total_time:.1f} s")

cmap     = plt.cm.RdYlGn
norm_vel = mcolors.Normalize(vmin=V_MIN, vmax=V_MAX)

# --------------------------------------------------
# SIMULACIÓN DEL LAZO DE CONTROL
# --------------------------------------------------
from collections import deque

p   = np.array(path[0], dtype=float)
v   = np.zeros(3)
buf = deque([np.zeros(3)] * max(int(round(SIM_LAG / CONTROL_RATE)), 1))

speed_cmd = float(speeds[0])
sim_traj, sim_z_list, lookahead_points = [p[:2].copy()], [float(p[2])], []
sim_e_lat, sim_e_z, sim_s, sim_v, sim_L = [], [], [], [], []

sim_time = 0.0
pp.step(p[0], p[1])

while not pp.finished and sim_time < PP_MAX_TIME:
    pp.step(p[0], p[1], max_advance=0.06 + 2.5 * speed_cmd * CONTROL_RATE)
    if pp.finished:
        break

    v_target  = float(np.interp(pp.s_proj, s_arr, speeds))
    speed_cmd = ALPHA_V * v_target + (1.0 - ALPHA_V) * speed_cmd

    pp.adaptive_lookahead(speed_cmd, pp.curvature_ahead(0.0))
    p_look, T_look, _ = pp.lookahead_state()

    e_lat  = pp.e_lat
    e_norm = np.linalg.norm(e_lat)
    corr   = K_LAT_SAT * e_lat / max(e_norm, LAT_BOUNDARY)

    cmd_xy = speed_cmd * T_look[:2] + KP_LAT * e_lat + corr
    nn = np.linalg.norm(cmd_xy)
    if nn > V_XY_MAX_CMD:
        cmd_xy *= V_XY_MAX_CMD / nn

    z_ref  = float(pp.proj[2])
    cmd_z  = np.clip(speed_cmd * T_look[2] + KP_Z * (z_ref - p[2]),
                     -VZ_MAX_CMD, VZ_MAX_CMD)

    buf.append(np.array([cmd_xy[0], cmd_xy[1], cmd_z]))
    cmd = buf.popleft()

    v += (cmd - v) * (CONTROL_RATE / SIM_TAU)
    p += v * CONTROL_RATE

    sim_traj.append(p[:2].copy())
    sim_z_list.append(float(p[2]))
    lookahead_points.append(p_look[:2].copy())
    sim_e_lat.append(pp.e_lat_signed)
    sim_e_z.append(z_ref - p[2])
    sim_s.append(pp.s_proj)
    sim_v.append(speed_cmd)
    sim_L.append(pp.L)

    sim_time += CONTROL_RATE

sim_traj  = np.array(sim_traj)
sim_z_arr = np.array(sim_z_list)
sim_e_lat = np.array(sim_e_lat)
sim_e_z   = np.array(sim_e_z)
sim_s     = np.array(sim_s)

print(f"\nSimulación: {len(sim_traj)} puntos | tiempo {sim_time:.1f} s")
print(f"  Error lateral: medio {np.abs(sim_e_lat).mean()*100:.2f} cm | "
      f"máx {np.abs(sim_e_lat).max()*100:.2f} cm")
print(f"  Error de altura: medio {np.abs(sim_e_z).mean()*100:.2f} cm | "
      f"máx {np.abs(sim_e_z).max()*100:.2f} cm")
print("  (el vuelo real tendrá más error: ruido de MoCap, viento, "
      "modelo aproximado)")

# --------------------------------------------------
# HELPER: arco acumulado de cada waypoint sobre la spline
# --------------------------------------------------
dists_wp = []
for wp in wp_arr:
    idx = int(np.argmin(np.linalg.norm(path[:, :2] - wp[:2], axis=1)))
    dists_wp.append(s_arr[idx])

# ==========================================================
# VENTANA 1 — Vista superior (XY) + Perfil de altura
# ==========================================================
fig1 = plt.figure(figsize=(14, 6), num="Vista superior y perfil de altura")
fig1.suptitle(
    f"Preview — {len(waypoints)} waypoints | longitud={pp.length:.2f} m | "
    f"radio mín={r_min:.2f} m | tiempo≈{sim_time:.1f} s",
    fontsize=12, fontweight='bold'
)

gs1   = gridspec.GridSpec(1, 3, figure=fig1, width_ratios=[5, 0.12, 4], wspace=0.4)
ax_xy = fig1.add_subplot(gs1[0])
ax_cb = fig1.add_subplot(gs1[1])
ax_z  = fig1.add_subplot(gs1[2])

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_vel)
sm.set_array([])
cb = plt.colorbar(sm, cax=ax_cb, orientation='vertical')
cb.set_label('Velocidad [m/s]', fontsize=9)

ax_xy.set_title('Vista superior (XY)', fontweight='bold', fontsize=11)
ax_xy.set_xlabel('X [m]')
ax_xy.set_ylabel('Y [m]')
ax_xy.set_aspect('equal')
ax_xy.grid(True, alpha=0.4)

# Trayectoria coloreada por la velocidad del perfil
for i in range(len(path) - 1):
    ax_xy.plot(path[i:i+2, 0], path[i:i+2, 1],
               color=cmap(norm_vel(speeds[i])), lw=3.0)

ax_xy.plot(sim_traj[:, 0], sim_traj[:, 1], color='magenta', lw=1.6,
           label='Simulación del lazo', zorder=4)

ax_xy.plot(wp_arr[:, 0], wp_arr[:, 1], 'o', color='orange', markersize=10,
           markeredgecolor='black', markeredgewidth=1.5,
           zorder=5, label='Waypoints')

for i, wp in enumerate(waypoints):
    ax_xy.annotate(f'WP{i}\nz={wp[2]:.2f}m', (wp[0], wp[1]),
                   textcoords='offset points', xytext=(8, 5), fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))

mid = len(path) // 2
dx  = path[mid+1, 0] - path[mid-1, 0]
dy  = path[mid+1, 1] - path[mid-1, 1]
nrm = np.hypot(dx, dy)
ax_xy.annotate('', xy=(path[mid, 0] + 0.3*dx/nrm, path[mid, 1] + 0.3*dy/nrm),
               xytext=(path[mid, 0], path[mid, 1]),
               arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

ax_xy.scatter(*path[0,  :2], color='green', s=120, zorder=6,
              edgecolors='black', linewidths=1.5, label='Inicio')
ax_xy.scatter(*path[-1, :2], color='red', s=120, zorder=6,
              edgecolors='black', linewidths=1.5, label='Fin')
ax_xy.legend(loc='upper right', fontsize=8)

ax_z.set_title('Perfil de altura', fontweight='bold', fontsize=11)
ax_z.set_xlabel('Distancia recorrida [m]')
ax_z.set_ylabel('Altura Z [m]')
ax_z.grid(True, alpha=0.4)

ax_z.plot(s_arr, path[:, 2], 'b-', lw=2.5, label='Spline')
ax_z.fill_between(s_arr, path[:, 2], alpha=0.15, color='blue')
ax_z.plot(sim_s, sim_z_arr[1:], color='magenta', lw=1.4, label='Simulación')

ax_z.scatter(dists_wp, wp_arr[:, 2], color='orange', s=100,
             edgecolors='black', linewidths=1.5, zorder=5, label='Waypoints')
for i, (d, wp) in enumerate(zip(dists_wp, waypoints)):
    ax_z.annotate(f'WP{i}\n{wp[2]:.2f}m', (d, wp[2]),
                  textcoords='offset points', xytext=(5, 5), fontsize=8,
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))

ax_z.set_ylim(0, max(wp_arr[:, 2].max() + 0.4, 1.0))
ax_z.legend(fontsize=9)

# ==========================================================
# VENTANA 2 — Vista 3D + Velocidad/curvatura + Error simulado
# ==========================================================
fig2 = plt.figure(figsize=(15, 8), num="Vista 3D, velocidad y error")
fig2.suptitle(
    f"Preview — v_max={V_MAX} m/s | a_lat_max={A_LAT_MAX} m/s² | "
    f"L={LOOKAHEAD_TIME}s·v | KP_LAT={KP_LAT}",
    fontsize=12, fontweight='bold'
)

gs2    = gridspec.GridSpec(2, 2, figure=fig2, wspace=0.3, hspace=0.35)
ax_3d  = fig2.add_subplot(gs2[:, 0], projection='3d')
ax_spd = fig2.add_subplot(gs2[0, 1])
ax_err = fig2.add_subplot(gs2[1, 1])

ax_3d.set_title('Vista 3D', fontweight='bold', fontsize=11)
for i in range(0, len(path) - 1, 2):
    ax_3d.plot(path[i:i+3, 0], path[i:i+3, 1], path[i:i+3, 2],
               color=cmap(norm_vel(speeds[i])), lw=2)

ax_3d.plot(sim_traj[1:, 0], sim_traj[1:, 1], sim_z_arr[1:],
           color='magenta', lw=1.4, label='Simulación')
ax_3d.plot(wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2], 'o', color='orange',
           markersize=8, markeredgecolor='black', zorder=5, label='Waypoints')
for i, wp in enumerate(waypoints):
    ax_3d.text(wp[0], wp[1], wp[2] + 0.05, f'WP{i}', fontsize=8)

ax_3d.scatter(*path[0],  color='green', s=80, zorder=6, label='Inicio')
ax_3d.scatter(*path[-1], color='red',   s=80, zorder=6, label='Fin')
ax_3d.set_xlabel('X [m]'); ax_3d.set_ylabel('Y [m]'); ax_3d.set_zlabel('Z [m]')
ax_3d.legend(fontsize=8)
ax_3d.view_init(elev=25, azim=135)

ax_spd.set_title('Perfil de velocidad y curvatura', fontweight='bold', fontsize=11)
ax_spd.set_xlabel('Distancia recorrida [m]')
ax_spd.set_ylabel('Velocidad [m/s]', color='green')
ax_spd.grid(True, alpha=0.4)
ax_spd.plot(s_arr, speeds, lw=2.2, color='green', label='Perfil')
ax_spd.fill_between(s_arr, speeds, alpha=0.18, color='green')
ax_spd.plot(sim_s, sim_v, lw=1.2, color='darkgreen', ls='--', label='Comandada')
ax_spd.axhline(V_MAX, color='red', ls='--', lw=1.2, label=f'V_MAX {V_MAX}')
ax_spd.set_ylim(0, V_MAX * 1.25)
ax_spd.legend(fontsize=8, loc='lower left')

ax_k = ax_spd.twinx()
ax_k.plot(s_arr, pp.kappa, color='darkorange', lw=1.2, alpha=0.8)
ax_k.set_ylabel('Curvatura [1/m]', color='darkorange')

for i, d in enumerate(dists_wp):
    ax_spd.axvline(d, color='orange', ls=':', lw=1.2, alpha=0.9)
    ax_spd.text(d, V_MAX * 1.10, f'WP{i}', ha='center', fontsize=7,
                color='darkorange', fontweight='bold')

ax_err.set_title(
    f'Error simulado  (lateral medio {np.abs(sim_e_lat).mean()*100:.2f} cm, '
    f'altura {np.abs(sim_e_z).mean()*100:.2f} cm)',
    fontweight='bold', fontsize=11)
ax_err.set_xlabel('Distancia recorrida [m]')
ax_err.set_ylabel('Error [cm]')
ax_err.grid(True, alpha=0.4)
ax_err.axhline(0, color='k', lw=0.8)
ax_err.plot(sim_s, sim_e_lat * 100, color='purple', lw=1.3, label='Lateral')
ax_err.plot(sim_s, sim_e_z * 100, color='teal', lw=1.3, label='Altura')
ax_err.legend(fontsize=8)

for d in dists_wp:
    ax_err.axvline(d, color='orange', ls=':', lw=1.0, alpha=0.7)

# ==========================================================
# GUARDAR Y MOSTRAR
# ==========================================================
out1 = os.path.join(script_dir, "preview_xy_altura.png")
out2 = os.path.join(script_dir, "preview_3d_velocidad.png")

fig1.savefig(out1, dpi=150, bbox_inches='tight')
fig2.savefig(out2, dpi=150, bbox_inches='tight')

print(f"\nGuardado: {out1}")
print(f"Guardado: {out2}")

plt.show()
