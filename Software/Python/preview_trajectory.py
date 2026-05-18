"""
preview_trajectory.py — Previsualiza la trayectoria antes de volar.

Ventana 1: Vista superior (XY) + Perfil de altura
Ventana 2: Vista 3D + Perfil de velocidad adaptativa
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
import sys

try:
    from trajectory import TrajectoryGenerator
except ImportError:
    print("ERROR: No se encontró trajectory.py en la misma carpeta.")
    sys.exit(1)

# --------------------------------------------------
# PARÁMETROS — mantener iguales que en crazyflie_controller.py
# --------------------------------------------------
SPLINE_POINTS  = 50
METHOD         = 'catmull_rom'
TENSION        = 0.5
MAX_VELOCITY   = 0.5
MIN_VELOCITY   = 0.1
CURVE_SLOWDOWN = 0.5

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
# GENERAR TRAYECTORIA
# --------------------------------------------------
traj         = TrajectoryGenerator(waypoints)
path         = traj.generate_spline(SPLINE_POINTS, method=METHOD, tension=TENSION)
path_visual  = traj.generate_spline(300,           method=METHOD, tension=TENSION)
wp_arr       = np.array(waypoints)

print(f"\nSpline: {len(path)} puntos de control")
print(f"Longitud total: {traj.total_length():.3f} m")

# --------------------------------------------------
# VELOCIDAD ADAPTATIVA
# --------------------------------------------------
def compute_adaptive_speeds(path, v_max, v_min, curve_slowdown):
    n      = len(path)
    speeds = np.full(n, v_max)
    for i in range(1, n - 1):
        d1      = path[i + 1] - path[i - 1]
        d2      = path[i + 1] - 2 * path[i] + path[i - 1]
        norm_d1 = np.linalg.norm(d1)
        if norm_d1 < 1e-6:
            continue
        curvature = np.linalg.norm(np.cross(d1, d2)) / (norm_d1 ** 3)
        factor    = 1.0 / (1.0 + curve_slowdown * 10 * curvature)
        speeds[i] = v_min + (v_max - v_min) * factor
    smoothed = speeds.copy()
    for i in range(1, n - 1):
        smoothed[i] = (speeds[i-1] + speeds[i] + speeds[i+1]) / 3.0
    return smoothed

speeds     = compute_adaptive_speeds(path, MAX_VELOCITY, MIN_VELOCITY, CURVE_SLOWDOWN)
seg_times  = np.linalg.norm(np.diff(path, axis=0), axis=1) / speeds[:-1]
total_time = seg_times.sum()

print(f"Velocidad promedio: {speeds.mean():.2f} m/s")
print(f"Tiempo estimado:    {total_time:.1f} s")

cmap = plt.cm.RdYlGn
norm = plt.Normalize(vmin=MIN_VELOCITY, vmax=MAX_VELOCITY)

# Helper: encontrar posición de cada WP en la spline
def wp_distances_on_spline(wp_arr, path_visual):
    dists_visual = np.concatenate([[0],
        np.cumsum(np.linalg.norm(np.diff(path_visual, axis=0), axis=1))])
    dists_wp = []
    for wp in wp_arr:
        idx = np.argmin(np.linalg.norm(path_visual[:, :2] - wp[:2], axis=1))
        dists_wp.append(dists_visual[idx])
    return dists_visual, dists_wp

dists_visual, dists_wp = wp_distances_on_spline(wp_arr, path_visual)

# ==========================================================
# VENTANA 1 — Vista superior (XY) + Perfil de altura
# ==========================================================
fig1 = plt.figure(figsize=(14, 6),
                  num="Vista superior y perfil de altura")
fig1.suptitle(
    f"Preview de Trayectoria — {len(waypoints)} waypoints | "
    f"longitud={traj.total_length():.2f}m | tiempo≈{total_time:.1f}s",
    fontsize=12, fontweight='bold'
)

gs1   = gridspec.GridSpec(1, 3, figure=fig1,
                           width_ratios=[5, 0.12, 4], wspace=0.4)
ax_xy = fig1.add_subplot(gs1[0])
ax_cb = fig1.add_subplot(gs1[1])   # colorbar fija
ax_z  = fig1.add_subplot(gs1[2])

# -- Colorbar --
import matplotlib.colorbar
import matplotlib.colors as mcolors
cb = matplotlib.colorbar.ColorbarBase(
    ax_cb, cmap=cmap,
    norm=mcolors.Normalize(vmin=MIN_VELOCITY, vmax=MAX_VELOCITY),
    orientation='vertical'
)
cb.set_label('Velocidad [m/s]', fontsize=9)

# -- Vista XY --
ax_xy.set_title('Vista superior (XY)', fontweight='bold', fontsize=11)
ax_xy.set_xlabel('X [m]'); ax_xy.set_ylabel('Y [m]')
ax_xy.set_aspect('equal'); ax_xy.grid(True, alpha=0.4)

# Trayectoria coloreada por velocidad
for i in range(len(path_visual) - 1):
    t     = i / len(path_visual)
    idx   = int(t * len(speeds))
    color = cmap(norm(speeds[min(idx, len(speeds)-1)]))
    ax_xy.plot(path_visual[i:i+2, 0], path_visual[i:i+2, 1],
               color=color, lw=2.5)

# Waypoints
ax_xy.plot(wp_arr[:, 0], wp_arr[:, 1], 'o--',
           color='orange', lw=1.5, markersize=10,
           markeredgecolor='black', markeredgewidth=1.5,
           zorder=5, label='Waypoints')

for i, wp in enumerate(waypoints):
    ax_xy.annotate(f'WP{i}\nz={wp[2]:.2f}m',
                   (wp[0], wp[1]), textcoords='offset points',
                   xytext=(8, 5), fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))

# Puntos de control de la spline
ax_xy.scatter(path[:, 0], path[:, 1],
              c=speeds, cmap=cmap, vmin=MIN_VELOCITY, vmax=MAX_VELOCITY,
              s=25, zorder=4, alpha=0.6, label=f'{len(path)} pts spline')

# Flecha de dirección a mitad de trayectoria
mid = len(path_visual) // 2
dx  = path_visual[mid+1, 0] - path_visual[mid-1, 0]
dy  = path_visual[mid+1, 1] - path_visual[mid-1, 1]
ax_xy.annotate('', xy=(path_visual[mid, 0] + dx*3,
                        path_visual[mid, 1] + dy*3),
               xytext=(path_visual[mid, 0], path_visual[mid, 1]),
               arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

ax_xy.scatter(*path[0, :2],  color='green', s=120, zorder=6,
              edgecolors='black', linewidths=1.5, label='Inicio')
ax_xy.scatter(*path[-1, :2], color='red',   s=120, zorder=6,
              edgecolors='black', linewidths=1.5, label='Fin')
ax_xy.legend(loc='upper right', fontsize=8)

# -- Perfil de altura --
ax_z.set_title('Perfil de altura', fontweight='bold', fontsize=11)
ax_z.set_xlabel('Distancia recorrida [m]')
ax_z.set_ylabel('Altura Z [m]')
ax_z.grid(True, alpha=0.4)

ax_z.plot(dists_visual, path_visual[:, 2], 'b-', lw=2.5, label='Spline')
ax_z.fill_between(dists_visual, path_visual[:, 2], alpha=0.15, color='blue')

ax_z.scatter(dists_wp, wp_arr[:, 2], color='orange', s=100,
             edgecolors='black', linewidths=1.5, zorder=5, label='Waypoints')
for i, (d, wp) in enumerate(zip(dists_wp, waypoints)):
    ax_z.annotate(f'WP{i}\n{wp[2]:.2f}m', (d, wp[2]),
                  textcoords='offset points', xytext=(5, 5), fontsize=8,
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))

ax_z.set_ylim(0, max(wp_arr[:, 2].max() + 0.4, 1.0))
ax_z.legend(fontsize=9)

# ==========================================================
# VENTANA 2 — Vista 3D + Perfil de velocidad
# ==========================================================
fig2 = plt.figure(figsize=(14, 6),
                  num="Vista 3D y perfil de velocidad")
fig2.suptitle(
    f"Preview de Trayectoria — velocidad máx={MAX_VELOCITY} m/s | "
    f"mín={MIN_VELOCITY} m/s | curve_slowdown={CURVE_SLOWDOWN}",
    fontsize=12, fontweight='bold'
)

gs2    = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.35)
ax_3d  = fig2.add_subplot(gs2[0], projection='3d')
ax_spd = fig2.add_subplot(gs2[1])

# -- Vista 3D --
ax_3d.set_title('Vista 3D', fontweight='bold', fontsize=11)

for i in range(len(path_visual) - 1):
    t     = i / len(path_visual)
    idx   = int(t * len(speeds))
    color = cmap(norm(speeds[min(idx, len(speeds)-1)]))
    ax_3d.plot(path_visual[i:i+2, 0], path_visual[i:i+2, 1],
               path_visual[i:i+2, 2], color=color, lw=2)

ax_3d.plot(wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2],
           'o--', color='orange', lw=1.5, markersize=8,
           markeredgecolor='black', zorder=5, label='Waypoints')

for i, wp in enumerate(waypoints):
    ax_3d.text(wp[0], wp[1], wp[2] + 0.05, f'WP{i}', fontsize=8)

ax_3d.scatter(*path[0],  color='green', s=80, zorder=6, label='Inicio')
ax_3d.scatter(*path[-1], color='red',   s=80, zorder=6, label='Fin')

ax_3d.set_xlabel('X [m]'); ax_3d.set_ylabel('Y [m]'); ax_3d.set_zlabel('Z [m]')
ax_3d.legend(fontsize=8)
ax_3d.view_init(elev=25, azim=135)

# -- Perfil de velocidad --
ax_spd.set_title('Perfil de velocidad adaptativa\n(baja en curvas, sube en rectas)',
                 fontweight='bold', fontsize=11)
ax_spd.set_xlabel('Índice de punto en spline')
ax_spd.set_ylabel('Velocidad [m/s]')
ax_spd.grid(True, alpha=0.4)

ax_spd.plot(speeds, lw=2.5, color='green', label='Velocidad objetivo')
ax_spd.fill_between(range(len(speeds)), speeds, alpha=0.2, color='green')
ax_spd.axhline(MAX_VELOCITY, color='red',  linestyle='--', lw=1.5,
               label=f'Máx ({MAX_VELOCITY} m/s)')
ax_spd.axhline(MIN_VELOCITY, color='blue', linestyle='--', lw=1.5,
               label=f'Mín ({MIN_VELOCITY} m/s)')
ax_spd.set_ylim(0, MAX_VELOCITY * 1.25)

# Líneas verticales en cada waypoint
for i, wp in enumerate(waypoints):
    diffs = np.linalg.norm(path[:, :2] - np.array(wp[:2]), axis=1)
    idx   = int(np.argmin(diffs))
    ax_spd.axvline(idx, color='orange', linestyle=':', lw=1.5, alpha=0.9)
    ax_spd.text(idx, MAX_VELOCITY * 1.08, f'WP{i}',
                ha='center', fontsize=8, color='darkorange', fontweight='bold')

ax_spd.legend(fontsize=9)

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
