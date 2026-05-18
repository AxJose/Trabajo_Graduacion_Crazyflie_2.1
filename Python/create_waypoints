"""
create_waypoints.py — Editor de waypoints con control de altura por punto.

Controles:
  Click izquierdo  → agregar waypoint
  Click derecho    → seleccionar waypoint para editar altura
  Scroll / ↑↓      → cambiar altura del punto seleccionado
  Backspace        → borrar último punto
  Enter            → guardar waypoints.json
  R                → reiniciar todo
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import json
import os
import numpy as np

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------
Z_MIN     = 0.1
Z_MAX     = 2.0
Z_DEFAULT = 0.5
Z_STEP    = 0.05
XLIM      = (-2, 2)
YLIM      = (-2, 2)

# --------------------------------------------------
# ESTADO
# --------------------------------------------------
points         = []
selected_index = None

# --------------------------------------------------
# FIGURA — layout fijo con GridSpec para evitar drift
# Reservamos una columna estrecha para la colorbar
# --------------------------------------------------
fig = plt.figure(figsize=(13, 6))
fig.suptitle("Editor de Waypoints — Crazyflie", fontsize=12, fontweight='bold')

gs      = fig.add_gridspec(1, 3, width_ratios=[4, 0.15, 3], wspace=0.35)
ax_xy   = fig.add_subplot(gs[0])     # vista superior XY
ax_cb   = fig.add_subplot(gs[1])     # colorbar fija (no se regenera)
ax_xz   = fig.add_subplot(gs[2])     # perfil de alturas

# Colorbar estática — se crea UNA SOLA VEZ
cmap     = cm.RdYlGn
norm     = mcolors.Normalize(vmin=Z_MIN, vmax=Z_MAX)
cb       = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm,
                                             orientation='vertical')
cb.set_label('Altura Z [m]', fontsize=9)

# --------------------------------------------------
# DIBUJO
# --------------------------------------------------

def redraw():
    # Limpiar solo los ejes de datos, nunca ax_cb
    ax_xy.cla()
    ax_xz.cla()

    # -- Configurar ax_xy --
    ax_xy.set_xlim(XLIM); ax_xy.set_ylim(YLIM)
    ax_xy.set_xlabel('X [m]'); ax_xy.set_ylabel('Y [m]')
    ax_xy.set_title('Vista superior (XY)\n'
                    'Click izq=añadir  |  Click der=seleccionar  |  '
                    'Scroll/↑↓=altura  |  Backspace=borrar  |  Enter=guardar',
                    fontsize=8)
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.4)

    # -- Configurar ax_xz --
    ax_xz.set_xlabel('Índice de waypoint')
    ax_xz.set_ylabel('Altura Z [m]')
    ax_xz.set_title('Perfil de altura')
    ax_xz.set_ylim(0, Z_MAX + 0.2)
    ax_xz.grid(True, alpha=0.4)

    if not points:
        ax_xz.set_xlim(-0.5, 1.5)
        _update_title()
        fig.canvas.draw_idle()
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    # -- Vista XY: línea + puntos coloreados por z --
    ax_xy.plot(xs, ys, 'b--', lw=1, alpha=0.4, zorder=1)

    if len(points) >= 2:
        dx = xs[-1] - xs[-2]
        dy = ys[-1] - ys[-2]
        ax_xy.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                       arrowprops=dict(arrowstyle='->', color='royalblue', lw=1.5),
                       zorder=2)

    for i, (xi, yi, zi) in enumerate(zip(xs, ys, zs)):
        color  = cmap(norm(zi))
        is_sel = (i == selected_index)
        marker = 'D' if is_sel else 'o'
        size   = 130  if is_sel else 80
        edge   = 'black'
        lw     = 2.0  if is_sel else 0.8
        ax_xy.scatter(xi, yi, c=[color], s=size, marker=marker,
                      edgecolors=edge, linewidths=lw, zorder=5)
        ax_xy.annotate(f'{i}  z={zi:.2f}m',
                       (xi, yi), textcoords='offset points',
                       xytext=(7, 5), fontsize=7.5,
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))

    # -- Perfil de altura --
    indices = list(range(len(points)))
    colors  = [cmap(norm(z)) for z in zs]
    bars    = ax_xz.bar(indices, zs, color=colors,
                        edgecolor='black', linewidth=0.8, width=0.6)

    if selected_index is not None and selected_index < len(points):
        bars[selected_index].set_linewidth(2.5)
        bars[selected_index].set_hatch('//')
        ax_xz.axhline(y=zs[selected_index], color='red',
                      linestyle='--', lw=1, alpha=0.6)

    ax_xz.set_xlim(-0.5, max(len(points), 1) + 0.5)
    for i, z in enumerate(zs):
        ax_xz.text(i, z + 0.03, f'{z:.2f}', ha='center', fontsize=8)

    _update_title()
    fig.canvas.draw_idle()


def _update_title():
    n      = len(points)
    status = f"{n} waypoint{'s' if n != 1 else ''}"
    if selected_index is not None and selected_index < len(points):
        status += (f"  |  Seleccionado: WP{selected_index} "
                   f"(z={points[selected_index][2]:.2f}m)  |  "
                   f"Scroll para cambiar altura")
    else:
        status += "  |  Clic derecho para seleccionar y editar altura"
    fig.suptitle(f"Editor de Waypoints — Crazyflie\n{status}",
                 fontsize=10, fontweight='bold')


# --------------------------------------------------
# EVENTOS
# --------------------------------------------------

def onclick(event):
    global selected_index
    if event.inaxes != ax_xy or event.xdata is None:
        return

    if event.button == 1:
        z = points[selected_index][2] if selected_index is not None else Z_DEFAULT
        points.append([event.xdata, event.ydata, z])
        selected_index = len(points) - 1
        print(f"  + WP{selected_index}: x={event.xdata:.3f}, "
              f"y={event.ydata:.3f}, z={z:.2f}")
        redraw()

    elif event.button == 3:
        if not points:
            return
        dists   = [np.hypot(p[0] - event.xdata, p[1] - event.ydata)
                   for p in points]
        closest = int(np.argmin(dists))
        if dists[closest] < 0.3:
            selected_index = closest
            print(f"  Seleccionado WP{selected_index} "
                  f"z={points[selected_index][2]:.2f}m — scroll para editar")
            redraw()


def onscroll(event):
    global selected_index
    if event.inaxes not in (ax_xy, ax_xz):
        return
    if selected_index is None or selected_index >= len(points):
        print("  Primero selecciona un waypoint con clic derecho")
        return
    delta = Z_STEP if event.button == 'up' else -Z_STEP
    old_z = points[selected_index][2]
    new_z = round(float(np.clip(old_z + delta, Z_MIN, Z_MAX)), 3)
    points[selected_index][2] = new_z
    print(f"  WP{selected_index} z: {old_z:.2f} → {new_z:.2f} m")
    redraw()


def onkey(event):
    global selected_index

    if event.key == 'enter':
        if not points:
            print("  Sin waypoints para guardar.")
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ruta     = os.path.join(base_dir, "waypoints.json")
        with open(ruta, 'w') as f:
            json.dump(points, f, indent=4)
        print(f"\n✓ Guardado: {ruta}")
        for i, p in enumerate(points):
            print(f"    WP{i}: x={p[0]:.3f}, y={p[1]:.3f}, z={p[2]:.3f}")

    elif event.key == 'backspace' and points:
        removed = points.pop()
        print(f"  - Borrado: {removed}")
        selected_index = len(points) - 1 if points else None
        redraw()

    elif event.key == 'r':
        points.clear()
        selected_index = None
        print("  Reiniciado.")
        redraw()

    elif event.key in ('up', 'down') and selected_index is not None:
        delta = Z_STEP if event.key == 'up' else -Z_STEP
        old_z = points[selected_index][2]
        new_z = round(float(np.clip(old_z + delta, Z_MIN, Z_MAX)), 3)
        points[selected_index][2] = new_z
        print(f"  WP{selected_index} z: {old_z:.2f} → {new_z:.2f} m")
        redraw()


# --------------------------------------------------
# ARRANCAR
# --------------------------------------------------
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('scroll_event',       onscroll)
fig.canvas.mpl_connect('key_press_event',    onkey)

redraw()

print("=" * 55)
print("  Editor de Waypoints — Crazyflie")
print("=" * 55)
print("  Click izq   → añadir waypoint")
print("  Click der   → seleccionar waypoint")
print("  Scroll      → cambiar altura del seleccionado")
print("  ↑ / ↓       → cambiar altura (teclado)")
print("  Backspace   → borrar último")
print("  Enter       → guardar waypoints.json")
print("  R           → reiniciar todo")
print("=" * 55)

plt.show()
