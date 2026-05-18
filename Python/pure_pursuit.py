"""
pure_pursuit.py — Algoritmo Pure Pursuit para seguimiento de trayectorias.

Referencia: Coulter, R.C. (1992). Implementation of the Pure Pursuit
            Path Tracking Algorithm. Carnegie Mellon University.

"""

import numpy as np


class PurePursuit:

    def __init__(self, lookahead_distance=0.3, min_lookahead=0.15,
                 max_lookahead=0.6):
        self.L         = lookahead_distance
        self.L_min     = min_lookahead
        self.L_max     = max_lookahead
        self.path      = None
        self.index     = 0
        self._finished = False

    # ------------------------------------------------------------------
    # CONFIGURACIÓN
    # ------------------------------------------------------------------

    def set_path(self, path):
        self.path      = np.array(path, dtype=float)
        self.index     = 0
        self._finished = False
        print(f"[PurePursuit] Trayectoria cargada: {len(self.path)} puntos")

    def set_lookahead(self, distance):
        self.L = float(np.clip(distance, self.L_min, self.L_max))

    def adaptive_lookahead(self, speed, curvature=None):
        k_speed = 0.6
        L_speed = k_speed * speed

        if curvature is not None:
            k_curve = 5.0
            L_curve = L_speed / (1 + k_curve * curvature)
        else:
            L_curve = L_speed

        self.L = float(np.clip(L_curve, self.L_min, self.L_max))
        return self.L

    @property
    def finished(self):
        return self._finished

    # ------------------------------------------------------------------
    # PUNTO MÁS CERCANO
    # Solo avanza hacia adelante. Llamar UNA VEZ por ciclo de control
    # antes de get_velocity_command y get_desired_height.
    # ------------------------------------------------------------------

    def _update_closest_index(self, x, y):
        if self.path is None:
            return
        search_window = min(self.index + 30, len(self.path))
        segment       = self.path[self.index:search_window, :2]
        pos           = np.array([x, y])
        dists         = np.linalg.norm(segment - pos, axis=1)
        local_idx     = int(np.argmin(dists))
        self.index    = self.index + local_idx

    # ------------------------------------------------------------------
    # INTERSECCIÓN CÍRCULO-SEGMENTO
    # Núcleo del Pure Pursuit correcto según Coulter (1992).
    #
    # Para cada segmento (p1 → p2) desde self.index hacia adelante,
    # calcula analíticamente si el círculo de radio L centrado en el
    # dron intersecta el segmento. Si hay dos intersecciones, toma la
    # más cercana a p2 (más adelante en la trayectoria).
    # ------------------------------------------------------------------

    def _circle_segment_intersection(self, p1, p2, cx, cy):
        """
        Retorna el parámetro t en [0,1] del punto de intersección
        del círculo (cx,cy,L) con el segmento p1→p2 más cercano a p2,
        o None si no hay intersección dentro del segmento.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        fx = p1[0] - cx
        fy = p1[1] - cy

        a = dx*dx + dy*dy
        if a < 1e-12:
            return None

        b = 2.0 * (fx*dx + fy*dy)
        c = fx*fx + fy*fy - self.L*self.L

        disc = b*b - 4.0*a*c
        if disc < 0:
            return None

        sq = disc ** 0.5
        t1 = (-b - sq) / (2.0*a)
        t2 = (-b + sq) / (2.0*a)

        # Preferir t2 (más hacia p2 = más adelante en la spline)
        for t in (t2, t1):
            if 0.0 <= t <= 1.0:
                return t

        return None

    # ------------------------------------------------------------------
    # LOOKAHEAD POINT
    # ------------------------------------------------------------------

    def get_lookahead_point(self, x, y):
        """
        Encuentra el punto lookahead usando intersección geométrica.
        Llama a _update_closest_index internamente — no llamar desde
        fuera para evitar doble actualización del índice.
        """
        if self.path is None or self._finished:
            return None

        self._update_closest_index(x, y)

        # Buscar intersección en cada segmento desde self.index
        for i in range(self.index, len(self.path) - 1):
            p1 = self.path[i,   :2]
            p2 = self.path[i+1, :2]

            t = self._circle_segment_intersection(p1, p2, x, y)
            if t is not None:
                # Interpolar la posición XY y también la Z
                point_xy = p1 + t * (p2 - p1[:2])  # 2D
                z = self.path[i, 2] + t * (self.path[i+1, 2] - self.path[i, 2])
                return np.array([point_xy[0], point_xy[1], z])

        # Sin intersección: comprobar si estamos cerca del final
        dist_to_end = np.linalg.norm(self.path[-1, :2] - np.array([x, y]))
        if dist_to_end < self.L * 0.5:
            self._finished = True
            print("[PurePursuit] Trayectoria completada.")
            return None

        # Fallback: apuntar al último punto
        return self.path[-1]

    # ------------------------------------------------------------------
    # COMANDO DE VELOCIDAD
    # ------------------------------------------------------------------

    def get_velocity_command(self, x, y, yaw, speed):
        """
        Calcula vx, vy en frame del mundo.
        Internamente llama get_lookahead_point (que llama
        _update_closest_index). No llamar _update_closest_index
        de forma separada en el mismo ciclo.
        """
        target = self.get_lookahead_point(x, y)

        if target is None:
            return 0.0, 0.0, yaw, None

        dx = target[0] - x
        dy = target[1] - y

        dist = np.linalg.norm([dx, dy])
        if dist < 1e-6:
            return 0.0, 0.0, yaw, target

        vx = speed * dx / dist
        vy = speed * dy / dist

        yaw_desired = np.arctan2(dy, dx)

        return vx, vy, yaw_desired, target

    # ------------------------------------------------------------------
    # ALTURA DESEADA
    # Usa self.index ya actualizado por get_velocity_command.
    # No llama _update_closest_index para evitar doble avance.
    # ------------------------------------------------------------------

    def get_desired_height(self, x, y):
        """
        Interpola Z usando self.index actualizado por
        get_velocity_command. Llamar DESPUÉS de get_velocity_command
        en el mismo ciclo de control.
        """
        if self.path is None:
            return None

        i = self.index
        if i >= len(self.path) - 1:
            return float(self.path[-1, 2])

        p1 = self.path[i]
        p2 = self.path[i + 1]

        pos     = np.array([x, y])
        seg     = p2[:2] - p1[:2]
        seg_len = np.linalg.norm(seg)

        if seg_len < 1e-6:
            return float(p1[2])

        t = np.dot(pos - p1[:2], seg) / (seg_len ** 2)
        t = float(np.clip(t, 0.0, 1.0))

        return float(p1[2] + t * (p2[2] - p1[2]))

    # ------------------------------------------------------------------
    # ESTADO
    # ------------------------------------------------------------------

    def get_progress(self):
        if self.path is None:
            return 0.0
        return 100.0 * self.index / max(len(self.path) - 1, 1)

    def reset(self):
        self.index     = 0
        self._finished = False
