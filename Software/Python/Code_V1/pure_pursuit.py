"""
pure_pursuit.py — Seguimiento de trayectorias 3D para un vehículo
holonómico (Crazyflie comandado con velocidad en el marco del mundo).

Base: Coulter, R.C. (1992). Implementation of the Pure Pursuit Path
Tracking Algorithm. Carnegie Mellon University.

Diferencias respecto a la versión clásica:
  - El punto de anticipación se obtiene avanzando L metros sobre la curva
    desde la proyección del dron (longitud de arco), no intersectando un
    círculo de radio L con la trayectoria.
  - El punto más cercano se calcula proyectando sobre los segmentos, no
    eligiendo el vértice más cercano.
  - La tangente y la curvatura se precalculan respecto a la longitud de
    arco.
  - El avance de la proyección por ciclo está acotado.
"""

import numpy as np

from trajectory import remove_duplicates, path_geometry


class PurePursuit:

    def __init__(self,
                 lookahead_distance=0.25,
                 min_lookahead=0.12,
                 max_lookahead=0.45,
                 lookahead_time=0.40,
                 k_curve=0.0,
                 goal_tolerance=0.10,
                 search_window=0.50):
        """
        lookahead_distance : [m] valor inicial de L
        min/max_lookahead  : [m] límites de L
        lookahead_time     : [s] L = lookahead_time * v
        k_curve            : acorta L en curva (0 = desactivado)
        goal_tolerance     : [m] distancia al final para dar por terminado
        search_window      : [m] ventana de búsqueda hacia adelante
        """
        self.L       = float(lookahead_distance)
        self.L_min   = float(min_lookahead)
        self.L_max   = float(max_lookahead)
        self.T_look  = float(lookahead_time)
        self.k_curve = float(k_curve)

        self.goal_tolerance = float(goal_tolerance)
        self.search_window  = float(search_window)

        self.path   = None
        self.s      = None      # longitud de arco acumulada
        self.T      = None      # tangente unitaria 3D
        self.kappa  = None      # curvatura XY
        self.length = 0.0

        self.index        = 0            # índice del segmento actual
        self.t_seg        = 0.0          # parámetro dentro del segmento
        self.s_proj       = 0.0          # arco recorrido en la proyección
        self.proj         = None         # punto proyectado (3D)
        self.e_lat        = np.zeros(2)  # vector proyección - dron  [m]
        self.e_lat_signed = 0.0          # error lateral con signo   [m]
        self._finished    = False

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------

    def set_path(self, path):
        self.path = remove_duplicates(path)
        self.s, self.T, self.kappa = path_geometry(self.path)
        self.length = float(self.s[-1])
        self.reset()

        paso = self.length / max(len(self.path) - 1, 1)
        print(f"[PurePursuit] Trayectoria: {len(self.path)} puntos | "
              f"{self.length:.2f} m | paso medio {paso*100:.1f} cm | "
              f"kappa max {self.kappa.max():.2f} 1/m "
              f"(radio min {1.0/max(self.kappa.max(), 1e-6):.2f} m)")

    def reset(self):
        self.index        = 0
        self.t_seg        = 0.0
        self.s_proj       = 0.0
        self.proj         = self.path[0].copy() if self.path is not None else None
        self.e_lat        = np.zeros(2)
        self.e_lat_signed = 0.0
        self._finished    = False

    def set_lookahead(self, distance):
        self.L = float(np.clip(distance, self.L_min, self.L_max))

    def adaptive_lookahead(self, speed, curvature=None):
        """L proporcional a la velocidad, acotado a [L_min, L_max]."""
        L = self.T_look * max(float(speed), 1e-3)
        if curvature is not None and self.k_curve > 0.0:
            L = L / (1.0 + self.k_curve * float(curvature))
        self.L = float(np.clip(L, self.L_min, self.L_max))
        return self.L

    def adaptive_lookahead_v2(self, speed, curvature=None):
        return self.adaptive_lookahead(speed, curvature)

    @property
    def finished(self):
        return self._finished

    # ------------------------------------------------------------------
    # Paso de control (llamar una vez por ciclo, antes de los getters)
    # ------------------------------------------------------------------

    def step(self, x, y, max_advance=None):
        """
        Proyecta el dron sobre la trayectoria y actualiza índice, arco
        recorrido, punto proyectado y error lateral.

        max_advance : [m] avance máximo de la proyección en este ciclo
        """
        if self.path is None:
            return

        n = len(self.path)
        if self.index > n - 2:
            self.index = n - 2

        pos    = np.array([float(x), float(y)])
        window = self.search_window if max_advance is None else float(max_advance)

        i_hi = int(np.searchsorted(self.s, self.s[self.index] + max(window, 1e-3),
                                   side='right'))
        i_hi = min(max(i_hi, self.index + 2), n)

        A   = self.path[self.index:i_hi - 1, :2]
        B   = self.path[self.index + 1:i_hi, :2]
        seg = B - A
        w   = pos - A

        denom = np.einsum('ij,ij->i', seg, seg)
        num   = np.einsum('ij,ij->i', w, seg)
        t     = np.clip(np.where(denom > 1e-12, num / np.maximum(denom, 1e-12), 0.0),
                        0.0, 1.0)

        proj_xy = A + t[:, None] * seg
        j       = int(np.argmin(np.linalg.norm(pos - proj_xy, axis=1)))

        self.index  = self.index + j
        self.t_seg  = float(t[j])
        i           = self.index
        self.s_proj = float(self.s[i] + self.t_seg * (self.s[i + 1] - self.s[i]))
        self.proj   = self.path[i] + self.t_seg * (self.path[i + 1] - self.path[i])

        # Vector del dron hacia su proyección
        self.e_lat = self.proj[:2] - pos

        Tp = self.state_at(self.s_proj)[1]
        self.e_lat_signed = float(Tp[0] * self.e_lat[1] - Tp[1] * self.e_lat[0])

        if (self.length - self.s_proj) <= self.goal_tolerance:
            if not self._finished:
                print("[PurePursuit] Trayectoria completada.")
            self._finished = True

    # ------------------------------------------------------------------
    # Consultas sobre la trayectoria
    # ------------------------------------------------------------------

    def state_at(self, s_query):
        """Punto, tangente unitaria y curvatura a una longitud de arco dada."""
        s_query = float(np.clip(s_query, 0.0, self.length))

        i  = int(np.searchsorted(self.s, s_query, side='right')) - 1
        i  = int(np.clip(i, 0, len(self.s) - 2))
        ds = self.s[i + 1] - self.s[i]
        t  = 0.0 if ds < 1e-12 else (s_query - self.s[i]) / ds

        p = self.path[i] + t * (self.path[i + 1] - self.path[i])

        T   = self.T[i] + t * (self.T[i + 1] - self.T[i])
        nrm = np.linalg.norm(T)
        T   = T / nrm if nrm > 1e-9 else self.T[i]

        k = float(self.kappa[i] + t * (self.kappa[i + 1] - self.kappa[i]))
        return p, T, k

    def lookahead_state(self):
        """Punto, tangente y curvatura L metros adelante de la proyección."""
        return self.state_at(self.s_proj + self.L)

    def curvature_ahead(self, distance=0.0):
        """Curvatura a `distance` metros por delante de la proyección."""
        return self.state_at(self.s_proj + distance)[2]

    def max_curvature_ahead(self, distance):
        """Curvatura máxima en los próximos `distance` metros."""
        if self.path is None:
            return 0.0
        i1 = int(np.searchsorted(self.s, self.s_proj + distance, side='right'))
        i1 = min(max(i1, self.index + 1), len(self.kappa))
        return float(np.max(self.kappa[self.index:i1]))

    def compute_local_curvature(self):
        return self.curvature_ahead(0.0)

    def get_desired_height(self, x=None, y=None):
        """Altura de la trayectoria en la proyección actual."""
        if self.proj is None:
            return None
        return float(self.proj[2])

    def get_progress(self):
        if self.path is None or self.length < 1e-9:
            return 0.0
        return 100.0 * self.s_proj / self.length

    def distance_to_end(self):
        return max(self.length - self.s_proj, 0.0)

    # ------------------------------------------------------------------
    # Pure Pursuit clásico (apunta directamente al punto de anticipación)
    # ------------------------------------------------------------------

    def get_velocity_command(self, x, y, yaw, speed):
        """Velocidad de magnitud `speed` en la dirección dron → anticipación."""
        if self.path is None or self._finished:
            return 0.0, 0.0, yaw, None

        self.step(x, y)
        if self._finished:
            return 0.0, 0.0, yaw, None

        target = self.lookahead_state()[0]

        dx = target[0] - x
        dy = target[1] - y
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0, yaw, target

        vx = speed * dx / dist
        vy = speed * dy / dist
        return vx, vy, np.arctan2(dy, dx), target

    def get_lookahead_point(self, x, y):
        if self.path is None or self._finished:
            return None
        self.step(x, y)
        if self._finished:
            return None
        return self.lookahead_state()[0]
