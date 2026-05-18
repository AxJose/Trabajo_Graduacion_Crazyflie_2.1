import numpy as np
from scipy.interpolate import splprep, splev


class TrajectoryGenerator:
    """
    Genera trayectorias suaves a partir de waypoints.

    Métodos disponibles (pasar en generate()):
      'catmull_rom'  — pasa exactamente por los waypoints, sin overshoot (recomendado)
      'cubic'        — spline cúbica scipy, muy suave pero con overshoot
      'quadratic'    — spline cuadrática, menos overshoot que cubic
      'linear'       — líneas rectas entre waypoints, sin curvas
    """

    def __init__(self, waypoints, fixed_height=None):
        self.waypoints    = np.array(waypoints, dtype=float)
        self.fixed_height = fixed_height
        self.path         = None

        if fixed_height is not None:
            self.waypoints[:, 2] = fixed_height

    # ------------------------------------------------------------------
    # MÉTODO PRINCIPAL — llama al método elegido
    # ------------------------------------------------------------------

    def generate_spline(self, num_points=50, smoothing=0,
                        method='catmull_rom', tension=0.5):
        """
        num_points : puntos interpolados en la trayectoria final
        smoothing  : solo para métodos scipy (cubic/quadratic)
        method     : 'catmull_rom' | 'cubic' | 'quadratic' | 'linear'
        tension    : solo para catmull_rom (0=relajada, 1=líneas rectas)
                     0.5 es el valor estándar
        """
        if method == 'catmull_rom':
            self.path = self._catmull_rom(num_points, tension)
        elif method == 'linear':
            self.path = self._linear(num_points)
        elif method in ('cubic', 'quadratic'):
            k = 3 if method == 'cubic' else 2
            self.path = self._scipy_spline(num_points, smoothing, k)
        else:
            raise ValueError(f"Método desconocido: {method}. "
                             f"Usa: catmull_rom, cubic, quadratic, linear")

        # Forzar altura constante si se pidió
        if self.fixed_height is not None:
            self.path[:, 2] = self.fixed_height

        return self.path

    # ------------------------------------------------------------------
    # CATMULL-ROM — pasa exactamente por los waypoints
    # ------------------------------------------------------------------

    def _catmull_rom(self, num_points, tension=0.5):
        """
        Catmull-Rom spline.

        Propiedades:
        - Pasa EXACTAMENTE por cada waypoint (sin overshoot)
        - Curvas suaves y naturales en los tramos
        - tension=0.0 → curvas más abiertas y suaves
        - tension=0.5 → estándar (Cardinal spline)
        - tension=1.0 → líneas rectas (equivale a linear)

        Añade puntos fantasma al inicio y al final para que la
        curva también sea suave en el primer y último waypoint.
        """
        pts = self.waypoints.copy()

        # Puntos fantasma: extender la trayectoria en los extremos
        # para que el primer y último waypoint también sean suaves
        p_start = pts[0]  + (pts[0]  - pts[1])
        p_end   = pts[-1] + (pts[-1] - pts[-2])
        pts     = np.vstack([p_start, pts, p_end])

        # Puntos por segmento (entre par de waypoints consecutivos)
        n_segments   = len(pts) - 3
        pts_per_seg  = max(2, num_points // n_segments)
        total_points = pts_per_seg * n_segments

        result = []
        for i in range(n_segments):
            p0 = pts[i]
            p1 = pts[i + 1]
            p2 = pts[i + 2]
            p3 = pts[i + 3]

            # No repetir el último punto (evita duplicados en la unión)
            end = pts_per_seg if i < n_segments - 1 else pts_per_seg + 1
            for j in range(end):
                t = j / pts_per_seg

                # Fórmula Catmull-Rom (Cardinal spline con alpha=tension)
                t2 = t * t
                t3 = t2 * t

                q = (
                    (-tension * t3 + 2*tension * t2 - tension * t) * p0
                  + ((2 - tension) * t3 + (tension - 3) * t2 + 1) * p1
                  + ((tension - 2) * t3 + (3 - 2*tension) * t2 + tension * t) * p2
                  + (tension * t3 - tension * t2) * p3
                )
                result.append(q)

        return np.array(result)

    # ------------------------------------------------------------------
    # LINEAR — líneas rectas entre waypoints
    # ------------------------------------------------------------------

    def _linear(self, num_points):
        """Interpolación lineal simple entre waypoints."""
        pts      = self.waypoints
        n_segs   = len(pts) - 1
        pps      = max(2, num_points // n_segs)
        result   = []

        for i in range(n_segs):
            end = pps if i < n_segs - 1 else pps + 1
            for j in range(end):
                t = j / pps
                result.append((1 - t) * pts[i] + t * pts[i + 1])

        return np.array(result)

    # ------------------------------------------------------------------
    # SCIPY SPLINE — cúbica o cuadrática
    # ------------------------------------------------------------------

    def _scipy_spline(self, num_points, smoothing, k):
        pts      = self.waypoints.T
        k        = min(k, len(pts[0]) - 1)
        tck, u   = splprep(pts, s=smoothing, k=k)
        u_fine   = np.linspace(0, 1, num_points)
        x, y, z  = splev(u_fine, tck)
        return np.vstack((x, y, z)).T

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------

    def get_point(self, index):
        if self.path is None:
            raise RuntimeError("Llama generate_spline() primero.")
        return self.path[index]

    def get_full_path(self):
        if self.path is None:
            raise RuntimeError("Llama generate_spline() primero.")
        return self.path

    def total_length(self):
        """Longitud 3D total de la trayectoria."""
        if self.path is None:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(self.path, axis=0), axis=1)))

    def total_length_xy(self):
        """Longitud solo en el plano XY."""
        if self.path is None:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(self.path[:, :2], axis=0), axis=1)))
