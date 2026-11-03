import numpy as np
from scipy.interpolate import splprep, splev


# ----------------------------------------------------------------------
# Utilidades de parametrización por longitud de arco
# ----------------------------------------------------------------------

def remove_duplicates(path, tol=1e-6):
    """Elimina puntos repetidos para que la longitud de arco sea creciente."""
    P = np.asarray(path, dtype=float)
    if len(P) < 2:
        return P
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > tol:
            keep.append(i)
    if len(keep) < 2:
        return P[:2].copy()
    return P[keep]


def resample_uniform(path, spacing=0.02):
    """Reparametriza una polilínea 3D a paso de arco constante [m]."""
    P = remove_duplicates(path)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s   = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])

    if total < 1e-9:
        return P.copy()

    n     = max(int(np.ceil(total / spacing)) + 1, 3)
    s_new = np.linspace(0.0, total, n)

    out = np.empty((n, 3))
    for k in range(3):
        out[:, k] = np.interp(s_new, s, P[:, k])
    return out


def path_geometry(path, kappa_smooth_m=0.12, kappa_max=25.0):
    """
    Geometría de una polilínea 3D.

    Retorna:
        s      : (n,)  longitud de arco acumulada
        T      : (n,3) tangente unitaria 3D
        kappa  : (n,)  curvatura en el plano XY [1/m]

    La curvatura se deriva respecto a s y se promedia sobre kappa_smooth_m
    para reducir el ruido de la segunda derivada.
    """
    P   = remove_duplicates(path)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s   = np.concatenate([[0.0], np.cumsum(seg)])

    # Tangente unitaria 3D
    d = np.gradient(P, s, axis=0)
    T = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)

    # Curvatura XY
    dx, dy = d[:, 0], d[:, 1]
    ddx    = np.gradient(dx, s)
    ddy    = np.gradient(dy, s)
    den    = (dx * dx + dy * dy) ** 1.5
    kappa  = np.abs(dx * ddy - dy * ddx) / np.maximum(den, 1e-9)
    kappa  = np.clip(kappa, 0.0, kappa_max)

    # Suavizado de la curvatura
    ds = float(np.median(np.diff(s))) if len(s) > 2 else 1.0
    w  = max(3, int(round(kappa_smooth_m / max(ds, 1e-6))) | 1)
    if w < len(kappa):
        pad   = w // 2
        kappa = np.convolve(np.pad(kappa, pad, mode='edge'),
                            np.ones(w) / w, mode='valid')

    return s, T, kappa


def compute_speed_profile(s, kappa, T,
                          v_max=0.55, v_min=0.12,
                          a_lat=0.55, a_tan=0.40,
                          vz_max=0.30,
                          v_start=0.12, v_end=0.10):
    """
    Perfil de rapidez a lo largo de la trayectoria.

    Limita la rapidez por aceleración lateral (v <= sqrt(a_lat / k)) y por
    velocidad vertical (v <= vz_max / |dz/ds|). Después aplica una pasada
    hacia atrás y otra hacia adelante para que la aceleración tangencial
    no supere a_tan, de modo que el frenado empiece antes de cada curva.
    """
    s     = np.asarray(s, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    T     = np.asarray(T, dtype=float)
    n     = len(s)

    v = np.full(n, float(v_max))

    # Límite por aceleración lateral
    v = np.minimum(v, np.sqrt(a_lat / np.maximum(kappa, 1e-6)))

    # Límite por velocidad vertical
    dzds = np.abs(T[:, 2])
    v    = np.minimum(v, vz_max / np.maximum(dzds, 1e-6))

    v = np.clip(v, v_min, v_max)

    # Extremos
    v[0]  = min(v[0],  float(v_start))
    v[-1] = min(v[-1], float(v_end))

    # Pasada hacia atrás (frenado anticipado)
    for i in range(n - 2, -1, -1):
        ds   = s[i + 1] - s[i]
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_tan * ds))

    # Pasada hacia adelante (aceleración limitada)
    for i in range(1, n):
        ds   = s[i] - s[i - 1]
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_tan * ds))

    return v


class TrajectoryGenerator:
    """
    Genera trayectorias a partir de waypoints.

    Métodos: 'catmull_rom', 'cubic', 'quadratic' y 'linear'.
    """

    def __init__(self, waypoints, fixed_height=None):
        self.waypoints    = np.array(waypoints, dtype=float)
        self.fixed_height = fixed_height
        self.path         = None

        if fixed_height is not None:
            self.waypoints[:, 2] = fixed_height

    # ------------------------------------------------------------------
    # Generación
    # ------------------------------------------------------------------

    def generate_spline(self, num_points=50, smoothing=0,
                        method='catmull_rom', tension=0.5):
        """
        num_points : puntos de la trayectoria
        smoothing  : solo para cubic/quadratic
        tension    : solo para catmull_rom (0.5 = cardinal estándar)
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

        if self.fixed_height is not None:
            self.path[:, 2] = self.fixed_height

        return self.path

    def generate_uniform(self, spacing=0.02, method='catmull_rom',
                         tension=0.5, dense_points=4000):
        """
        Genera la trayectoria densa y la remuestrea a paso de arco
        constante (spacing, en metros).
        """
        dense     = self.generate_spline(dense_points, method=method,
                                         tension=tension)
        self.path = resample_uniform(dense, spacing)

        if self.fixed_height is not None:
            self.path[:, 2] = self.fixed_height

        return self.path

    # ------------------------------------------------------------------
    # Catmull-Rom
    # ------------------------------------------------------------------

    def _catmull_rom(self, num_points, tension=0.5):
        """Spline Catmull-Rom que pasa exactamente por los waypoints."""
        pts = self.waypoints.copy()

        # Puntos fantasma en los extremos
        p_start = pts[0]  + (pts[0]  - pts[1])
        p_end   = pts[-1] + (pts[-1] - pts[-2])
        pts     = np.vstack([p_start, pts, p_end])

        n_segments   = len(pts) - 3
        pts_per_seg  = max(2, num_points // n_segments)
        total_points = pts_per_seg * n_segments

        result = []
        for i in range(n_segments):
            p0 = pts[i]
            p1 = pts[i + 1]
            p2 = pts[i + 2]
            p3 = pts[i + 3]

            # El último punto solo se incluye en el último segmento
            end = pts_per_seg if i < n_segments - 1 else pts_per_seg + 1
            for j in range(end):
                t = j / pts_per_seg

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
    # Lineal
    # ------------------------------------------------------------------

    def _linear(self, num_points):
        """Interpolación lineal entre waypoints."""
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
    # Spline de scipy (cúbica o cuadrática)
    # ------------------------------------------------------------------

    def _scipy_spline(self, num_points, smoothing, k):
        pts      = self.waypoints.T
        k        = min(k, len(pts[0]) - 1)
        tck, u   = splprep(pts, s=smoothing, k=k)
        u_fine   = np.linspace(0, 1, num_points)
        x, y, z  = splev(u_fine, tck)
        return np.vstack((x, y, z)).T

    # ------------------------------------------------------------------
    # Utilidades
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
