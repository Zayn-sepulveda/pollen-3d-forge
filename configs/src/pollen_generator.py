"""
Pollen 3D Forge - Generador procedural de granos de polen
Basado en datos morfométricos estandarizados (Erdtman / Punt).
"""

import math
import struct
import numpy as np
import yaml


def load_pollen_config(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pollen_mesh(cfg: dict, n_lat: int = 100, n_lon: int = 100):
    # 1. Semiejes del elipsoide base (en micrómetros)
    p_axis = cfg["biometrics"]["polar_axis"]
    e_axis = cfg["biometrics"]["equatorial_axis"]
    c = p_axis / 2.0  # Semieje vertical (Z)
    a = b = e_axis / 2.0  # Semiejes ecuatoriales (X, Y)

    # 2. Malla paramétrica latitud / longitud
    phi = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    theta = np.linspace(-np.pi, np.pi, n_lon)
    Phi, Theta = np.meshgrid(phi, theta)

    # Coordenadas base del elipsoide
    X = a * np.cos(Phi) * np.cos(Theta)
    Y = b * np.cos(Phi) * np.sin(Theta)
    Z = c * np.sin(Phi)

    # 3. Modelado de aperturas (Poros ecuatoriales de Anthurium)
    if cfg["apertures"]["type"] == "porate":
        pore_count = cfg["apertures"]["count"]
        pore_radius = cfg["apertures"]["pore_diameter"] / 2.0
        pore_angles = [2 * np.pi * k / pore_count for k in range(pore_count)]

        for p_angle in pore_angles:
            # Distancia angular sobre la esfera respecto al centro del poro
            d_theta = np.arctan2(np.sin(Theta - p_angle), np.cos(Theta - p_angle))
            d_phi = Phi
            ang_dist = np.sqrt(d_theta**2 + d_phi**2)
            metric_dist = ang_dist * ((a + b) / 2.0)

            # Depresión gaussiana del poro hacia la intina/nexina
            invagination = -1.5 * np.exp(-(metric_dist**2) / (2 * (pore_radius * 0.8)**2))

            norm = np.sqrt(X**2 + Y**2 + Z**2) + 1e-8
            X += (X / norm) * invagination
            Y += (Y / norm) * invagination
            Z += (Z / norm) * invagination

    # 4. Microescultura superficial (microrreticulado homogéneo)
    if cfg["sculpture"]["type"] == "microreticulate":
        relief_h = cfg["sculpture"].get("relief_height", 0.3)
        # Frecuencia espacial del retículo
        freq = 35.0
        reticulum = (np.sin(freq * Theta) * np.cos(freq * Phi)) ** 2
        ret_displacement = reticulum * relief_h

        norm = np.sqrt(X**2 + Y**2 + Z**2) + 1e-8
        X += (X / norm) * ret_displacement
        Y += (Y / norm) * ret_displacement
        Z += (Z / norm) * ret_displacement

    return X, Y, Z


def export_to_binary_stl(X, Y, Z, filename: str):
    """Exporta las coordenadas de superficie a una malla STL binaria."""
    n_lon, n_lat = X.shape
    triangles = []

    for i in range(n_lon - 1):
        for j in range(n_lat - 1):
            p1 = np.array([X[i, j], Y[i, j], Z[i, j]])
            p2 = np.array([X[i + 1, j], Y[i + 1, j], Z[i + 1, j]])
            p3 = np.array([X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]])
            p4 = np.array([X[i, j + 1], Y[i, j + 1], Z[i, j + 1]])

            triangles.append((p1, p2, p3))
            triangles.append((p1, p3, p4))

    # Escritura del archivo STL en formato binario
    with open(filename, "wb") as f:
        f.write(b"\0" * 80)  # Encabezado de 80 bytes
        f.write(struct.pack("<I", len(triangles)))

        for v1, v2, v3 in triangles:
            # Vector normal por producto cruz
            normal = np.cross(v2 - v1, v3 - v1)
            norm_val = np.linalg.norm(normal)
            if norm_val > 1e-9:
                normal /= norm_val
            else:
                normal = np.array([0.0, 0.0, 0.0])

            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<3f", *v3))
            f.write(struct.pack("<H", 0))

    print(f"[OK] Modelo exportado con exito: {filename}")


if __name__ == "__main__":
    config = load_pollen_config("configs/anthurium_nigrescens.yaml")
    X, Y, Z = build_pollen_mesh(config)
    export_to_binary_stl(X, Y, Z, "anthurium_nigrescens.stl")
