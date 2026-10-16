"""
Pollen 3D Forge - Generador procedural de alta fidelidad palinológica
Modelado isotrópico microrreticulado y aparato apertural diferenciado.
"""

import math
import struct
import numpy as np
import yaml


def load_pollen_config(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_isotropic_dirs(n_dirs: int = 48, seed: int = 42) -> np.ndarray:
    """Genera vectores unitarios distribuidos uniformemente (esfera de Fibonacci)
    para evitar cuadrículas artificiales en los polos y el ecuador."""
    rng = np.random.default_rng(seed)
    indices = np.arange(0, n_dirs, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_dirs)
    theta = np.pi * (1 + 5**0.5) * indices
    
    dirs = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=-1)
    
    phases = rng.uniform(0, 2 * np.pi, size=n_dirs)
    return dirs, phases


def build_pollen_mesh(cfg: dict, n_lat: int = 220, n_lon: int = 220):
    # 1. Dimensiones biométricas base (en micrómetros)
    p_axis = float(cfg["biometrics"]["polar_axis_P"])      # 19.2 um
    e_axis = float(cfg["biometrics"]["equatorial_axis_E"]) # 22.2 um
    c = p_axis / 2.0
    a = b = e_axis / 2.0

    # Malla esférica parametrizada
    phi = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    theta = np.linspace(-np.pi, np.pi, n_lon, endpoint=False)
    Phi, Theta = np.meshgrid(phi, theta)

    # Coordenadas del elipsoide suboblado base
    X = a * np.cos(Phi) * np.cos(Theta)
    Y = b * np.cos(Phi) * np.sin(Theta)
    Z = c * np.sin(Phi)

    # Normales unitarias hacia el exterior para deformación radial limpia
    R_norm = np.sqrt(X**2 + Y**2 + Z**2) + 1e-9
    Nx, Ny, Nz = X / R_norm, Y / R_norm, Z / R_norm

    # 2. Aparato apertural: 3 poros ecuatoriales (Zonoaperturado)
    n_pores = int(cfg["apertures"]["count"])
    pore_diam = float(cfg["apertures"]["diameter_um"])
    pore_radius = pore_diam / 2.0  # 1.0 um
    pore_angles = [2 * np.pi * k / n_pores for k in range(n_pores)]

    aperture_displacement = np.zeros_like(X)
    sculpture_mask = np.ones_like(X)  # Atenúa el retículo dentro del poro

    for p_ang in pore_angles:
        # Posición 3D del centro de cada poro en el ecuador
        px = a * np.cos(p_ang)
        py = b * np.sin(p_ang)
        pz = 0.0
        
        # Distancia métrica real sobre la superficie hacia el centro del poro
        dist = np.sqrt((X - px)**2 + (Y - py)**2 + (Z - pz)**2)
        
        # Ligero engrosamiento anular (margo suave) alrededor del poro
        rim = 0.22 * np.exp(-((dist - pore_radius * 1.35)**2) / (2 * (0.45)**2))
        
        # Depresión circular cóncava del poro (invaginación de la exina)
        pit = -1.35 / (1.0 + np.exp((dist - pore_radius * 0.95) / 0.18))
        
        aperture_displacement += (rim + pit)
        
        # Suavizar el microrretículo dentro del lumen del poro
        local_mask = 1.0 / (1.0 + np.exp(-(dist - pore_radius * 1.15) / 0.2))
        sculpture_mask = np.minimum(sculpture_mask, local_mask)

    # 3. Escultura microrreticulada homobrocada orgánica (Sin efecto cuadrícula)
    dirs, phases = generate_isotropic_dirs(n_dirs=36, seed=101)
    spatial_freq = 2.6  # Controla el diámetro de los lúmenes (< 1 um)
    
    # Campo celular isotrópico evaluado directamente en coordenadas 3D (X, Y, Z)
    field = np.zeros_like(X)
    for (dx, dy, dz), phase in zip(dirs, phases):
        proj = X * dx + Y * dy + Z * dz
        field += np.cos(spatial_freq * proj + phase)
    
    field = field / np.sqrt(len(dirs))
    
    # Transformación no lineal para crear muros (muri) conectados y lúmenes (lumina) cóncavos
    reticulum = 1.0 - np.abs(field)
    reticulum = np.clip(reticulum, 0.0, 1.0) ** 1.6
    
    # Relieve micrométrico de la sexina (0.38 um de profundidad de lumen)
    exine_relief = (reticulum - 0.5) * 0.42 * sculpture_mask

    # 4. Desplazamiento total combinado a lo largo de la normal
    total_disp = aperture_displacement + exine_relief
    X_final = X + Nx * total_disp
    Y_final = Y + Ny * total_disp
    Z_final = Z + Nz * total_disp

    # 5. Triangulación cerrada de la malla (incluyendo cierre de costuras)
    triangles = []
    for i in range(n_lon):
        i_next = (i + 1) % n_lon
        for j in range(n_lat - 1):
            p1 = (X_final[i, j], Y_final[i, j], Z_final[i, j])
            p2 = (X_final[i_next, j], Y_final[i_next, j], Z_final[i_next, j])
            p3 = (X_final[i_next, j + 1], Y_final[i_next, j + 1], Z_final[i_next, j + 1])
            p4 = (X_final[i, j + 1], Y_final[i, j + 1], Z_final[i, j + 1])

            triangles.append((p1, p2, p3))
            triangles.append((p1, p3, p4))

    return triangles


def export_binary_stl(triangles: list, filename: str):
    header = b"Pollen 3D Forge - Anthurium nigrescens STL" + b" " * 38
    with open(filename, "wb") as f:
        f.write(header[:80])
        f.write(struct.pack("<I", len(triangles)))
        for p1, p2, p3 in triangles:
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p3) - np.array(p1)
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-12:
                normal = normal / norm_len
            else:
                normal = np.array([0.0, 0.0, 0.0])

            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *p1))
            f.write(struct.pack("<3f", *p2))
            f.write(struct.pack("<3f", *p3))
            f.write(struct.pack("<H", 0))


if __name__ == "__main__":
    config_file = "configs/anthurium_nigrescens.yaml"
    cfg = load_pollen_config(config_file)
    tris = build_pollen_mesh(cfg)
    out_file = "anthurium_nigrescens.stl"
    export_binary_stl(tris, out_file)
    print(f"[OK] Modelo de alta resolucion exportado con exito: {out_file} ({len(tris)} triangulos)")
