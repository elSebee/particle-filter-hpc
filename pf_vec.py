import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def run_particle_filter(N=1000, T=50, velocity=(1.0, 0.5), 
                           sigma_motion=0.5, sigma_sensor=1.0):
    """
    Ejecuta un filtro de partículas 2D.
    
    Retorna:
        estimate: estimación final (x, y)
        true_pos: posición real final (x, y)
        particles: posiciones finales de las partículas
        last_measurement: última observación (x, y)
    """
    true_pos = np.array([0.0, 0.0])
    velocity = np.array(velocity)

    particles = np.random.uniform(low=0, high=10, size=(N, 2))
    weights = np.ones(N) / N

    for t in range(T):
        true_pos += velocity + np.random.normal(0, sigma_motion, size=2)
        z = true_pos + np.random.normal(0, sigma_sensor, size=2)

        particles += velocity + np.random.normal(0, sigma_motion, size=(N, 2))
        dists = np.linalg.norm(particles - z, axis=1)
        weights = np.exp(-0.5 * (dists**2) / sigma_sensor**2)
        weights += 1e-300
        weights /= np.sum(weights)

        indices = np.random.choice(N, size=N, p=weights)
        particles = particles[indices]
        weights = np.ones(N) / N

    estimate = particles.mean(axis=0)
    return estimate, true_pos, particles, z


def plot_particles_2d(particles, estimate, true_pos, measurement):
    """
    Grafica la distribución final de partículas en 2D.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(particles[:, 0], particles[:, 1], s=2, alpha=0.3, label="Partículas")
    plt.scatter(*true_pos, color='red', label=f"Posición real ({true_pos[0]:.1f}, {true_pos[1]:.1f})")
    plt.scatter(*estimate, color='green', label=f"Estimación ({estimate[0]:.1f}, {estimate[1]:.1f})")
    plt.scatter(*measurement, color='orange', alpha=0.5, label=f"Medición ({measurement[0]:.1f}, {measurement[1]:.1f})")
    plt.title("Filtro de Partículas 2D")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def main():
    # if len(sys.argv) != 2:
    #     print("Uso: python script.py <N>")
    #     print("Donde N es el número de partículas")
    #     sys.exit(1)
    
    # try:
    #     N = int(sys.argv[1])
    #     if N <= 0:
    #         print("Error: N debe ser un número positivo")
    #         sys.exit(1)
    # except ValueError:
    #     print("Error: N debe ser un número entero")
    #     sys.exit(1)
    for N in [1000000, 5000000, 10000000]:
        print(f"Ejecutando filtro de partículas con N={N}...")
    
        start = time.time()
        estimate, true_pos, particles, measurement = run_particle_filter(N=N, T=100)
        end = time.time()
        print(f"Estimación final: {estimate}, Posición real: {true_pos}, Tiempo: {end - start:.4f} segundos")


if __name__ == "__main__":
    main()

# # Graficar si lo deseas
# plot_particles_2d(particles, estimate, true_pos, measurement)
