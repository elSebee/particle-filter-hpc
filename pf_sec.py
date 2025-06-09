import random
import math
import time
import sys

def run_particle_filter(N=1000, T=50, velocity=(1.0, 0.5),
                                      sigma_motion=0.5, sigma_sensor=1.0):
    """
    Versión secuencial del filtro de partículas 2D, sin vectorización.
    Ideal para benchmarking de optimizaciones futuras.
    """
    true_pos = [0.0, 0.0]
    vel = velocity

    # Partículas inicializadas al azar
    particles = [[random.uniform(0, 10), random.uniform(0, 10)] for _ in range(N)]
    weights = [1.0 / N] * N

    for t in range(T):
        # Movimiento del estado real con ruido
        true_pos[0] += vel[0] + random.gauss(0, sigma_motion)
        true_pos[1] += vel[1] + random.gauss(0, sigma_motion)

        # Medición con ruido
        z = [true_pos[0] + random.gauss(0, sigma_sensor),
             true_pos[1] + random.gauss(0, sigma_sensor)]

        # Movimiento de partículas
        for i in range(N):
            particles[i][0] += vel[0] + random.gauss(0, sigma_motion)
            particles[i][1] += vel[1] + random.gauss(0, sigma_motion)

        # Actualización de pesos (Bayes)
        total_weight = 0.0
        for i in range(N):
            dx = particles[i][0] - z[0]
            dy = particles[i][1] - z[1]
            dist2 = dx * dx + dy * dy
            weight = math.exp(-0.5 * dist2 / (sigma_sensor ** 2))
            weights[i] = weight
            total_weight += weight

        # Normalización de pesos
        if total_weight == 0:
            weights = [1.0 / N] * N  # evitar división por cero
        else:
            weights = [w / total_weight for w in weights]

        # Reamostrado
        new_particles = []
        indices = random.choices(range(N), weights=weights, k=N)
        for idx in indices:
            new_particles.append(particles[idx][:])  # copia
        particles = new_particles
        weights = [1.0 / N] * N

    # Estimación final: promedio manual
    sum_x = sum(p[0] for p in particles)
    sum_y = sum(p[1] for p in particles)
    estimate = [sum_x / N, sum_y / N]

    return estimate, true_pos, particles, z

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

    for N in [10000000]:
        print(f"Ejecutando filtro de partículas con N={N}...")
    
        start = time.time()
        estimate, true_pos, particles, measurement = run_particle_filter(N=N, T=100)
        end = time.time()
        print(f"Estimación final: {estimate}, Posición real: {true_pos}, Tiempo: {end - start:.4f} segundos")

if __name__ == "__main__":
    main()