import hidet
import numpy as np

@hidet.script
def pf_step(particles, weights, velocity, sigma_motion, sigma_sensor):
    """
    Un paso del filtro de partículas en GPU.
    particles: float32[N, 2]
    weights: float32[N]
    """
    N = particles.shape[0]
    # Predicción con ruido
    motion = hidet.randn((N, 2), dtype='float32', stddev=sigma_motion)
    particles += velocity + motion

    # Observación simulada por línea antes del kernel
    return particles * 0  # dummy para sistema de flujo.


def run_particle_filter_hidet(N=10000, T=100,
                              velocity=(1.0,0.5),
                              sigma_motion=0.5,
                              sigma_sensor=1.0):
    # Inicialización
    particles = hidet.randn((N,2), dtype='float32', device='cuda') * 10.0
    weights = hidet.full((N,), 1.0/N, device='cuda', dtype='float32')
    velocity = hidet.from_numpy(np.array(velocity, dtype='float32')).cuda()

    for _ in range(T):
        # Predicción
        particles = pf_step(particles, weights, velocity, sigma_motion, sigma_sensor)

        # Peso (kernel separado)
        dists = hidet.linalg.norm(particles - ... , axis=1)
        weights = hidet.exp(-0.5 * dists*dists / (sigma_sensor**2))
        weights = weights / hidet.sum(weights)

        # Remuestreo sistemático (kernel personalizado)
        indices = systematic_resample_hidet(weights)
        particles = particles[indices]
        weights = hidet.full((N,), 1.0/N, device='cuda')

    estimate = hidet.mean(particles, axis=0)
    return estimate, particles
