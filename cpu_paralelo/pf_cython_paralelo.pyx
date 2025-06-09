import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange
from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time

ctypedef cnp.float64_t DTYPE_t

# Inicializar generador de números aleatorios
srand(time(NULL))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void systematic_resample(cnp.ndarray[DTYPE_t, ndim=1] weights, 
                             cnp.ndarray[long, ndim=1] indices, int N):
    """Remuestreo sistemático - más eficiente que np.random.choice"""
    cdef double r = (<double>rand() / RAND_MAX) / N
    cdef double c = weights[0]
    cdef int i = 0
    cdef int j
    
    for j in range(N):
        while c < r:
            i += 1
            if i >= N:
                i = N - 1
                break
            c += weights[i]
        indices[j] = i
        r += 1.0 / N

@cython.boundscheck(False)
@cython.wraparound(False)
def run_particle_filter(int N=1000, int T=50, tuple velocity=(1.0, 0.5),
                       double sigma_motion=0.5, double sigma_sensor=1.0):
    """
    Ejecuta un filtro de partículas 2D optimizado.
    """
    # Usar arrays de NumPy para operaciones vectorizadas donde es eficiente
    cdef cnp.ndarray[DTYPE_t, ndim=1] true_pos = np.array([0.0, 0.0], dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] vel = np.array(velocity, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=2] particles = np.random.uniform(low=0, high=10, size=(N, 2)).astype(np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] weights = np.ones(N, dtype=np.float64) / N
    cdef cnp.ndarray[DTYPE_t, ndim=1] z = np.zeros(2, dtype=np.float64)
    cdef cnp.ndarray[long, ndim=1] indices = np.zeros(N, dtype=np.int64)
    
    cdef int t, i
    cdef double weight_sum, dist_sq
    cdef double sigma_sensor_sq = sigma_sensor * sigma_sensor
    cdef double inv_sigma_sensor_sq = 1.0 / sigma_sensor_sq
    
    # Variables temporales
    cdef cnp.ndarray[DTYPE_t, ndim=2] motion_noise
    cdef cnp.ndarray[DTYPE_t, ndim=1] sensor_noise
    
    for t in range(T):
        # Actualizar posición real (vectorizado)
        sensor_noise = np.random.normal(0, sigma_motion, size=2)
        true_pos[0] += vel[0] + sensor_noise[0]
        true_pos[1] += vel[1] + sensor_noise[1]
        
        # Generar observación (vectorizado)
        sensor_noise = np.random.normal(0, sigma_sensor, size=2)
        z[0] = true_pos[0] + sensor_noise[0]
        z[1] = true_pos[1] + sensor_noise[1]
        
        # Actualizar partículas (vectorizado - NumPy es eficiente aquí)
        motion_noise = np.random.normal(0, sigma_motion, size=(N, 2))
        for i in prange(N, nogil=True):
            particles[i, 0] += vel[0] + motion_noise[i, 0]
            particles[i, 1] += vel[1] + motion_noise[i, 1]
        
        # Calcular pesos (bucle optimizado)
        weight_sum = 0.0
        for i in prange(N, schedule='static', nogil=True):
            dist_sq = (particles[i, 0] - z[0])**2 + (particles[i, 1] - z[1])**2
            weights[i] = exp(-0.5 * dist_sq * inv_sigma_sensor_sq)
    
        # Normalizar pesos
        weight_sum += 1e-300  # Evitar división por cero
        for i in range(N):
            weights[i] /= weight_sum
        
        # Remuestreo sistemático (más eficiente que np.random.choice)
        systematic_resample(weights, indices, N)
        particles = particles[indices]
        
        # Resetear pesos
        for i in range(N):
            weights[i] = 1.0 / N
    
    # Estimación final (vectorizada)
    estimate = particles.mean(axis=0)
    
    return estimate, true_pos, particles, z