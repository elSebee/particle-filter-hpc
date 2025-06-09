import matplotlib.pyplot as plt
import numpy as np

# --- Datos ---
# Eje X: Cantidad de partículas
particulas = [1000000, 5000000, 10000000]

# Eje Y: Tiempo en segundos para cada algoritmo
tiempo_secuencial = [418.1, 2636.7, 5553.0]
tiempo_vectorizado = [60.9, 538.6, 1182.8]
tiempo_cbinds = [44.9, 365.7, 851.1]
tiempo_cython = [6.9, 37.5, 75.6]
tiempo_cython_paralelo = [5.8, 28.9, 57.3]
# Los datos de GPU se pueden agregar aquí cuando los tengas
# tiempo_gpu = [y1, y2, y3]

# --- Creación del Gráfico ---
fig, ax = plt.subplots(figsize=(10, 6))

# Dibujar una línea para cada algoritmo
ax.plot(particulas, tiempo_secuencial, marker='o', linestyle='-', label='Secuencial')
ax.plot(particulas, tiempo_vectorizado, marker='o', linestyle='-', label='Vectorizado')
ax.plot(particulas, tiempo_cbinds, marker='o', linestyle='-', label='CBinds')
ax.plot(particulas, tiempo_cython, marker='o', linestyle='-', label='Cython')
ax.plot(particulas, tiempo_cython_paralelo, marker='s', linestyle='--', label='Cython-Paralelo')
# Descomenta la siguiente línea cuando tengas los datos de la GPU
# ax.plot(particulas, tiempo_gpu, marker='^', linestyle=':', label='GPU')

# --- Personalización del Gráfico ---
ax.set_title('Rendimiento de Algoritmos por Cantidad de Partículas', fontsize=16)
ax.set_xlabel('Cantidad de Partículas', fontsize=12)
ax.set_ylabel('Tiempo de Ejecución (segundos en escala logarítmica)', fontsize=12)

# Usar una escala logarítmica para el eje Y para apreciar mejor las diferencias
ax.set_yscale('log')

# Formatear las etiquetas del eje X para que sean más legibles
ax.set_xticks(particulas)
ax.set_xticklabels(['1 Millón', '5 Millones', '10 Millones'])

# Añadir una leyenda para identificar cada línea
ax.legend()

# Ajustar el diseño y mostrar el gráfico
plt.tight_layout()
plt.show()