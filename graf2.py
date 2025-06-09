import matplotlib.pyplot as plt
import numpy as np

# --- Datos ---
# Eje X: Algoritmos
algoritmos = ['Secuencial', 'Vectorizado', 'CBinds', 'Cython', 'Cython-Paralelo', 'GPU']

# Eje Y: Tiempo en segundos para 5 millones de partículas
tiempos = [5553.0, 1182.8, 851.1, 75.6, 57.3, 0] # Se usa 0 para GPU como marcador

# --- Creación del Gráfico ---
fig, ax = plt.subplots(figsize=(11, 7))

# Asignar colores (gris para el dato pendiente de GPU)
colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 'grey']

# Crear las barras
barras = ax.bar(algoritmos, tiempos, color=colores)

# --- Personalización del Gráfico ---
ax.set_title('Tiempo de Ejecución para 5 Millones de Partículas', fontsize=16, pad=20)
ax.set_ylabel('Tiempo de Ejecución (segundos) - Escala Logarítmica', fontsize=12)
ax.set_xlabel('Algoritmo', fontsize=12)

# Usar una escala logarítmica para apreciar mejor las grandes diferencias
ax.set_yscale('log')

# Añadir etiquetas con el valor exacto sobre cada barra
for barra in barras:
    yval = barra.get_height()
    if yval > 0:
        ax.text(barra.get_x() + barra.get_width()/2.0, yval * 1.1, f'{yval} s', ha='center', va='bottom')

# Anotación especial para la barra de GPU
ax.text(len(algoritmos) - 1, 1, 'Dato Pendiente', ha='center', va='bottom', color='black', fontsize=10, style='italic')

# Mejorar el límite del eje Y para que los textos no se corten
ax.set_ylim(bottom=1, top=max(tiempos) * 5)

# Rotar las etiquetas del eje X si es necesario para evitar superposición
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()