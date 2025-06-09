import sys
import time
import pf_cbind

def main():
    # if len(sys.argv) != 2:
    #     print("Uso: python main_cpp.py <N>")
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
        estimate, true_pos = pf_cbind.run_particle_filter(N=N, T=100)
        end = time.time()

        print(f"Estimación: {estimate}")
        print(f"Real:       {true_pos}")
        print(f"Tiempo:     {end - start:.3f} s")

if __name__ == "__main__":
    main()