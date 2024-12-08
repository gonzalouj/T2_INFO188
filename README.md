# Sorting Paralelo CPU vs GPU

Implementación y comparación de algoritmos de ordenamiento paralelos:

- CPU: Parallel Merge Sort con OpenMP
- GPU: Radix Sort con CUDA Thrust

## Requisitos

- NVIDIA CUDA Toolkit
- GCC 9+
- OpenMP
- Python 3.x con numpy y matplotlib

## Compilación

make

## Ejecución

./prog <n> <modo> <hilos>

Donde:

- n: Tamaño del arreglo
- modo: 0 (CPU) o 1 (GPU)
- hilos: Número de hilos CPU (ignorado en modo GPU)

Ejemplo:

    ./prog 1000000 0 8  # CPU con 8 hilos
    ./prog 1000000 1 1  # GPU
