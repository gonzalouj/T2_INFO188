// =============================================================================
// INFO188: Tarea 2 - Batalla de sorting paralelo
// Implementación de algoritmos de ordenamiento paralelo para CPU y GPU
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

// Definimos threads por bloque según recomendaciones de NVIDIA para RTX/GTX
#define THREADS_PER_BLOCK 256

// Funciones auxiliares
bool check_sorting(int* arr, long long n) {
    for(long long i = 1; i < n; i++) {
        if(arr[i] < arr[i-1]) return false;
    }
    return true;
}

void generate_random_data(int *arr, long long n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, n * 2);
    
    for(long long i = 0; i < n; i++) {
        arr[i] = dis(gen);
    }
}

// =================== ALGORITMO 1: ORDENAMIENTO POR MEZCLA PARALELO (CPU) ===================
/*
 * Implementación paralela de Ordenamiento por Mezcla usando OpenMP
 * El algoritmo divide el arreglo en segmentos y los procesa en paralelo
 * usando múltiples hilos de CPU. La fase de mezcla se realiza
 * de manera iterativa, combinando pares de subarreglos ordenados.
 */

/**
 * Función merge: Combina dos subarreglos ordenados en uno solo
 * @param arr: Arreglo original
 * @param temp: Arreglo temporal para la mezcla
 * @param start: Índice inicial
 * @param mid: Índice medio (donde comienza el segundo subarreglo)
 * @param end: Índice final
 */
void merge(int *arr, int *temp, int start, int mid, int end) {
    int i = start, j = mid, k = start;
    
    while (i < mid && j < end) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    while (i < mid) {
        temp[k++] = arr[i++];
    }
    while (j < end) {
        temp[k++] = arr[j++];
    }
    
    for (i = start; i < end; i++) {
        arr[i] = temp[i];
    }
}

/**
 * Implementación paralela del Merge Sort
 * - Usa OpenMP para paralelizar la fase de merge
 * - Cada thread procesa un chunk del array independientemente
 * @param arr: Array a ordenar
 * @param temp: Array temporal necesario para el merge
 * @param n: Tamaño del array
 * @param num_threads: Número de threads a usar
 */
void parallel_merge_sort(int *arr, int *temp, int n, int num_threads) {
    omp_set_num_threads(num_threads);
    
    for (int chunk_size = 1; chunk_size < n; chunk_size *= 2) {
        #pragma omp parallel for
        for (int i = 0; i < n; i += 2 * chunk_size) {
            int start = i;
            int mid = std::min(i + chunk_size, n);
            int end = std::min(i + 2 * chunk_size, n);
            merge(arr, temp, start, mid, end);
        }
    }
}

// =================== ALGORITMO 2: ORDENAMIENTO RADIX (GPU) ===================
/*
 * Implementación de Radix Sort en GPU utilizando CUDA Thrust
 * El algoritmo aprovecha la biblioteca Thrust que provee una 
 * implementación optimizada del ordenamiento Radix para GPUs NVIDIA.
 */

/**
 * Wrapper para el Radix Sort de Thrust
 * Maneja la transferencia de memoria entre CPU y GPU, y la ejecución
 * del algoritmo de ordenamiento en la GPU
 * @param data: Array a ordenar
 * @param n: Tamaño del array
 * @param num_blocks: Número de bloques CUDA a usar
 */
void radix_sort_cuda(int* data, int n, int num_blocks) {
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Usar thrust::sort que implementa Radix Sort paralelo para enteros
    thrust::device_ptr<int> dev_ptr(d_data);
    thrust::sort(dev_ptr, dev_ptr + n);
    
    cudaMemcpy(data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

/**
 * Programa principal:
 * Maneja la lógica de entrada, memoria y ejecución de los algoritmos
 * Uso: ./prog <n> <modo> <hilos>
 * - n: tamaño del arreglo
 * - modo: 0 para CPU (Ordenamiento por Mezcla), 1 para GPU (Radix Sort)
 * - hilos: número de hilos para CPU (ignorado en GPU)
 */
int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Uso: %s <n> <modo> <hilos>\n", argv[0]);
        printf("  n: tamaño del arreglo\n");
        printf("  modo: CPU -> 0, GPU -> 1\n");
        printf("  hilos: número de hilos CPU\n");
        return 1;
    }
    
    long long n = atoll(argv[1]);
    int mode = atoi(argv[2]);
    int threads = atoi(argv[3]);
    
    if (n <= 0 || (mode != 0 && mode != 1) || threads <= 0) {
        printf("Error: argumentos inválidos\n");
        return 1;
    }

    int *data = (int *)malloc(n * sizeof(int));
    if (data == NULL) {
        printf("Error: no se pudo asignar memoria\n");
        return 1;
    }

    generate_random_data(data, n);

    if (n <= 20) {
        printf("Arreglo original: ");
        for(int i = 0; i < n; i++) printf("%d ", data[i]);
        printf("\n");
    }

    int *temp = (int *)malloc(n * sizeof(int));
    if (temp == NULL) {
        printf("Error: no se pudo asignar memoria temporal\n");
        free(data);
        return 1;
    }

    double start = omp_get_wtime();
    
    if (mode == 0) {
        parallel_merge_sort(data, temp, n, threads);
    } else {
        int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        radix_sort_cuda(data, n, num_blocks);
    }
    
    double end = omp_get_wtime();
    double total_time = end - start;
    
    if (n <= 20) {
        printf("Arreglo ordenado: ");
        for(int i = 0; i < n; i++) printf("%d ", data[i]);
        printf("\n");
    }

    if (!check_sorting(data, n)) {
        printf("Error: el arreglo no está ordenado correctamente\n");
        return 1;
    }
    
    printf("%.4f\n", total_time);

    free(temp);
    free(data);
    return 0;
}