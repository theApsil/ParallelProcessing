#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <assert.h>
#include <unistd.h>
#include <time.h>


#define R_DIM 512
#define N_DIM 512
#define C_DIM 512
#define VEC_SIZE (sizeof(__m256d) / sizeof(double))

#define SPEEDTEST(func, exp) \
    speedtest_mul_matrix(#func, func, exp)

void speedtest_mul_matrix(
    const char* name,
    void (*func)(double* D, const double* A, const double* B, 
                      size_t n, size_t R, size_t C),
    size_t exp_count
) {
    printf("======= SPEEDTEST OPEN MP AVG ==========\n");

    FILE* output = fopen("matrix_output.csv", "a");
    if (!output) {
        printf("Error while opening file\n");
        return;
    }

    double time_sum_1 = 0.0;

    double time_sum = 0.0;
    double t0 = omp_get_wtime();

    for (size_t exp = 0; exp < exp_count; exp++) {

        size_t R = R_DIM, n = N_DIM, C = C_DIM;
        size_t size_A = R * n;
        size_t size_B = n * C;
        size_t size_D = R * C;

        double *A = (double*)aligned_alloc(32, size_A * sizeof(double));
        double *B = (double*)aligned_alloc(32, size_B * sizeof(double));
        double *D1 = (double*)aligned_alloc(32, size_D * sizeof(double));
        double *D2 = (double*)aligned_alloc(32, size_D * sizeof(double));

        for (size_t k = 0; k < n; k++) {
            for (size_t i = 0; i < R; i++) {
                A[k * R + i] = i;
            }
        }

        for (size_t j = 0; j < C; j++) {
            for (size_t k = 0; k < n; k++) {
                B[j * n + k] = k;
            }
        }

        double t1 = omp_get_wtime();
        func(D1, A, B, n, R, C);
        double t2 = omp_get_wtime();

        time_sum += (t2 - t1) * 1000.0;

        free(A);
        free(B);
        free(D1);
        free(D2);
    }

    double total_time = (omp_get_wtime() - t0) * 1000.0;

    double avg = time_sum / exp_count;

    printf(
        "AVG: Method = %s\t| total experiment time: %.2f ms\t| avg time = %.2f ms\n",
        name, total_time, avg
    );

    fprintf(
        output, "%s,%.2f,%.2f\n",
        name, total_time, avg
    );
    

    fclose(output);
}

void matrix_mul(double* D, const double* A, const double* B, 
               size_t n, size_t R, size_t C)
{
    for (size_t i = 0; i < R; i++)
        for (size_t j = 0; j < C; j++) {
            double accum = 0;
            
            for (size_t k = 0; k < n; k++)
                accum += A[i * n + k] * B[k * C + j];
            D[i * C + j] = accum;
        }
}

void matrix_mul_avx256(double* D, const double* A, const double* B, 
                      size_t n, size_t R, size_t C) {
                        printf("%u = %u\n", R, (sizeof(__m256) / sizeof(double)));
    assert(R % (sizeof(__m256) / sizeof(double)) == 0);

    for (size_t i = 0; i < R / sizeof(__m256); i++) {
        for (size_t j = 0; j < C; j++) {
            __m256d sum = _mm256_setzero_pd();
            for (size_t k = 0; k < n; k++) {
                __m256d x = _mm256_load_pd(&A[k * R + i * sizeof(__m256d)]);
                __m256d y = _mm256_set1_pd(B[j * n + k]);
                sum = _mm256_fmadd_pd(x, y, sum);
            }
            _mm256_storeu_pd(&D[j * R + i * sizeof(__m256d)], sum);
        }
    }
}

void matrix_mul_colmajor(double* D, const double* A, const double* B, 
                        size_t n, size_t R, size_t C)
{
    for (size_t i = 0; i < R; i++) {
        for (size_t j = 0; j < C; j++) {
            double accum = 0.0;
            for (size_t k = 0; k < n; k++) {
                accum += A[k * R + i] * B[j * n + k];
            }
            D[j * R + i] = accum;
        }
    }
}

void create_file() {
    FILE* output = fopen("matrix_output.csv", "w");
    if (!output) {
        printf("Error while opening file\n");
        return;
    }
    fprintf(output, "Method,Time,Avg\n");
    fclose(output);
}

int main() {
    size_t test_count = 15;
    create_file();
    SPEEDTEST(matrix_mul, test_count);
    SPEEDTEST(matrix_mul_colmajor, test_count);
    SPEEDTEST(matrix_mul_avx256, test_count);
    return 0;
}
