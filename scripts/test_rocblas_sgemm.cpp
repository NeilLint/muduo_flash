#include <hip/hip_runtime.h>
#include <hipblas.h> // rocBLAS/hipBLAS header
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath> // For fabs

// Helper macro for checking HIP API calls
#define HIP_CHECK(command) { \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s (%d) at %s:%d\n", \
                hipGetErrorString(status), status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Helper macro for checking hipBLAS API calls
#define HIPBLAS_CHECK(command) { \
    hipblasStatus_t status = command; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLAS Error: Status %d at %s:%d\n", \
                status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}


int main() {
    // --- 1. Problem Definition ---
    int d = 3; // Rows of matrix W / Size of output vector O
    int n_vec = 4; // Columns of matrix W / Size of input vector X

    // Host data (CPU memory)
    std::vector<float> h_w = {
        1.0f, 0.0f, 1.0f, 0.0f,  // Row 0 (d=0)
        0.0f, 1.0f, 0.0f, 1.0f,  // Row 1 (d=1)
        1.0f, 1.0f, 1.0f, 1.0f   // Row 2 (d=2)
    };
    std::vector<float> h_x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> h_o_gpu(d);
    std::vector<float> h_o_expected(d);

    // --- 2. Calculate Expected Result on CPU ---
    printf("Calculating expected result on CPU...\n");
    for (int i = 0; i < d; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n_vec; ++j) {
            sum += h_w[i * n_vec + j] * h_x[j];
        }
        h_o_expected[i] = sum;
    }
    printf("Expected output o: [ ");
    for(int i=0; i<d; ++i) printf("%.2f ", h_o_expected[i]);
    printf("]\n");

    // --- 3. Initialize hipBLAS ---
    hipblasHandle_t blas_handle;
    printf("Creating hipBLAS handle...\n");
    HIPBLAS_CHECK(hipblasCreate(&blas_handle));

    // --- 4. Allocate GPU Memory ---
    printf("Allocating memory on GPU...\n");
    float *d_w = nullptr, *d_x = nullptr, *d_o = nullptr;
    size_t w_size = (size_t)d * n_vec * sizeof(float);
    size_t x_size = n_vec * sizeof(float);
    size_t o_size = d * sizeof(float);

    HIP_CHECK(hipMalloc(&d_w, w_size));
    HIP_CHECK(hipMalloc(&d_x, x_size));
    HIP_CHECK(hipMalloc(&d_o, o_size));

    // --- 5. Copy Data from Host to GPU ---
    printf("Copying data from host to GPU...\n");
    HIP_CHECK(hipMemcpy(d_w, h_w.data(), w_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size, hipMemcpyHostToDevice));

    // --- 6. Call hipblasSgemm ---
    printf("Calling hipblasSgemm (for GEMV)...\n");
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Parameters for C[M, N] = A[M, K] * B[K, N]
    int M = d;      // Rows of op(A) (which is W) and C (which is O)
    int N_gemm = 1; // Columns of op(B) (which is X) and C (which is O)
    int K = n_vec;  // Columns of op(A) (W) and Rows of op(B) (X)

    // Call: O = alpha * W * X + beta * O
    // A = W (row-major M x K passed), use OP_T -> op(A) is M x K
    // B = X (vector K x 1 passed), use OP_N -> op(B) is K x 1
    // C = O (vector M x 1 output)
    HIPBLAS_CHECK(hipblasSgemm(
        blas_handle,        // Handle
        HIPBLAS_OP_T,       // Transpose A (W) because input is row-major
        HIPBLAS_OP_N,       // No transpose for B (X)
        M,                  // Rows of op(A) and C
        N_gemm,             // Columns of op(B) and C
        K,                  // Columns of op(A), Rows of op(B)
        &alpha,             // Scalar alpha
        d_w,                // Pointer to A (W)
        K,                  // Leading dimension of A (if col-major K x M) -> K
        d_x,                // Pointer to B (X)
        K,                  // Leading dimension of B (if col-major K x 1) -> K
        &beta,              // Scalar beta
        d_o,                // Pointer to C (O)
        M                   // Leading dimension of C (if col-major M x 1) -> M
    ));


    // --- 7. Copy Result from GPU to Host ---
    printf("Copying result from GPU to host...\n");
    HIP_CHECK(hipMemcpy(h_o_gpu.data(), d_o, o_size, hipMemcpyDeviceToHost));

    // --- 8. Verify Result ---
    printf("Verifying result...\n");
    printf("GPU output o:      [ ");
    for(int i=0; i<d; ++i) printf("%.2f ", h_o_gpu[i]);
    printf("]\n");

    bool passed = true;
    float epsilon = 1e-5f; // Tolerance
    float max_error = 0.0f;
    for (int i = 0; i < d; ++i) {
        float error = std::fabs(h_o_gpu[i] - h_o_expected[i]);
        if (error > max_error) {
            max_error = error;
        }
        if (error > epsilon) {
            passed = false;
            printf("Mismatch at index %d: Expected=%.6f, Got=%.6f, Error=%.6f\n",
                   i, h_o_expected[i], h_o_gpu[i], error);
        }
    }

    printf("Maximum error: %.6f\n", max_error);
    if (passed) {
        printf("Test PASSED!\n");
    } else {
        printf("Test FAILED!\n");
    }

    // --- 9. Cleanup ---
    printf("Cleaning up resources...\n");
    HIP_CHECK(hipFree(d_w));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_o));
    HIPBLAS_CHECK(hipblasDestroy(blas_handle));

    return passed ? 0 : 1;
}