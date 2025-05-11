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
        /* You might want to map status codes to strings if needed */ \
        exit(EXIT_FAILURE); \
    } \
}


int main() {
    // --- 1. Problem Definition ---
    int d = 3; // Rows of matrix W / Size of output vector O
    int n = 4; // Columns of matrix W / Size of input vector X

    // Host data (CPU memory)
    // Matrix W (d x n = 3 x 4), Row-Major Order
    std::vector<float> h_w = {
        1.0f, 0.0f, 1.0f, 0.0f,  // Row 0
        0.0f, 1.0f, 0.0f, 1.0f,  // Row 1
        1.0f, 1.0f, 1.0f, 1.0f   // Row 2
    };
    // Vector X (n x 1 = 4 x 1)
    std::vector<float> h_x = {1.0f, 2.0f, 3.0f, 4.0f};
    // Output Vector O (d x 1 = 3 x 1) - GPU result will be stored here
    std::vector<float> h_o_gpu(d);
    // Expected Output Vector O (calculated manually on CPU)
    std::vector<float> h_o_expected(d);

    // --- 2. Calculate Expected Result on CPU ---
    printf("Calculating expected result on CPU...\n");
    for (int i = 0; i < d; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum += h_w[i * n + j] * h_x[j];
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
    size_t w_size = (size_t)d * n * sizeof(float);
    size_t x_size = n * sizeof(float);
    size_t o_size = d * sizeof(float);

    HIP_CHECK(hipMalloc(&d_w, w_size));
    HIP_CHECK(hipMalloc(&d_x, x_size));
    HIP_CHECK(hipMalloc(&d_o, o_size));

    // --- 5. Copy Data from Host to GPU ---
    printf("Copying data from host to GPU...\n");
    HIP_CHECK(hipMemcpy(d_w, h_w.data(), w_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size, hipMemcpyHostToDevice));
    // No need to copy h_o_gpu to d_o if beta is 0

    // --- 6. Call hipblasSgemv ---
    printf("Calling hipblasSgemv...\n");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Operation: o = alpha * W * x + beta * o
    // Since W is row-major (d x n), we pass it as 'A' but use HIPBLAS_OP_T.
    // hipBLAS sees 'A' as conceptually column-major n x d.
    // The operation uses A^T, which is d x n.
    // m = rows of op(A) = d
    // n = cols of op(A) = n
    // lda = leading dimension of A (conceptual n x d) = n
    HIPBLAS_CHECK(hipblasSgemv(
        blas_handle,        // Handle
        HIPBLAS_OP_T,       // Transpose W (passed as A)
        d,                  // Rows of operator W^T (d)
        n,                  // Cols of operator W^T (n)
        &alpha,             // Pointer to alpha
        d_w,                // Device pointer to W (A)
        n,                  // Leading dimension of W if column-major (n rows)
        d_x,                // Device pointer to X
        1,                  // Increment for X
        &beta,              // Pointer to beta
        d_o,                // Device pointer to O (Y)
        1                   // Increment for O
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
    float epsilon = 1e-5f; // Tolerance for floating point comparison
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

    return passed ? 0 : 1; // Return 0 on success, 1 on failure
}