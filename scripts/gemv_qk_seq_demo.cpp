#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // For sqrtf
// 用于HIP错误检查的辅助宏
#define HIP_CHECK(command) { \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s (%d) at %s:%d\n", \
                hipGetErrorString(status), status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define HIPBLAS_CHECK(command) { \
    hipblasStatus_t status = command; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLAS Error: Status %d at %s:%d\n", \
                status, __FILE__, __LINE__); \
        /* You might want a function to map status codes to strings */ \
        exit(EXIT_FAILURE); \
    } \
}


// Assume HIP_CHECK and HIPBLAS_CHECK macros are defined as before

class GPU_Backend {
private:
    hipblasHandle_t blas_handle; // hipBLAS handle managed by the class

public:
    GPU_Backend() {
        HIPBLAS_CHECK(hipblasCreate(&blas_handle));
    }

    ~GPU_Backend() {
        if (blas_handle) {
            HIPBLAS_CHECK(hipblasDestroy(blas_handle));
        }
    }

    // --- Original CPU-based loop (for reference or CPU fallback) ---
    void gemvQkSeq_cpu_loop(float *h_q,             // Host query vector
                            float *h_key,            // Host key cache
                            float *h_attentionScores,// Host output scores
                            int pos,
                            int kvDim,
                            int headSize,
                            // Assuming a CPU dot function exists or using the original CBackend::dot
                            void (*dot_func)(float*, float*, float*, int))
    {
        if (pos < 0) return; // Handle empty sequence case
        for (int timestep = 0; timestep <= pos; timestep++) {
            float* k = h_key + timestep * kvDim; // Pointer to current key in host memory
            float score = 0.0f;
            dot_func(&score, h_q, k, headSize); // Call the appropriate dot function
            score /= sqrtf(headSize);
            h_attentionScores[timestep] = score; // Write to host memory
        }
    }


    // --- Optimized GPU version using hipblasSgemv ---
    void gemvQkSeq_gpu(float* d_attentionScores, // Output: Device pointer for scores (size >= pos + 1)
                       const float* d_q,         // Input: Device pointer for query vector (size headSize)
                       const float* d_key,       // Input: Device pointer for key cache
                       int pos,                 // Current position index (0-based)
                       int kvDim,               // Stride between keys in d_key cache
                       int headSize)            // Dimension for the GEMV operation
    {
        if (pos < 0) {
            // Handle the case where there are no previous keys (e.g., first token)
            // Often, you might not even call this function if pos < 0.
            // Or you might want to explicitly zero out the score array if needed.
            // HIP_CHECK(hipMemset(d_attentionScores, 0, (pos + 1) * sizeof(float))); // Example if needed
            return;
        }

        if (d_attentionScores == nullptr || d_q == nullptr || d_key == nullptr || headSize <= 0 || kvDim <= 0) {
            fprintf(stderr, "Error: Invalid pointers or dimensions for gemvQkSeq_gpu.\n");
            return; // Or handle error appropriately
        }

        // Number of key vectors to multiply against = number of rows in the matrix A
        int num_keys = pos + 1;

        // The matrix A consists of the first 'num_keys' key vectors from d_key.
        // The vector x is the query vector d_q.
        // The result vector y is d_attentionScores.
        // We want: y = alpha * A * x + beta * y
        // Where A is (num_keys x headSize), x is (headSize x 1), y is (num_keys x 1)

        // Set scalar parameters (alpha for scaling, beta for initialization)
        float alpha = 1.0f / sqrtf((float)headSize); // Scaling factor
        float beta = 0.0f; // Overwrite the output buffer d_attentionScores

        // Set pointer mode for scalar alpha/beta (passed by value from host)
        HIPBLAS_CHECK(hipblasSetPointerMode(blas_handle, HIPBLAS_POINTER_MODE_HOST));

        // Perform the GEMV operation: d_attentionScores = (1/sqrt(headSize)) * KeyMatrix * d_q
        // KeyMatrix dimensions: m = num_keys, n = headSize
        // KeyMatrix storage: Assumed row-major relative to GEMV (keys stacked),
        //                   with stride `kvDim` between rows (keys).
        HIPBLAS_CHECK(hipblasSgemv(blas_handle,
                                   HIPBLAS_OP_N,        // Operation Non-Transpose (use keys as rows)
                                   num_keys,            // m: Number of rows in matrix A (number of keys)
                                   headSize,            // n: Number of columns in matrix A (dimension of keys/query)
                                   &alpha,              // alpha scaling factor
                                   d_key,               // A: Pointer to the key matrix (start of cache) on device
                                   kvDim,               // lda: Leading dimension of A (stride between keys)
                                   d_q,                 // x: Pointer to the query vector on device
                                   1,                   // incx: Increment for x (query vector)
                                   &beta,               // beta scaling factor
                                   d_attentionScores,   // y: Pointer to the result vector on device
                                   1));                 // incy: Increment for y (scores vector)

        // Result is now in d_attentionScores on the GPU.
        // No data transfer or memory allocation/deallocation happens inside this function.
    }
};

// --- Example Usage (Conceptual) ---

int main() {
    int batch_size = 1; // Assuming batch size 1 for simplicity
    int seq_len = 512;
    int num_heads = 8;
    int embed_dim = 256;
    int head_size = embed_dim / num_heads; // e.g., 32
    int kv_dim = embed_dim; // Often kv_dim matches embed_dim for the whole cache

    // --- GPU Memory Allocation (Done ONCE outside the loop) ---
    float *d_query_buffer, *d_key_cache, *d_attention_scores;
    size_t query_bytes = head_size * sizeof(float);
    // Key cache stores keys for all positions, potentially across heads/layers
    // Size depends on context, let's assume enough for seq_len
    size_t key_cache_bytes = seq_len * kv_dim * sizeof(float);
    size_t scores_bytes = seq_len * sizeof(float); // Max scores needed

    HIP_CHECK(hipMalloc(&d_query_buffer, query_bytes));
    HIP_CHECK(hipMalloc(&d_key_cache, key_cache_bytes));
    HIP_CHECK(hipMalloc(&d_attention_scores, scores_bytes));

    GPU_Backend backend; // Creates hipBLAS handle

    // --- Inside the generation loop (for each position 'current_pos') ---
    int current_pos = 10; // Example position

    // Assume d_query_buffer contains the query vector for the current head at 'current_pos'
    // Assume d_key_cache contains the keys for all previous positions (0 to current_pos)
    // for the current head, stored with stride kv_dim.
    // (Data needs to be placed into these buffers by preceding GPU operations or HtoD copies)

    // Calculate attention scores for the current query against all previous keys
    backend.gemvQkSeq_gpu(d_attention_scores, // Output buffer (will contain scores 0 to current_pos)
                          d_query_buffer,     // Input query vector for current position
                          d_key_cache,        // Input key cache (contains keys 0 to current_pos)
                          current_pos,        // Current position index
                          kv_dim,             // Stride in key cache
                          head_size);         // Dimension of query/key for dot product

    // Now d_attention_scores (from index 0 to current_pos) contains the computed scores on the GPU
    // This can be used directly in subsequent GPU operations (like softmax).

    // --- Cleanup (Done ONCE after the loop) ---
    HIP_CHECK(hipFree(d_query_buffer));
    HIP_CHECK(hipFree(d_key_cache));
    HIP_CHECK(hipFree(d_attention_scores));
    // backend object goes out of scope, destroying hipBLAS handle
    return 0;
}
