#ifndef HIP_UTILS_HPP
#define HIP_UTILS_HPP

#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(command)                                                  \
    do                                                                      \
    {                                                                       \
        hipError_t status = (command);                                      \
        if (status != hipSuccess)                                           \
        {                                                                   \
            std::fprintf(stderr, "HIP Error: %s (%d) at %s:%d\n",         \
                         hipGetErrorString(status), status, __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

#define HIPBLAS_CHECK(command)                                              \
    do                                                                      \
    {                                                                       \
        hipblasStatus_t status = (command);                                 \
        if (status != HIPBLAS_STATUS_SUCCESS)                               \
        {                                                                   \
            std::fprintf(stderr, "hipBLAS Error: Status %d at %s:%d\n",   \
                         status, __FILE__, __LINE__);                       \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

#endif
