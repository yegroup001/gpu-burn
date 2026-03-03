/*
 * Copyright (c) 2022, Ville Timonen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *	this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 *those of the authors and should not be interpreted as representing official
 *policies, either expressed or implied, of the FreeBSD Project.
 */

// Matrices are SIZE*SIZE..  POT should be efficiently implemented in CUBLAS
#define SIZE 8192ul
#define USEMEM 0.9 // Try to allocate 90% of memory
#define COMPARE_KERNEL "compare.ptx"

// Used to report op/s, measured through Visual Profiler, CUBLAS from CUDA 7.5
// (Seems that they indeed take the naive dim^3 approach)
//#define OPS_PER_MUL 17188257792ul // Measured for SIZE = 2048
#define OPS_PER_MUL 1100048498688ul // Extrapolated for SIZE = 8192

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <exception>
#include <fstream>
#include <signal.h>
#include <stdexcept>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <regex>
#include <type_traits>
#include <cstdint>

#define SIGTERM_TIMEOUT_THRESHOLD_SECS 30 // number of seconds for sigterm to kill child processes before forcing a sigkill

#include "cublas_v2.h"
#include "cublasLt.h"
#define CUDA_ENABLE_DEPRECATED
#include <cuda.h>
#include <cuda_fp16.h>
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
#include <cuda_fp8.h>
#endif

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
// Lightweight host-side wrappers to distinguish FP8 and FP4 modes
struct fp8_e4m3_t { unsigned char v; };
struct fp4_e2m1_t { unsigned char v; };
#endif

void _checkError(int rCode, std::string file, int line, std::string desc = "") {
    if (rCode != CUDA_SUCCESS) {
        const char *err;
        cuGetErrorString((CUresult)rCode, &err);

        throw std::runtime_error(
            (desc == "" ? std::string("Error (")
                        : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}

void _checkError(cublasStatus_t rCode, std::string file, int line, std::string desc = "") {
    if (rCode != CUBLAS_STATUS_SUCCESS) {
#if CUBLAS_VER_MAJOR >= 12
		const char *err = cublasGetStatusString(rCode);
#else
		const char *err = "";
#endif
        throw std::runtime_error(
            (desc == "" ? std::string("Error (")
                        : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}

#define checkError(rCode, ...)                                                 \
    _checkError(rCode, __FILE__, __LINE__, ##__VA_ARGS__)

double getTime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec / 1e6;
}

bool g_running = false;

struct BurnProgressPacket {
    int processed;
    float gemmMs;
    int errors;
};

template <class T> class GPU_Test {
  public:
    GPU_Test(int dev, bool tensors, const char *kernelFile)
    : d_devNumber(dev), d_tensors(tensors), d_kernelFile(kernelFile) {
    d_useLt = false;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
    d_isFp4 = false;
    d_ltHandle = nullptr;
    d_ltMatmulDesc = nullptr;
    d_Adesc = nullptr;
    d_Bdesc = nullptr;
    d_Cdesc = nullptr;
    d_AscaleData = 0;
    d_BscaleData = 0;
    d_scaleBytes = 0;
#endif

    // configure type-specific settings
        if (std::is_same<T, double>::value) {
            d_dataType = CUDA_R_64F;
            d_computeType = CUBLAS_COMPUTE_64F;
            d_typeName = "using DOUBLES";
            d_compareKernelName = "compareD";
        } else if (std::is_same<T, float>::value) {
            d_dataType = CUDA_R_32F;
            d_computeType = CUBLAS_COMPUTE_32F;
            d_typeName = "using FLOATS";
            d_compareKernelName = "compare";
        } else if (std::is_same<T, __half>::value) {
            d_dataType = CUDA_R_16F;
            d_computeType = CUBLAS_COMPUTE_32F;
            d_typeName = "using FP16";
            d_compareKernelName = "compareH";
        }
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
        else if (std::is_same<T, __nv_bfloat16>::value) {
            d_dataType = CUDA_R_16BF;
            d_computeType = CUBLAS_COMPUTE_32F;
            d_typeName = "using BF16";
            d_compareKernelName = "compareBF16";
        }
#endif
    #if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
        else if (std::is_same<T, fp8_e4m3_t>::value) {
            d_dataType = CUDA_R_8F_E4M3;
            d_computeType = CUBLAS_COMPUTE_32F;
            d_typeName = "using FP8 (E4M3)";
            // For FP8/FP4 we accumulate and store results in FP32 and
            // reuse the float compare kernel.
            d_compareKernelName = "compare";
            d_useLt = true;
        } else if (std::is_same<T, fp4_e2m1_t>::value) {
            d_dataType = CUDA_R_4F_E2M1;
            d_computeType = CUBLAS_COMPUTE_32F;
            d_typeName = "using FP4 (E2M1)";
            d_compareKernelName = "compare";
            d_useLt = true;
            d_isFp4 = true;
        }
    #endif

        checkError(cuDeviceGet(&d_dev, d_devNumber));
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
            checkError(cuCtxCreate(&d_ctx, nullptr, 0, d_dev));
#else
            checkError(cuCtxCreate(&d_ctx, 0, d_dev));
#endif

        bind();

        // checkError(cublasInit());
        checkError(cublasCreate(&d_cublas), "init");

    #if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
        if (d_isFp4)
            initFp4Scales();
        if (d_useLt)
            initLtGemm();
    #endif

        if (d_tensors)
            checkError(cublasSetMathMode(d_cublas, CUBLAS_TENSOR_OP_MATH));

        checkError(cuMemAllocHost((void **)&d_faultyElemsHost, sizeof(int)));
        d_error = 0;

        g_running = true;

        struct sigaction action;
        memset(&action, 0, sizeof(struct sigaction));
        action.sa_handler = termHandler;
        sigaction(SIGTERM, &action, NULL);
    }
    ~GPU_Test() {
        bind();
        checkError(cuMemFree(d_Cdata), "Free A");
        checkError(cuMemFree(d_Adata), "Free B");
        checkError(cuMemFree(d_Bdata), "Free C");
        cuMemFreeHost(d_faultyElemsHost);
        printf("Freed memory for dev %d\n", d_devNumber);

        cublasDestroy(d_cublas);
        printf("Uninitted cublas\n");

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
        if (d_useLt) {
            if (d_ltMatmulDesc)
                cublasLtMatmulDescDestroy(d_ltMatmulDesc);
            if (d_Adesc)
                cublasLtMatrixLayoutDestroy(d_Adesc);
            if (d_Bdesc)
                cublasLtMatrixLayoutDestroy(d_Bdesc);
            if (d_Cdesc)
                cublasLtMatrixLayoutDestroy(d_Cdesc);
            if (d_ltHandle)
                cublasLtDestroy(d_ltHandle);
        }
        if (d_AscaleData)
            cuMemFree(d_AscaleData);
        if (d_BscaleData)
            cuMemFree(d_BscaleData);
#endif
    }

    static void termHandler(int signum) { g_running = false; }

    unsigned long long int getErrors() {
        if (*d_faultyElemsHost) {
            d_error += (long long int)*d_faultyElemsHost;
        }
        unsigned long long int tempErrs = d_error;
        d_error = 0;
        return tempErrs;
    }

    size_t getIters() { return d_iters; }

    void bind() { checkError(cuCtxSetCurrent(d_ctx), "Bind CTX"); }

    size_t totalMemory() {
        bind();
        size_t freeMem, totalMem;
        checkError(cuMemGetInfo(&freeMem, &totalMem));
        return totalMem;
    }

    size_t availMemory() {
        bind();
        size_t freeMem, totalMem;
        checkError(cuMemGetInfo(&freeMem, &totalMem));
        return freeMem;
    }

    void initBuffers(T *A, T *B, ssize_t useBytes = 0) {
        bind();

        if (useBytes == 0)
            useBytes = (ssize_t)((double)availMemory() * USEMEM);
        if (useBytes < 0)
            useBytes = (ssize_t)((double)availMemory() * (-useBytes / 100.0));

         printf("Initialized device %d with %lu MB of memory (%lu MB available, "
             "using %lu MB of it), %s%s\n",
             d_devNumber, totalMemory() / 1024ul / 1024ul,
             availMemory() / 1024ul / 1024ul, useBytes / 1024ul / 1024ul,
             d_typeName,
             d_tensors ? ", using Tensor Cores" : "");

        // For FP8/FP4 (Lt path) we keep A/B in low precision but store C in FP32.
        size_t elemSizeA = sizeof(T);
        size_t elemSizeB = sizeof(T);
        size_t elemSizeC = sizeof(T);
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
        if (d_useLt) {
            elemSizeC = sizeof(float);
        }
#endif

        d_AbytesPerIter = elemSizeA * SIZE * SIZE;
        d_BbytesPerIter = elemSizeB * SIZE * SIZE;
        d_CbytesPerIter = elemSizeC * SIZE * SIZE;

        if ((size_t)useBytes < d_AbytesPerIter + d_BbytesPerIter + d_CbytesPerIter)
            throw std::string("Low mem for result. aborting.\n");

        d_iters = (useBytes - d_AbytesPerIter - d_BbytesPerIter) / d_CbytesPerIter;

        printf("Results are %zu bytes each, thus performing %zu iterations\n",
               d_CbytesPerIter, d_iters);

        checkError(cuMemAlloc(&d_Cdata, d_iters * d_CbytesPerIter), "C alloc");
        checkError(cuMemAlloc(&d_Adata, d_AbytesPerIter), "A alloc");
        checkError(cuMemAlloc(&d_Bdata, d_BbytesPerIter), "B alloc");

        checkError(cuMemAlloc(&d_faultyElemData, sizeof(int)), "faulty data");

        // Populating matrices A and B
        checkError(cuMemcpyHtoD(d_Adata, A, d_AbytesPerIter), "A -> device");
        checkError(cuMemcpyHtoD(d_Bdata, B, d_BbytesPerIter), "B -> device");

        initCompareKernel();
    }

    void compute() {
        bind();
        static const float alpha = 1.0f;
        static const float beta = 0.0f;
        static const double alphaD = 1.0;
        static const double betaD = 0.0;

        for (size_t i = 0; i < d_iters; ++i) {
            const void *alphaPtr = &alpha;
            const void *betaPtr = &beta;
            if (d_computeType == CUBLAS_COMPUTE_64F) {
                alphaPtr = &alphaD;
                betaPtr = &betaD;
            }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
            if (d_useLt) {
                void *Cptr = (void *)((float *)d_Cdata + i * SIZE * SIZE);
                checkError(
                    cublasLtMatmul(d_ltHandle, d_ltMatmulDesc,
                                   alphaPtr,
                                   (const void *)d_Adata, d_Adesc,
                                   (const void *)d_Bdata, d_Bdesc,
                                   betaPtr,
                                   Cptr, d_Cdesc,
                                   Cptr, d_Cdesc,
                                   &d_ltAlgo,
                                   nullptr, 0, 0),
                    "LtMatmul");
            } else
#endif
            {
                checkError(
                    cublasGemmEx(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 SIZE, SIZE, SIZE,
                                 alphaPtr,
                                 (const void *)d_Adata, d_dataType, SIZE,
                                 (const void *)d_Bdata, d_dataType, SIZE,
                                 betaPtr,
                                 (void *)((T *)d_Cdata + i * SIZE * SIZE), d_dataType, SIZE,
                                 d_computeType,
                                 d_tensors ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT),
                    "GEMMEx");
            }
        }
    }

    void initCompareKernel() {
        {
            std::ifstream f(d_kernelFile);
            checkError(f.good() ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND,
                       std::string("couldn't find compare kernel: ") + d_kernelFile);
        }
        checkError(cuModuleLoad(&d_module, d_kernelFile), "load module");
        checkError(cuModuleGetFunction(&d_function, d_module,
                           d_compareKernelName),
               "get func");

        checkError(cuFuncSetCacheConfig(d_function, CU_FUNC_CACHE_PREFER_L1),
                   "L1 config");
        checkError(cuParamSetSize(d_function, __alignof(T *) +
                                                  __alignof(int *) +
                                                  __alignof(size_t)),
                   "set param size");
        checkError(cuParamSetv(d_function, 0, &d_Cdata, sizeof(T *)),
                   "set param");
        checkError(cuParamSetv(d_function, __alignof(T *), &d_faultyElemData,
                               sizeof(T *)),
                   "set param");
        checkError(cuParamSetv(d_function, __alignof(T *) + __alignof(int *),
                               &d_iters, sizeof(size_t)),
                   "set param");

        checkError(cuFuncSetBlockShape(d_function, g_blockSize, g_blockSize, 1),
                   "set block size");
    }

    void compare() {
        checkError(cuMemsetD32Async(d_faultyElemData, 0, 1, 0), "memset");
        checkError(cuLaunchGridAsync(d_function, SIZE / g_blockSize,
                                     SIZE / g_blockSize, 0),
                   "Launch grid");
        checkError(cuMemcpyDtoHAsync(d_faultyElemsHost, d_faultyElemData,
                                     sizeof(int), 0),
                   "Read faultyelemdata");
    }

    bool shouldRun() { return g_running; }

  private:
    bool d_tensors;
    int d_devNumber;
    const char *d_kernelFile;
        const char *d_typeName;
        const char *d_compareKernelName;
        cudaDataType_t d_dataType;
        cublasComputeType_t d_computeType;
        bool d_useLt;
        size_t d_AbytesPerIter;
        size_t d_BbytesPerIter;
        size_t d_CbytesPerIter;
    size_t d_iters;
    long long int d_error;

    static const int g_blockSize = 16;

    CUdevice d_dev;
    CUcontext d_ctx;
    CUmodule d_module;
    CUfunction d_function;

    CUdeviceptr d_Cdata;
    CUdeviceptr d_Adata;
    CUdeviceptr d_Bdata;
    CUdeviceptr d_faultyElemData;
    int *d_faultyElemsHost;

    cublasHandle_t d_cublas;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
    // cuBLASLt state for FP8/FP4 paths
    bool d_isFp4;
    cublasLtHandle_t d_ltHandle;
    cublasLtMatmulDesc_t d_ltMatmulDesc;
    cublasLtMatrixLayout_t d_Adesc;
    cublasLtMatrixLayout_t d_Bdesc;
    cublasLtMatrixLayout_t d_Cdesc;
    cublasLtMatmulAlgo_t d_ltAlgo;
    CUdeviceptr d_AscaleData;
    CUdeviceptr d_BscaleData;
    size_t d_scaleBytes;

    void initFp4Scales() {
        const size_t fp4Elems = SIZE * SIZE;
        const size_t scaleElems = (fp4Elems + 15) / 16;
        d_scaleBytes = scaleElems * sizeof(unsigned char);

        checkError(cuMemAlloc(&d_AscaleData, d_scaleBytes), "ltFP4AScaleAlloc");
        checkError(cuMemAlloc(&d_BscaleData, d_scaleBytes), "ltFP4BScaleAlloc");

        // UE4M3 encoding for 1.0f is 0x38. We use unit scaling for all blocks.
        checkError(cuMemsetD8(d_AscaleData, 0x38, d_scaleBytes), "ltFP4AScaleInit");
        checkError(cuMemsetD8(d_BscaleData, 0x38, d_scaleBytes), "ltFP4BScaleInit");
    }

    void initLtGemm() {
        checkError(cublasLtCreate(&d_ltHandle), "ltCreate");

        // computeType and scaleType are both 32F for FP8/FP4
        checkError(cublasLtMatmulDescCreate(&d_ltMatmulDesc, d_computeType, CUDA_R_32F), "ltMatmulDescCreate");

        cublasLtPointerMode_t pointerMode = CUBLASLT_POINTER_MODE_HOST;
        checkError(cublasLtMatmulDescSetAttribute(d_ltMatmulDesc,
                                                  CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                                  &pointerMode, sizeof(pointerMode)),
                   "ltSetPointerMode");

        cublasOperation_t opN = CUBLAS_OP_N;
        checkError(cublasLtMatmulDescSetAttribute(d_ltMatmulDesc,
                                                  CUBLASLT_MATMUL_DESC_TRANSA,
                                                  &opN, sizeof(opN)),
                   "ltSetTransA");
        checkError(cublasLtMatmulDescSetAttribute(d_ltMatmulDesc,
                                                  CUBLASLT_MATMUL_DESC_TRANSB,
                                                  &opN, sizeof(opN)),
                   "ltSetTransB");

        // Column-major SIZE x SIZE with leading dimension SIZE
        checkError(cublasLtMatrixLayoutCreate(&d_Adesc, d_dataType,
                                              SIZE, SIZE, SIZE),
                   "ltAdesc");
        checkError(cublasLtMatrixLayoutCreate(&d_Bdesc, d_dataType,
                                              SIZE, SIZE, SIZE),
                   "ltBdesc");
        // C (and D) are stored in FP32
        checkError(cublasLtMatrixLayoutCreate(&d_Cdesc, CUDA_R_32F,
                              SIZE, SIZE, SIZE),
                   "ltCdesc");

        cublasLtMatmulPreference_t pref;
        checkError(cublasLtMatmulPreferenceCreate(&pref), "ltPrefCreate");
        size_t workspaceSize = 0;
        checkError(cublasLtMatmulPreferenceSetAttribute(
                       pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                       &workspaceSize, sizeof(workspaceSize)),
                   "ltPrefWorkspace");

        cublasLtMatmulHeuristicResult_t heuristic;
        int returnedResults = 0;

        if (d_isFp4) {
            if (!d_AscaleData || !d_BscaleData)
                throw std::runtime_error("FP4 scale buffers were not allocated");

            void *aScalePtr = reinterpret_cast<void *>(static_cast<uintptr_t>(d_AscaleData));
            void *bScalePtr = reinterpret_cast<void *>(static_cast<uintptr_t>(d_BscaleData));
            cublasLtMatmulMatrixScale_t scalar32fMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
            checkError(cublasLtMatmulDescSetAttribute(d_ltMatmulDesc,
                                                      CUBLASLT_MATMUL_DESC_C_SCALE_MODE,
                                                      &scalar32fMode, sizeof(scalar32fMode)),
                       "ltSetCScaleMode");
            checkError(cublasLtMatmulDescSetAttribute(d_ltMatmulDesc,
                                                      CUBLASLT_MATMUL_DESC_D_SCALE_MODE,
                                                      &scalar32fMode, sizeof(scalar32fMode)),
                       "ltSetDScaleMode");
            checkError(cublasLtMatmulDescSetAttribute(d_ltMatmulDesc,
                                                      CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                      &aScalePtr, sizeof(aScalePtr)),
                       "ltSetAScalePtr");
            checkError(cublasLtMatmulDescSetAttribute(d_ltMatmulDesc,
                                                      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                      &bScalePtr, sizeof(bScalePtr)),
                       "ltSetBScalePtr");

            struct Fp4HeuristicConfig {
                cublasOperation_t transA;
                cublasLtMatmulMatrixScale_t scaleMode;
                unsigned char scaleOne;
                const char *name;
            };

            std::vector<Fp4HeuristicConfig> fp4Configs;
            fp4Configs.push_back({CUBLAS_OP_T, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, 0x38, "T+VEC16_UE4M3"});
            fp4Configs.push_back({CUBLAS_OP_N, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, 0x38, "N+VEC16_UE4M3"});
#ifdef CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
            fp4Configs.push_back({CUBLAS_OP_T, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, 0x7F, "T+VEC32_UE8M0"});
            fp4Configs.push_back({CUBLAS_OP_N, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, 0x7F, "N+VEC32_UE8M0"});
#endif

            cublasStatus_t lastStatus = CUBLAS_STATUS_NOT_SUPPORTED;
            bool found = false;
            for (size_t i = 0; i < fp4Configs.size(); ++i) {
                const auto &cfg = fp4Configs[i];
                checkError(cuMemsetD8(d_AscaleData, cfg.scaleOne, d_scaleBytes), "ltFP4AScaleInitTry");
                checkError(cuMemsetD8(d_BscaleData, cfg.scaleOne, d_scaleBytes), "ltFP4BScaleInitTry");

                checkError(cublasLtMatmulDescSetAttribute(d_ltMatmulDesc,
                                                          CUBLASLT_MATMUL_DESC_TRANSA,
                                                          &cfg.transA, sizeof(cfg.transA)),
                           "ltSetTransA");
                checkError(cublasLtMatmulDescSetAttribute(d_ltMatmulDesc,
                                                          CUBLASLT_MATMUL_DESC_TRANSB,
                                                          &opN, sizeof(opN)),
                           "ltSetTransB");
                checkError(cublasLtMatmulDescSetAttribute(d_ltMatmulDesc,
                                                          CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                          &cfg.scaleMode, sizeof(cfg.scaleMode)),
                           "ltSetAScaleMode");
                checkError(cublasLtMatmulDescSetAttribute(d_ltMatmulDesc,
                                                          CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                          &cfg.scaleMode, sizeof(cfg.scaleMode)),
                           "ltSetBScaleMode");

                returnedResults = 0;
                lastStatus = cublasLtMatmulAlgoGetHeuristic(
                    d_ltHandle, d_ltMatmulDesc,
                    d_Adesc, d_Bdesc, d_Cdesc, d_Cdesc,
                    pref, 1, &heuristic, &returnedResults);
                if (lastStatus == CUBLAS_STATUS_SUCCESS && returnedResults > 0) {
                    fprintf(stderr, "FP4 heuristic selected config: %s\n", cfg.name);
                    found = true;
                    break;
                }
                fprintf(stderr, "FP4 heuristic rejected config %s (status=%d, results=%d)\n",
                        cfg.name, (int)lastStatus, returnedResults);
            }
            if (!found) {
                checkError(lastStatus, "ltGetHeuristic");
                throw std::runtime_error("No suitable cuBLASLt FP4 matmul algorithm found");
            }
        } else {
            checkError(cublasLtMatmulAlgoGetHeuristic(
                           d_ltHandle, d_ltMatmulDesc,
                           d_Adesc, d_Bdesc, d_Cdesc, d_Cdesc,
                           pref, 1, &heuristic, &returnedResults),
                       "ltGetHeuristic");
        }
        cublasLtMatmulPreferenceDestroy(pref);

        if (returnedResults == 0)
            throw std::runtime_error("No suitable cuBLASLt matmul algorithm found for FP8/FP4");

        d_ltAlgo = heuristic.algo;
    }
#endif
};

// Returns the number of devices
int initCuda() {
    try {
        CUresult initResult = cuInit(0);
        const char *initErrStr = "<unavailable>";
        if (cuGetErrorString(initResult, &initErrStr) != CUDA_SUCCESS ||
            initErrStr == nullptr) {
                initErrStr = "<unavailable>";
            }
        fprintf(stderr, "cuInit returned %d (%s)\n", initResult,
            initErrStr);
        checkError(initResult);
    } catch (std::runtime_error e) {
        fprintf(stderr, "Couldn't init CUDA: %s\n", e.what());
        return 0;
    }
    int deviceCount = 0;
    checkError(cuDeviceGetCount(&deviceCount));

    if (!deviceCount)
        throw std::string("No CUDA devices");

#ifdef USEDEV
    if (USEDEV >= deviceCount)
        throw std::string("Not enough devices for USEDEV");
#endif

    return deviceCount;
}

template <class T>
void startBurn(int index, int writeFd, T *A, T *B, bool tensors,
               ssize_t useBytes, const char *kernelFile) {
    GPU_Test<T> *our;
    try {
        our = new GPU_Test<T>(index, tensors, kernelFile);
        our->initBuffers(A, B, useBytes);
    } catch (const std::exception &e) {
        fprintf(stderr, "Couldn't init a GPU test: %s\n", e.what());
        exit(EMEDIUMTYPE);
    }

    // The actual work
    try {
        int eventIndex = 0;
        const int maxEvents = 2;
        CUevent doneEvents[maxEvents];
        CUevent computeStartEvents[maxEvents];
        CUevent computeStopEvents[maxEvents];
        for (int i = 0; i < maxEvents; ++i)
        {
            checkError(cuEventCreate(doneEvents + i, 0), "Create done event");
            checkError(cuEventCreate(computeStartEvents + i, 0), "Create compute start event");
            checkError(cuEventCreate(computeStopEvents + i, 0), "Create compute stop event");
        }

        int nonWorkIters = maxEvents;

        while (our->shouldRun()) {
            int submitIndex = eventIndex;
            checkError(cuEventRecord(computeStartEvents[submitIndex], 0), "Record compute start");
            our->compute();
            checkError(cuEventRecord(computeStopEvents[submitIndex], 0), "Record compute stop");
            our->compare();
            checkError(cuEventRecord(doneEvents[submitIndex], 0), "Record done event");

            eventIndex = ++eventIndex % maxEvents;
            int readyIndex = eventIndex;

            while (cuEventQuery(doneEvents[readyIndex]) != CUDA_SUCCESS)
                usleep(1000);

            if (--nonWorkIters > 0)
                continue;

            float gemmMs = 0.0f;
            checkError(cuEventElapsedTime(&gemmMs,
                                          computeStartEvents[readyIndex],
                                          computeStopEvents[readyIndex]),
                       "Compute elapsed");

            BurnProgressPacket packet;
            packet.processed = (int)our->getIters();
            packet.gemmMs = gemmMs;
            packet.errors = (int)our->getErrors();
            write(writeFd, &packet, sizeof(packet));
        }

        for (int i = 0; i < maxEvents; ++i)
            checkError(cuEventSynchronize(doneEvents[i]), "Sync done event");
        for (int i = 0; i < maxEvents; ++i) {
            cuEventDestroy(doneEvents[i]);
            cuEventDestroy(computeStartEvents[i]);
            cuEventDestroy(computeStopEvents[i]);
        }
        delete our;
    } catch (const std::exception &e) {
        fprintf(stderr, "Failure during compute: %s\n", e.what());
        BurnProgressPacket packet = {-1, 0.0f, -1};
        // Signalling that we failed
        write(writeFd, &packet, sizeof(packet));
        exit(ECONNREFUSED);
    }
}

int pollTemp(pid_t *p) {
    int tempPipe[2];
    pipe(tempPipe);

    pid_t myPid = fork();

    if (!myPid) {
        close(tempPipe[0]);
        dup2(tempPipe[1], STDOUT_FILENO);
#if IS_JETSON
        execlp("tegrastats", "tegrastats", "--interval", "5000", NULL);
        fprintf(stderr, "Could not invoke tegrastats, no temps available\n");
#else
        execlp("nvidia-smi", "nvidia-smi", "-l", "5", "-q", "-d", "TEMPERATURE",
               NULL);
        fprintf(stderr, "Could not invoke nvidia-smi, no temps available\n");
#endif

        exit(ENODEV);
    }

    *p = myPid;
    close(tempPipe[1]);

    return tempPipe[0];
}

void updateTemps(int handle, std::vector<int> *temps) {
    const int readSize = 10240;
    static int gpuIter = 0;
    char data[readSize + 1];

    int curPos = 0;
    do {
        read(handle, data + curPos, sizeof(char));
    } while (data[curPos++] != '\n');

    data[curPos - 1] = 0;

#if IS_JETSON
    std::string data_str(data);
    std::regex pattern("GPU@([0-9]+)C");
    std::smatch matches;
    if (std::regex_search(data_str, matches, pattern)) {
        if (matches.size() > 1) {
            int tempValue = std::stoi(matches[1]);
            temps->at(gpuIter) = tempValue;
            gpuIter = (gpuIter + 1) % (temps->size());
        }
    }
#else
    // FIXME: The syntax of this print might change in the future..
    int tempValue;
    if (sscanf(data,
               "		GPU Current Temp			: %d C",
               &tempValue) == 1) {
        temps->at(gpuIter) = tempValue;
        gpuIter = (gpuIter + 1) % (temps->size());
    } else if (!strcmp(data, "		Gpu				"
                             "	 : N/A"))
        gpuIter =
            (gpuIter + 1) %
            (temps->size()); // We rotate the iterator for N/A values as well
#endif
}

int pollPower(pid_t *p) {
    int powerPipe[2];
    pipe(powerPipe);

    pid_t myPid = fork();

    if (!myPid) {
        close(powerPipe[0]);
        dup2(powerPipe[1], STDOUT_FILENO);
#if IS_JETSON
        execlp("tegrastats", "tegrastats", "--interval", "5000", NULL);
        fprintf(stderr, "Could not invoke tegrastats, no power data available\n");
#else
        execlp("nvidia-smi", "nvidia-smi", "-l", "5", "-q", "-d", "POWER",
               NULL);
        fprintf(stderr, "Could not invoke nvidia-smi, no power data available\n");
#endif

        exit(ENODEV);
    }

    *p = myPid;
    close(powerPipe[1]);

    return powerPipe[0];
}

void updatePower(int handle, std::vector<int> *powers) {
    const int readSize = 10240;
    static int gpuIter = 0;
    char data[readSize + 1];

    int curPos = 0;
    do {
        read(handle, data + curPos, sizeof(char));
    } while (data[curPos++] != '\n');

    data[curPos - 1] = 0;

#if IS_JETSON
    std::string data_str(data);
    std::regex pattern("POM_5V_GPU\\s+([0-9]+)mW");
    std::smatch matches;
    if (std::regex_search(data_str, matches, pattern)) {
        if (matches.size() > 1) {
            int powerMw = std::stoi(matches[1]);
            int powerW = powerMw / 1000; // rough conversion
            powers->at(gpuIter) = powerW;
            gpuIter = (gpuIter + 1) % (powers->size());
        }
    }
#else
    // Example line (spacing may vary across drivers):
    // "        Power Draw                    : 75.34 W"
    int powerValueInt = 0;
    float powerValue = 0.0f;
    if (strstr(data, "Power Draw") != NULL) {
        if (sscanf(data, "%*[^:]: %f W", &powerValue) == 1) {
            powerValueInt = (int)(powerValue + 0.5f);
            powers->at(gpuIter) = powerValueInt;
            gpuIter = (gpuIter + 1) % (powers->size());
        }
    }
#endif
}

void listenClients(std::vector<int> clientFd, std::vector<pid_t> clientPid,
                   int runTime, std::chrono::seconds sigterm_timeout_threshold_secs) {
    fd_set waitHandles;

    pid_t tempPid;
    int tempHandle = pollTemp(&tempPid);
    pid_t powerPid;
    int powerHandle = pollPower(&powerPid);
    int maxHandle = tempHandle > powerHandle ? tempHandle : powerHandle;

    FD_ZERO(&waitHandles);
    FD_SET(tempHandle, &waitHandles);
    FD_SET(powerHandle, &waitHandles);

    for (size_t i = 0; i < clientFd.size(); ++i) {
        if (clientFd.at(i) > maxHandle)
            maxHandle = clientFd.at(i);
        FD_SET(clientFd.at(i), &waitHandles);
    }

    std::vector<int> clientTemp;
    std::vector<int> clientErrors;
    std::vector<int> clientCalcs;
    std::vector<int> clientPower;
    std::vector<float> clientGflops;
    std::vector<bool> clientFaulty;

    time_t startTime = time(0);

    for (size_t i = 0; i < clientFd.size(); ++i) {
        clientTemp.push_back(0);
        clientErrors.push_back(0);
        clientCalcs.push_back(0);
        clientPower.push_back(0);
        clientGflops.push_back(0.0f);
        clientFaulty.push_back(false);
    }

    int changeCount;
    float nextReport = 10.0f;
    bool childReport = false;
    while (
        (changeCount = select(maxHandle + 1, &waitHandles, NULL, NULL, NULL))) {
        size_t thisTime = time(0);

        // Going through all descriptors
        for (size_t i = 0; i < clientFd.size(); ++i)
            if (FD_ISSET(clientFd.at(i), &waitHandles)) {
                BurnProgressPacket packet;
                int res = read(clientFd.at(i), &packet, sizeof(packet));
                if (res < (int)sizeof(packet)) {
                    fprintf(stderr, "read[%zu] error %d", i, res);
                    packet.processed = -1;
                    packet.gemmMs = 0.0f;
                    packet.errors = -1;
                }

                if (packet.processed == -1)
                    clientCalcs.at(i) = -1;
                else {
                    clientErrors.at(i) += packet.errors;
                    if (packet.gemmMs > 0.0f) {
                        clientGflops.at(i) =
                            (double)((unsigned long long int)packet.processed *
                                     OPS_PER_MUL) /
                            ((double)packet.gemmMs * 1000000.0);
                    } else {
                        clientGflops.at(i) = 0.0f;
                    }
                    clientCalcs.at(i) += packet.processed;
                }

                childReport = true;
            }

        if (FD_ISSET(tempHandle, &waitHandles))
            updateTemps(tempHandle, &clientTemp);

        if (FD_ISSET(powerHandle, &waitHandles))
            updatePower(powerHandle, &clientPower);

        // Resetting the listeners
        FD_ZERO(&waitHandles);
        FD_SET(tempHandle, &waitHandles);
        FD_SET(powerHandle, &waitHandles);
        for (size_t i = 0; i < clientFd.size(); ++i)
            FD_SET(clientFd.at(i), &waitHandles);

        // Printing progress (if a child has initted already)
        if (childReport) {
            float elapsed =
                fminf((float)(thisTime - startTime) / (float)runTime * 100.0f,
                      100.0f);
            printf("\r%.1f%%  ", elapsed);
            printf("proc'd: ");
            for (size_t i = 0; i < clientCalcs.size(); ++i) {
                printf("%d (%.0f Gflop/s GEMM) ", clientCalcs.at(i),
                       clientGflops.at(i));
                if (i != clientCalcs.size() - 1)
                    printf("- ");
            }
            printf("  errors: ");
            for (size_t i = 0; i < clientErrors.size(); ++i) {
                std::string note = "%d ";
                if (clientCalcs.at(i) == -1)
                    note += " (DIED!)";
                else if (clientErrors.at(i))
                    note += " (WARNING!)";

                printf(note.c_str(), clientErrors.at(i));
                if (i != clientCalcs.size() - 1)
                    printf("- ");
            }
            printf("  temps: ");
            for (size_t i = 0; i < clientTemp.size(); ++i) {
                printf(clientTemp.at(i) != 0 ? "%d C " : "-- ",
                       clientTemp.at(i));
                if (i != clientCalcs.size() - 1)
                    printf("- ");
            }

            printf("  power: ");
            for (size_t i = 0; i < clientPower.size(); ++i) {
                printf(clientPower.at(i) != 0 ? "%d W " : "-- ",
                       clientPower.at(i));
                if (i != clientCalcs.size() - 1)
                    printf("- ");
            }

            fflush(stdout);

            for (size_t i = 0; i < clientErrors.size(); ++i)
                if (clientErrors.at(i))
                    clientFaulty.at(i) = true;

            if (nextReport < elapsed) {
                nextReport = elapsed + 10.0f;
                printf("\n\tSummary at:   ");
                fflush(stdout);
                system("date"); // Printing a date
                fflush(stdout);
                printf("\n");
                for (size_t i = 0; i < clientErrors.size(); ++i)
                    clientErrors.at(i) = 0;
            }
        }

        // Checking whether all clients are dead
        bool oneAlive = false;
        for (size_t i = 0; i < clientCalcs.size(); ++i)
            if (clientCalcs.at(i) != -1)
                oneAlive = true;
        if (!oneAlive) {
            fprintf(stderr, "\n\nNo clients are alive!  Aborting\n");
            exit(ENOMEDIUM);
        }

        if (startTime + runTime < thisTime)
            break;
    }

    printf("\nKilling processes with SIGTERM (soft kill)\n");
    fflush(stdout);
    for (size_t i = 0; i < clientPid.size(); ++i)
        kill(clientPid.at(i), SIGTERM);

    kill(tempPid, SIGTERM);
    kill(powerPid, SIGTERM);

    // processes should be terminated by SIGTERM within threshold time (so wait and then check pids)
    std::this_thread::sleep_for(sigterm_timeout_threshold_secs);

    // check each process and see if they are alive
    std::vector<int> killed_processes; // track the number of killed processes
    // loop through pids for each client / GPU
    for (size_t i = 0; i < clientPid.size(); ++i) {
        int status;
        pid_t return_pid = waitpid(clientPid.at(i), &status, WNOHANG);
        if (return_pid == clientPid.at(i)) {
            /* child is finished. exit status in status */
            killed_processes.push_back(return_pid);
        }
    }
    // handle the tempPid
    int status;
    pid_t return_pid = waitpid(tempPid, &status, WNOHANG);
    if (return_pid == tempPid) {
        /* child is finished. exit status in status */
        killed_processes.push_back(return_pid);
    }

    // handle the powerPid
    return_pid = waitpid(powerPid, &status, WNOHANG);
    if (return_pid == powerPid) {
        /* child is finished. exit status in status */
        killed_processes.push_back(return_pid);
    }

    // number of killed process should be number GPUs + 2 (tempPid + powerPid)
    if (killed_processes.size() != clientPid.size() + 2) {
        printf("\nKilling processes with SIGKILL (force kill)\n");

        for (size_t i = 0; i < clientPid.size(); ++i) {
            // check if pid was already killed with SIGTERM before using SIGKILL
            if (std::find(killed_processes.begin(), killed_processes.end(), clientPid.at(i)) == killed_processes.end())
                kill(clientPid.at(i), SIGKILL);
        }

        // check if pid was already killed with SIGTERM before using SIGKILL
        if (std::find(killed_processes.begin(), killed_processes.end(), tempPid) == killed_processes.end())
            kill(tempPid, SIGKILL);

        if (std::find(killed_processes.begin(), killed_processes.end(), powerPid) == killed_processes.end())
            kill(powerPid, SIGKILL);
    }

    close(tempHandle);
    close(powerHandle);

    while (wait(NULL) != -1)
        ;
    printf("done\n");

    printf("\nTested %d GPUs:\n", (int)clientPid.size());
    for (size_t i = 0; i < clientPid.size(); ++i)
        printf("\tGPU %d: %s\n", (int)i, clientFaulty.at(i) ? "FAULTY" : "OK");
}

template <class T>
struct InitHostMatrices {
    static void init(T *A, T *B) {
        memset(A, 0, sizeof(T) * SIZE * SIZE);
        memset(B, 0, sizeof(T) * SIZE * SIZE);
    }
};

template <>
struct InitHostMatrices<float> {
    static void init(float *A, float *B) {
        for (size_t i = 0; i < SIZE * SIZE; ++i) {
            A[i] = (float)((double)(rand() % 1000000) / 100000.0);
            B[i] = (float)((double)(rand() % 1000000) / 100000.0);
        }
    }
};

template <>
struct InitHostMatrices<double> {
    static void init(double *A, double *B) {
        for (size_t i = 0; i < SIZE * SIZE; ++i) {
            A[i] = (double)(rand() % 1000000) / 100000.0;
            B[i] = (double)(rand() % 1000000) / 100000.0;
        }
    }
};

template <class T>
void launch(int runLength, bool useTensorCores,
            ssize_t useBytes, int device_id, const char * kernelFile,
            std::chrono::seconds sigterm_timeout_threshold_secs) {
#if IS_JETSON
    std::ifstream f_model("/proc/device-tree/model");
    std::stringstream ss_model;
    ss_model << f_model.rdbuf();
    printf("%s\n", ss_model.str().c_str());
#else
    system("nvidia-smi -L");
#endif

    // Initting A and B
    T *A = (T *)malloc(sizeof(T) * SIZE * SIZE);
    T *B = (T *)malloc(sizeof(T) * SIZE * SIZE);
    srand(10);
    InitHostMatrices<T>::init(A, B);

    // Forking a process..  This one checks the number of devices to use,
    // returns the value, and continues to use the first one.
    int mainPipe[2];
    pipe(mainPipe);
    int readMain = mainPipe[0];
    std::vector<int> clientPipes;
    std::vector<pid_t> clientPids;
    clientPipes.push_back(readMain);

    if (device_id > -1) {
        pid_t myPid = fork();
        if (!myPid) {
            // Child
            close(mainPipe[0]);
            int writeFd = mainPipe[1];
            initCuda();
            int devCount = 1;
            write(writeFd, &devCount, sizeof(int));
            startBurn<T>(device_id, writeFd, A, B, useTensorCores,
                         useBytes, kernelFile);
            close(writeFd);
            return;
        } else {
            clientPids.push_back(myPid);
            close(mainPipe[1]);
            int devCount;
            read(readMain, &devCount, sizeof(int));
            listenClients(clientPipes, clientPids, runLength, sigterm_timeout_threshold_secs);
        }
        for (size_t i = 0; i < clientPipes.size(); ++i)
            close(clientPipes.at(i));
    } else {
        pid_t myPid = fork();
        if (!myPid) {
            // Child
            close(mainPipe[0]);
            int writeFd = mainPipe[1];
            int devCount = initCuda();
            write(writeFd, &devCount, sizeof(int));

            startBurn<T>(0, writeFd, A, B, useTensorCores,
                         useBytes, kernelFile);

            close(writeFd);
            return;
        } else {
            clientPids.push_back(myPid);

            close(mainPipe[1]);
            int devCount;
            read(readMain, &devCount, sizeof(int));

            if (!devCount) {
                fprintf(stderr, "No CUDA devices\n");
                exit(ENODEV);
            } else {
                for (int i = 1; i < devCount; ++i) {
                    int slavePipe[2];
                    pipe(slavePipe);
                    clientPipes.push_back(slavePipe[0]);

                    pid_t slavePid = fork();

                    if (!slavePid) {
                        // Child
                        close(slavePipe[0]);
                        initCuda();
                        startBurn<T>(i, slavePipe[1], A, B,
                                     useTensorCores, useBytes, kernelFile);

                        close(slavePipe[1]);
                        return;
                    } else {
                        clientPids.push_back(slavePid);
                        close(slavePipe[1]);
                    }
                }

                listenClients(clientPipes, clientPids, runLength, sigterm_timeout_threshold_secs);
            }
        }
        for (size_t i = 0; i < clientPipes.size(); ++i)
            close(clientPipes.at(i));
    }

    free(A);
    free(B);
}

void showHelp() {
    printf("GPU Burn\n");
    printf("Usage: gpu-burn [OPTIONS] [TIME]\n\n");
    printf("-m X\tUse X MB of memory.\n");
    printf("-m N%%\tUse N%% of the available GPU memory.  Default is %d%%\n",
           (int)(USEMEM * 100));
    printf("-t TYPE\tData type: fp32 (default), fp64, fp16, bf16, fp8, fp4 (fp8/fp4 are Tensor Core only)\n");
    printf("-tc\tTry to use Tensor cores (required for fp8/fp4)\n");
    printf("-l\tLists all GPUs in the system\n");
    printf("-i N\tExecute only on GPU N\n");
    printf("-c FILE\tUse FILE as compare kernel.  Default is %s\n",
           COMPARE_KERNEL);
    printf("-stts T\tSet timeout threshold to T seconds for using SIGTERM to abort child processes before using SIGKILL.  Default is %d\n",
           SIGTERM_TIMEOUT_THRESHOLD_SECS);
    printf("-h\tShow this help message\n\n");
    printf("Examples:\n");
    printf("  gpu-burn -t fp64 3600 # burns all GPUs with doubles for an hour\n");
    printf(
        "  gpu-burn -m 50%% # burns using 50%% of the available GPU memory\n");
    printf("  gpu-burn -l # list GPUs\n");
    printf("  gpu-burn -i 2 # burns only GPU of index 2\n");
}

// NNN MB
// NN% <0
// 0 --- error
ssize_t decodeUSEMEM(const char *s) {
    char *s2;
    int64_t r = strtoll(s, &s2, 10);
    if (s == s2)
        return 0;
    if (*s2 == '%')
        return (s2[1] == 0) ? -r : 0;
    return (*s2 == 0) ? r * 1024 * 1024 : 0;
}

int main(int argc, char **argv) {
    int runLength = 10;
    bool useTensorCores = false;
    int thisParam = 0;
    ssize_t useBytes = 0; // 0 == use USEMEM% of free mem
    int device_id = -1;
    char *kernelFile = (char *)COMPARE_KERNEL;
    std::chrono::seconds sigterm_timeout_threshold_secs = std::chrono::seconds(SIGTERM_TIMEOUT_THRESHOLD_SECS);

    enum class DataTypeMode { FP32, FP64, FP16, BF16, FP8, FP4 };
    DataTypeMode dataTypeMode = DataTypeMode::FP32;

    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 1; i < args.size(); ++i) {
        if (argc >= 2 && std::string(argv[i]).find("-h") != std::string::npos) {
            showHelp();
            return 0;
        }
        if (argc >= 2 && std::string(argv[i]).find("-l") != std::string::npos) {
            int count = initCuda();
            if (count == 0) {
                throw std::runtime_error("No CUDA capable GPUs found.\n");
            }
            for (int i_dev = 0; i_dev < count; i_dev++) {
                CUdevice device_l;
                char device_name[255];
                checkError(cuDeviceGet(&device_l, i_dev));
                checkError(cuDeviceGetName(device_name, 255, device_l));
                size_t device_mem_l;
                checkError(cuDeviceTotalMem(&device_mem_l, device_l));
                printf("ID %i: %s, %ldMB\n", i_dev, device_name,
                       device_mem_l / 1000 / 1000);
            }
            thisParam++;
            return 0;
        }
        if (argc >= 2 &&
            std::string(argv[i]).find("-tc") != std::string::npos) {
            useTensorCores = true;
            thisParam++;
        }
        if (argc >= 2 && strcmp(argv[i], "-t") == 0) {
            thisParam++;

            if (i + 1 < args.size()) {
                i++;
                thisParam++;
                std::string t(argv[i]);
                if (t == "fp32") {
                    dataTypeMode = DataTypeMode::FP32;
                } else if (t == "fp64") {
                    dataTypeMode = DataTypeMode::FP64;
                } else if (t == "fp16") {
                    dataTypeMode = DataTypeMode::FP16;
                } else if (t == "bf16") {
                    dataTypeMode = DataTypeMode::BF16;
                } else if (t == "fp8") {
                    dataTypeMode = DataTypeMode::FP8;
                } else if (t == "fp4") {
                    dataTypeMode = DataTypeMode::FP4;
                } else {
                    fprintf(stderr, "Unknown data type for -t: %s\n", argv[i]);
                    exit(EINVAL);
                }
            } else {
                fprintf(stderr, "Syntax error near -t\n");
                exit(EINVAL);
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-m", 2) == 0) {
            thisParam++;

            // -mNNN[%]
            // -m NNN[%]
            if (argv[i][2]) {
                useBytes = decodeUSEMEM(argv[i] + 2);
            } else if (i + 1 < args.size()) {
                i++;
                thisParam++;
                useBytes = decodeUSEMEM(argv[i]);
            } else {
                fprintf(stderr, "Syntax error near -m\n");
                exit(EINVAL);
            }
            if (useBytes == 0) {
                fprintf(stderr, "Syntax error near -m\n");
                exit(EINVAL);
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-i", 2) == 0) {
            thisParam++;

            if (argv[i][2]) {
                device_id = strtol(argv[i] + 2, NULL, 0);
            } else if (i + 1 < args.size()) {
                i++;
                thisParam++;
                device_id = strtol(argv[i], NULL, 0);
            } else {
                fprintf(stderr, "Syntax error near -i\n");
                exit(EINVAL);
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-c", 2) == 0) {
            thisParam++;

            if (argv[i + 1]) {
                kernelFile = argv[i + 1];
                thisParam++;
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-stts", 2) == 0) {
            thisParam++;

            if (argv[i + 1]) {
                sigterm_timeout_threshold_secs = std::chrono::seconds(atoi(argv[i + 1]));
                thisParam++;
            }
        }
    }

    if (argc - thisParam < 2)
        printf("Run length not specified in the command line. ");
    else
        runLength = atoi(argv[1 + thisParam]);
    printf("Using compare file: %s\n", kernelFile);
    printf("Burning for %d seconds.\n", runLength);

    switch (dataTypeMode) {
    case DataTypeMode::FP64:
        launch<double>(runLength, useTensorCores, useBytes,
                       device_id, kernelFile, sigterm_timeout_threshold_secs);
        break;
    case DataTypeMode::FP16:
        launch<__half>(runLength, useTensorCores, useBytes,
                       device_id, kernelFile, sigterm_timeout_threshold_secs);
        break;
    case DataTypeMode::BF16:
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
        launch<__nv_bfloat16>(runLength, useTensorCores, useBytes,
                              device_id, kernelFile, sigterm_timeout_threshold_secs);
#else
        fprintf(stderr, "BF16 not supported by this CUDA toolkit.\n");
        return EINVAL;
#endif
        break;
    case DataTypeMode::FP8:
    #if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
        launch<fp8_e4m3_t>(runLength, useTensorCores, useBytes,
                           device_id, kernelFile, sigterm_timeout_threshold_secs);
    #else
        fprintf(stderr, "FP8 not supported by this CUDA toolkit.\n");
        return EINVAL;
    #endif
        break;
    case DataTypeMode::FP4:
    #if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
        launch<fp4_e2m1_t>(runLength, useTensorCores, useBytes,
                           device_id, kernelFile, sigterm_timeout_threshold_secs);
    #else
        fprintf(stderr, "FP4 not supported by this CUDA toolkit.\n");
        return EINVAL;
    #endif
        break;
    case DataTypeMode::FP32:
    default:
        launch<float>(runLength, useTensorCores, useBytes,
                      device_id, kernelFile, sigterm_timeout_threshold_secs);
        break;
    }

    return 0;
}
