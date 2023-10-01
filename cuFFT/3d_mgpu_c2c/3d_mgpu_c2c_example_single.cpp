/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <array>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cufftXt.h>

#include "cufft_utils.h"

#include <chrono>
using namespace std::chrono;
using namespace std;

using cpudata_t = std::vector<std::complex<float>>;
using gpus_t = std::vector<int>;
using dim_t = std::array<size_t, 3>;

void fill_array(cpudata_t &array) {
    for (size_t i = 0; i < array.size(); ++i) {
        float real = 0.0;
        float imag = 0.0;
        array[i] = {real, imag};
    };
};

void spmg(dim_t fft, gpus_t gpus, cpudata_t &h_data_in, cpudata_t &h_data_out,
          cufftXtSubFormat_t subformat) {

    // Initiate cufft plan
    cufftHandle plan{};
    CUFFT_CALL(cufftCreate(&plan));

#if CUFFT_VERSION >= 10400
    // Create CUDA Stream
    cudaStream_t stream{};
    CUDA_RT_CALL(cudaStreamCreate(&stream));
    CUFFT_CALL(cufftSetStream(plan, stream));
#endif

    // Define which GPUS are to be used
    CUFFT_CALL(cufftXtSetGPUs(plan, gpus.size(), gpus.data()));

    // Create the plan
    // With multiple gpus, worksize will contain multiple sizes
    size_t workspace_sizes[gpus.size()];
    CUFFT_CALL(cufftMakePlan3d(plan, fft[0], fft[1], fft[2], CUFFT_C2C, workspace_sizes));

    cudaLibXtDesc *indesc;

    // Copy input data to GPUs
    CUFFT_CALL(cufftXtMalloc(plan, &indesc, subformat));
    CUFFT_CALL(cufftXtMemcpy(plan, reinterpret_cast<void *>(indesc),
                             reinterpret_cast<void *>(h_data_in.data()),
                             CUFFT_COPY_HOST_TO_DEVICE));

// Execute the plan
// warmup
    CUFFT_CALL(cufftXtExecDescriptor(plan, indesc, indesc, CUFFT_FORWARD));

CUDA_RT_CALL(cudaDeviceSynchronize());
auto start = high_resolution_clock::now();
for(int i=0; i<20; ++i)
  {
    CUFFT_CALL(cufftXtExecDescriptor(plan, indesc, indesc, CUFFT_FORWARD));
  }
CUDA_RT_CALL(cudaDeviceSynchronize());
auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop-start);

std::cout << fft[0] << ", " << duration.count()/20. << std::endl;

    CUFFT_CALL(cufftXtFree(indesc));
    CUFFT_CALL(cufftDestroy(plan));

#if CUFFT_VERSION >= 10400
    CUDA_RT_CALL(cudaStreamDestroy(stream));
#endif
};

/** Runs single and multi-GPU version of cuFFT plan then compares results.
 * Maximum FFT size limited by single GPU memory.
 * */
int main(int argc, char *argv[]) {

    int m = atoi(argv[1]);
    size_t n = static_cast<size_t>(m);

    dim_t fft = {n, n, n};
    // can be {0, 0} to run on single-GPU system or if GPUs are not of same architecture
    gpus_t gpus = {0, 1};

    size_t element_count = fft[0] * fft[1] * fft[2];

    cpudata_t data_in(element_count);
    fill_array(data_in);

    cpudata_t data_out_reference(element_count, {-1.0f, -1.0f});
    cpudata_t data_out_test(element_count, {-0.5f, -0.5f});

    cufftXtSubFormat_t decomposition = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;

    spmg(fft, gpus, data_in, data_out_test, decomposition);

    return EXIT_SUCCESS;
};
