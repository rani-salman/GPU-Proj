#include <cuda_runtime.h>
#include <assert.h>
#include "common.h"
#include "timer.h"

#define SEQUENCE_LENGTH 1024 

_global_ void nw_kernel(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, int numSequences, int* ref_d, int* ref_hv, int* cur) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < SEQUENCE_LENGTH) {
        if (threadIdx.x == 0) {
            ref_d[0] = 0;
            ref_hv[0] = INSERTION;
        }
        if (threadIdx.x + 1 <= SEQUENCE_LENGTH) {
            ref_d[threadIdx.x + 1] = (threadIdx.x + 1) * INSERTION;
            ref_hv[threadIdx.x + 1] = (threadIdx.x + 1) * INSERTION;
        }
        __syncthreads();
        for (int d = 2; d < 2 * SEQUENCE_LENGTH; ++d) {
            int start = max(1, d - SEQUENCE_LENGTH + 1);
            int end = min(d, SEQUENCE_LENGTH - 1);

            if (tid >= start && tid < end) {
                if (tid <= SEQUENCE_LENGTH && d - tid - 1 <= SEQUENCE_LENGTH) {
                    int i1 = tid;
                    int i2 = d - tid - 1;
                    int score = (sequence1_d[i1] == sequence2_d[i2]) ? MATCH : MISMATCH;
                    int top = ref_hv[i1];
                    int left = ref_d[i1 + 1];
                    int topleft = ref_d[i1];
                    cur[i1 + 1] = max(topleft + score, max(top + INSERTION, left + DELETION));
                }
            }
            __syncthreads();
            if (tid < end && (tid + 1) < SEQUENCE_LENGTH) {
                ref_d[tid + 1] = cur[tid + 1];
                ref_hv[tid + 1] = cur[tid + 1];
            }
            __syncthreads();
        }
        if (blockIdx.x < numSequences) {
            scores_d[blockIdx.x] = cur[SEQUENCE_LENGTH -1 ];
        }

    }
}


void nw_gpu0(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences, int* ref_d, int* ref_hv, int* cur) {
    int blocksPerGrid = numSequences;
    nw_kernel<<<blocksPerGrid, SEQUENCE_LENGTH>>>(sequence1_d, sequence2_d, scores_d, numSequences, ref_d, ref_hv, cur);
    cudaDeviceSynchronize();
}