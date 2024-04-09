#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"
#include "timer.h"

__global__ void nw_kernel(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, int numSequences) {
    int bid = blockIdx.x;
    if (bid < numSequences) {
        int seqIndex1 = bid * SEQUENCE_LENGTH;
        int seqIndex2 = bid * SEQUENCE_LENGTH;

        __shared__ int ref_d_shared[SEQUENCE_LENGTH];
        __shared__ int ref_hv_shared[SEQUENCE_LENGTH + 1]; 
        __shared__ int cur_shared[SEQUENCE_LENGTH];

        if (threadIdx.x == 0) {
            ref_d_shared[0] = 0;
            ref_hv_shared[0] = INSERTION;
            ref_hv_shared[1] = INSERTION;
        }

        for (int i = threadIdx.x; i < SEQUENCE_LENGTH; i += blockDim.x) {
            cur_shared[i] = 0;
        }
        __syncthreads();

        for (int d = 2; d < 2 * SEQUENCE_LENGTH; ++d) {
            int ad_length;
            if (d <= SEQUENCE_LENGTH) {
                ad_length = d - 1; 
            } else {
                ad_length = 2 * SEQUENCE_LENGTH - d - 1;
            }

            if (threadIdx.x < ad_length) {
                int i1 = threadIdx.x < SEQUENCE_LENGTH ? threadIdx.x : SEQUENCE_LENGTH - 1;
                int i2 = threadIdx.x - i1;
                int score = (sequence1_d[seqIndex1 + i1] == sequence2_d[seqIndex2 + i2]) ? MATCH : MISMATCH;

                if (d < SEQUENCE_LENGTH) { 
                    cur_shared[threadIdx.x] = max(ref_d_shared[threadIdx.x - 1] + score, max(ref_hv_shared[threadIdx.x - 1] + INSERTION, ref_hv_shared[threadIdx.x] + INSERTION));
                } else if (d == SEQUENCE_LENGTH) { 
                    cur_shared[threadIdx.x] = max(ref_d_shared[threadIdx.x - 1] + score, max(ref_hv_shared[threadIdx.x] + INSERTION, ref_hv_shared[threadIdx.x + 1] + INSERTION));
                } else { 
                    cur_shared[threadIdx.x] = max(ref_d_shared[threadIdx.x - 1] + score, max(ref_hv_shared[threadIdx.x] + INSERTION, ref_hv_shared[threadIdx.x + 1] + INSERTION));
                }
            }
            __syncthreads();

            if (threadIdx.x < ad_length && d < 2 * SEQUENCE_LENGTH - 1) {
                ref_d_shared[threadIdx.x] = ref_hv_shared[threadIdx.x];
                ref_hv_shared[threadIdx.x] = cur_shared[threadIdx.x];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            scores_d[bid] = cur_shared[SEQUENCE_LENGTH - 1];
        }
    }
}


void nw_gpu1(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {


    nw_kernel<<<numSequences, SEQUENCE_LENGTH>>>(sequence1_d, sequence2_d, scores_d, numSequences);

}
