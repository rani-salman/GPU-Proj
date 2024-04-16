
#include <assert.h>

#include "common.h"
#include "timer.h"



#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"
#include "timer.h"

__global__ void nw_kernel1(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, int numSequences) {
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

        int num_elements = (SEQUENCE_LENGTH + blockDim.x - 1) / blockDim.x; // Calculate the number of elements each thread should handle

        // Initialize multiple elements per thread
        for (int i = threadIdx.x * num_elements; i < min((threadIdx.x + 1) * num_elements, SEQUENCE_LENGTH); i++) {
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

            // Process multiple elements per thread
            for (int i = threadIdx.x * num_elements; i < min((threadIdx.x + 1) * num_elements, ad_length); i++) {
                int i1 = i < SEQUENCE_LENGTH ? i : SEQUENCE_LENGTH - 1;
                int i2 = i - i1;
                int score = (sequence1_d[seqIndex1 + i1] == sequence2_d[seqIndex2 + i2]) ? MATCH : MISMATCH;

                if (d < SEQUENCE_LENGTH) {
                    cur_shared[i] = max(ref_d_shared[i - 1] + score, max(ref_hv_shared[i - 1] + INSERTION, ref_hv_shared[i] + INSERTION));
                } else if (d == SEQUENCE_LENGTH) {
                    cur_shared[i] = max(ref_d_shared[i - 1] + score, max(ref_hv_shared[i] + INSERTION, ref_hv_shared[i + 1] + INSERTION));
                } else {
                    cur_shared[i] = max(ref_d_shared[i - 1] + score, max(ref_hv_shared[i] + INSERTION, ref_hv_shared[i + 1] + INSERTION));
                }
            }
            __syncthreads();

            // Update ref_d_shared and ref_hv_shared for multiple elements per thread
            for (int i = threadIdx.x * num_elements; i < min((threadIdx.x + 1) * num_elements, ad_length); i++) {
                ref_d_shared[i] = ref_hv_shared[i];
                ref_hv_shared[i] = cur_shared[i];
            }
            __syncthreads();
        }

        // Only the first thread writes the final score
        if (threadIdx.x == 0) {
            scores_d[bid] = cur_shared[SEQUENCE_LENGTH - 1];
        }
    }
}



void nw_gpu2(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    nw_kernel1<<<numSequences, SEQUENCE_LENGTH>>>(sequence1_d, sequence2_d, scores_d, numSequences);



}

