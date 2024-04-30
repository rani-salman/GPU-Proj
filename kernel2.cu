#include <assert.h>

#include "common.h"
#include "timer.h"

#define COARSENING_FACTOR 1

__global__ void kernel_nw2(unsigned char* sequence1, unsigned char* sequence2, int* scores_d, unsigned int numSequences)
{
    __shared__ int curr[SEQUENCE_LENGTH];
    __shared__ int ref_b[SEQUENCE_LENGTH];
    __shared__ int ref_hv[SEQUENCE_LENGTH];

    int threadIteration = 1;

    for (unsigned int index = 0; index <= (2 * SEQUENCE_LENGTH) - 1; index += COARSENING_FACTOR) {
        int column = threadIdx.x * COARSENING_FACTOR + 1; 
        int row = threadIteration; 

        if (column <= min(SEQUENCE_LENGTH, index + 1) && row <= SEQUENCE_LENGTH && column <= SEQUENCE_LENGTH) {
            ++threadIteration;

            unsigned int seq1_idx = blockIdx.x * SEQUENCE_LENGTH + (column - 1);
            unsigned int seq2_idx = blockIdx.x * SEQUENCE_LENGTH + row - 1;

            unsigned char seq1_val = sequence1[seq1_idx];
            unsigned char seq2_val = sequence2[seq2_idx];

            int top = (row == 1) ? (column) * DELETION : (ref_b[column - 1]);
            int left = (column == 1) ? (row) * INSERTION : (ref_b[column - 2]);
            int topleft = (row == 1) ? (column - 1) * DELETION : (column == 1) ? (row - 1) * INSERTION : (ref_hv[column - 2]);
            int insertion = top + INSERTION;
            int deletion = left + DELETION;
            int match = topleft + ((seq2_val == seq1_val) ? MATCH : MISMATCH);
            int max = (insertion > deletion) ? insertion : deletion;
            max = (match > max) ? match : max;
            curr[column - 1] = max; 
        }

        __syncthreads();

if (column <= min(SEQUENCE_LENGTH, index + 1)) {
    for (int i = 0; i < COARSENING_FACTOR; ++i) {
        int idx = column + i - 1;
        if (idx < SEQUENCE_LENGTH) {
            ref_hv[idx] = ref_b[idx];
            ref_b[idx] = curr[idx];
        }
    }
}

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scores_d[blockIdx.x] = curr[SEQUENCE_LENGTH - 1];
    }
}



void nw_gpu2(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    const unsigned int numThreadsPerBlock = (SEQUENCE_LENGTH + COARSENING_FACTOR - 1) / COARSENING_FACTOR; 
    const unsigned int numBlocks = numSequences;
    cudaDeviceSynchronize();
    kernel_nw2 <<< numBlocks, numThreadsPerBlock >>> (sequence1_d, sequence2_d, scores_d, numSequences);
}
