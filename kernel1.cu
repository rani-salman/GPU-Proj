#include <assert.h>

#include "common.h"
#include "timer.h"

__global__ void kernel_nw1 (unsigned char* sequence1, unsigned char* sequence2, int* scores_d, unsigned int numSequences)
{
    __shared__ int curr[SEQUENCE_LENGTH];
    __shared__ int ref_b[SEQUENCE_LENGTH];
    __shared__ int ref_hv[SEQUENCE_LENGTH];

    int Iteration = 1;

    for (unsigned int i = 0; i <= (2 * SEQUENCE_LENGTH) - 1; ++i) {
        int column = threadIdx.x + 1;
        int row = Iteration;

        if (column <= i + 1 && row <= SEQUENCE_LENGTH && column <= SEQUENCE_LENGTH) {
            ++Iteration;
            int top = (row == 1) ? (column) * DELETION : (ref_b[column - 1]); 
            int left = (column == 1) ? (row) * INSERTION : (ref_b[column - 2]); 
            int topleft = (row == 1) ? (column - 1) * DELETION : (column == 1) ? (row - 1) * INSERTION : (ref_hv[column - 2]);
            int insertion = top + INSERTION;
            int deletion = left + DELETION;
            int match = topleft + ((sequence2[blockIdx.x * SEQUENCE_LENGTH + (row - 1)] == sequence1[blockIdx.x * SEQUENCE_LENGTH + (column - 1)]) ? MATCH : MISMATCH); 
            int max = (insertion > deletion) ? insertion : deletion;
            max = (match > max) ? match : max;
            curr[column - 1] = max;
        }

        __syncthreads();
        if (column <= min(SEQUENCE_LENGTH, i + 1)) {
                if (column <= min(SEQUENCE_LENGTH, i + 1)) {
                    ref_hv[column - 1] = ref_b[column - 1];
                    ref_b[column - 1] = curr[column - 1];
                }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scores_d[blockIdx.x] = curr[SEQUENCE_LENGTH - 1];
    }
}



void nw_gpu1(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {
    const unsigned int numThreadsPerBlock = SEQUENCE_LENGTH;
    const unsigned int numBlocks = numSequences;
    cudaDeviceSynchronize();
    kernel_nw1 <<< numBlocks, numThreadsPerBlock >>> (sequence1_d, sequence2_d, scores_d, numSequences);
}
