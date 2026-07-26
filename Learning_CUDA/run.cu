#include "basic_vector_add.cuh"
#include "unified_vector_add.cuh"
#include "basic_matmul.cuh"
#include "unified_basic_matmul.cuh"
#include "tiled_matmul.cuh"
#include "unified_tiled_matmul.cuh"
#include "cublas_matmul.cuh"
#include "transposition_sort.cuh"
#include "tcore_basic_matmul.cuh"
#include "custom_matmul.cuh"
#include "custom_matmul_v0.cuh"
#include "tcore_custom_matmul_v1.cuh"
#include "tcore_custom_matmul_v2.cuh"
#include "tcore_custom_matmul_v3.cuh"
#include "tcore_custom_matmul_v4.cuh"
#include "tcore_cublas_matmul.cuh"

int main()
{
	custmat_v0::matmul();
}