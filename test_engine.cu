#include "utils.h"
#include <cuda_runtime.h>

#include "test_cases/baseline_cases.h"
#include "test_cases/datapath_cases.h"
#include "test_cases/interconnection_cases.h"

__global__ void wmma_kernel_fp32_acc(half *d_a, half *d_b, float *d_c,
                                     float *d_d) {
  using namespace nvcuda;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> d_frag;

  wmma::load_matrix_sync(a_frag, d_a, 16);
  wmma::load_matrix_sync(b_frag, d_b, 16);
  wmma::load_matrix_sync(c_frag, d_c, 16, wmma::mem_row_major);

  wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

  wmma::store_matrix_sync(d_d, d_frag, 16, wmma::mem_row_major);
}

__global__ void wmma_kernel_fp16_acc(half *d_a, half *d_b, half *d_c,
                                     half *d_d) {
  using namespace nvcuda;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> d_frag;

  wmma::load_matrix_sync(a_frag, d_a, 16);
  wmma::load_matrix_sync(b_frag, d_b, 16);
  wmma::load_matrix_sync(c_frag, d_c, 16, wmma::mem_row_major);

  wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

  wmma::store_matrix_sync(d_d, d_frag, 16, wmma::mem_row_major);
}

void run_test(TestCase *tc) {
  printf("\n");
  printf("----------------------------------------\n");
  printf("Test: %s\n", tc->name);
  printf("Description: %s\n", tc->description);
  printf("Accumulator Type: %s\n", tc->acc_type == ACC_FP32 ? "FP32" : "FP16");

  half *d_a, *d_b;
  size_t size_half = MATRIX_SIZE * MATRIX_SIZE * sizeof(half);

  cudaMalloc(&d_a, size_half);
  cudaMalloc(&d_b, size_half);
  cudaMemcpy(d_a, tc->h_mat_a, size_half, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, tc->h_mat_b, size_half, cudaMemcpyHostToDevice);

  if (tc->acc_type == ACC_FP32) {
    float *d_c, *d_d, *h_d;
    size_t size_float = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    cudaMalloc(&d_c, size_float);
    cudaMalloc(&d_d, size_float);
    h_d = (float *)malloc(size_float);

    cudaMemcpy(d_c, tc->h_mat_c_fp32, size_float, cudaMemcpyHostToDevice);
    wmma_kernel_fp32_acc<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();
    cudaMemcpy(h_d, d_d, size_float, cudaMemcpyDeviceToHost);

    int idx = tc->focus_row * MATRIX_SIZE + tc->focus_col;
    printf("\nResult[%d][%d] = %f\n", tc->focus_row, tc->focus_col, h_d[idx]);
    print_fp32_bits(h_d[idx]);

    cudaFree(d_c);
    cudaFree(d_d);
    free(h_d);
  } else {
    half *d_c, *d_d, *h_d;

    cudaMalloc(&d_c, size_half);
    cudaMalloc(&d_d, size_half);
    h_d = (half *)malloc(size_half);

    cudaMemcpy(d_c, tc->h_mat_c_fp16, size_half, cudaMemcpyHostToDevice);
    wmma_kernel_fp16_acc<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();
    cudaMemcpy(h_d, d_d, size_half, cudaMemcpyDeviceToHost);

    int idx = tc->focus_row * MATRIX_SIZE + tc->focus_col;
    printf("\nResult[%d][%d] = %f (FP16)\n", tc->focus_row, tc->focus_col,
           __half2float(h_d[idx]));
    print_fp16_bits(h_d[idx]);

    cudaFree(d_c);
    cudaFree(d_d);
    free(h_d);
  }

  cudaFree(d_a);
  cudaFree(d_b);
}

int main(int argc, char **argv) {
  printf("Tensor Core Reverse Engineering\n");

  if (argc < 3) {
    printf("Usage: %s <phase> <FP32|FP16>\n", argv[0]);
    printf("Example: %s 1 FP32\n", argv[0]);
    printf("Example: %s 1 FP16\n", argv[0]);
    return 1;
  }

  int phase = atoi(argv[1]);
  AccumulatorType acc_type;

  if (strcmp(argv[2], "FP32") == 0) {
    acc_type = ACC_FP32;
  } else if (strcmp(argv[2], "FP16") == 0) {
    acc_type = ACC_FP16;
  } else {
    printf("Error: Accumulator type must be FP32 or FP16\n");
    return 1;
  }

  switch (phase) {
  case 1:
    printf("----- Phase 1: Baseline Tests -----\n");
    printf("Accumulator Mode: %s\n\n", argv[2]);
    run_baseline_tests(acc_type);
    break;
  case 2:
    printf("----- Phase 2: Datapath Tests -----\n");
    printf("Accumulator Mode: %s\n\n", argv[2]);
    run_datapath_tests(acc_type);
    break;
  case 3:
    printf("----- Phase 3: Interconnection Tests -----\n");
    printf("Accumulator Mode: %s\n\n", argv[2]);
    run_interconnection_tests(acc_type);
    break;
  case 4:
    printf("----- Phase 4: Grouping Tests -----\n");
    printf("Not implemented yet.\n");
    break;
  case 5:
    printf("----- Phase 5: Edge Cases -----\n");
    printf("Not implemented yet.\n");
    break;
  default:
    printf("Invalid phase. Use 1-5.\n");
  }

  return 0;
}

void print_fp16_bits(half h) {
  unsigned short bits = *reinterpret_cast<unsigned short *>(&h);
  printf("Hexadecimal: 0x%04x ,", bits);
  printf("Binary: ");
  for (int i = 15; i >= 0; i--) {
    printf("%d", (bits >> i) & 1);
    if (i == 15 || i == 10)
      printf(" ");
  }
  printf("\n");
  printf("----------------------------------------\n");
}

void print_fp32_bits(float f) {
  unsigned int bits = *reinterpret_cast<unsigned int *>(&f);
  printf("Hexadecimal: 0x%08x: ,", bits);
  printf("Binary: ");
  for (int i = 31; i >= 0; i--) {
    printf("%d", (bits >> i) & 1);
    if (i == 31 || i == 23)
      printf(" ");
  }
  printf("\n");
  printf("----------------------------------------\n");
}

void init_test_case(TestCase *tc, const char *name, const char *desc,
                    AccumulatorType acc_type) {
  strncpy(tc->name, name, 255);
  tc->name[255] = '\0';
  strncpy(tc->description, desc, 255);
  tc->description[255] = '\0';
  tc->acc_type = acc_type;
  tc->focus_row = 0;
  tc->focus_col = 0;

  memset(tc->h_mat_a, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(half));
  memset(tc->h_mat_b, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(half));

  if (acc_type == ACC_FP32) {
    memset(tc->h_mat_c_fp32, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
  } else {
    memset(tc->h_mat_c_fp16, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(half));
  }
}

void set_element(half *mat, int row, int col, half value) {
  mat[row * MATRIX_SIZE + col] = value;
}

void set_element_fp32(float *mat, int row, int col, float value) {
  mat[row * MATRIX_SIZE + col] = value;
}

void set_element_c(TestCase *tc, int row, int col, float value) {
  int idx = row * MATRIX_SIZE + col;
  if (tc->acc_type == ACC_FP32) {
    tc->h_mat_c_fp32[idx] = value;
  } else {
    tc->h_mat_c_fp16[idx] = __float2half(value);
  }
}

void set_focus(TestCase *tc, int row, int col) {
  tc->focus_row = row;
  tc->focus_col = col;
}
