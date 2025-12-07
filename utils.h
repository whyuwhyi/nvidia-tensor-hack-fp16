#ifndef UTILS_H
#define UTILS_H

#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <string.h>

#define MATRIX_SIZE 16

enum AccumulatorType { ACC_FP32, ACC_FP16 };

struct TestCase {
  char name[256];
  char description[256];
  half h_mat_a[MATRIX_SIZE * MATRIX_SIZE];
  half h_mat_b[MATRIX_SIZE * MATRIX_SIZE];

  union {
    float h_mat_c_fp32[MATRIX_SIZE * MATRIX_SIZE];
    half h_mat_c_fp16[MATRIX_SIZE * MATRIX_SIZE];
  };

  AccumulatorType acc_type;
  int focus_row;
  int focus_col;
};

void print_fp16_bits(half h);
void print_fp32_bits(float f);
void init_test_case(TestCase *tc, const char *name, const char *desc,
                    AccumulatorType acc_type);
void set_element(half *mat, int row, int col, half value);
void set_element_fp32(float *mat, int row, int col, float value);
void set_element_c(TestCase *tc, int row, int col, float value);
void set_focus(TestCase *tc, int row, int col);
void run_test(TestCase *tc);

#endif
