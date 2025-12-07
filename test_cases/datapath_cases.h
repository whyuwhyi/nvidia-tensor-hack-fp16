#ifndef DATAPATH_CASES_H
#define DATAPATH_CASES_H

#include "../utils.h"

TestCase create_big_small_permutation_test(int pos0, int pos1, int pos2,
                                           AccumulatorType acc_type) {
  TestCase tc;
  char name[100];
  sprintf(name, "Big-Small Perm [%d,%d,%d]", pos0, pos1, pos2);
  init_test_case(&tc, name, "Test no internal rounding points", acc_type);

  half sqrt_big = __float2half(ldexpf(1.0f, 15));
  half neg_sqrt_big = __float2half(-ldexpf(1.0f, 15));
  half sqrt_small = __float2half(ldexpf(1.0f, -7));

  set_element(tc.h_mat_a, 0, pos0, sqrt_big);
  set_element(tc.h_mat_a, 0, pos1, sqrt_big);
  set_element(tc.h_mat_a, 0, pos2, sqrt_small);
  set_element(tc.h_mat_b, pos0, 0, sqrt_big);
  set_element(tc.h_mat_b, pos1, 0, neg_sqrt_big);
  set_element(tc.h_mat_b, pos2, 0, sqrt_small);

  return tc;
}

TestCase create_datapath_width_test(int n, AccumulatorType acc_type) {
  TestCase tc;
  char name[100];
  sprintf(name, "Datapath Width N=%d", n);
  init_test_case(&tc, name, "Scan to find datapath width", acc_type);

  half sqrt_big = __float2half(ldexpf(1.0f, 15));
  half neg_sqrt_big = __float2half(-ldexpf(1.0f, 15));

  int exp_a = n / 2;
  int exp_b = (n + 1) / 2;

  half a = __float2half(ldexpf(1.0f, exp_a));
  half b = __float2half(ldexpf(1.0f, exp_b));

  set_element(tc.h_mat_a, 0, 0, sqrt_big);
  set_element(tc.h_mat_a, 0, 1, neg_sqrt_big);
  set_element(tc.h_mat_a, 0, 2, a);
  set_element(tc.h_mat_b, 0, 0, sqrt_big);
  set_element(tc.h_mat_b, 1, 0, sqrt_big);
  set_element(tc.h_mat_b, 2, 0, b);

  return tc;
}

TestCase create_integer_overflow_test(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "Integer Overflow", "Test integer overflow truncation",
                 acc_type);

  half one = __float2half(1.0f);
  half val1 = __float2half(ldexpf(1.0f, -23));
  half val2 = __float2half(ldexpf(1.0f, -24));

  set_element(tc.h_mat_a, 0, 0, one);
  set_element(tc.h_mat_a, 0, 1, one);
  set_element(tc.h_mat_a, 0, 2, val1);
  set_element(tc.h_mat_a, 0, 3, val2);
  set_element(tc.h_mat_b, 0, 0, one);
  set_element(tc.h_mat_b, 1, 0, one);
  set_element(tc.h_mat_b, 2, 0, one);
  set_element(tc.h_mat_b, 3, 0, one);

  return tc;
}

TestCase create_fp32_truncation_positive_test(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "FP32 Truncation Positive",
                 "Test FP32 truncation with positive values", acc_type);

  half one = __float2half(1.0f);
  half val1 = __float2half(ldexpf(1.0f, -23));
  half val2 = __float2half(ldexpf(1.0f, -24));

  set_element(tc.h_mat_a, 0, 0, one);
  set_element(tc.h_mat_a, 0, 1, val1);
  set_element(tc.h_mat_a, 0, 2, val2);
  set_element(tc.h_mat_b, 0, 0, one);
  set_element(tc.h_mat_b, 1, 0, one);
  set_element(tc.h_mat_b, 2, 0, one);

  return tc;
}

TestCase create_fp32_truncation_negative_test(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "FP32 Truncation Negative",
                 "Test FP32 truncation with negative values", acc_type);

  half pos_one = __float2half(1.0f);
  half neg_one = __float2half(-1.0f);
  half val1 = __float2half(-ldexpf(1.0f, -23));
  half val2 = __float2half(-ldexpf(1.0f, -24));

  set_element(tc.h_mat_a, 0, 0, neg_one);
  set_element(tc.h_mat_a, 0, 1, val1);
  set_element(tc.h_mat_a, 0, 2, val2);
  set_element(tc.h_mat_b, 0, 0, pos_one);
  set_element(tc.h_mat_b, 1, 0, pos_one);
  set_element(tc.h_mat_b, 2, 0, pos_one);

  return tc;
}

TestCase create_fp16_rne_positive_test(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "FP16 RNE Positive",
                 "Test FP16 RNE rounding with positive values", acc_type);

  half one = __float2half(1.0f);
  half val1 = __float2half(ldexpf(1.0f, -10));
  half val2 = __float2half(ldexpf(1.0f, -11));

  set_element(tc.h_mat_a, 0, 0, one);
  set_element(tc.h_mat_a, 0, 1, val1);
  set_element(tc.h_mat_a, 0, 2, val2);
  set_element(tc.h_mat_b, 0, 0, one);
  set_element(tc.h_mat_b, 1, 0, one);
  set_element(tc.h_mat_b, 2, 0, one);

  return tc;
}

TestCase create_fp16_rne_negative_test(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "FP16 RNE Negative",
                 "Test FP16 RNE rounding with negative values", acc_type);

  half pos_one = __float2half(1.0f);
  half neg_one = __float2half(-1.0f);
  half val1 = __float2half(-ldexpf(1.0f, -10));
  half val2 = __float2half(-ldexpf(1.0f, -11));

  set_element(tc.h_mat_a, 0, 0, neg_one);
  set_element(tc.h_mat_a, 0, 1, val1);
  set_element(tc.h_mat_a, 0, 2, val2);
  set_element(tc.h_mat_b, 0, 0, pos_one);
  set_element(tc.h_mat_b, 1, 0, pos_one);
  set_element(tc.h_mat_b, 2, 0, pos_one);

  return tc;
}

TestCase create_fp16_datapath_sticky_test(int n, AccumulatorType acc_type) {
  TestCase tc;
  char name[100];
  sprintf(name, "FP16 Datapath Sticky N=%d", n);
  init_test_case(&tc, name, "Test FP16 uses FP32 datapath with sticky bit",
                 acc_type);

  half one = __float2half(1.0f);
  half val1 = __float2half(ldexpf(1.0f, -11));

  half val2 = __float2half(ldexpf(1.0f, n));

  set_element(tc.h_mat_a, 0, 0, one);
  set_element(tc.h_mat_a, 0, 1, val1);
  set_element(tc.h_mat_a, 0, 2, val2);
  set_element(tc.h_mat_b, 0, 0, one);
  set_element(tc.h_mat_b, 1, 0, one);
  set_element(tc.h_mat_b, 2, 0, one);

  return tc;
}

void run_datapath_tests(AccumulatorType acc_type) {
  TestCase tc;

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      if (j == i)
        continue;
      for (int k = 0; k < 16; k++) {
        if (k == i || k == j)
          continue;
        tc = create_big_small_permutation_test(i, j, k, acc_type);
        run_test(&tc);
      }
    }
  }

  for (int n = -28; n <= 30; n++) {
    tc = create_datapath_width_test(n, acc_type);
    run_test(&tc);
  }

  tc = create_integer_overflow_test(acc_type);
  run_test(&tc);

  tc = create_fp32_truncation_positive_test(acc_type);
  run_test(&tc);

  tc = create_fp32_truncation_negative_test(acc_type);
  run_test(&tc);

  tc = create_fp16_rne_positive_test(acc_type);
  run_test(&tc);

  tc = create_fp16_rne_negative_test(acc_type);
  run_test(&tc);

  for (int n = -30; n <= -12; n++) {
    tc = create_fp16_datapath_sticky_test(n, acc_type);
    run_test(&tc);
  }
}

#endif
