#ifndef INTERCONNECTION_CASES_H
#define INTERCONNECTION_CASES_H

#include "../utils.h"

TestCase create_fp16_c_addition_positive_test(int pos0, int pos1, int pos2,
                                              AccumulatorType acc_type) {
  TestCase tc;
  char name[100];
  sprintf(name, "FP16 C Add Pos [%d,%d,%d]", pos0, pos1, pos2);
  init_test_case(&tc, name, "C matrix addition with RNE rounding", acc_type);

  half one = __float2half(1.0f);
  half val1 = __float2half(ldexpf(1.0f, -9));
  half val2 = __float2half(ldexpf(1.0f, -10));

  set_element(tc.h_mat_a, 0, pos0, one);
  set_element(tc.h_mat_a, 0, pos1, val1);
  set_element(tc.h_mat_a, 0, pos2, val2);
  set_element(tc.h_mat_b, pos0, 0, one);
  set_element(tc.h_mat_b, pos1, 0, one);
  set_element(tc.h_mat_b, pos2, 0, one);

  set_element_c(&tc, 0, 0, 1.0f);

  return tc;
}

TestCase create_fp16_c_addition_negative_test(int pos0, int pos1, int pos2,
                                              AccumulatorType acc_type) {
  TestCase tc;
  char name[100];
  sprintf(name, "FP16 C Add Neg [%d,%d,%d]", pos0, pos1, pos2);
  init_test_case(&tc, name, "C matrix addition with RNE rounding", acc_type);

  half one = __float2half(1.0f);
  half neg_one = __float2half(-1.0f);
  half val1 = __float2half(-ldexpf(1.0f, -9));
  half val2 = __float2half(-ldexpf(1.0f, -10));

  set_element(tc.h_mat_a, 0, pos0, one);
  set_element(tc.h_mat_a, 0, pos1, val1);
  set_element(tc.h_mat_a, 0, pos2, val2);
  set_element(tc.h_mat_b, pos0, 0, neg_one);
  set_element(tc.h_mat_b, pos1, 0, one);
  set_element(tc.h_mat_b, pos2, 0, one);

  set_element_c(&tc, 0, 0, -1.0f);

  return tc;
}

TestCase create_fp32_c_addition_positive_test(int pos0, int pos1, int pos2,
                                              AccumulatorType acc_type) {
  TestCase tc;
  char name[100];
  sprintf(name, "FP32 C Add Pos [%d,%d,%d]", pos0, pos1, pos2);
  init_test_case(&tc, name, "C matrix addition with RNE rounding", acc_type);

  half one = __float2half(1.0f);
  half val1 = __float2half(ldexpf(1.0f, -22));
  half val2 = __float2half(ldexpf(1.0f, -23));

  set_element(tc.h_mat_a, 0, pos0, one);
  set_element(tc.h_mat_a, 0, pos1, val1);
  set_element(tc.h_mat_a, 0, pos2, val2);
  set_element(tc.h_mat_b, pos0, 0, one);
  set_element(tc.h_mat_b, pos1, 0, one);
  set_element(tc.h_mat_b, pos2, 0, one);

  set_element_c(&tc, 0, 0, 1.0f);

  return tc;
}

TestCase create_fp32_c_addition_negative_test(int pos0, int pos1, int pos2,
                                              AccumulatorType acc_type) {
  TestCase tc;
  char name[100];
  sprintf(name, "FP32 C Add Neg [%d,%d,%d]", pos0, pos1, pos2);
  init_test_case(&tc, name, "C matrix addition with RNE rounding", acc_type);

  half one = __float2half(1.0f);
  half neg_one = __float2half(-1.0f);
  half val1 = __float2half(-ldexpf(1.0f, -22));
  half val2 = __float2half(-ldexpf(1.0f, -23));

  set_element(tc.h_mat_a, 0, pos0, one);
  set_element(tc.h_mat_a, 0, pos1, val1);
  set_element(tc.h_mat_a, 0, pos2, val2);
  set_element(tc.h_mat_b, pos0, 0, neg_one);
  set_element(tc.h_mat_b, pos1, 0, one);
  set_element(tc.h_mat_b, pos2, 0, one);

  set_element_c(&tc, 0, 0, -1.0f);

  return tc;
}

TestCase create_grouping_test(int pos0, int pos1, int pos2,
                              AccumulatorType acc_type) {
  TestCase tc;
  char name[100];
  sprintf(name, "Grouping [%d,%d,%d]", pos0, pos1, pos2);
  init_test_case(&tc, name, "Test 8-element grouping (T0,T1) vs (T2,T3)",
                 acc_type);

  half big = __float2half(ldexpf(1.0f, 15));
  half neg_big = __float2half(-ldexpf(1.0f, 15));
  half small = __float2half(ldexpf(1.0f, -7));

  set_element(tc.h_mat_a, 0, pos0, big);
  set_element(tc.h_mat_a, 0, pos1, big);
  set_element(tc.h_mat_a, 0, pos2, small);
  set_element(tc.h_mat_b, pos0, 0, big);
  set_element(tc.h_mat_b, pos1, 0, neg_big);
  set_element(tc.h_mat_b, pos2, 0, small);

  return tc;
}

TestCase create_c_addition_position_test(int n, AccumulatorType acc_type) {
  TestCase tc;
  char name[100];
  sprintf(name, "C addition position test [%d]", n);
  init_test_case(&tc, name, "Test if C is addition together with input values",
                 acc_type);

  half big = __float2half(ldexpf(1.0f, 15));
  half neg_big = __float2half(-ldexpf(1.0f, 15));
  half small = __float2half(ldexpf(1.0f, n));

  set_element(tc.h_mat_a, 0, 0, big);
  set_element(tc.h_mat_a, 0, 1, big);
  set_element(tc.h_mat_b, 0, 0, big);
  set_element(tc.h_mat_b, 1, 0, neg_big);

  set_element_c(&tc, 0, 0, small);

  return tc;
}

void run_interconnection_tests(AccumulatorType acc_type) {
  TestCase tc;

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      if (j == i)
        continue;
      for (int k = 0; k < 16; k++) {
        if (k == i || k == j)
          continue;

        tc = create_fp16_c_addition_positive_test(i, j, k, acc_type);
        run_test(&tc);
        tc = create_fp16_c_addition_negative_test(i, j, k, acc_type);
        run_test(&tc);
        tc = create_fp32_c_addition_positive_test(i, j, k, acc_type);
        run_test(&tc);
        tc = create_fp32_c_addition_negative_test(i, j, k, acc_type);
        run_test(&tc);

        tc = create_grouping_test(i, j, k, acc_type);
        run_test(&tc);
      }
    }
  }
  for (int n = -7; n <= 7; n++) {
    tc = create_c_addition_position_test(n, acc_type);
    run_test(&tc);
  }
}

#endif
