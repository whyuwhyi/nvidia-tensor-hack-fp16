#ifndef BASELINE_CASES_H
#define BASELINE_CASES_H

#include "../utils.h"

TestCase create_nan_test1(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "NaN Propagation 1",
                 "Test if NaN payload is propagated or fixed qNaN is produced",
                 acc_type);

  unsigned short nan_bits = 0x7C01;
  half a = *reinterpret_cast<half *>(&nan_bits);
  half b = __float2half(1.0f);

  set_element(tc.h_mat_a, 0, 0, a);
  set_element(tc.h_mat_b, 0, 0, b);

  return tc;
}

TestCase create_nan_test2(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "NaN Propagation 2",
                 "Test if NaN payload is propagated or fixed qNaN is produced",
                 acc_type);

  half a = __float2half(0.0f);
  half b = __float2half(100000.0f);

  set_element(tc.h_mat_a, 0, 0, a);
  set_element(tc.h_mat_b, 0, 0, b);

  return tc;
}

TestCase create_nan_test3(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "NaN Propagation 3",
                 "Test if NaN payload is propagated or fixed qNaN is produced",
                 acc_type);

  half a = __float2half(100000.0f);
  half b = __float2half(-100000.0f);
  half c = __float2half(1.0f);

  set_element(tc.h_mat_a, 0, 0, a);
  set_element(tc.h_mat_a, 0, 1, b);
  set_element(tc.h_mat_b, 0, 0, c);
  set_element(tc.h_mat_b, 1, 0, c);

  return tc;
}

TestCase create_inf_test1(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "Infinity Handling 1",
                 "Test infinity output with correct sign", acc_type);

  half a = __float2half(65500.0f);

  unsigned inf_bits = 0x7F800000;
  float c = *reinterpret_cast<float *>(&inf_bits);

  set_element(tc.h_mat_a, 0, 0, a);
  set_element(tc.h_mat_b, 0, 0, a);
  set_element_c(&tc, 0, 0, c);

  return tc;
}

TestCase create_inf_test2(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "Infinity Handling 2",
                 "Test infinity output with correct sign", acc_type);

  half a = __float2half(65500.0f);
  half b = __float2half(-65500.0f);

  unsigned inf_bits = 0xFF800000;
  float c = *reinterpret_cast<float *>(&inf_bits);

  set_element(tc.h_mat_a, 0, 0, a);
  set_element(tc.h_mat_b, 0, 0, b);
  set_element_c(&tc, 0, 0, c);

  return tc;
}

TestCase create_subnormal_test1(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "Subnormal Input 1",
                 "Test input/output subnormals are correctly computed",
                 acc_type);

  unsigned short subnormal_bits = 0x0001;
  half a = *reinterpret_cast<half *>(&subnormal_bits);
  half b = __float2half(1.0f);

  set_element(tc.h_mat_a, 0, 0, a);
  set_element(tc.h_mat_b, 0, 0, b);

  return tc;
}

TestCase create_subnormal_test2(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "Subnormal Input 2",
                 "Test input/output subnormals are correctly computed",
                 acc_type);

  half a = __float2half(0.0f);
  half b = __float2half(1.0f);

  unsigned subnormal_fp32_bits = 0x00000100;
  float c = *reinterpret_cast<float *>(&subnormal_fp32_bits);

  set_element(tc.h_mat_a, 0, 0, a);
  set_element(tc.h_mat_b, 0, 0, b);
  set_element_c(&tc, 0, 0, c);

  return tc;
}

TestCase create_zero_test1(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "Zero Sign 1", "Test sign of zero output", acc_type);

  unsigned short neg_zero_bits = 0x8000;
  half a = *reinterpret_cast<half *>(&neg_zero_bits);
  half b = __float2half(1.0f);

  set_element(tc.h_mat_a, 0, 0, a);
  set_element(tc.h_mat_a, 0, 1, a);
  set_element(tc.h_mat_b, 0, 0, b);
  set_element(tc.h_mat_b, 1, 0, b);

  return tc;
}

TestCase create_zero_test2(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "Zero Sign 2", "Test sign of zero output", acc_type);

  unsigned short neg_zero_bits = 0x8000;
  half a = *reinterpret_cast<half *>(&neg_zero_bits);
  half b = __float2half(1.0f);

  set_element(tc.h_mat_a, 0, 0, a);
  set_element(tc.h_mat_b, 0, 0, b);

  return tc;
}

TestCase create_zero_test3(AccumulatorType acc_type) {
  TestCase tc;
  init_test_case(&tc, "Zero Sign 3", "Test sign of zero output", acc_type);

  unsigned short neg_zero_bits = 0x8000;
  half a = *reinterpret_cast<half *>(&neg_zero_bits);
  half b = __float2half(1.0f);

  for (int i = 0; i < MATRIX_SIZE; i++) {
    set_element(tc.h_mat_a, 0, i, a);
    set_element(tc.h_mat_b, i, 0, b);
  }

  set_element_c(&tc, 0, 0, -0.0f);

  return tc;
}

void run_baseline_tests(AccumulatorType acc_type) {
  TestCase tc;

  tc = create_nan_test1(acc_type);
  run_test(&tc);
  tc = create_nan_test2(acc_type);
  run_test(&tc);
  tc = create_nan_test3(acc_type);
  run_test(&tc);

  tc = create_inf_test1(acc_type);
  run_test(&tc);
  tc = create_inf_test2(acc_type);
  run_test(&tc);

  tc = create_subnormal_test1(acc_type);
  run_test(&tc);
  tc = create_subnormal_test2(acc_type);
  run_test(&tc);

  tc = create_zero_test1(acc_type);
  run_test(&tc);
  tc = create_zero_test2(acc_type);
  run_test(&tc);
  tc = create_zero_test3(acc_type);
  run_test(&tc);
}

#endif
