# nvidia-tensor-hack-fp16

A reverse engineering project to understand the internal behavior and implementation details of NVIDIA Tensor Cores using FP16 matrix operations.

## Overview

This project uses the CUDA WMMA (Warp Matrix Multiply-Accumulate) API to probe Tensor Core behavior through carefully designed test cases. It performs 16x16 FP16 matrix multiplications with both FP32 and FP16 accumulators to analyze:

- Edge case handling (NaN, Infinity, Subnormal, Zero)
- Internal datapath architecture
- Interconnection patterns between processing elements
- Rounding and precision behavior

## Requirements

- NVIDIA GPU with Tensor Core support (sm_120 architecture - Blackwell)
- CUDA Toolkit
- nvcc compiler

## Build

```bash
make
```

## Usage

Run tests by specifying a phase number and accumulator type:

```bash
./build/test_engine <phase> <FP32|FP16>
```

### Examples

```bash
# Run Phase 1 baseline tests with FP32 accumulator
./build/test_engine 1 FP32

# Run Phase 1 baseline tests with FP16 accumulator
./build/test_engine 1 FP16
```

### Convenience Targets

```bash
# Run specific phase with specific accumulator
make phase1_fp16    # Phase 1 with FP16 accumulator
make phase1_fp32    # Phase 1 with FP32 accumulator
make phase2_fp16    # Phase 2 with FP16 accumulator
make phase2_fp32    # Phase 2 with FP32 accumulator
make phase3_fp16    # Phase 3 with FP16 accumulator
make phase3_fp32    # Phase 3 with FP32 accumulator

# Run all tests for a specific phase
make phase1
make phase2
make phase3

# Run all tests
make all
```

## Test Phases

### Phase 1: Baseline Tests

Tests fundamental edge cases and special value handling:

- NaN propagation and payload preservation
- Infinity handling with correct signs
- Subnormal number processing
- Zero sign handling

### Phase 2: Datapath Tests

Explores the internal datapath architecture and computational flow.

### Phase 3: Interconnection Tests

Investigates how processing elements are interconnected within Tensor Cores.

### Phase 4-5: (Not yet implemented)

Planned for grouping patterns and additional edge cases.

## Output

Test results are automatically saved to the `output/` directory with detailed information including:

- Test name and description
- Result values
- Hexadecimal and binary representations of output

## Project Structure

```
nvidia-tensor-hack-fp16/
├── test_engine.cu              # Main test runner
├── utils.h                     # Utility functions and test infrastructure
├── test_cases/
│   ├── baseline_cases.h        # Phase 1 test cases
│   ├── datapath_cases.h        # Phase 2 test cases
│   └── interconnection_cases.h # Phase 3 test cases
├── build/                      # Compiled binaries
└── output/                     # Test results
```

## Clean

```bash
make clean
```
