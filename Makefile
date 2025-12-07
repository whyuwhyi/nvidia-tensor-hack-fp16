NVCC = nvcc
ARCH = sm_120
CFLAGS = -arch=$(ARCH) -I.
TARGET = build/test_engine
OUTDIR = output

$(shell mkdir -p build)
$(shell mkdir -p $(OUTDIR))

run: $(TARGET)

$(TARGET): test_engine.cu utils.h test_cases/*.h
	$(NVCC) $(CFLAGS) test_engine.cu -o $(TARGET)

phase1_fp16: $(TARGET)
	@./$(TARGET) 1 FP16 | tee $(OUTDIR)/results_phase1_fp16.txt

phase1_fp32: $(TARGET)
	@./$(TARGET) 1 FP32 | tee $(OUTDIR)/results_phase1_fp32.txt

phase2_fp16: $(TARGET)
	@./$(TARGET) 2 FP16 | tee $(OUTDIR)/results_phase2_fp16.txt

phase2_fp32: $(TARGET)
	@./$(TARGET) 2 FP32 | tee $(OUTDIR)/results_phase2_fp32.txt

phase3_fp16: $(TARGET)
	@./$(TARGET) 3 FP16 | tee $(OUTDIR)/results_phase3_fp16.txt

phase3_fp32: $(TARGET)
	@./$(TARGET) 3 FP32 | tee $(OUTDIR)/results_phase3_fp32.txt

phase1: phase1_fp16 phase1_fp32

phase2: phase2_fp16 phase2_fp32

phase3: phase3_fp16 phase3_fp32

all: phase1 phase2 phase3

clean:
	@rm -rf build
