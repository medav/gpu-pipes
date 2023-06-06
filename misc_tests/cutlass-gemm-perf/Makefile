
M?=128
N?=128
K?=128
RELU?=0

EXTRA_FLAGS=-DMM=$(M) -DNN=$(N) -DKK=$(K) -DRELU=$(RELU)

APP=cutlass-gemm
include ../../common/app.mk

bench: $(APP).elf
	LD_LIBRARY_PATH=$(CUTLASS_LIB) ./$(APP).elf
	rm $(APP).elf
