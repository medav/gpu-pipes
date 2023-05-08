
CUTLASS_PATH=/nobackup/medavies/cutlass
CUTLASS_INC=$(CUTLASS_PATH)/include
CUTLASS_LIB=$(CUTLASS_PATH)/build/tools/library

LIBS?=
LIBS+=-L$(CUTLASS_LIB) -lcutlass

INCS?=
INCS+=-I../common -I../../common -I../../../common
INCS+=-I$(CUTLASS_INC) -I$(CUTLASS_PATH)/tools/util/include/
INCS+=-I$(CUTLASS_PATH)/tools/util/include


.PHONY: default run objdump gdb

default: $(APP).elf

HDRS=$(wildcard *.*h) $(wildcard ../common/*.*h) $(wildcard ../../common/*.*h)
APP?=UNKNOWN

$(APP).elf: $(APP).cu $(HDRS)
	/usr/local/cuda/bin/nvcc \
		--expt-relaxed-constexpr \
		-O3 \
		-std=c++17 \
		-o $@ \
		-arch=sm_80 \
		$(LIBS) \
		$(INCS) \
		$<

clean:
	rm -f $(APP).elf

run: $(APP).elf
	LD_LIBRARY_PATH=$(CUTLASS_LIB) ./$(APP).elf

gdb: $(APP).elf
	LD_LIBRARY_PATH=$(CUTLASS_LIB) /usr/local/cuda/bin/cuda-gdb ./$(APP).elf

memcheck: $(APP).elf
	LD_LIBRARY_PATH=$(CUTLASS_LIB) /usr/local/cuda/bin/cuda-memcheck ./$(APP).elf

objdump: $(APP).elf
	/usr/local/cuda/bin/cuobjdump --dump-ptx $(APP).elf
