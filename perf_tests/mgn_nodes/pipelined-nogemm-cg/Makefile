
LIBS?=
LIBS+=

INCS?=
INCS+=
INCS+=


.PHONY: default run objdump gdb

default: run

HDRS=$(wildcard *.cuh)
# APP=test_mgn_node_pipe
APP=test_mgn_node_pipe

$(APP): $(APP).cu $(HDRS)
	/usr/local/cuda/bin/nvcc \
		-DCUTLASS_ENABLE_L2_PREFETCH \
		--expt-relaxed-constexpr \
		-O3 \
		-std=c++17 \
		-o $@ \
		-arch=sm_80 \
		$(LIBS) \
		$(INCS) \
		$<

clean:
	rm -f $(APP)

run: $(APP)
	LD_LIBRARY_PATH=$(CUTLASS_LIB) ./$(APP)

gdb: $(APP)
	LD_LIBRARY_PATH=$(CUTLASS_LIB) /usr/local/cuda/bin/cuda-gdb ./$(APP)

memcheck: $(APP)
	LD_LIBRARY_PATH=$(CUTLASS_LIB) /usr/local/cuda/bin/cuda-memcheck ./$(APP)

objdump: $(APP)
	/usr/local/cuda/bin/cuobjdump --dump-ptx $(APP)
