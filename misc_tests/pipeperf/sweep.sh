
set -x

for P in 1 2 4 8 16 32 64 128 256 512 1024 2048; do
    make clean
    make PLKB=$P NWARPS=32 NOSYNC=-DNOSYNC

    for NQ in 54; do
        echo "==== Payload: $P KB, # Queues: $NQ ===="
        # ./../../run_ncu.sh
        ./pipeperf.elf 100000 $NQ
    done
done


# for P in 8192; do
#     make clean
#     make PLKB=$P NOSYNC=-DNOSYNC

#     for NQ in 1 27 54; do
#         echo "==== Payload: $P KB, # Queues: $NQ ===="
#         ./pipeperf.elf 100000 $NQ 32
#     done
# done
