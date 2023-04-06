#pragma once
#include "mgn_node_pipe.cuh"


template<typename Block_, size_t NumWarps_>
class Mlp2s0 {
public:
    using Block = Block_;

    static constexpr size_t num_warps = NumWarps_;
    static constexpr size_t nblk_per_warp = Block::kK / num_warps;
    static constexpr size_t read_bytes = Block::kM * Block::kK * sizeof(half);
    static constexpr size_t write_bytes = Block::kM * Block::kN * sizeof(half);
    static constexpr size_t buf_len = 3;

    using GemmOp = ThreadBlockGemmTensorOp<
        num_warps, cutlass::gemm::GemmShape<Block::kM, Block::kN, Block::kK>>;

    using SharedState =
        cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, 2>;

    using Buffer = SmemBuffers<
        SharedState,
        half,
        buf_len,
        Block::kM,
        Block::kN,
        Block::kK>;

    static constexpr size_t smem_bytes = sizeof(Buffer);

private:
    const size_t lane_id;
    const size_t warp_id;
    cooperative_groups::thread_block this_block;

    GemmOp gemm_op;
    const size_t num_iters;
    Buffer * buf;

public:
    __device__ Mlp2s0(half * smem, size_t prob_m, size_t _lane_id, size_t _warp_id) :
        lane_id(_lane_id),
        warp_id(_warp_id),
        this_block(cooperative_groups::this_thread_block()),
        gemm_op(),
        num_iters(prob_m / Block::kM),
        buf((Buffer *)smem) { }

    __device__ void load_weights_async(half * weight) {
        #pragma unroll
        for (size_t i = 0; i < Block::kN; i++) {
            cooperative_groups::memcpy_async(
                this_block,
                &buf->weight[i][0],
                weight,
                Block::kK * sizeof(half));
        }
    }

    template<typename QT>
    __device__ void read_input(ssize_t seq_n_, QT& in_q) {
        if (seq_n_ >= num_iters) return;
        const size_t mbase = seq_n_ * Block::kM;

        auto& slot = in_q.read_wait(seq_n_);
        cooperative_groups::memcpy_async(
            this_block,
            (void *)&buf->in[seq_n_ % buf_len][0][0],
            (void *)&slot.data.buf[0][0],
            Block::kM * Block::kK * sizeof(half)
        );
    };

    __device__ void read_accum(ssize_t seq_n_) { };

    __device__ void write_output(ssize_t seq_n_, half * output) {
        if (seq_n_ < 0) return;
        const size_t wbytes = write_bytes;

        cooperative_groups::memcpy_async(
            this_block,
            (void *)&output[seq_n_ * Block::kM * Block::kN],
            (void *)&buf->out[seq_n_ % buf_len][0][0],
            wbytes
        );
    };

    template<typename QT>
    __device__ void read_commit(ssize_t seq_n_, QT& in_q) {
        if (seq_n_ < 0) return;
        in_q.read_commit(seq_n_);
    }

    template<typename QT>
    __device__ void write_commit(ssize_t seq_n_, QT& out_q) { }

    __device__ void compute(ssize_t seq_n_) {
        if (seq_n_ >= num_iters) return;
        size_t bufi = seq_n_ % buf_len;
        // gemm_op(
        //     (cutlass::half_t *)&buf->in[bufi][0][0],
        //     (cutlass::half_t *)&buf->weight[0][0],
        //     (cutlass::half_t *)&buf->out[bufi][0][0],
        //     warp_id,
        //     lane_id);
    };


    template<typename IQ>
    __device__ void run(half * weight, IQ& in_q, half * output) {
        load_weights_async(weight);
        this_block.sync();

        read_input(0, in_q);
        read_accum(0);

        for (ssize_t seq_n = 1; seq_n < num_iters + 3; seq_n++) {
            this_block.sync();              // Barrier
            read_commit(seq_n - 1, in_q);         // Non-blocking
            write_commit(seq_n - 3, output); // Non-blocking

            read_input(seq_n, in_q);       // Asynchronous / Non-Blocking
            read_accum(seq_n);              // Asynchronous / Non-Blocking
            write_output(seq_n - 2, output); // Asynchronous / Non-Blocking

            compute(seq_n - 1);             // Blocks
        }
    }
};
