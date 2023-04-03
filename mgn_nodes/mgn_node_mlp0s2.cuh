#pragma once
#include "mgn_node_pipe.cuh"


template<typename Block_, size_t NumWarps_>
class Mlp0s2 {
public:
    using Block = Block_;

    static const size_t num_warps = NumWarps_;
    static const size_t nblk_per_warp = Block::kK / num_warps;
    static const size_t read_bytes = Block::kM * Block::kK * sizeof(half);
    static const size_t write_bytes = Block::kM * Block::kN * sizeof(half);
    static const size_t buf_len = 3;

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
    __device__ Mlp0s2(half * smem, size_t prob_m, size_t _lane_id, size_t _warp_id) :
        lane_id(_lane_id),
        warp_id(_warp_id),
        this_block(cooperative_groups::this_thread_block()),
        gemm_op(),
        num_iters(prob_m / Block::kM),
        buf((Buffer *)smem) { }

    __device__ void load_weights_async(half * weight) {
        for (size_t i = 0; i < Block::kN; i++) {
            cooperative_groups::memcpy_async(
                this_block,
                &buf->weight[i][0],
                weight,
                Block::kK * sizeof(half));
        }
    }

    __device__ void read_input(ssize_t seq_n_, half * input) {
        if (seq_n_ >= num_iters) return;
        const size_t mbase = seq_n_ * Block::kM;

        cooperative_groups::memcpy_async(
            this_block,
            (void *)&buf->in[seq_n_ % buf_len][0][0],
            (void *)&input[mbase * Block::kK],
            Block::kM * Block::kK * sizeof(half)
        );
    };

    template<typename QT>
    __device__ void read_accum(ssize_t seq_n_, QT& in_q) {
        if (seq_n_ >= num_iters) return;
        const size_t wbytes = write_bytes;

        auto& slot = in_q.read_wait(seq_n_);
        cooperative_groups::memcpy_async(
            this_block,
            (void *)&slot.data.buf[0][0],
            (void *)&buf->out[seq_n_ % buf_len][0][0],
            wbytes
        );
    };

    template<typename QT>
    __device__ void write_output(ssize_t seq_n_, QT& out_q) {
        if (seq_n_ < 0) return;
        const size_t wbytes = write_bytes;

        auto& slot = out_q.write_wait(seq_n_);
        cooperative_groups::memcpy_async(
            this_block,
            (void *)&slot.data.buf[0][0],
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
    __device__ void write_commit(ssize_t seq_n_, QT& out_q) {
        if (seq_n_ < 0) return;
        out_q.write_commit(seq_n_);
    }

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


    template<typename QIN, typename QOUT>
    __device__ void run(half * weight, half * input, QIN& accum_in_q, QOUT& out_q) {
        out_q.reset();

        load_weights_async(weight);
        this_block.sync();

        read_input(0, input);
        read_accum(0, accum_in_q);

        for (ssize_t seq_n = 1; seq_n < num_iters + 3; seq_n++) {
            this_block.sync();
            read_commit(seq_n - 1, accum_in_q);
            write_commit(seq_n - 3, out_q);

            read_input(seq_n, input);
            read_accum(seq_n, accum_in_q);

            write_output(seq_n - 2, out_q);

            compute(seq_n - 1);
        }
    }
};

