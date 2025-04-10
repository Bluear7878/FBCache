#include "gemm_w4a4.cuh"

namespace nunchaku::kernels {

template<typename Config>
class GEMM_W4A4_Launch {
    using GEMM = GEMM_W4A4<Config>;
//     using LoraRanks = std::integer_sequence<int, 0, 32>;
    using LoraRanks = std::integer_sequence<int, 0, 32, 48, 64, 80, 96, 112, 128, 160, 176, 224>;
//     using LoraRanks = std::integer_sequence<int,
//     0, 32, 48, 64, 80, 96, 112, 128, 144, 160,
//     176, 192, 208, 224, 240, 256, 272, 288, 304, 320,
//     336, 352, 368, 384, 400, 416, 432, 448, 464, 480,
//     496, 512>;

    using packed_act_t    = typename GEMM::packed_act_t;
    using packed_wgt_t    = typename GEMM::packed_wgt_t;
    using packed_ascale_t = typename GEMM::packed_ascale_t;
    using packed_wscale_t = typename GEMM::packed_wscale_t;
    using packed_amscale_t = typename GEMM::packed_amscale_t;
    using packed_wmscale_t = typename GEMM::packed_wmscale_t;
    using packed_fpsum_t  = typename GEMM::packed_fpsum_t;
    using half_t          = typename GEMM::half_t;

public:
    static void gemm_w4a4(
        Tensor act,          // packed act [M, K / 2]
        Tensor wgt,          // packed act [N, K / 2]
        Tensor out,          // linear     [M, N]
        Tensor qout,         // packed act [M, N / 2]
        Tensor ascales,      // packed as  [K / 64, M]
        Tensor wscales,      // packed ws  [K / 64, N]
        Tensor oscales,      // packed as  [N / 64, M]
        Tensor poolout,      // linear     [M / PoolSize, N]
        Tensor lora_act_in,  // packed lora_act [M, R]
        Tensor lora_up,      // packed lora_wgt [N, R]
        Tensor lora_down,    // packed lora_wgt [N, R]
        Tensor lora_act_out, // packed lora_act [M, R]
        Tensor norm_q,       // linear     [HEAD_DIM]
        Tensor norm_k,       // linear     [HEAD_DIM]
        Tensor rotary_emb,   // linear     [M, HEAD_DIM / 2, 2, 2]
        Tensor bias,         // packed ws  [N]
        Tensor smooth_factor, // packed ws  [N], for quantization of the next layer
        Tensor out_vk,        // linear     [B, num_heads, head_dim + 1, head_dim]
        Tensor out_linearattn,// linear     [B, (M), N / 3]
        bool act_unsigned,
        std::vector<float> lora_scales,  // [R / 16]
        bool fuse_silu,
        bool fp4,
        float alpha,
        Tensor wcscales       // packed ws  [N]  
    );
    static void quantize_w4a4_act_fuse_lora(Tensor input, Tensor output, Tensor oscales, Tensor lora_down, Tensor lora_act_out, Tensor smooth, bool fuse_glu, bool fp4);
    static void quantize_w4a4_act(Tensor input, Tensor output, Tensor oscales);
    static void quantize_w4a4_wgt(Tensor input, Tensor output, Tensor oscales);

    static void linearattn_vk_mul_q(Tensor q, Tensor vk);
};


};  // namespace nunchaku::kernels