#include "FluxModel.h"
#include "kernels/misc_kernels.h"
#include "kernels/gemm_batched.h"
#include "flash_api.h"
#include "activation.h"

#include <nvtx3/nvToolsExt.h>

#include <iostream>

using spdlog::fmt_lib::format;
using namespace nunchaku;



Tensor forward_mlp(GEMM_W4A4 &fc1, GEMM_W4A4 &fc2, Tensor norm_hidden_states) {
    Tensor ff_output = fc2.forward_quant(
        std::get<GEMM_W4A4::QuantizedActivation>(fc1.forward(norm_hidden_states, GEMM_W4A4::FuseOptions::GELU_QUANT, &fc2))
    );
    return ff_output;
}

// Tensor forward_mlp(GEMM_W8A8 &fc1, GEMM_W8A8 &fc2, Tensor norm_hidden_states) {
//     Tensor ff_output = fc2.forward(fc1.forward(norm_hidden_states), GEMM_W8A8::FuseOptions::GELU);
//     return ff_output;
// }


Tensor forward_fc(GEMM_W4A4 &fc, Tensor x) {
    return fc.forward(x);
    // return std::get<Tensor>(fc.forward(x));
}

// Tensor forward_fc(GEMM_W8A8 &fc, Tensor x) {
//     return fc.forward(x);
// }


AdaLayerNormZeroSingle::AdaLayerNormZeroSingle(int dim, Tensor::ScalarType dtype, Device device) :
    dim(dim),
    linear(dim, 3 * dim, true, dtype, device),
    norm(dim, 1e-6, false, dtype, device) 
{
    registerChildren
        (linear, "linear")
        (norm, "norm")
    ;
}

AdaLayerNormZeroSingle::Output AdaLayerNormZeroSingle::forward(Tensor x, Tensor emb) {
    debug("emb_input", emb);
    emb = linear.forward(Silu::forward(emb));
    debug("emb_linear", emb);
    auto &&[shift_msa, scale_msa, gate_msa] = kernels::split_mod<3>(emb);
    debug("scale_msa", scale_msa);
    debug("shift_msa", shift_msa);

    debug("x", x);
    Tensor norm_x = norm.forward(x);
    debug("norm_x", norm_x);
    
    kernels::mul_add(norm_x, scale_msa, shift_msa);
    return Output{norm_x, gate_msa};
}

AdaLayerNormZero::AdaLayerNormZero(int dim, bool pre_only, Tensor::ScalarType dtype, Device device) : 
    dim(dim), pre_only(pre_only),
    linear(dim, pre_only ? 2 * dim : 6 * dim, true, dtype, device),
    norm(dim, 1e-6, false, dtype, device)
{
    registerChildren
        (linear, "linear")
        (norm, "norm")
    ;
}

AdaLayerNormZero::Output AdaLayerNormZero::forward(Tensor x, Tensor emb) {
    debug("x", x);

    debug("emb_input", emb);
    emb = linear.forward(Silu::forward(emb));
    debug("emb_linear", emb);

    if (pre_only) {
        auto &&[shift_msa, scale_msa] = kernels::split_mod<2>(emb);
        debug("shift_msa", shift_msa);

        Tensor norm_x = norm.forward(x);
        debug("norm_x", norm_x);

        kernels::mul_add(norm_x, scale_msa, shift_msa);
        debug("norm_x_scaled", norm_x);
        
        return Output{norm_x};
    } else {
        auto &&[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] = kernels::split_mod<6>(emb);
        debug("shift_msa", shift_msa);

        Tensor norm_x = norm.forward(x);
        debug("norm_x", norm_x);

        kernels::mul_add(norm_x, scale_msa, shift_msa);
        debug("norm_x_scaled", norm_x);

        return Output{norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp};
    }
}


Attention::Attention(int num_heads, int dim_head, Device device) : 
    num_heads(num_heads), dim_head(dim_head), force_fp16(false)
{
    headmask_type = Tensor::allocate({num_heads}, Tensor::INT32, Device::cpu());
    for (int i = 0; i < num_heads; i++) {
        headmask_type.data_ptr<int32_t>()[i] = i + 1;
    }
    headmask_type = headmask_type.copy(device);
}

Tensor Attention::forward(Tensor qkv, Tensor pool_qkv, float sparsityRatio) {
    const bool cast_fp16 = this->force_fp16 && qkv.scalar_type() != Tensor::FP16;

    assert(qkv.ndims() == 3);

    const Device device = qkv.device();
    const int batch_size = qkv.shape[0];
    const int num_tokens = qkv.shape[1];
    assert(qkv.shape[2] == num_heads * dim_head * 3);

    constexpr int POOL_SIZE = 128;
    const int pool_tokens = ceilDiv(num_tokens, POOL_SIZE);

    Tensor blockmask;

    if (pool_qkv.valid()) {
        assert(pool_qkv.shape[0] == batch_size);
        assert(pool_qkv.shape[1] == pool_tokens);
        assert(pool_qkv.shape[2] == num_heads * dim_head * 3);
    }

    Tensor pool_score = Tensor::allocate({batch_size, num_heads, pool_tokens, pool_tokens}, Tensor::FP32, device);

    if (pool_qkv.valid() && sparsityRatio > 0) {
        pool_qkv = pool_qkv.view({batch_size, pool_tokens, 3, num_heads, dim_head});
        pool_qkv = pool_qkv.transpose(1, 2).transpose(2, 3);    // [batch_size, 3, num_heads, poolTokens, dim_head]
        for (int i = 0; i < batch_size; i++) {
            Tensor pool_q = pool_qkv.slice(0, i, i+1).slice(1, 0, 1);
            Tensor pool_k = pool_qkv.slice(0, i, i+1).slice(1, 1, 2);
            Tensor pool_s = pool_score.slice(0, i, i+1);
            gemm_batched_fp16(pool_q, pool_k, pool_s);
        }
    }
    
    blockmask = kernels::topk(pool_score, pool_tokens * (1 - sparsityRatio));

    if (cu_seqlens_cpu.valid()) {
        if (cu_seqlens_cpu.shape[0] != batch_size + 1) {
            cu_seqlens_cpu = Tensor{};
        } else {
            for (int i = 0; i <= batch_size; i++) {
                if (cu_seqlens_cpu.data_ptr<int32_t>()[i] != num_tokens * i) {
                    cu_seqlens_cpu = Tensor{};
                    break;
                }
            }
        }
    }
    if (!cu_seqlens_cpu.valid()) {
        cu_seqlens_cpu = Tensor::allocate({batch_size + 1}, Tensor::INT32, Device::cpu());
        cu_seqlens_cpu.data_ptr<int32_t>()[0] = 0;
        for (int i = 1; i <= batch_size; i++) {
            cu_seqlens_cpu.data_ptr<int32_t>()[i] = cu_seqlens_cpu.data_ptr<int32_t>()[i - 1] + num_tokens;
        }
    }

    if (cast_fp16) {
        Tensor tmp = Tensor::empty(qkv.shape.dataExtent, Tensor::FP16, qkv.device());
        kernels::cast(qkv, tmp);
        qkv = tmp;
    }

    debug("qkv", qkv);

    Tensor cu_seqlens = cu_seqlens_cpu.copy(device);

    Tensor reshaped = qkv.view({batch_size * num_tokens, num_heads * 3, dim_head});
    Tensor q = reshaped.slice(1, 0, num_heads);
    Tensor k = reshaped.slice(1, num_heads, num_heads * 2);
    Tensor v = reshaped.slice(1, num_heads * 2, num_heads * 3);

    spdlog::debug("q,k,v={}", q.shape.str());

    Tensor raw_attn_output = mha_fwd_block(
        q, k, v,
        cu_seqlens, cu_seqlens,
        POOL_SIZE, POOL_SIZE,
        headmask_type,
        {},
        blockmask,
        num_tokens,
        num_tokens,
        0.0f,
        pow(q.shape[-1], (-0.5)),
        false, false, false, -1, -1
    ).front();

    debug("raw_attn_output", raw_attn_output);

    if (cast_fp16) {
        Tensor tmp = Tensor::empty(raw_attn_output.shape.dataExtent, Tensor::BF16, raw_attn_output.device());
        kernels::cast(raw_attn_output, tmp);
        raw_attn_output = tmp;
    }

    /**
    Tensor raw_attn_output = mha_varlen_fwd(q, k, v,
        cu_seqlens,
        cu_seqlens,
        concat.shape[1],
        concat.shape[1],
        0.0f,
        pow(q.shape[-1], (-0.5)),
        false,
        true,
        -1, -1,
        false
    ).front();

    Tensor raw_attn_output = mha_fwd(q, k, v, 
        0.0f, 
        pow(q.shape[-1], (-0.5)), 
        false, -1, -1, false
    ).front();

    Tensor raw_attn_output = mha_varlen_fwd(
        q, k, v,
        cu_seqlens, cu_seqlens,
        num_tokens_img + num_tokens_context, num_tokens_img + num_tokens_context,
        0.0f,
        pow(q.shape[-1], (-0.5)),
        false, false, -1, -1, false
    ).front();
    **/

    assert(raw_attn_output.shape[0] == batch_size * num_tokens);
    assert(raw_attn_output.shape[1] == num_heads);
    assert(raw_attn_output.shape[2] == dim_head);

    return raw_attn_output;
}

void Attention::setForceFP16(Module *module, bool value) {
    spdlog::info("{} force fp16 attention", value ? "Enable" : "Disable");

    module->traverse([&](Module *m) {
        if (Attention *attn = dynamic_cast<Attention *>(m)) {
            attn->force_fp16 = value;
        }
    });
}

FluxSingleTransformerBlock::FluxSingleTransformerBlock(int dim, int num_attention_heads, int attention_head_dim, int mlp_ratio, bool use_fp4, Tensor::ScalarType dtype, Device device) :
    dim(dim), 
    dim_head(attention_head_dim / num_attention_heads),
    num_heads(num_attention_heads),
    mlp_hidden_dim(dim * mlp_ratio),
    norm(dim, dtype, device),
    mlp_fc1(dim, mlp_hidden_dim, true, use_fp4, dtype, device),
    mlp_fc2(mlp_hidden_dim, dim, true, use_fp4, dtype, device),
    qkv_proj(dim, dim * 3, true, use_fp4, dtype, device),
    norm_q(dim_head, 1e-6, false, dtype, device),
    norm_k(dim_head, 1e-6, false, dtype, device),
    attn(num_attention_heads, attention_head_dim / num_attention_heads, device),
    out_proj(dim, dim, true, use_fp4, dtype, device)
{
    registerChildren
        (norm, "norm")
        (mlp_fc1, "mlp_fc1")
        (mlp_fc2, "mlp_fc2")
        (qkv_proj, "qkv_proj")
        (norm_q, "norm_q")
        (norm_k, "norm_k")
        (attn, "attn")
        (out_proj, "out_proj")
    ;
}

Tensor FluxSingleTransformerBlock::forward(Tensor hidden_states, Tensor temb, Tensor rotary_emb) {

    nvtxRangePushA("FluxSingleTransformerBlock");

    const int batch_size = hidden_states.shape[0];
    const int num_tokens = hidden_states.shape[1];

    auto &&[norm_hidden_states, gate] = this->norm.forward(hidden_states, temb);
    debug("norm_hidden_states", norm_hidden_states);
    debug("gate", gate);

    Tensor residual = hidden_states;

    Tensor qkv = Tensor::allocate({batch_size, num_tokens, dim * 3}, norm_hidden_states.scalar_type(), norm_hidden_states.device());
    // qkv_proj.forward(norm_hidden_states, qkv, {});
    // debug("qkv_raw", qkv);

    debug("rotary_emb", rotary_emb);
    qkv_proj.forward(norm_hidden_states, qkv, {}, norm_q.weight, norm_k.weight, rotary_emb);
    debug("qkv", qkv);
    // Tensor qkv = forward_fc(qkv_proj, norm_hidden_states);
    
    Tensor attn_output = attn.forward(qkv, {}, 0);
    attn_output = attn_output.reshape({batch_size, num_tokens, num_heads * dim_head});
    debug("raw_attn_output", attn_output);

    attn_output = forward_fc(out_proj, attn_output);
    debug("attn_output", attn_output);

    Tensor ff_output = forward_mlp(mlp_fc1, mlp_fc2, norm_hidden_states);
    debug("ff_output", ff_output);

    hidden_states = kernels::add(attn_output, ff_output);
    debug("attn_ff_output", hidden_states);
    
    kernels::mul_add(hidden_states, gate, residual);

    nvtxRangePop();

    return hidden_states;
}

JointTransformerBlock::JointTransformerBlock(int dim, int num_attention_heads, int attention_head_dim, bool context_pre_only, bool use_fp4, Tensor::ScalarType dtype, Device device) : 
    dim(dim),
    dim_head(attention_head_dim / num_attention_heads),
    num_heads(num_attention_heads),
    context_pre_only(context_pre_only),
    norm1(dim, false, dtype, device),
    norm1_context(dim, context_pre_only, dtype, device),
    qkv_proj(dim, dim * 3, true, use_fp4, dtype, device),
    qkv_proj_context(dim, dim * 3, true, use_fp4, dtype, device),
    norm_q(dim_head, 1e-6, false, dtype, device),
    norm_k(dim_head, 1e-6, false, dtype, device),
    norm_added_q(dim_head, 1e-6, false, dtype, device),
    norm_added_k(dim_head, 1e-6, false, dtype, device),
    attn(num_attention_heads, attention_head_dim / num_attention_heads, device),
    out_proj(dim, dim, true, use_fp4, dtype, device),
    out_proj_context(dim, dim, true, use_fp4, dtype, device),
    norm2(dim, 1e-6, false, dtype, device),
    norm2_context(dim, 1e-6, false, dtype, device),
    mlp_fc1(dim, dim * 4, true, use_fp4, dtype, device),
    mlp_fc2(dim * 4, dim, true, use_fp4, dtype, device),
    mlp_context_fc1(dim, dim * 4, true, use_fp4, dtype, device),
    mlp_context_fc2(dim * 4, dim, true, use_fp4, dtype, device)
{
    registerChildren
        (norm1, "norm1")
        (norm1_context, "norm1_context")
        (qkv_proj, "qkv_proj")
        (qkv_proj_context, "qkv_proj_context")
        (norm_q, "norm_q")
        (norm_k, "norm_k")
        (norm_added_q, "norm_added_q")
        (norm_added_k, "norm_added_k")
        (attn, "attn")
        (out_proj, "out_proj")
        (out_proj_context, "out_proj_context")
        (norm2, "norm2")
        (norm2_context, "norm2_context")
        (mlp_fc1, "mlp_fc1")
        (mlp_fc2, "mlp_fc2")
        (mlp_context_fc1, "mlp_context_fc1")
        (mlp_context_fc2, "mlp_context_fc2")
    ;
}


// hidden_states: [Batch, Width * Height, dim]
// encoder_hidden_states: [Batch, Token, dim]
std::tuple<Tensor, Tensor> JointTransformerBlock::forward(Tensor hidden_states, Tensor encoder_hidden_states, Tensor temb, Tensor rotary_emb, Tensor rotary_emb_context, float sparsityRatio) {
    int batch_size = hidden_states.shape[0];
    assert(encoder_hidden_states.shape[0] == batch_size);

    nvtxRangePushA("JointTransformerBlock");

    nvtxRangePushA("AdaNorm");


    int num_tokens_img = hidden_states.shape[1];
    int num_tokens_context = encoder_hidden_states.shape[1];
    
    assert(hidden_states.shape[2] == dim);
    assert(encoder_hidden_states.shape[2] == dim);

    spdlog::debug("hidden_states={} encoder_hidden_states={} temb={}", hidden_states.shape.str(), encoder_hidden_states.shape.str(), temb.shape.str());
    spdlog::debug("batch_size={} num_tokens_img={} num_tokens_context={}", batch_size, num_tokens_img, num_tokens_context);

    //std::cerr << "norm1forward" << std::endl;
    //std::cerr << "hidden_states shape: " << hidden_states.shape.str() << std::endl;
    //std::cerr << "encoder_hidden_states shape: " << encoder_hidden_states.shape.str() << std::endl;
    //std::cerr << "temb shape: " << temb.shape.str() << std::endl;


    auto norm1_output = norm1.forward(hidden_states, temb);

    auto norm1_context_output = norm1_context.forward(encoder_hidden_states, temb);
   
#if 0
    norm1_output.x = hidden_states;
    norm1_context_output.x = encoder_hidden_states;
#endif

    debug("norm_hidden_states", norm1_output.x);
    debug("norm_encoder_hidden_states", norm1_context_output.x);


    

    constexpr int POOL_SIZE = Attention::POOL_SIZE;

    nvtxRangePop();

    auto stream = getCurrentCUDAStream();
    Tensor concat;
    Tensor pool;
    
    {
        nvtxRangePushA("qkv_proj");

        const bool blockSparse = sparsityRatio > 0;

        

        const int poolTokens = num_tokens_img / POOL_SIZE + num_tokens_context / POOL_SIZE;
        concat = Tensor::allocate({batch_size, num_tokens_img + num_tokens_context, dim * 3}, norm1_output.x.scalar_type(), norm1_output.x.device());

        pool = blockSparse
            ? Tensor::allocate({batch_size, poolTokens, dim * 3}, norm1_output.x.scalar_type(), norm1_output.x.device())
            : Tensor{};
        
        for (int i = 0; i < batch_size; i++) 
        {
            
            // img first
            Tensor qkv = concat.slice(0, i, i + 1).slice(1, 0, num_tokens_img);
            Tensor qkv_context = concat.slice(0, i, i + 1).slice(1, num_tokens_img, num_tokens_img + num_tokens_context);

            Tensor pool_qkv = pool.valid() 
                ? pool.slice(0, i, i + 1).slice(1, 0, num_tokens_img / POOL_SIZE) 
                : Tensor{};
            Tensor pool_qkv_context = pool.valid() 
                ? concat.slice(0, i, i + 1).slice(1, num_tokens_img / POOL_SIZE, num_tokens_img / POOL_SIZE + num_tokens_context / POOL_SIZE)
                : Tensor{};

            // qkv_proj.forward(norm1_output.x.slice(0, i, i + 1), qkv);
            // debug("qkv_raw", qkv);
            //std::cerr << "FluxModel::slice2" << std::endl;
            debug("rotary_emb", rotary_emb);
            //std::cerr << norm1_output.x.shape.str() << std::endl; //**error**
            Tensor rotary_emb_slice = rotary_emb.slice(0, i, i + 1);

            qkv_proj.forward(norm1_output.x.slice(0, i, i + 1), qkv, pool_qkv, norm_q.weight, norm_k.weight, rotary_emb_slice);
            debug("qkv", qkv);

            // qkv_proj_context.forward(norm1_context_output.x.slice(0, i, i + 1), qkv_context);
            // debug("qkv_context_raw", qkv_context);

            debug("rotary_emb_context", rotary_emb_context);
            //std::cerr << "FluxModel::slice3" << std::endl;
            Tensor rotary_emb_context_slice = rotary_emb_context.slice(0, i, i + 1);
            qkv_proj_context.forward(norm1_context_output.x.slice(0, i, i + 1), qkv_context, pool_qkv_context, norm_added_q.weight, norm_added_k.weight, rotary_emb_context_slice);
            debug("qkv_context", qkv_context);
            //std::cerr << "FluxModel::slice4" << std::endl;
        }

        nvtxRangePop();
    }
    //std::cerr << "FluxModel::Concat" << std::endl;

    spdlog::debug("concat={}", concat.shape.str());
    debug("concat", concat);

    assert(concat.shape[2] == num_heads * dim_head * 3);

    nvtxRangePushA("Attention");

    //std::cerr << "FluxModel::raw_attn_output" << std::endl;
    Tensor raw_attn_output = attn.forward(concat, pool, sparsityRatio);
    

    nvtxRangePop();

    spdlog::debug("raw_attn_output={}", raw_attn_output.shape.str());

    raw_attn_output = raw_attn_output.view({batch_size, num_tokens_img + num_tokens_context, num_heads, dim_head});
    debug("raw_attn_output", raw_attn_output);


    {
        nvtxRangePushA("o_proj");

        auto &&[_, gate_msa, shift_mlp, scale_mlp, gate_mlp] = norm1_output;

        // raw_attn_output: [batch_size, num_tokens_img + num_tokens_context, num_heads * dim_head]


        //std::cerr << "FluxModel::o_proj" << std::endl;
        Tensor raw_attn_output_split;
        if (batch_size == 1) {
            raw_attn_output_split = raw_attn_output.slice(1, 0, num_tokens_img).reshape({batch_size, num_tokens_img, num_heads * dim_head});
        } else {
            raw_attn_output_split = Tensor::allocate({batch_size, num_tokens_img, num_heads * dim_head}, raw_attn_output.scalar_type(), raw_attn_output.device());
            checkCUDA(cudaMemcpy2DAsync(
                raw_attn_output_split.data_ptr(), 
                num_tokens_img * num_heads * dim_head * raw_attn_output_split.scalar_size(),
               raw_attn_output.data_ptr(),
                (num_tokens_img + num_tokens_context) * num_heads * dim_head * raw_attn_output.scalar_size(),
                num_tokens_img * num_heads * dim_head * raw_attn_output_split.scalar_size(),
                batch_size,
                cudaMemcpyDeviceToDevice, 
                stream));


            /*
            raw_attn_output_split = Tensor::allocate(
                {batch_size, num_tokens_img, num_heads * dim_head},
                raw_attn_output.scalar_type(),
                raw_attn_output.device()
            );
            //std::cerr << "raw_attn_output_split shape: " << raw_attn_output_split.shape.str() << std::endl;

            auto element_size = raw_attn_output.scalar_size();
            auto stream = getCurrentCUDAStream();

            for (int b = 0; b < batch_size; b++) 
            {
                size_t src_offset_bytes = static_cast<size_t>(b)
                    * (num_tokens_img + num_tokens_context) * num_heads * dim_head
                    * element_size;

                size_t dst_offset_bytes = static_cast<size_t>(b)
                    * (num_tokens_img) * (num_heads * dim_head)
                    * element_size;
                
                size_t copy_bytes_per_batch =
                    static_cast<size_t>(num_tokens_img)
                * (num_heads * dim_head)
                * element_size;

                void* src_ptr = static_cast<char*>(raw_attn_output.data_ptr()) + src_offset_bytes;
                void* dst_ptr = static_cast<char*>(raw_attn_output_split.data_ptr()) + dst_offset_bytes;

                checkCUDA(cudaMemcpyAsync(
                    dst_ptr,
                    src_ptr,
                    copy_bytes_per_batch,
                    cudaMemcpyDeviceToDevice,
                    stream
                ));
            }
            */    
        }
        

        spdlog::debug("raw_attn_output_split={}", raw_attn_output_split.shape.str());
        debug("img.raw_attn_output_split", raw_attn_output_split);

        Tensor attn_output = forward_fc(out_proj, raw_attn_output_split); // std::get<Tensor>(out_proj.forward(raw_attn_output_split));
        debug("img.attn_output", attn_output);
        //std::cerr << "FluxModol::o_proj " << std::endl;

#if 1
        kernels::mul_add(attn_output, gate_msa, hidden_states);
        hidden_states = std::move(attn_output);

        nvtxRangePop();
        nvtxRangePushA("MLP");

        spdlog::debug("attn_output={}", hidden_states.shape.str());

        Tensor norm_hidden_states = norm2.forward(hidden_states);
        debug("scale_mlp", scale_mlp);
        debug("shift_mlp", shift_mlp);
        kernels::mul_add(norm_hidden_states, scale_mlp, shift_mlp);

        spdlog::debug("norm_hidden_states={}", norm_hidden_states.shape.str());
#else
        Tensor norm_hidden_states = hidden_states;
#endif

        // Tensor ff_output = mlp_fc2.forward(GELU::forward(mlp_fc1.forward(norm_hidden_states)));
        debug("img.ff_input", norm_hidden_states);
        Tensor ff_output = forward_mlp(mlp_fc1, mlp_fc2, norm_hidden_states);
        debug("img.ff_output", ff_output);

        debug("gate_mlp", gate_mlp);
        kernels::mul_add(ff_output, gate_mlp, hidden_states);
        hidden_states = std::move(ff_output);

        nvtxRangePop();

        spdlog::debug("ff_output={}", hidden_states.shape.str());
    }

    if (context_pre_only) {
        return { hidden_states, encoder_hidden_states };
    }

    {
        nvtxRangePushA("o_proj_context");

        auto &&[_, gate_msa, shift_mlp, scale_mlp, gate_mlp] = norm1_context_output;

        Tensor raw_attn_output_split;
        if (batch_size == 1) {
            raw_attn_output_split = raw_attn_output.slice(1, num_tokens_img, num_tokens_img + num_tokens_context).reshape({batch_size, num_tokens_context, num_heads * dim_head});
        } else 
        {
            raw_attn_output_split = Tensor::allocate({batch_size, num_tokens_context, num_heads * dim_head}, raw_attn_output.scalar_type(), raw_attn_output.device());
            checkCUDA(cudaMemcpy2DAsync(
                raw_attn_output_split.data_ptr(), 
                num_tokens_context * num_heads * dim_head * raw_attn_output_split.scalar_size(),
                raw_attn_output.data_ptr<char>() + num_tokens_img * num_heads * dim_head * raw_attn_output_split.scalar_size(),
                (num_tokens_img + num_tokens_context) * num_heads * dim_head * raw_attn_output.scalar_size(),
                num_tokens_context * num_heads * dim_head * raw_attn_output_split.scalar_size(),
                batch_size,
               cudaMemcpyDeviceToDevice, 
                stream));
  
        }
        

        spdlog::debug("raw_attn_output_split={}", raw_attn_output_split.shape.str());
        debug("context.raw_attn_output_split", raw_attn_output_split);

        Tensor attn_output = forward_fc(out_proj_context, raw_attn_output_split); // std::get<Tensor>(out_proj_context.forward(raw_attn_output_split));
        debug("context.attn_output", attn_output);
        
        //std::cerr << "FluxModol::o_proj_context " << std::endl;

#if 1
        kernels::mul_add(attn_output, gate_msa, encoder_hidden_states);
        encoder_hidden_states = std::move(attn_output);

        nvtxRangePop();
        nvtxRangePushA("MLP");

        spdlog::debug("attn_output={}", encoder_hidden_states.shape.str());

        Tensor norm_hidden_states = norm2_context.forward(encoder_hidden_states);
        debug("c_scale_mlp", scale_mlp);
        debug("c_shift_mlp", shift_mlp);
        kernels::mul_add(norm_hidden_states, scale_mlp, shift_mlp);

        spdlog::debug("norm_hidden_states={}", norm_hidden_states.shape.str());
#else
        auto norm_hidden_states = encoder_hidden_states;
#endif
        

        // Tensor ff_output = mlp_context_fc2.forward(GELU::forward(mlp_context_fc1.forward(norm_hidden_states)));
        // Tensor ff_output = mlp_context_fc2.forward_quant(quant_static_fuse_gelu(mlp_context_fc1.forward(norm_hidden_states), 1.0));
        debug("context.ff_input", norm_hidden_states);
        Tensor ff_output = forward_mlp(mlp_context_fc1, mlp_context_fc2, norm_hidden_states);
        debug("context.ff_output", ff_output);

        debug("c_gate_mlp", gate_mlp);
        kernels::mul_add(ff_output, gate_mlp, encoder_hidden_states);
        encoder_hidden_states = std::move(ff_output);

        nvtxRangePop();

        spdlog::debug("ff_output={}", encoder_hidden_states.shape.str());
    }

    nvtxRangePop();

    return { hidden_states, encoder_hidden_states };
}

FluxModel::FluxModel(bool use_fp4, bool offload, Tensor::ScalarType dtype, Device device) : offload(offload) {
    for (int i = 0; i < 19; i++) {
        transformer_blocks.push_back(std::make_unique<JointTransformerBlock>(3072, 24, 3072, false, use_fp4, dtype, device));
        registerChildren(*transformer_blocks.back(), format("transformer_blocks.{}", i));
        if (offload && i > 0) { // don't offload first block
            transformer_blocks.back()->setLazyLoad(true);
            transformer_blocks.back()->releaseLazyParams();
        }
    }
    for (int i = 0; i < 38; i++) {
        single_transformer_blocks.push_back(std::make_unique<FluxSingleTransformerBlock>(3072, 24, 3072, 4, use_fp4, dtype, Device::cuda()));
        registerChildren(*single_transformer_blocks.back(), format("single_transformer_blocks.{}", i));
        if (offload) {
            single_transformer_blocks.back()->setLazyLoad(true);
            single_transformer_blocks.back()->releaseLazyParams();
        }
    }
}

Tensor FluxModel::forward(Tensor hidden_states, Tensor encoder_hidden_states, Tensor temb, Tensor rotary_emb_img, Tensor rotary_emb_context, Tensor rotary_emb_single) {
    const int batch_size = hidden_states.shape[0];
    const Tensor::ScalarType dtype = hidden_states.dtype();
    const Device device = hidden_states.device();

    const int txt_tokens = encoder_hidden_states.shape[1];
    const int img_tokens = hidden_states.shape[1];

    const int numLayers = transformer_blocks.size() + single_transformer_blocks.size();

    Tensor concat;
    //std::cerr << "FluxModel::Forward1" << std::endl;

    auto compute = [&](int layer) {
        if (size_t(layer) < transformer_blocks.size()) {
            //std::cerr << "FluxModel::Forward2" << std::endl;
            auto &block = transformer_blocks.at(layer);
            std::tie(hidden_states, encoder_hidden_states) = block->forward(hidden_states, encoder_hidden_states, temb, rotary_emb_img, rotary_emb_context, 0.0f);
        } else {
            if (size_t(layer) == transformer_blocks.size()) {
                //std::cerr << "FluxModel::Forward3" << std::endl;
                // txt first, same as diffusers
                concat = Tensor::allocate({batch_size, txt_tokens + img_tokens, 3072}, dtype, device);
                //std::cerr << "FluxModel::Forward3::concat shape" << concat.shape.str() <<std::endl;
                for (int i = 0; i < batch_size; i++) {
                    /*
                    concat.slice(0, i, i + 1).slice(1, 0, txt_tokens).copy_(encoder_hidden_states);
                    concat.slice(0, i, i + 1).slice(1, txt_tokens, txt_tokens + img_tokens).copy_(hidden_states);
                    */
                   Tensor concat_txt_slice = concat.slice(0, i, i + 1)
                                       .slice(1, 0, txt_tokens);

                    Tensor enc_slice = encoder_hidden_states.slice(0, i, i + 1);

                    concat_txt_slice.copy_(enc_slice);
                    Tensor concat_img_slice = concat.slice(0, i, i + 1)
                                                    .slice(1, txt_tokens, txt_tokens + img_tokens);

                    Tensor hid_slice = hidden_states.slice(0, i, i + 1);

                    concat_img_slice.copy_(hid_slice);
                }
                hidden_states = concat;
                encoder_hidden_states = {};
                //std::cerr << "FluxModel::Concat" << std::endl;
            }

            auto &block = single_transformer_blocks.at(layer - transformer_blocks.size());
            hidden_states = block->forward(hidden_states, temb, rotary_emb_single);
        }
    };
    auto load = [&](int layer) {
        if (size_t(layer) < transformer_blocks.size()) {
            auto &block = transformer_blocks.at(layer);
            block->loadLazyParams();
        } else {
            auto &block = single_transformer_blocks.at(layer - transformer_blocks.size());
            block->loadLazyParams();
        }
    };
    auto unload = [&](int layer) {
        if (size_t(layer) < transformer_blocks.size()) {
            auto &block = transformer_blocks.at(layer);
            block->releaseLazyParams();
        } else {
            auto &block = single_transformer_blocks.at(layer - transformer_blocks.size());
            block->releaseLazyParams();
        }
    };

    LayerOffloadHelper helper(this->offload, numLayers, compute, load, unload);
    helper.run();

    return hidden_states;
}