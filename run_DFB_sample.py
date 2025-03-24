#env : nunchaku
#python3 run_DFB.py


import torch
from diffusers import FluxPipeline
import os
import time
import torch
from diffusers import FluxPipeline

from nunchaku.models.Dtransformer_flux import NunchakuFluxTransformer2dModel
from nunchaku.models.DFB_cache import *

device = torch.device("cuda:0")

flux_model_path = "/home/ict04/Lighten_DL/LLM/cp/cp/svdq/models--mit-han-lab--svdq-int4-flux.1-dev/snapshots/3af964b97d298d4ed9da753edcc1ba3b322031f2"
black_flux_model_path = "/home/ict04/Lighten_DL/LLM/cp/cp/black_forest/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44"

transformer,m = NunchakuFluxTransformer2dModel.from_pretrained(flux_model_path)
#transformer.transformer_blocks[0].m.load("/home/ict04/Lighten_DL/LLM/nunchaku_FB/nunchaku_loras/svdq-int4-flux.1-dev-ghibsky.safetensors", partial=True)

pipeline = FluxPipeline.from_pretrained(
    black_flux_model_path, transformer=transformer, torch_dtype=torch.bfloat16
).to(device)


pipeline2 = FluxPipeline.from_pretrained(
    black_flux_model_path, transformer=transformer, torch_dtype=torch.bfloat16
).to(device)
# 4) Threshold test values
threshold_values = [0.1] 
fixed = 0.1

num_inference_steps = 50
guidance_scale = 3.5
prompt = "A cute panda holding a sign that says hello world"
out_path_1 = f"results/Dfb_cache_results/flux_t{fixed*100}_first.png"
graph_save_path_1 = f"results/Dfb_cache_results/flux_t{int(fixed*100)}_threshold_graph_first.png"
out_path_2 = f"results/Dfb_cache_results/flux_t{fixed*100}_second.png"
graph_save_path_2 = f"results/Dfb_cache_results/flux_t{int(fixed*100)}_threshold_graph_second.png"

os.makedirs("results/Dfb_cache_results", exist_ok=True)

for th in threshold_values:
    with FBTransformerCacheContext() as fb_ctx:
        transformer.set_residual_diff_threshold(
            threshold_multi=fixed,
            threshold_single=th
            )

        start_time = time.time()
        image_1 = pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        end_time = time.time()
        
        
        transformer.transformer_blocks[0].plot_thresholds(save_path=graph_save_path_1)
        
        transformer.transformer_blocks[0].reset_threshold_logs()
        
        
        
        
    with FBTransformerCacheContext() as fb_ctx:
        transformer.set_residual_diff_threshold(
        threshold_multi=fixed,
        threshold_single=th
        )
        
        mid_time = time.time()
        image_2 = pipeline2(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        end_time = time.time()
        
        transformer.transformer_blocks[0].plot_thresholds(save_path=graph_save_path_2)
        
        transformer.transformer_blocks[0].reset_threshold_logs()

        time_first = end_time - start_time
        time_second = end_time - mid_time
        
        

        print(f"\n[Threshold={th}]")
        print(f" - 1st inference time : {time_first:.2f} s")
        print(f" - 2nd inference time : {time_second:.2f} s")

        

        
    safe_save(image_1,out_path_1)
    time.sleep(5)
    safe_save(image_2,out_path_2)
    print(f" * Saved images at: {out_path_1}, {out_path_2}")
