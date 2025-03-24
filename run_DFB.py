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


transformer,m = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")


pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to(device)


# 4) Threshold test values
threshold_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  
threshold_values = [0.2, 0.4, 0.6, 0.8, 1.0] 
fixed = 0.4

num_inference_steps = 50
guidance_scale = 3.5
prompt = "A cute panda holding a sign that says hello world"

os.makedirs("results/Dfb_cache_results", exist_ok=True)

for th in threshold_values:
    with FBTransformerCacheContext() as fb_ctx:
        transformer.set_residual_diff_threshold(
            threshold_multi=fixed,
            threshold_single=th
            )

        start_time = time.time()
        image_1 = pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        mid_time = time.time()
        image_2 = pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        end_time = time.time()

        time_first = mid_time - start_time
        time_second = end_time - mid_time
        total_time = end_time - start_time

        print(f"\n[Threshold={th}]")
        print(f" - 1st inference time : {time_first:.2f} s")
        print(f" - 2nd inference time : {time_second:.2f} s")
        print(f" => total time        : {total_time:.2f} s\n")

        out_path_1 = f"results/Dfb_cache_results/flux_t{th*100}_first.png"
        out_path_2 = f"results/Dfb_cache_results/flux_t{th*100}_second.png"
        image_1.save(out_path_1)
        image_2.save(out_path_2)

        print(f" * Saved images at: {out_path_1}, {out_path_2}")
