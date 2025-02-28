# Nunchaku FB-Cache Experiment

This repository contains an experiment demonstrating the **First-Block Cache (FBCache)** applied to a FluxPipeline with a quantized model on an **NVIDIA A6000** GPU. By adjusting the `residual_diff_threshold` parameter, we can see how caching affects subsequent inference times.

<br>

| Threshold | Single_layer 0.4 (s) | Multi_layer 0.4 (s) | Single_layer Image                                                                                      | Multi_layer Image                                                                                         |
|-----------|----------------------|---------------------|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| 0.2       | 7.06                | 6.54               | <img src="./results/Dfb_cache_results_sing_fix_multi_var/flux_t20.0_second.png" width="140">             | <img src="./results/Dfb_cache_results/flux_t20.0_second.png" width="140">                                  |
| 0.4       | 4.5                 | 4.49               | <img src="./results/Dfb_cache_results_sing_fix_multi_var/flux_t40.0_second.png" width="140">             | <img src="./results/Dfb_cache_results/flux_t40.0_second.png" width="140">                                  |
| 0.6       | 4.17                | 4.52               | <img src="./results/Dfb_cache_results_sing_fix_multi_var/flux_t60.0_second.png" width="140">             | <img src="./results/Dfb_cache_results/flux_t60.0_second.png" width="140">                                  |
| 0.8       | 3.69                | 4.2                | <img src="./results/Dfb_cache_results_sing_fix_multi_var/flux_t80.0_second.png" width="140">             | <img src="./results/Dfb_cache_results/flux_t80.0_second.png" width="140">                                  |
| 1         | 3.7                 | 3.88               | <img src="./results/Dfb_cache_results_sing_fix_multi_var/flux_t100.0_second.png" width="140">            | <img src="./results/Dfb_cache_results/flux_t100.0_second.png" width="140">                                 |
