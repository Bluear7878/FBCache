# Nunchaku FB-Cache Experiment

This repository contains an experiment demonstrating the **First-Block Cache (FBCache)** applied to a FluxPipeline with a quantized model on an **NVIDIA A6000** GPU. By adjusting the `residual_diff_threshold` parameter, we can see how caching affects subsequent inference times.

<br>

| Threshold | 1st Inference (s) | 2nd Inference (s) | Total (s) | First Inference Image                                                                                 | Second Inference Image                                                                                 |
|-----------|-------------------|-------------------|----------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **0**     | 22.18            | 20.58            | 42.76    | <img src="fb_cache_results/flux_t0_first.png" alt="flux_t0_first" width="140">                        | <img src="fb_cache_results/flux_t0_second.png" alt="flux_t0_second" width="140">                       |
| **0.01**  | 20.74            | 20.86            | 41.60    | <img src="fb_cache_results/flux_t0.01_first.png" alt="flux_t0.01_first" width="140">                   | <img src="fb_cache_results/flux_t0.01_second.png" alt="flux_t0.01_second" width="140">                 |
| **0.05**  | 14.88            | 14.89            | 29.77    | <img src="fb_cache_results/flux_t0.05_first.png" alt="flux_t0.05_first" width="140">                   | <img src="fb_cache_results/flux_t0.05_second.png" alt="flux_t0.05_second" width="140">                 |
| **0.1**   | 9.26             | 9.26             | 18.53    | <img src="fb_cache_results/flux_t0.1_first.png" alt="flux_t0.1_first" width="140">                     | <img src="fb_cache_results/flux_t0.1_second.png" alt="flux_t0.1_second" width="140">                   |
| **0.2**   | 5.24             | 5.25             | 10.49    | <img src="fb_cache_results/flux_t0.2_first.png" alt="flux_t0.2_first" width="140">                     | <img src="fb_cache_results/flux_t0.2_second.png" alt="flux_t0.2_second" width="140">                   |
| **0.5**   | 2.84             | 2.83             | 5.67     | <img src="fb_cache_results/flux_t0.5_first.png" alt="flux_t0.5_first" width="140">                     | <img src="fb_cache_results/flux_t0.5_second.png" alt="flux_t0.5_second" width="140">                   |
| **0.8**   | 2.03             | 2.03             | 4.06     | <img src="fb_cache_results/flux_t0.8_first.png" alt="flux_t0.8_first" width="140">                     | <img src="fb_cache_results/flux_t0.8_second.png" alt="flux_t0.8_second" width="140">                   |

