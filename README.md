# Nunchaku FB-Cache Experiment

This repository contains an experiment demonstrating the **First-Block Cache (FBCache)** applied to a FluxPipeline with a quantized model on an **NVIDIA A6000** GPU. By adjusting the `residual_diff_threshold` parameter, we can see how caching affects subsequent inference times.

<br>

## Results

| Threshold | 1st Inference (s) | 2nd Inference (s) | Total (s) |
|-----------|-------------------|-------------------|-----------|
| **0**     | 22.18             | 20.58             | 42.76     |
| **0.01**  | 20.74             | 20.86             | 41.60     |
| **0.05**  | 14.88             | 14.89             | 29.77     |
| **0.1**   | 9.26              | 9.26              | 18.53     |
| **0.2**   | 5.24              | 5.25              | 10.49     |
| **0.5**   | 2.84              | 2.83              | 5.67      |
| **0.8**   | 2.03              | 2.03              | 4.06      |

- **Threshold=0** effectively disables caching, so both inferences take roughly the same time.  
- As the threshold increases, the **residual difference** is more easily considered "similar," so the pipeline can skip deeper blocks on subsequent calls. This leads to **faster second inferences** and reduced total time.

---

## Images

Below are the images generated for each threshold during the **first** and **second** inference. The prompt was:

> A cute panda holding a sign that says hello world

### Threshold = 0
**First**  
![flux_t0_first.png](fb_cache_results/flux_t0_first.png)  

**Second**  
![flux_t0_second.png](fb_cache_results/flux_t0_second.png)

---

### Threshold = 0.01
**First**  
![flux_t0.01_first.png](fb_cache_results/flux_t0.01_first.png)

**Second**  
![flux_t0.01_second.png](fb_cache_results/flux_t0.01_second.png)

---

### Threshold = 0.05
**First**  
![flux_t0.05_first.png](fb_cache_results/flux_t0.05_first.png)

**Second**  
![flux_t0.05_second.png](fb_cache_results/flux_t0.05_second.png)

---

### Threshold = 0.1
**First**  
![flux_t0.1_first.png](fb_cache_results/flux_t0.1_first.png)

**Second**  
![flux_t0.1_second.png](fb_cache_results/flux_t0.1_second.png)

---

### Threshold = 0.2
**First**  
![flux_t0.2_first.png](fb_cache_results/flux_t0.2_first.png)

**Second**  
![flux_t0.2_second.png](fb_cache_results/flux_t0.2_second.png)

---

### Threshold = 0.5
**First**  
![flux_t0.5_first.png](fb_cache_results/flux_t0.5_first.png)

**Second**  
![flux_t0.5_second.png](fb_cache_results/flux_t0.5_second.png)

---

### Threshold = 0.8
**First**  
![flux_t0.8_first.png](fb_cache_results/flux_t0.8_first.png)

**Second**  
![flux_t0.8_second.png](fb_cache_results/flux_t0.8_second.png)

---

## Observations

1. **Low Threshold** (`0` or `0.01`):  
   - The model rarely reuses cached states, so we see little to no speed-up on the second inference.

2. **Medium Threshold** (`0.05` to `0.2`):  
   - The second inference becomes faster because partial blocks are skipped if the residual is similar enough.

3. **High Threshold** (`0.5` or `0.8`):  
   - The second inference is **much** faster. Most of the deeper blocks are skipped after the first run.

4. **Trade-off**:  
   - If the threshold is *too high*, you might risk skipping blocks that could significantly alter the output. However, if your application tolerates minimal changes in output, this approach can greatly reduce inference time.

---

**Enjoy faster inference with your Nunchaku-based Flux Pipeline, and tune the threshold to suit your performance vs. accuracy needs!**