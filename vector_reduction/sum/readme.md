
# Naive sum reduction metrics
 
Performance Metrics ---
Host-to-Device Transfer Time: 0.022528 ms (0.0454545 GB/s)
CUDA Kernel Execution Time:   0.244736 ms
Device-to-Host Transfer Time: 0.027456 ms (0.000145688 GB/s)
End-to-End CUDA Execution Time: 0.313216 ms

# Naive sum reduction no divergence


Performance Metrics ---
Host-to-Device Transfer Time: 0.01952 ms (0.052459 GB/s)
CUDA Kernel Execution Time:   0.162368 ms
Device-to-Host Transfer Time: 0.025376 ms (0.000157629 GB/s)
End-to-End CUDA Execution Time: 0.223808 ms

# convergent sum reduction metrics


Performance Metrics ---
Host-to-Device Transfer Time: 0.016352 ms (0.0626223 GB/s)
CUDA Kernel Execution Time:   0.231424 ms
Device-to-Host Transfer Time: 0.027072 ms (0.000147754 GB/s)
End-to-End CUDA Execution Time: 0.28976 ms

# shared memory reduction metrics

Performance Metrics ---
Host-to-Device Transfer Time: 0.023392 ms (0.0437757 GB/s)
CUDA Kernel Execution Time:   0.169984 ms
Device-to-Host Transfer Time: 0.028512 ms (0.000140292 GB/s)
End-to-End CUDA Execution Time: 0.239872 ms

# shared memory no diverge
 Performance Metrics ---
Host-to-Device Transfer Time: 0.016896 ms (0.0606061 GB/s)
CUDA Kernel Execution Time:   0.170976 ms
Device-to-Host Transfer Time: 0.028224 ms (0.000141723 GB/s)
End-to-End CUDA Execution Time: 0.233504 ms

# convergent shared memory reduction metrics
 

Performance Metrics ---
Host-to-Device Transfer Time: 0.022272 ms (0.045977 GB/s)
CUDA Kernel Execution Time:   0.172064 ms
Device-to-Host Transfer Time: 0.028704 ms (0.000139353 GB/s)
End-to-End CUDA Execution Time: 0.239232 ms


## Performance Metrics with N=256

| Kernel Name                   | End-to-End Execution Time (ms) |
|-------------------------------|--------------------------------|
| `naive_sum_reduction`         | 0.290496 ms                    |
| `shared_mem_sum_reduction`    | 0.235328 ms                    |
| `convergent_sum_reduction`    | 0.311776 ms                    |
| `convergent_shared_mem`       | 0.262048 ms                    |
| `naive_sum_reduction_noDiverge` | 0.239744 ms                    |
| `shared_mem_noDiverge`        | 0.289984 ms  