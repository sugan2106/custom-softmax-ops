**Custom CUDA Softmax vs PyTorch's Built-in Softmax Ops**

A CUDA C++ project focused on implementing a fused, numerically stable Softmax kernel and evaluating its performance against PyTorch's built-in Softmax operator.

**Project Description**

Designed and implemented a fused, numerically stable Softmax kernel in CUDA C++.

Used NVIDIA warp-level primitives to perform row-wise computation and reduce intermediate memory traffic.

Integrated the kernel as a native PyTorch operator using pybind11 and the CppExtension API.

Benchmarked the custom operator against PyTorch Softmax across tensor sizes, analyzing latency and performance.
