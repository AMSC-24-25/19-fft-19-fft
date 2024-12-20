# FFT Execution Time Comparison

This repository contains a comparison of the execution times for two different FFT methods:

1. **Normal FFT**
2. **Separated FFT**

## Description

The plot below shows the execution time (in microseconds) for different input sizes \(N\) ranging from \(2^5\) to \(2^{20}\). The execution times were measured for both the standard FFT algorithm (normal) and an optimized version of the FFT algorithm (separated).

- **FFT Normal:** This represents the execution time for the standard FFT algorithm.
- **FFT Separated:** This represents the execution time for a separated or optimized version of the FFT algorithm.

Both execution times are plotted on a **log-log scale** to better illustrate how the times scale with increasing input sizes.

## Results

- As the input size increases, the **normal FFT** execution time grows more rapidly compared to the **separated FFT**.
- The **separated FFT** method performs better for larger input sizes, as shown by its significantly lower execution times.

## Plot

The following plot compares the execution times for both FFT methods:

![FFT Execution Times](fft_execution_times.png)

## Files

- `fft_execution_times.png`: The plot showing the FFT execution times for both methods.
- `README.md`: This file providing an overview of the project.

## Conclusion

From the plot, we observe that the separated FFT method provides a significant speed-up for larger input sizes, making it a more efficient choice for larger FFT computations.

---

Feel free to explore and adjust the parameters for further analysis!
