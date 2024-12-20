import numpy as np
import matplotlib.pyplot as plt

# Range of N (2^5 to 2^20)
Ns = [2**i for i in range(5, 21)]

# Placeholder for execution times (replace with actual data)
fft_normal_times = [
    25 * (2**(i - 5)) for i in range(5, 21)  # Example: time grows with N, replace with actual data
]
fft_separated_times = [
    5 * (2**(i - 5)) for i in range(5, 21)  # Example: time grows with N, replace with actual data
]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(Ns, fft_normal_times, label="FFT Normal", marker='o', linestyle='-', color='b')
plt.plot(Ns, fft_separated_times, label="FFT Separated", marker='x', linestyle='--', color='r')

# Set log scale for x and y axes
plt.xscale('log')
plt.yscale('log')

# Adding labels and title
plt.xlabel('N (Size of FFT)', fontsize=12)
plt.ylabel('Execution Time (microseconds)', fontsize=12)
plt.title('FFT Execution Time for Different N', fontsize=14)

# Adding grid and legend
plt.grid(True, which="both", ls="--")
plt.legend()

# Save the plot as a PNG file
plt.savefig("fft_execution_times.png", format="png")

# Do not show the plot, just save it
plt.close()
