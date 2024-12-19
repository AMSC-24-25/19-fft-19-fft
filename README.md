# Fast Fourier Transform (FFT) Implementation

This project provides an efficient implementation of the Fast Fourier Transform (FFT) algorithm using the Cooley-Tukey radix-2 algorithm. The code supports both complex and real-valued sequences. Additionally, it includes functions for performing the bit-reversal operation and inverse FFT.

## Features
- Computes the FFT of both complex and real-valued sequences.
- Uses a radix-2 FFT algorithm for efficient computation.
- Includes a bit-reversal function to reorder sequence indices.
- Inverse FFT function to compute the inverse transform.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Implementation Details](#functions)
4. [Usage](#usage)
5. [Installation](#installation)

## Prerequisites
- C++11 or later for code compatibility.
- OpenMP for parallel processing of FFT computations.

## Mathematical Foundation

The **Fast Fourier Transform (FFT)** is an efficient algorithm to compute the **Discrete Fourier Transform (DFT)** of a sequence, or its inverse. The DFT is a fundamental operation in signal processing and many other fields such as image analysis, audio processing, and communications. 

### 1. **Discrete Fourier Transform (DFT)**

The DFT of a sequence $x[n]$ of length $N$ is defined as:


$$\large X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-2\pi i \frac{k n}{N}}, \quad k = 0, 1, 2, \dots, N-1$$

Where:
- $X[k]$ are the frequency domain components,
- $x[n]$ are the time domain samples,
- $i$ is the imaginary unit ( $i = \sqrt{-1}$ ),
- $N$ is the size of the sequence

### 2. **Cooley-Tukey Radix-2 FFT**

The **Cooley-Tukey Radix-2 algorithm** is the most commonly used FFT algorithm. It is used when $N$ is a power of two and its core idea is to break the sequence $x[n]$ into smaller sequences. 
From the formula for the DFT it can be easily seen that the computation can be divided in two parts: one for the **even-indexed** elements and another for the **odd-indexed** elements of the sequence.

#### 2.1. **Mathematical Formula for Radix-2 FFT**

Using the split approach, the DFT of a sequence $x[n]$ can be expressed as:

$$\large X[k] = \sum_{n = 0}^{N-1}x[n]\cdot e^{-i2\pi\frac{kn}{N}} = \sum_{n = 0}^{\frac{N}{2}-1}x[2n]\cdot e^{-i2\pi\frac{k\cdot2n}{N}} + \sum_{n = 0}^{\frac{N}{2}-1}x[2n+1]\cdot e^{-i2\pi\frac{k\cdot(2n+1)}{N}} $$

Where we just rearranged the terms. Rewriting the exponential terms:

$$\large X[k] = \sum_{n = 0}^{\frac{N}{2}-1}x[2n]\cdot e^{-i2\pi\frac{kn}{\frac{N}{2}}} + e^{-i2\pi \frac{k}{N}} \sum_{n = 0}^{\frac{N}{2}-1}x[2n+1]\cdot e^{-i2\pi\frac{kn}{\frac{N}{2}}} $$

Now we can call $\normalsize x[2n] = x_{even}[n]$ and $\normalsize x[2n+1] = x_{odd}[n]$. So: 

$$\large X[k] = \sum_{n = 0}^{\frac{N}{2}-1}x_{even}[n]\cdot e^{-i2\pi\frac{kn}{\frac{N}{2}}} + e^{-i2\pi \frac{k}{N}} \sum_{n = 0}^{\frac{N}{2}-1}x_{odd}[n]\cdot e^{-i2\pi\frac{kn}{\frac{N}{2}}} = X_{even}[k] + e^{-i2\pi \frac{k}{N}}\cdot X_{odd}[k] $$

So we reduced the computation of a DTF of size $N$ to the computation of two DTFs of size $\frac{N}{2}$

If $N$ is a power of two we can use this approach iteratively until the base case of lenght one.  
For each step of the FFT, the size of the problem is halved, resulting in a logarithmic depth of recursion. This allows us to compute the DFT in $O(N\log(n))$ time, which is a significant improvement over the direct DFT computation, which has a time complexity of $O(N^2)$.

#### 2.2 **Optimization for real sequences**

If $x[n]$ is a sequence of real numbers with length $N$, the DFT exhibits a symmetric property: $X[k] = X^*[N-k]$.  
We can use this property to optimize the computation of the DFT in case of real-valued sequences.

Let $x[n]$ and $y[n]$ be two real valued sequences of length $\frac{N}{2}$. We can create another sequence $z[n] = x[n] + i\cdot y[n]$  
We know that:

$$
x[n] = \frac{z[n] + z^*[n]}{2} \quad y[n] = \frac{z[n]-z^{\*}[n]}{2i}
$$

Since the DFT is linear: 


$$
X[k] = \frac{Z[k] + \mathcal{F}{\\{z^*[n]\\}}}{2} \quad Y[k] = \frac{Z[k]-\mathcal{F}{\\{z^{\*}[n]\\}}}{2i}
$$

Knowing that

$$
\mathcal{F}{\\{z^*[n]\\}} = \sum_{n=0}^{N-1}z^\*[n]e^{-i2\pi \frac{nk}{N}} = (\sum_{n=0}^{N-1}z[n]e^{i2\pi \frac{nk}{N}})^\* = (\sum_{n=0}^{N-1}z[n]e^{i2\pi \frac{nk}{N}}e^{-i2\pi \frac{nN}{N}})^\* = (\sum_{n=0}^{N-1}z[n]e^{-i2\pi\frac{n(N-k)}{N}})^\* = Z^\*[N-k]
$$

So we can write 

$$
X[k] = \frac{Z[k] + Z^\*[\frac{N}{2}-k]}{2} \quad Y[k] = \frac{Z[k]-Z^\*[\frac{N}{2}-k]}{2i}
$$

If we now take $x_{even}[n] = x[2n]$ and $x_{odd}[n] = x[2n+1]$ we can compute the DFT of $z[n] = x_{even}[n] + i\cdot x_{odd}[n]$ at the cost of one DFT of size $\frac{N}{2}$.  
Using these DFTs, we can then compute the DFT of $x[n]$ by exploiting the symmetries inherent in real-valued sequences.

### 3. **Inverse DFT (IDFT)**

The **Inverse DFT** is used to recover the original time-domain sequence from its frequency-domain representation. The formula for the IDFT has the same structure of the one for the DFT, minus the exponent sign.  
Mathematically, the inverse DFT is defined as:

$$
\large x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{2\pi i \frac{k n}{N}}, \quad n = 0, 1, 2, \dots, N-1
$$

It is obvious then that the same reasoning can be applied also to compute the Inverse DFT.

## Implementation Details

In this implementation, we chose an **in-place iterative** approach for the Fast Fourier Transform (FFT). While an additional array is used for storing the output to maintain separation, the algorithm could theoretically overwrite the input array itself during computation, achieving an in-place solution.

### In-Place Iterative FFT
The algorithm iteratively computes the FFT of the sequence by progressively breaking it down into smaller subsequences. Each iteration processes the sequence and updates it based on the FFT computation. By avoiding recursion, the iterative approach reduces the function call overhead, making it faster and more efficient. 
Although we use an auxiliary array to hold the results for clarity and separation, this is purely for organizational purposes. It is possible to modify the input sequence directly during computation (i.e., overwrite it), which would further reduce memory usage, making the implementation truly in-place. However, for clarity, we opted to keep the input and output separate during the FFT calculation.

### Bit-Reversal of the Sequence
The Cooley-Tukey Radix-2 FFT algorithm requires the input sequence to be rearranged according to bit-reversal order. To understand this, consider the sequence indices as binary numbers. At each stage of the algorithm, the sequence is split into two sub-sequences based on whether the indices are even or odd.  
In binary terms, this corresponds to right-shifting the index positions and separating those ending in 0 from those ending in 1. By recursively applying this process, we observe that the indices are reordered in a way that reflects the bit-reversal of their original positions.  
Ultimately, when the recursion reaches the base case, the sequence is fully rearranged according to the reversed binary order of the indices.

Let's understand it with an example: Let's take a sequence of lenght $8$  

$x[n] = [000,001,010,011,100,101,110,111]$

For the first step of the algorithm we should consider the two sequences:

$[00\textcolor{red}{0}, 01\textcolor{red}{0}, 10\textcolor{red}{0}, 11\textcolor{red}{0}]$ and $[00\textcolor{red}{1}, 01\textcolor{red}{1}, 10\textcolor{red}{1}, 11\textcolor{red}{1}]$

Now considering just the first sub-sequence with the successive step we would have:

$[0\textcolor{blue}{0}\textcolor{red}{0}, 1\textcolor{blue}{0}\textcolor{red}{0}]$ and $[0\textcolor{blue}{1}\textcolor{red}{0}, 1\textcolor{blue}{1}\textcolor{red}{0}]$ 

so we should rearrange our sequence $x[n]$ 

$[000,100,010,110,001,101,011,111]$

The new positions of the elements in x[n]x[n] correspond to the binary representation of their indices, with the bits read from right to left!

## Usage

### Generating a Random Sequence and Computing FFT
```cpp
unsigned int constexpr dim = 1048576;  // Set sequence size (must be a power of 2)
vector<complex<double>> sequence(dim);

// Fill the sequence with random values
for (unsigned int i = 0; i < dim; ++i) {
    double rand_real = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    double rand_imag = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    sequence[i] = complex<double>(rand_real, rand_imag);
}

// Compute the FFT of the sequence
vector<complex<double>> result = fft_radix2(sequence);
vector<complex<double>> inverse_result = inverse_fft_radix_2(result);
```
## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/fft-implementation.git
```
2. Compile the .cpp file with **OpenMP** support
```bash
g++ -fopenmp cooley-tukey.cpp -o cooley-tukey
```
3. Run the program
```bash
./cooley-tukey
```
