#ifndef __AMSC_FFT__H
#define __AMSC_FFT__H

#include <complex>
#include <vector>

namespace AMSC {

void
fft_recursive(
    const std::vector<std::complex<double>>& input,
    std::vector<std::complex<double>> &output
);

void
fft_recursive_parallel(
    const std::vector<std::complex<double>>& input,
    std::vector<std::complex<double>> &output
);

void
fft_radix2(
    const std::vector<std::complex<double>> &input,
    std::vector<std::complex<double>> &output
);

void
ifft_radix2(
    const std::vector<std::complex<double>> &input,
    std::vector<std::complex<double>> &output
);

void
fft_radix2_parallel(
    const std::vector<std::complex<double>> &input,
    std::vector<std::complex<double>> &output
);

void
ifft_radix2_parallel(
    const std::vector<std::complex<double>> &input,
    std::vector<std::complex<double>> &output
);

void
fft_radix2_lookup(
    const std::vector<std::complex<double>> &input,
    std::vector<std::complex<double>> &output
);


void
fft_radix2_lookup_parallel(
    const std::vector<std::complex<double>> &input,
    std::vector<std::complex<double>> &output
);

void
fft_radix2_real(
    const std::vector<double> &input,
    std::vector<std::complex<double>> &output
);

}; // namespace AMSC

#endif //__AMSC_FFT__H 
