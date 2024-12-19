#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <functional>
#include "../include/fft.hpp"

using namespace AMSC;
using namespace std;

vector<complex<double>>
time_function(
    function<void(const vector<complex<double>>&, vector<complex<double>>&)> f,
    const vector<complex<double>> &sequence,
    string name
) {
    auto start = std::chrono::high_resolution_clock::now();
    vector<complex<double>> fft;
    f(sequence, fft);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "Time taken by " << name << ":   \t" 
         << duration.count()/1000.0 << " milliseconds" << endl;
    return fft;
}

void
verify(
    const vector<complex<double>> &standard,
    const vector<complex<double>> &sequence,
    string name
){
    for(size_t i=0; i<standard.size(); ++i){
        if(abs(standard[i]-sequence[i])>1e-7){
            cout << "Error in " << name << " at index " << i << endl;
            cout << "Expected: " << standard[i] << " Got: " << sequence[i] << endl;
        }
    }
    cout << "Test passed for " << name << endl;
}

int main() {
    // 2^20 elements
    constexpr unsigned int dim = 1048576;
    vector<complex<double>> sequence(dim);
    for(size_t i=0; i<dim; ++i){
        double rand_real =
            static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        double rand_i =
            static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        sequence[i] = complex<double>(rand_real,rand_i);
    }
    auto fft_ref = time_function(fft_radix2, sequence, "normal");
    auto fft = time_function(fft_radix2_parallel, sequence, "parallel");
    verify(fft_ref, fft, "parallel");
    fft = time_function(fft_radix2_lookup, sequence, "lookup");
    verify(fft_ref, fft, "lookup");
    fft = time_function(fft_radix2_lookup_parallel, sequence, "lookup p");
    verify(fft_ref, fft, "lookup parallel");

    return 0;
}
