#include <execution>
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

template <typename T>
vector<complex<double>>
time_function(
    function<void(const vector<T>&, vector<complex<double>>&)> f,
    const vector<T> &sequence,
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
            return;
        }
    }
    cout << "Test passed for " << name << endl;
}

void test_speed_correctness_fft() {
    // 2^20
    constexpr size_t max_dim = 1048576;
    for(size_t dim = 4; dim <= max_dim; dim*=2) {
        vector<complex<double>> sequence(dim);
        vector<double> real_sequence(dim);
        
        for(size_t i=0; i<dim; ++i){
            double rand_real =
                static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
            double rand_i =
                static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
            sequence[i] = complex<double>(rand_real,rand_i);
            real_sequence[i] = rand_real;
        }

        cout << "[TEST dim=" << dim<< "]" << endl;
        auto fft_ref = time_function<complex<double>>(fft_radix2, sequence, "normal");
        auto fft = time_function<complex<double>>(fft_radix2_parallel, sequence, "parallel");
        verify(fft_ref, fft, "parallel");
        fft = time_function<complex<double>>(fft_radix2_lookup, sequence, "lookup");
        verify(fft_ref, fft, "lookup");
        fft = time_function<complex<double>>(fft_radix2_lookup_parallel, sequence, "lookup p");
        verify(fft_ref, fft, "lookup parallel");
        
        // We need to create a sequence with real values to test the fft_radix2_real_lookup function
        for(size_t i=0;i<dim;++i){
            sequence[i] = std::complex<double>(real_sequence[i],0.0);
        }
        fft_radix2_lookup(sequence,fft_ref);
        fft = time_function<double>(fft_radix2_real,real_sequence,"real optimization");
        verify(fft_ref,fft,"real optimization");
        cout << "\n\n"; 
    }
}

void test_fft_ifft() {
    constexpr size_t dim = 1048576;
    vector<complex<double>> sequence(dim);

    for(size_t i=0; i<dim; ++i){
        double rand_real =
            static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        double rand_i =
            static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        sequence[i] = complex<double>(rand_real,rand_i);
    }


    vector<complex<double>> ifft(dim);
    ifft_radix2(sequence, ifft);
    vector<complex<double>> output(dim);
    fft_radix2(ifft, output);

    for(size_t i = 0; i < output.size(); ++i) {
        if(abs(output[i]-sequence[i]) > 1.0e-7 ) {
            cout << "Error in test_ifft at index " << i << endl;
            return;
        }
    }

    cout << "Test test_fft_ifft passed" << endl;
}

void test_ifft_fft() {
    constexpr size_t dim = 1048576;
    vector<complex<double>> sequence(dim);

    for(size_t i=0; i<dim; ++i){
        double rand_real =
            static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        double rand_i =
            static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        sequence[i] = complex<double>(rand_real,rand_i);
    }

    vector<complex<double>> fft(dim);
    fft_radix2(sequence, fft);
    vector<complex<double>> output(dim);
    ifft_radix2(fft, output);

    for(size_t i = 0; i < output.size(); ++i) {
        if(abs(output[i]-sequence[i]) > 1.0e-7 ) {
            cout << "Error in test_ifft at index " << i << endl;
            return;
        }
    }

    cout << "Test test_ifft_fft passed" << endl;
}

int main() {
    // Speed and correctness test
    test_speed_correctness_fft();

    // fft(ifft(sequence)) must be equal to sequence
    test_fft_ifft();

    // ifft(fft(sequence)) must be equal to sequence
    test_ifft_fft();

    return 0;
}
