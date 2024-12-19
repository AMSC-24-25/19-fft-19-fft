#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <omp.h>
#include <chrono>
#include <functional>
#include <random>

using namespace std;

/**
 * @brief Reverses the bits of an unsigned integer.
 * 
 * This function takes an unsigned integer value and reverses its bit order.
 * For example, if the binary representation of `value` is `1011` with `num_bit = 4`,
 * the reversed representation would be `1101`.
 * 
 * @param value The unsigned integer to reverse.
 * @param num_bit The number of bits to consider in the reversal.
 * @return The unsigned integer value with reversed bits.
 * @throws std::invalid_argument if `num_bit` is zero.
 */
unsigned int
reverse_number_bit(
    const unsigned int &value,
    const unsigned int &num_bit
){
    unsigned int reversed = 0;
    if(value==0)
        return 0;
    if(num_bit==0){
        throw invalid_argument("The number of bits cannot be equal to zero");
    }
    for(int i=0;i<num_bit;++i){
        //Check if the value confronted bit by bit with 1...0 is different from zero
        if((value & (1<<i)) != 0){
            //Compute the reversed value adding the bit in the right position
            reversed |= (1<<(num_bit-1-i));
        }
    }

    return reversed;
}

/**
 * @brief Reverses the bit order of the indices of elements in a sequence.
 * 
 * This function rearranges the elements of the input sequence such that the indices
 * of the elements follow a bit-reversed order. The sequence size must be a power of 2.
 * 
 * @tparam T The type of elements in the sequence.
 * @param sequence A vector containing the elements to reorder.
 * @throws std::invalid_argument if the sequence size is zero or not a power of 2.
 * 
 * Example:
 * Given a sequence of size 8, with binary indices:
 * Index:     000  001  010  011  100  101  110  111
 * After reverse: 
 * Index:     000  100  010  110  001  101  011  111
 */
template <typename T>
void inline
reverse_bit_order(
    vector<T> &sequence
){
    unsigned int seq_size = sequence.size();
    if(sequence.size()==0){
        throw invalid_argument("The sequence size must be different from zero");
    }
    else if(seq_size&(seq_size-1)!=0){
        throw invalid_argument("The sequence size must be a power of 2");
    }
    unsigned int num_bit = static_cast<int> (log2(seq_size));
    unsigned int swap_position = 0;
    // sequence can be reversed in parallel
    #pragma omp parallel for schedule(static) shared(sequence) firstprivate(num_bit) private(swap_position)
    for(unsigned int i=0;i<seq_size;i++){
        swap_position = reverse_number_bit(i,num_bit);
        if(swap_position>i){
            swap(sequence[i],sequence[swap_position]);
        }
    }
}

/**
 * @brief Computes the Fast Fourier Transform (FFT) using the Cooley-Tukey radix-2 algorithm.
 * 
 * This function computes the Discrete Fourier Transform (DFT) of a given complex sequence
 * using the efficient radix-2 FFT algorithm. The input sequence size must be a power of 2.
 * 
 * @param sequence A vector of complex numbers representing the input sequence.
 * @return A vector of complex numbers representing the DFT of the input sequence.
 * @throws std::invalid_argument if the sequence size is zero or not a power of 2
 */
vector<complex<double>>
fft_radix2(
    const vector<complex<double>> &sequence
){
    vector<complex<double>> dft(sequence);
    if(sequence.size()==0){
        throw invalid_argument("The sequence size must be different from zero");
    }
    else if((sequence.size()&(sequence.size()-1))!=0){
        throw invalid_argument("The sequence size must be a power of 2");
    }
    else if(sequence.size()==1){
        return dft;
    }
    reverse_bit_order(dft);

    // Iteration on the different depths(sub_size) of the problem
    // We'll start from subproblems of size 1 and go on from there
    // sub_size is the current size of the computed partial DTF in our problem
    for(unsigned int sub_size=1;sub_size<dft.size();sub_size*=2){

        // At every inner iteration we consider two adjecent subproblems
        // j is the index of the current subproblem
        // Since we consider two problem for iteration j has to skip the adjecent subproblem
        for(unsigned int j=0;j<dft.size();j+=2*sub_size){
            
            //Iteration on the two adjecent subproblems
            //We just apply the formula
            for(unsigned int z=0;z<sub_size;z++){
                auto angle = complex<double> (0,-2*M_PI*z/(2*sub_size));
                auto phase_factor = exp(angle);
                auto even_term = dft[j+z];
                auto odd_term = dft[j+z+sub_size]*phase_factor;
                dft[j+z] = even_term + odd_term;
                dft[j+z+sub_size] = even_term - odd_term;
            }
        }
    }
    return dft;
}

vector<complex<double>>
fft_radix2_parallel(
    const vector<complex<double>> &sequence
){
    vector<complex<double>> dft(sequence);
    if(sequence.size()==0){
        throw invalid_argument("The sequence size must be different from zero");
    }
    else if((sequence.size()&(sequence.size()-1))!=0){
        throw invalid_argument("The sequence size must be a power of 2");
    }
    else if(sequence.size()==1){
        return dft;
    }
    reverse_bit_order(dft);

    // Iteration on the different depths(sub_size) of the problem
    // We'll start from subproblems of size 1 and go on from there
    // sub_size is the current size of the computed partial DTF in our problem
    for(unsigned int sub_size=1;sub_size<dft.size();sub_size*=2){

        // At every inner iteration we consider two adjecent subproblems
        // j is the index of the current subproblem
        // Since we consider two problem for iteration j has to skip the adjecent subproblem
        #pragma omp parallel for schedule(static) shared(dft) collapse(2)
        for(unsigned int j=0;j<dft.size();j+=2*sub_size){
            
            //Iteration on the two adjecent subproblems
            //We just apply the formula
            for(unsigned int z=0;z<sub_size;z++){
                auto angle = complex<double> (0,-2*M_PI*z/(2*sub_size));
                auto phase_factor = exp(angle);
                auto even_term = dft[j+z];
                auto odd_term = dft[j+z+sub_size]*phase_factor;
                dft[j+z] = even_term + odd_term;
                dft[j+z+sub_size] = even_term - odd_term;
            }
        }
    }
    return dft;
}

vector<complex<double>>
fft_radix2_lookup(
    const vector<complex<double>> &sequence
){
    vector<complex<double>> dft(sequence);
    vector<double> sin_table(sequence.size());

    for(unsigned int i=0;i<sequence.size();i++){
        sin_table[i] = sin(2*M_PI*static_cast<float>(i)/sequence.size());
    }

    
    if(sequence.size()==0){
        throw invalid_argument("The sequence size must be different from zero");
    }
    else if((sequence.size()&(sequence.size()-1))!=0){
        throw invalid_argument("The sequence size must be a power of 2");
    }
    else if(sequence.size()==1){
        return dft;
    }
    reverse_bit_order(dft);

    // Iteration on the different depths(sub_size) of the problem
    // We'll start from subproblems of size 1 and go on from there
    // sub_size is the current size of the computed partial DTF in our problem
    for(unsigned int sub_size=1;sub_size<dft.size();sub_size*=2){

        // At every inner iteration we consider two adjecent subproblems
        // j is the index of the current subproblem
        // Since we consider two problem for iteration j has to skip the adjecent subproblem
        // #pragma omp parallel for schedule(static) shared(dft)
        for(unsigned int j=0;j<dft.size();j+=2*sub_size){
            
            //Iteration on the two adjecent subproblems
            //We just apply the formula
            for(unsigned int z=0;z<sub_size;z++){
                unsigned int idx = z * dft.size() / (2*sub_size);
                auto phase_factor = complex<double>(sin_table[idx + dft.size()/4], -sin_table[idx]);             
                // cout << "At " << z << " " << phase_factor << " " << phase_factor_exp << endl;
                // cout << "J : " << j << " Z : " << z << " Subsize : " << sub_size << " z/2*sub_size " << static_cast<double>(z)/(2.0*static_cast<double>(sub_size))<< endl;
                // cout << "IDX: " << idx << endl;
                auto even_term = dft[j+z];
                auto odd_term = dft[j+z+sub_size] * phase_factor;
                dft[j+z] = even_term + odd_term;
                dft[j+z+sub_size] = even_term - odd_term;
            }
        }
    }
    return dft;
}

const vector<complex<double>>
fft_radix2_lookup_parallel(
    const vector<complex<double>> &sequence
){
    vector<complex<double>> dft(sequence);

    // List of all sin (and cos) values used in the computation (0, 2pi/N_SAMPLES, 4pi/N_SAMPLES, ..., (N-1)2pi/N_SAMPLES)
    vector<double> sin_table(sequence.size());
    // Create the list of sin values in parallel
    #pragma omp parallel for schedule(static) shared(sin_table)
    for(unsigned int i=0;i<sequence.size();i++){
        sin_table[i] = sin(2*M_PI*static_cast<float>(i)/sequence.size());
    }

    
    if(sequence.size()==0){
        throw invalid_argument("The sequence size must be different from zero");
    }
    else if((sequence.size()&(sequence.size()-1))!=0){
        throw invalid_argument("The sequence size must be a power of 2");
    }
    else if(sequence.size()==1){
        return dft;
    }
    reverse_bit_order(dft);

    // Iteration on the different depths(sub_size) of the problem
    // We'll start from subproblems of size 1 and go on from there
    // sub_size is the current size of the computed partial DTF in our problem
    for(unsigned int sub_size=1;sub_size<dft.size();sub_size*=2){
        // Rescale factor to access sin table using z
        unsigned int idx_rescale = dft.size() / (2*sub_size);

        // Parallelize both loops with collapse(2). Each work unit will compute two ouput values
        #pragma omp parallel for schedule(static) shared(dft) shared(idx_rescale) collapse(2)
        // At every inner iteration we consider two adjecent subproblems
        // j is the index of the current subproblem
        // Since we consider two problem for iteration j has to skip the adjecent subproblem
        for(unsigned int j=0;j<dft.size();j+=2*sub_size){
            
            
            //Iteration on the two adjecent subproblems
            //We just apply the formula
            for(unsigned int z=0;z<sub_size;z++){

                // phase factor e^(-i * 2 * M_PI * z / ( 2 * sub_size)) = cos(2 * M_PI * z / (2 * sub_size)) - i * sin(2 * M_PI * z / (2 * sub_size))
                // sin(2 * M_PI * z / (2 * sub_size)) = sin_table[z * dft.size() / (2 * sub_size)]
                // and, since sin and cos are shifted by pi/2, we have:
                // cos(2 * M_PI * z / (2 * sub_size)) = sin_table[z * dft.size() / (2 * sub_size) + dft.size() / 4]

                unsigned int idx = z * idx_rescale;
                auto phase_factor = complex<double>(sin_table[idx + dft.size()/4], -sin_table[idx]);
                auto even_term = dft[j+z];
                auto odd_term = dft[j+z+sub_size] * phase_factor;
                dft[j+z] = even_term + odd_term;
                dft[j+z+sub_size] = even_term - odd_term;
            }
        }
    }
    return dft;
}


const vector<complex<double>>
time_function(
    function<const vector<complex<double>>(const vector<complex<double>>)> f,
    const vector<complex<double>> &sequence,
    string name
) {
    auto start = std::chrono::high_resolution_clock::now();
    auto fft = f(sequence);
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
    for(unsigned int i=0;i<standard.size();++i){
        if(abs(standard[i]-sequence[i])>1e-7){
            cout << "Error in " << name << " at index " << i << endl;
            cout << "Expected: " << standard[i] << " Got: " << sequence[i] << endl;
        }
    }
    cout << "Test passed for " << name << endl;
}

int
main(){
    unsigned int constexpr dim = 1048576;
    vector<complex<double>> sequence(dim);
    for(unsigned int i=0; i<dim; ++i){
        double rand_real = static_cast <double>(rand()) / static_cast<double>(RAND_MAX);
        double rand_i = static_cast <double>(rand()) / static_cast<double>(RAND_MAX);
        sequence[i] = complex<double>(rand_real,rand_i);
    }
    auto fft_ref = time_function(fft_radix2, sequence, "normal");
    auto fft = time_function(fft_radix2_parallel, sequence, "parallel");
    verify(fft_ref, fft, "parallel");
    fft = time_function(fft_radix2_lookup, sequence, "lookup");
    verify(fft_ref, fft, "lookup");
    fft = time_function(fft_radix2_lookup_parallel, sequence, "lookup p");
    verify(fft_ref, fft, "lookup parallel");

}
