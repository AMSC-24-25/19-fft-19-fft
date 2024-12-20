#include "../include/fft.hpp"
#include <algorithm>
#include <map>

/* PRIVATE UTILITY FUNCTIONS */
template <typename T>
void inline
reverse_bit_order(
    std::vector<T> &sequence
);

unsigned int
reverse_number_bit(
    size_t value,
    size_t num_bit
);

template <typename T>
void inline
reverse_bit_order_parallel(
    std::vector<T> &sequence
);

unsigned int
reverse_number_bit(
    size_t value,
    size_t num_bit
) {
    unsigned int reversed = 0;
    if(value==0) {
        return 0;
    }
    if(num_bit==0) {
        throw std::invalid_argument(
                "The number of bits cannot be equal to zero");
    }
    for(int i=0;i<num_bit;++i){
        //Check if the value confronted bit by 
        //bit with 1...0 is different from zero
        if((value & (1<<i)) != 0){
            //Compute the reversed value adding 
            //the bit in the right position
            reversed |= (1<<(num_bit-1-i));
        }
    }

    return reversed;
}

template <typename T>
void inline
reverse_bit_order(
    std::vector<T> &sequence
){
    size_t seq_size = sequence.size();
    if(seq_size==0){
        throw std::invalid_argument(
                "The sequence size must be different from zero");
    }
    else if((seq_size&(seq_size-1))!=0){
        throw std::invalid_argument(
                "The sequence size must be a power of 2");
    }
    size_t num_bit = static_cast<int>(std::log2(seq_size));
    size_t swap_position = 0;

    // sequence can be reversed in parallel
    for(size_t i=0;i<seq_size;i++){
        swap_position = reverse_number_bit(i,num_bit);
        if(swap_position>i){
            swap(sequence[i],sequence[swap_position]);
        }
    }
}

template <typename T>
void inline
reverse_bit_order_parallel(
    std::vector<T> &sequence
){
    size_t seq_size = sequence.size();
    if(seq_size==0){
        throw std::invalid_argument(
                "The sequence size must be different from zero");
    }
    else if((seq_size&(seq_size-1))!=0){
        throw std::invalid_argument(
                "The sequence size must be a power of 2");
    }
    size_t num_bit = static_cast<int>(std::log2(seq_size));
    size_t swap_position = 0;

    // sequence can be reversed in parallel
#pragma omp parallel for schedule(static) shared(sequence) firstprivate(num_bit) private(swap_position)
    for(size_t i=0;i<seq_size;i++){
        swap_position = reverse_number_bit(i,num_bit);
        if(swap_position>i){
            swap(sequence[i],sequence[swap_position]);
        }
    }
}
/*********************/


void
AMSC::fft_radix2(
    const std::vector<std::complex<double>> &input,
    std::vector<std::complex<double>> &output
) {
    output = input;
    if(input.size()==0){
        throw std::invalid_argument(
                "The sequence size must be different from zero");
    }
    else if((input.size()&(input.size()-1))!=0){
        throw std::invalid_argument(
                "The sequence size must be a power of 2");
    }
    else if(input.size()==1){
        return;
    }
    reverse_bit_order(output);

    // Iteration on the different depths(sub_size) of the problem
    // We'll start from subproblems of size 1 and go on from there
    // sub_size is the current size of the computed partial DTF 
    // in our problem
    for(size_t sub_size=1; sub_size<output.size(); sub_size*=2) {

        // At every inner iteration we consider two adjecent 
        // subproblems
        // j is the index of the current subproblem
        // Since we consider two problem for iteration j has 
        // to skip the adjecent subproblem
// #pragma omp parallel for schedule(static) shared(dft) collapse(2)
        for(size_t j=0; j<output.size(); j+=2*sub_size){
            
            //Iteration on the two adjecent subproblems
            //We just apply the formula
            for(size_t z=0; z<sub_size; z++){
                auto angle =
                    std::complex<double>(0,-2*M_PI*z/(2*sub_size));
                auto phase_factor = exp(angle);
                auto even_term = output[j+z];
                auto odd_term = output[j+z+sub_size]*phase_factor;
                output[j+z] = even_term + odd_term;
                output[j+z+sub_size] = even_term - odd_term;
            }
        }
    }
}

void
AMSC::ifft_radix2(
    const std::vector<std::complex<double>> &input,
    std::vector<std::complex<double>> &output
) {
    output = input;

    if(input.size()==0){
        throw std::invalid_argument(
                "The sequence size must be different from zero");
    }
    else if((input.size()&(input.size()-1))!=0){
        throw std::invalid_argument(
                "The sequence size must be a power of 2");
    }
    else if(input.size()==1){
        return;
    }

    reverse_bit_order(output);

    // Iteration on the different depths(sub_size) of the problem
    // We'll start from subproblems of size 1 and go on from there
    // sub_size is the current size of the computed partial DTF 
    // in our problem
    for(size_t sub_size=1; sub_size<output.size(); sub_size*=2) {

        // At every inner iteration we consider two adjecent 
        // subproblems
        // j is the index of the current subproblem
        // Since we consider two problem for iteration j has 
        // to skip the adjecent subproblem
        for(size_t j=0; j<output.size(); j+=2*sub_size){
            
            //Iteration on the two adjecent subproblems
            //We just apply the formula
            for(size_t z=0; z<sub_size; z++){
                auto angle =
                    std::complex<double>(0,2*M_PI*z/(2*sub_size));
                auto phase_factor = exp(angle);
                auto even_term = output[j+z];
                auto odd_term = output[j+z+sub_size]*phase_factor;
                output[j+z] = even_term + odd_term;
                output[j+z+sub_size] = even_term - odd_term;
            }
        }
    }

    for(size_t i = 0; i < output.size(); ++i) {
        output[i] *= 1.0/input.size();
    }
}

void
AMSC::fft_radix2_parallel(
    const std::vector<std::complex<double>> &input,
    std::vector<std::complex<double>> &output
) {
    output = input;
    if(input.size()==0){
        throw std::invalid_argument(
                "The sequence size must be different from zero");
    }
    else if((input.size()&(input.size()-1))!=0){
        throw std::invalid_argument(
                "The sequence size must be a power of 2");
    }
    else if(input.size()==1){
        return;
    }
    reverse_bit_order_parallel(output);

    // Iteration on the different depths(sub_size) of the problem
    // We'll start from subproblems of size 1 and go on from there
    // sub_size is the current size of the computed partial DTF 
    // in our problem
    for(size_t sub_size=1; sub_size<output.size(); sub_size*=2) {

        // At every inner iteration we consider two adjecent 
        // subproblems
        // j is the index of the current subproblem
        // Since we consider two problem for iteration j has 
        // to skip the adjecent subproblem
#pragma omp parallel for schedule(static) shared(output) collapse(2)
        for(size_t j=0; j<output.size(); j+=2*sub_size){
            
            //Iteration on the two adjecent subproblems
            //We just apply the formula
            for(size_t z=0; z<sub_size; z++){
                auto angle =
                    std::complex<double>(0,-2*M_PI*z/(2*sub_size));
                auto phase_factor = exp(angle);
                auto even_term = output[j+z];
                auto odd_term = output[j+z+sub_size]*phase_factor;
                output[j+z] = even_term + odd_term;
                output[j+z+sub_size] = even_term - odd_term;
            }
        }
    }
}

void
AMSC::ifft_radix2_parallel(
    const std::vector<std::complex<double>> &input,
    std::vector<std::complex<double>> &output
) {
    output = input;

    if(input.size()==0){
        throw std::invalid_argument(
                "The sequence size must be different from zero");
    }
    else if((input.size()&(input.size()-1))!=0){
        throw std::invalid_argument(
                "The sequence size must be a power of 2");
    }
    else if(input.size()==1){
        return;
    }

    reverse_bit_order_parallel(output);

    // Iteration on the different depths(sub_size) of the problem
    // We'll start from subproblems of size 1 and go on from there
    // sub_size is the current size of the computed partial DTF 
    // in our problem
    for(size_t sub_size=1; sub_size<output.size(); sub_size*=2) {

        // At every inner iteration we consider two adjecent 
        // subproblems
        // j is the index of the current subproblem
        // Since we consider two problem for iteration j has 
        // to skip the adjecent subproblem
#pragma omp parallel for schedule(static) shared(output) collapse(2)
        for(size_t j=0; j<output.size(); j+=2*sub_size){
            
            //Iteration on the two adjecent subproblems
            //We just apply the formula
            for(size_t z=0; z<sub_size; z++){
                auto angle =
                    std::complex<double>(0,2*M_PI*z/(2*sub_size));
                auto phase_factor = exp(angle);
                auto even_term = output[j+z];
                auto odd_term = output[j+z+sub_size]*phase_factor;
                output[j+z] = even_term + odd_term;
                output[j+z+sub_size] = even_term - odd_term;
            }
        }
    }

#pragma omp parallel for schedule(static)
    for(size_t i = 0; i < output.size(); ++i) {
        output[i] *= 1.0/input.size();
    }
}

void
AMSC::fft_radix2_lookup(
    const std::vector<std::complex<double>> &input,
    std::vector<std::complex<double>> &output
) {
    output = input;
    std::vector<double> sin_table(input.size());

    for(size_t i=0; i<input.size(); i++){
        sin_table[i] =
            sin(2*M_PI*static_cast<float>(i)/input.size());
    }

    
    if(input.size()==0){
        throw std::invalid_argument(
                "The sequence size must be different from zero");
    }
    else if((input.size()&(input.size()-1))!=0){
        throw std::invalid_argument(
                "The sequence size must be a power of 2");
    }
    else if(input.size()==1){
        return;
    }
    reverse_bit_order(output);

    // Iteration on the different depths(sub_size) of the problem
    // We'll start from subproblems of size 1 and go on from there
    // sub_size is the current size of the computed partial DTF
    // in our problem
    for(size_t sub_size=1; sub_size<output.size(); sub_size*=2) {

        // At every inner iteration we consider two adjecent
        // subproblems
        // j is the index of the current subproblem
        // Since we consider two problem for iteration j has
        // to skip the adjecent subproblem
        for(size_t j=0; j<output.size(); j+=2*sub_size) {
            
            //Iteration on the two adjecent subproblems
            //We just apply the formula
            for(size_t z=0; z<sub_size; z++){
                size_t idx = z * output.size() / (2*sub_size);
                auto phase_factor =
                    std::complex<double>(
                        sin_table[idx + output.size()/4],
                        -sin_table[idx]
                    );
                auto even_term = output[j+z];
                auto odd_term = output[j+z+sub_size] * phase_factor;
                output[j+z] = even_term + odd_term;
                output[j+z+sub_size] = even_term - odd_term;
            }
        }
    }
}

void
AMSC::fft_radix2_lookup_parallel(
    const std::vector<std::complex<double>> &input,
    std::vector<std::complex<double>> &output
) {
    output = input;

    // List of all sin (and cos) values used in the
    // computation (0, 2pi/N_SAMPLES, 4pi/N_SAMPLES, 
    // ..., (N-1)2pi/N_SAMPLES)
    std::vector<double> sin_table(input.size());

    // Create the list of sin values in parallel
    #pragma omp parallel for schedule(static) shared(sin_table)
    for(size_t i=0; i<input.size(); i++){
        sin_table[i] =
            sin(2*M_PI*static_cast<float>(i)/input.size());
    }
    
    if(input.size()==0){
        throw std::invalid_argument(
                "The sequence size must be different from zero");
    }
    else if((input.size()&(input.size()-1))!=0){
        throw std::invalid_argument(
                "The sequence size must be a power of 2");
    }
    else if(input.size()==1){
        return;
    }
    reverse_bit_order_parallel(output);

    // Iteration on the different depths(sub_size) of the problem
    // We'll start from subproblems of size 1 and go on from there
    // sub_size is the current size of the computed partial DTF
    // in our problem
    for(size_t sub_size=1; sub_size<output.size(); sub_size*=2){
        // Rescale factor to access sin table using z
        size_t idx_rescale = output.size() / (2*sub_size);

        // Parallelize both loops with collapse(2). Each work
        // unit will compute two ouput values
#pragma omp parallel for schedule(static) shared(output) shared(idx_rescale) collapse(2)
        // At every inner iteration we consider two adjecent
        // subproblems
        // j is the index of the current subproblem
        // Since we consider two problem for iteration j has to
        // skip the adjecent subproblem
        for(size_t j=0; j<output.size(); j+=2*sub_size){
            //Iteration on the two adjecent subproblems
            //We just apply the formula
            for(size_t z=0; z<sub_size; z++){
                // phase factor e^(-i * 2 * M_PI * z / 
                // ( 2 * sub_size)) = cos(2 * M_PI * z / 
                // (2 * sub_size)) - i * sin(2 * M_PI * z /
                // (2 * sub_size))
                //
                // sin(2 * M_PI * z / (2 * sub_size)) =
                // sin_table[z * dft.size() / (2 * sub_size)]
                //
                // and, since sin and cos are shifted by pi/2, 
                // we have:
                //
                // cos(2 * M_PI * z / (2 * sub_size)) =
                //  sin_table[z * dft.size() / (2 * sub_size) +
                //  dft.size() / 4]
                size_t idx = z * idx_rescale;
                auto phase_factor = std::complex<double>(
                    sin_table[idx + output.size()/4],
                    -sin_table[idx]
                );
                auto even_term = output[j+z];
                auto odd_term = output[j+z+sub_size] * phase_factor;
                output[j+z] = even_term + odd_term;
                output[j+z+sub_size] = even_term - odd_term;
            }
        }
    }
}

void 
AMSC::fft_radix2_real_lookup(
    const std::vector<double> &input,
    std::vector<std::complex<double>> &output
) {
    const unsigned int N = input.size();
    output.resize(N); //Make sure that the output can hold N values
    
    // Input validation
    if(N==0){
        throw std::invalid_argument("Size must be non zero");
    }
    else if( (N&(N-1))!=0 ){
        throw std::invalid_argument("Size must be a power of 2");
    }
    else if(N==1){
        output[0] = std::complex<double>(input[0],0.0); // If the sequence has length one then the DFT is the same sequence
        return;
    }

    // Let's take two real valued vectors w[n] and g[n] of size N
    // We create the vector z[n] = w[n] + i*g[n]
    // We have that w[n] = (z[n] + z*[n])/2 applying the DTF to both sides
    // W[k] = (Z[k] + Z*[N-k])/2 (knowing that DTF{x*[n]} = X*[N-k])
    // We now take as w[n] = x[2n] = x_even[n] and g[n] = x[2n+1] = x_odd[n]

    // Construct a complex sequence z of half the size from the input sequence.
    // The real parts are the even-indexed elements, and the imaginary parts are the odd-indexed elements.
    std::vector<std::complex<double>> z(N/2);
    for(unsigned int i=0;i<N/2;++i){
        z[i] = std::complex<double>(input[2*i],input[2*i+1]);
    }

    std::vector<std::complex<double>> Z(z.size());
    // Computing the FFT of z
    fft_radix2_lookup(z,Z);

    std::complex<double> X_even,X_odd;


    X_even = (Z[0] + conj(Z[0]))/2.0;
    X_odd = std::complex<double>(0.0,-1.0)*(Z[0] - conj(Z[0]))/2.0;

    output[0] = X_even + X_odd;  // Handling the zero component
    output[N/2] = X_even - X_odd; // Handling the Nyquist component 

    for(unsigned int i=1;i<N/2;++i){
        // Even and odd parts are derived from symmetry properties.
        X_even = (Z[i] + conj(Z[N/2-i]))/2.0;
        X_odd = std::complex<double>(0.0,-1.0)*(Z[i] - conj(Z[N/2-i]))/2.0;

        // Apply the phase factor to the odd component and combine. 
        std::complex<double> phase_factor = exp(std::complex<double>(0,-2*M_PI*i/N));
        output[i] = X_even + phase_factor*X_odd;
        output[N-i] = std::conj(output[i]);
    }
}
