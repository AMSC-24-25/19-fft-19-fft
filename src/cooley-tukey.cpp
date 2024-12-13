#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <stdexcept>

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
unsigned int reverse_number_bit(const unsigned int &value,const unsigned int &num_bit){
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
void inline reverse_bit_order(vector<T> &sequence){
    unsigned int seq_size = sequence.size();
    if(sequence.size()==0){
        throw invalid_argument("The sequence size must be different from zero");
    }
    else if(seq_size&(seq_size-1)!=0){
        throw invalid_argument("The sequence size must be a power of 2");
    }
    unsigned int num_bit = static_cast<int> (log2(seq_size));
    unsigned int swap_position = 0;
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
const vector<complex<double>> fft_radix2(const vector<complex<double>> &sequence){
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

int main(){
    unsigned int constexpr dim = 32;
    vector<complex<double>> sequence(dim);
    for(unsigned int i=0; i<dim; ++i){
        sequence[i] = complex<double>(i,0.0);
    }
    auto fft = fft_radix2(sequence);
    for(auto &data : fft){
        cout << data << " ";
    }
    cout << endl;
}