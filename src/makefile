INC_DIR=../include

test: test.o fft.o
	g++ -fopenmp -o test test.o fft.o

test.o: test.cpp
	g++ -fopenmp -o test.o -c test.cpp

fft.o: fft.cpp
	g++ -fopenmp -o fft.o -c fft.cpp

clean:
	rm -f *.o
	rm -f test
