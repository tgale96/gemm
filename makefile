test:
	g++ -std=c++11 -I. -I/usr/local/cuda/include test.cc sgemm.cc -o test

clean:
	rm -rf test
