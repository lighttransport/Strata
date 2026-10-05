CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -mavx2 -fopenmp -Wall -Wextra

membw: membw.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f membw

.PHONY: clean
