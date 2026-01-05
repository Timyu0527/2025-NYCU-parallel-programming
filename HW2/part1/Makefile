C := g++
CFLAGS := -O3 -Wall -Wextra -pthread

TARGET := pi.out
SRC := pi.cpp

SIMD_FLAGS ?= -mavx2 -mfma

all: $(TARGET)

$(TARGET): $(SRC)
	$(C) $(CFLAGS) $(SIMD_FLAGS) -o $@ $<
check:
	run -c 4 -- bash -c "time (./pi.out 3 100000000; ./pi.out 4 100000000)"

nosimd:
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

