#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_mask mask, maskExpPos;
  __pp_vec_float clamp;
  clamp = _pp_vset_float(9.999999f);
  int width;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    width = min(VECTOR_WIDTH, N - i);
    mask = _pp_init_ones(width);

    __pp_vec_float result, base;
    __pp_vec_int exp, zeros, ones;
    _pp_vset_float(result, 1.0f, mask);
    _pp_vload_float(base, values + i, mask);
    _pp_vload_int(exp, exponents + i, mask);
    zeros = _pp_vset_int(0);
    ones = _pp_vset_int(1);

    while(true) {
      _pp_vgt_int(maskExpPos, exp, zeros, mask);
      int bitCnt = _pp_cntbits(maskExpPos);
      if (bitCnt == 0) {
        break;
      }

      _pp_vmult_float(result, result, base, maskExpPos);
      _pp_vsub_int(exp, exp, ones, maskExpPos);
    }
    __pp_mask maskClamp;
    _pp_vgt_float(maskClamp, result, clamp, mask);
    _pp_vmove_float(result, clamp, maskClamp);

    _pp_vstore_float(output + i, result, mask);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  __pp_mask mask;
  __pp_vec_float sum;
  int width;
  sum = _pp_vset_float(0.0f);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    width = min(N - i, VECTOR_WIDTH);
    mask = _pp_init_ones(width);

    __pp_vec_float x;
    _pp_vload_float(x, values + i, mask);
    _pp_vadd_float(sum, sum, x, mask);
    
  }
  
  for (int i = VECTOR_WIDTH; i > 1; i >>= 1) {
    _pp_hadd_float(sum, sum);
    _pp_interleave_float(sum, sum);
  }
  return sum.value[0];
}
