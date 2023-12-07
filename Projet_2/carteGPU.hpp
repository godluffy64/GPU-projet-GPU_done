#ifndef __CARTE_GPU__
#define __CARTE_GPU__


#include "utils/commonCUDA.hpp"
#include "utils/chronoGPU.hpp"
#include "los/ppm.hpp"

using namespace los;

__global__ void kernelMap(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy);
void carteGPU(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy);



#endif //__CARTE_GPU__