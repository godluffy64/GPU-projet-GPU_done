#include "utils/commonCUDA.hpp"
#include "utils/chronoGPU.hpp"

#include "carteGPU.hpp"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <iomanip>

using namespace std;

/*__global__ void kernelMap(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy)
{
    int Px = blockIdx.x * blockDim.x + threadIdx.x;
    int Py = blockIdx.y * blockDim.y + threadIdx.y;

    if (Px < MapWidth && Py < MapHeight)
    {
        float Dx = Px - Cx;
        float Dy = Py - Cy;
        float Dz = h_in[Py * MapWidth + Px] - h_in[Cy * MapWidth + Cx];
        float D = fmaxf(fabsf(Dx), fabsf(Dy));
        double angle_ref = atan2f(Dz, sqrt((Dx * Dx) + (Dy * Dy)));

        float Cx_dda = static_cast<float>(Cx), Cy_dda = static_cast<float>(Cy);
        float incX = Dx / D;
        float incY = Dy / D;

        h_out[Py * MapWidth + Px] = 244;

        for (int i = 0; i < D - 1; i++)
        {
            Cx_dda += incX;
            Cy_dda += incY;
            int Lx = static_cast<int>(round(Cx_dda));
            int Ly = static_cast<int>(round(Cy_dda));

            Dx = Px - Lx;
            Dy = Py - Ly;
            Dz = h_in[Py * MapWidth + Px] - h_in[Ly * MapWidth + Lx];

            double angle = atan2f(Dz, sqrt((Dx * Dx) + (Dy * Dy)));

            if (angle_ref >= angle)
            {
                h_out[Py * MapWidth + Px] = 0;
                break;
            }
        }
    }
}*/

__global__ void kernelMap(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy)
{

    for(int Py = blockDim.y * blockIdx.y + threadIdx.y; Py < MapHeight; Py += blockDim.y * gridDim.y)
    {
        for(int Px = blockDim.x * blockIdx.x + threadIdx.x; Px < MapWidth; Px += blockDim.x * gridDim.x)
        {
        // DDA entre le point c (Cx, Cy) et le point P (indexX, indexY);
            
            float Dx = Px - Cx;
            float Dy = Py - Cy;
            float Dz = h_in[Py * MapWidth + Px] - h_in[Cy * MapWidth + Cx];
            float D = max(abs(Dx), abs(Dy));
            double angle_ref = atan(Dz / sqrt((Dx * Dx) + (Dy * Dy)));

            float Cx_dda = static_cast<float>(Cx), Cy_dda = static_cast<float>(Cy);
            float incX = Dx / D;
            float incY = Dy / D;

            h_out[Py * MapWidth + Px] = 244;

            for (int i = 0; i < D - 1; i++)
            {
                Cx_dda += incX;
                Cy_dda += incY;
                int Lx = static_cast<int>(round(Cx_dda));
                int Ly = static_cast<int>(round(Cy_dda));

                Dx = Px - Lx;
                Dy = Py - Ly;
                Dz = h_in[Py * MapWidth + Px] - h_in[Ly * MapWidth + Lx];

                double angle = atan(Dz / sqrt((Dx * Dx) + (Dy * Dy)));

                if (angle_ref >= angle)
                {
                    h_out[Py * MapWidth + Px] = 0;
                    break;
                }                     
            } 
        }
    }
}

void carteGPU(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy)
{
    uint8_t *d_in, *d_out;
    cudaMalloc((void**)&d_in, MapWidth * MapHeight * sizeof(uint8_t));
    cudaMalloc((void**)&d_out, MapWidth * MapHeight * sizeof(uint8_t));

    // Copie des données vers le GPU
    cudaMemcpy(d_in, h_in, MapWidth * MapHeight * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Définir la taille des blocs et des grilles
    dim3 blockSize(16, 16);
    dim3 gridSize((MapWidth + blockSize.x - 1) / blockSize.x, (MapHeight + blockSize.y - 1) / blockSize.y);

    // Appel du kernel
    kernelMap<<<gridSize, blockSize>>>(d_in, d_out, MapWidth, MapHeight, Cx, Cy);

    // Copie des résultats depuis le GPU
    cudaMemcpy(h_out, d_out, MapWidth * MapHeight * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Libération de la mémoire GPU
    cudaFree(d_in);
    cudaFree(d_out);
}