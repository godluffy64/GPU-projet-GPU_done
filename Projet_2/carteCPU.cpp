#include <cmath>

#include "carteCPU.hpp"
#include "los/ppm.hpp"

using namespace std;


void drawMap(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx,const int Cy)
{
    for(int Py = 0; Py < MapHeight; Py++)
    {
        for(int Px = 0; Px < MapWidth; Px++)
        {

        // DDA entre le point c (Cx, Cy) et le point P (Px, Py);
            float Dx, Dy, Dz,  D;

            Dx = Px - Cx;   // delta x
            Dy = Py - Cy;   // delta y
            Dz = h_in[Py * MapWidth + Px] - h_in[Cy * MapWidth + Cx];   // delta z
            D = max(abs(Dx), abs(Dy));  // delta positif max entre Dx et Dy
            double angle, angle_ref = atan(Dz / sqrt((Dx * Dx) + (Dy * Dy)));

            float Cx_dda = (float) Cx, Cy_dda = (float) Cy;
            float incX = Dx / D;
            float incY = Dy / D;
            int Lx, Ly;

            h_out[Py * MapWidth + Px] = 244;
            for (int i = 0; i < D - 1; i++)
            {
                Cx_dda += incX;
                Cy_dda += incY;
                Lx = (int)round(Cx_dda);
                Ly = (int)round(Cy_dda);
                Dx = Px - Lx;
                Dy = Py - Ly;    
                Dz = h_in[Py * MapWidth + Px] - h_in[Ly * MapWidth + Lx];  
                // Calcule Angle 
                angle = atan(Dz / sqrt((Dx * Dx) + (Dy * Dy)));     
                if (angle_ref >= angle)
                {
                    //h_out.setPixel(Px, Py, 0);
                    h_out[Py * MapWidth + Px] = 0;
                    break;
                }                         
            }   
        }
    }
}