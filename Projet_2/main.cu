#include <iostream>
#include <cstdlib>
#include <iomanip>

#include "utils/chronoCPU.hpp"
#include "utils/chronoGPU.hpp"
#include "los/ppm.hpp"
#include "carteCPU.hpp"
#include "carteGPU.hpp"

#define Cx 245
#define Cy 497

int main(int argc, char **argv)
{
	// Parse program arguments
	// ================================================================================================================
	// Allocation and initialization

	// ================================================================================================================

	// ================================================================================================================
	// CPU sequential
	std::cout << "============================================" << std::endl;
	std::cout << "         Sequential version on CPU          " << std::endl;
	std::cout << "============================================" << std::endl;



    Heightmap h_inCPU("img/input/1.input.ppm");
    Heightmap h_outCPU(h_inCPU.getWidth(), h_inCPU.getHeight());

	ChronoGPU chrCPU;
	chrCPU.start();		// CPU method
	drawMap(h_inCPU.getPtr(), h_outCPU.getPtr(), h_inCPU.getWidth(), h_inCPU.getHeight(),  Cx, Cy);
	//void drawMap(int *data,uint8_t *h_data, Heightmap h_out, const int MapSize, const int MapWidth, const int MapHeight,const int Cx,const int Cy)
	h_outCPU.saveTo("img/Result/CPU/LimousinCPU.ppm");
	chrCPU.stop();

	const float timeComputeCPU = chrCPU.elapsedTime();
	std::cout << "-> Done : " << std::fixed << std::setprecision(2) << timeComputeCPU << " ms" << std::endl
			  << std::endl;

	// ================================================================================================================

	// ================================================================================================================
	// GPU CUDA
	std::cout << "============================================" << std::endl;
	std::cout << "          Parallel version on GPU           " << std::endl;
	std::cout << "============================================" << std::endl;

	// data GPU

    Heightmap h_inGPU("img/input/1.input.ppm");
    Heightmap h_outGPU(h_inGPU.getWidth(), h_inGPU.getHeight());

	// data GPU

	// GPU allocation
	ChronoGPU chrGPU;
	chrGPU.start();	

	carteGPU(h_inGPU.getPtr(), h_outGPU.getPtr(), h_inGPU.getWidth(), h_inGPU.getHeight(), Cx, Cy);
	h_outGPU.saveTo("img/Result/GPU/LimousinGPU.ppm");

	chrGPU.stop();
	// ======================

	const float timeAllocGPU = chrGPU.elapsedTime();
	std::cout << "-> Done : " << std::fixed << std::setprecision(2) << timeAllocGPU << " ms" << std::endl
			  << std::endl;

	// Copy from host to device
	

	
	// Launch kernel

	// copy from device to host

	// Free GPU memory

	// ================================================================================================================

	std::cout << "============================================" << std::endl;
	std::cout << "              Checking results              " << std::endl;
	std::cout << "============================================" << std::endl;

	std::cout << h_inCPU.getHeight() << "   " << h_inCPU.getWidth() << std::endl;
	for (int i = 0; i < h_inCPU.getHeight() - 1; i++)
	{
		for (int j = 0; i < h_inCPU.getWidth() - 1; j++)
			{
				if (h_outCPU.getPixel(j, i) != h_outGPU.getPixel(j, i))
				{
					std::cout << "error on index (" << i << ", " << j << ")" << std::endl;
					std::cout << "value CPU : " << +h_outCPU.getPixel(j, i) << ", value GPU : " << +h_outGPU.getPixel(j, i) << std::endl;

					return EXIT_FAILURE;
				} 
			}
	}



	/*std::cout << "Congratulations! Job's done!" << std::endl
			  << std::endl;

	std::cout << "============================================" << std::endl;
	std::cout << "            Times recapitulation            " << std::endl;
	std::cout << "============================================" << std::endl;
	std::cout << "-> CPU	Sequential" << std::endl;
	std::cout << "   - Computation:    " << std::fixed << std::setprecision(2)
			  << timeComputeCPU << " ms" << std::endl;
	std::cout << "-> GPU	" << std::endl;
	std::cout << "   - Allocation:     " << std::fixed << std::setprecision(2)
			  << timeAllocGPU << " ms " << std::endl;
	std::cout << "   - Host to Device: " << std::fixed << std::setprecision(2)
			  << timeHtoDGPU << " ms" << std::endl;
	std::cout << "   - Computation:    " << std::fixed << std::setprecision(2)
			  << timeComputeGPU << " ms" << std::endl;
	std::cout << "   - Device to Host: " << std::fixed << std::setprecision(2)
			  << timeDtoHGPU << " ms " << std::endl
			  << std::endl;
*/
	return EXIT_SUCCESS;
}