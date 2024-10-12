
/** 
 * @file metodo_directo_grav.c
 * @author Santiago Salas santiago.salas@estudiante.uam.es
 * 
 * File that does the operations for the acceletation of each body suffered by the effects of the gravitational pull of all other bodies
 * 
 * This is the point where most of the time of the simulation happens, optimizing this file even by a little has grat impact on the 
 * total performance
 * 
 * In this case we are compiling this file with -O3 for maximum optimizations, this does not affect results but speeds performance
 */

#include "../inc/medoto_directo_defs.h"

//Macros to correctly access multidimensional arrays that has been flatten 
#define S(size_i, size_j, cord, i, j, pointer) pointer[(size_i * size_j * cord) + (size_j * i) + j]

//Cuda kernels
__device__ void calculate_acceleration_values(realptr d_masses, realptr position, realptr sdata, int n);

template <unsigned int blockSize>
__device__ void warpReduce(volatile realptr sdata, int baseIndex0, int baseIndex1, int baseIndex2);

template <unsigned int blockSize>
__global__ void calculate_acceleration_values_block_reduce(realptr d_masses, realptr position, realptr d_block_holder, int n, unsigned int number_of_blocks_j);

template <unsigned int blockSize>
__device__ void full_block_reduction (realptr d_block_holder, realptr sdata, int n, unsigned int number_of_blocks_j);

__global__ void finish_block_reduce (realptr acceleration, realptr d_block_holder, int n, unsigned int number_of_blocks_j);

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**
 * Function to check errors in CUDA. 
 * 
 * Extracted from StackOverflow 
 * @link https://stackoverflow.com/a/14038590
 */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int calculate_acceleration(Simulation* simulation)
/**
 * Funtion to calculate the velocity and acceleration of the bodies using the current positions and velocities. 
 * It uses a cuda kernel to do GPU acceleration and calculate the values
 * 
 * @param simulation(Simulation*): a pointer to the simulation object we are simulating
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 *         The resulting acceleration is stored inside acceleration atribute of the simulation
**/
{   
    //Error checking
    if(simulation == NULL)
        return STATUS_ERROR;

    //Set up memory for cuda as 0 for reduccions
    cudaMemset( simulation->d_block_holder, 0.0, 3 * simulation->n  * simulation->gridDimsGrav.y * sizeof(simulation->d_block_holder[0]));
    cudaMemset( simulation->d_acceleration, 0.0, 3 * simulation->n  * sizeof(simulation->d_acceleration[0]));

    //Call cuda with the correct block size
    switch (simulation->threadBlockDimsGrav.y)
    {   
        case 1024:
            calculate_acceleration_values_block_reduce<1024><<<simulation->gridDimsGrav, simulation->threadBlockDimsGrav, 3 * simulation->threadBlockDimsGrav.x * simulation->threadBlockDimsGrav.y * sizeof(real)>>>(simulation->d_masses, simulation->d_positions, simulation->d_block_holder ,simulation->n, simulation->gridDimsGrav.y);
            break;
        case 512:
            calculate_acceleration_values_block_reduce<512><<<simulation->gridDimsGrav, simulation->threadBlockDimsGrav, 3 * simulation->threadBlockDimsGrav.x * simulation->threadBlockDimsGrav.y * sizeof(real)>>>(simulation->d_masses, simulation->d_positions, simulation->d_block_holder ,simulation->n, simulation->gridDimsGrav.y);
            break;
        case 256:
            calculate_acceleration_values_block_reduce<256><<<simulation->gridDimsGrav, simulation->threadBlockDimsGrav, 3 * simulation->threadBlockDimsGrav.x * simulation->threadBlockDimsGrav.y * sizeof(real)>>>(simulation->d_masses, simulation->d_positions, simulation->d_block_holder ,simulation->n, simulation->gridDimsGrav.y);
            break;
        case 128:
            calculate_acceleration_values_block_reduce<128><<<simulation->gridDimsGrav, simulation->threadBlockDimsGrav, 3 * simulation->threadBlockDimsGrav.x * simulation->threadBlockDimsGrav.y * sizeof(real)>>>(simulation->d_masses, simulation->d_positions, simulation->d_block_holder ,simulation->n, simulation->gridDimsGrav.y);
            break;
        case 64:
            calculate_acceleration_values_block_reduce<64><<<simulation->gridDimsGrav, simulation->threadBlockDimsGrav, 3 * simulation->threadBlockDimsGrav.x * simulation->threadBlockDimsGrav.y * sizeof(real)>>>(simulation->d_masses, simulation->d_positions, simulation->d_block_holder ,simulation->n, simulation->gridDimsGrav.y);
            break;            
        case 32:
            calculate_acceleration_values_block_reduce<32><<<simulation->gridDimsGrav, simulation->threadBlockDimsGrav, 3 * simulation->threadBlockDimsGrav.x * simulation->threadBlockDimsGrav.y * sizeof(real)>>>(simulation->d_masses, simulation->d_positions, simulation->d_block_holder ,simulation->n, simulation->gridDimsGrav.y);
            break;
    }
    
    
    cudaError_t status = cudaGetLastError();
    cudaErrorCheck(status);

    finish_block_reduce<<<simulation->gridDimsLeap,simulation->threadBlockLeap>>>(simulation->d_acceleration, simulation->d_block_holder, simulation->n, simulation->gridDimsGrav.y);
    
    status = cudaGetLastError();
    cudaErrorCheck(status);
    return STATUS_OK;
}

template <unsigned int blockSize>
__global__ void calculate_acceleration_values_block_reduce(realptr d_masses, realptr position, realptr d_block_holder, int n, unsigned int number_of_blocks_j)
/**
 * A cuda kernel that calculate the 3n**2 acceleration values and reduce them to an array of size of 3n * gridDim.y. 
 * @param d_masses (realptr): A cuda array of size n with the masses of each body
 * @param d_position (realptr): A cuda array of size 3n with the current position of all of the bodies stored in the following matter x1,y1,z1,x2,y2,z2...xn,yn,zn
 * @param d_block_holder (realptr): An array where the block-reduced results will be stored. It must be of size 3n * gridDim.y
 * @param n (int): the number of bodies
 * @param number_of_blocks_j (unsigned int): The number of blocks in the j dimension. Used for the block reduction
 */
{   
    //Array where all values of this block will be stored
    extern __shared__ real sdata[];

    //Calculate aceleration values
    calculate_acceleration_values(d_masses, position, sdata, n);
        
    //Reduce all values of this block
    full_block_reduction<blockSize>(d_block_holder, sdata, n, number_of_blocks_j);
}


__device__ void calculate_acceleration_values(realptr d_masses, realptr position, realptr sdata, int n)
/**
 * Cuda kernel that calculates the acceleration values that each body suffers from every other body
 * @param d_masses (realptr): A cuda array of size n with the masses of each body
 * @param d_position (realptr): A cuda array of size 3n with the current position of all of the bodies stored in the following matter x1,y1,z1,x2,y2,z2...xn,yn,zn
 * @param sdata (realptr): A shared array in which to store all of the acceleration values
 * @param n (int): the number of bodies
 */
{
    //Get position in the block
    int x = threadIdx.x;
    int y = threadIdx.y;

    //Get universal position
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;    
    
    real softening2 = softening * softening;

    //Calculate pull of one body by other body
    if(i < n && j < n && i != j)
    {   
        real dx = position[j] - position[i]; //rx body 2 - rx body 1
        real dy = position[j + n] - position[i + n]; //ry body 2 - ry body 1
        real dz = position[j + n + n] - position[i + n + n]; //rz body 2 - rz body 1
        
        real r = rsqrt(dx * dx + dy * dy + dz * dz + softening2); //distance magnitud with some softening
        r = (G * d_masses[j] * r * r * r ); //Acceleration formula

        S(blockDim.x, blockDim.y, 0, x, y, sdata) =  r * dx; //Acceleration formula for x
        S(blockDim.x, blockDim.y, 1, x, y, sdata) =  r * dy; //Acceleration formula for y
        S(blockDim.x, blockDim.y, 2, x, y, sdata) =  r * dz; //Acceleration formula for z
    }
    //Fill with 0 the remaining values in the array with 0
    else
    {  
        S(blockDim.x, blockDim.y, 0, x, y, sdata) = 0.0;    //x
        S(blockDim.x, blockDim.y, 1, x, y, sdata) = 0.0;    //y
        S(blockDim.x, blockDim.y, 2, x, y, sdata) = 0.0;    //z
    }
    
}

template <unsigned int blockSize>
__device__ void full_block_reduction (realptr d_block_holder, realptr sdata, int n, unsigned int number_of_blocks_j)
/**
 * A cuda kernel that reduces the acceleration values of this block to one for every body in this block. 
 * It implements a modified version of reduction6 in https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * 
 * @param d_block_holder (realptr): An array where the block-reduced results will be stored. It must be of size 3n * gridDim.y
 * @param sdata (realptr): A shared array in which all of the aceleration values of this block are stored
 * @param n (int): the number of bodies
 * @param number_of_blocks_j (unsigned int): The number of blocks in the j dimension.
 */
{
    //Get position in the block
    int x = threadIdx.x;
    int y = threadIdx.y;

    //Get universal position
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int baseIndex0 = (blockDim.y * x) + y;
    int baseIndex1 = (blockDim.x * blockDim.y) + (blockDim.y * x) + y;
    int baseIndex2 = (blockDim.x * blockDim.y * 2) + (blockDim.y * x) + y;

    __syncthreads();
    
    //Unrolled reduction
    if (blockSize >= 1024) 
    {   
        if (y < 512) 
        { 
            sdata[baseIndex0] += sdata[baseIndex0 + 512];
            sdata[baseIndex1] += sdata[baseIndex1 + 512];
            sdata[baseIndex2] += sdata[baseIndex2 + 512];
        }
        __syncthreads();
    }

    if (blockSize >= 512) 
    {   
        if (y < 256) 
        { 
            sdata[baseIndex0] += sdata[baseIndex0 + 256];
            sdata[baseIndex1] += sdata[baseIndex1 + 256];
            sdata[baseIndex2] += sdata[baseIndex2 + 256];
        }
        __syncthreads();
    }

    if (blockSize >= 256) 
    {   
        if (y < 128) 
        { 
            sdata[baseIndex0] += sdata[baseIndex0 + 128];
            sdata[baseIndex1] += sdata[baseIndex1 + 128];
            sdata[baseIndex2] += sdata[baseIndex2 + 128];
        }
        __syncthreads();
    }

    if (blockSize >= 128) 
    {   
        if (y < 64) 
        { 
            sdata[baseIndex0] += sdata[baseIndex0 + 64];
            sdata[baseIndex1] += sdata[baseIndex1 + 64];
            sdata[baseIndex2] += sdata[baseIndex2 + 64];
        }
        __syncthreads();
    }

    if (y < 32 && i < n) 
        warpReduce<blockSize>(sdata, baseIndex0, baseIndex1, baseIndex2);
    
    // write result for this block to global mem
    if (y == 0)
    {
        S(n, number_of_blocks_j, 0, i, blockIdx.y, d_block_holder) += sdata[baseIndex0];
        S(n, number_of_blocks_j, 1, i, blockIdx.y, d_block_holder) += sdata[baseIndex1];
        S(n, number_of_blocks_j, 2, i, blockIdx.y, d_block_holder) += sdata[baseIndex2];
    } 
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile realptr sdata, int baseIndex0, int baseIndex1, int baseIndex2)
/**
 * A cuda kernel that helps in the reduction process. It completly reduces a warp
 * It implements a modified version of warpReduce from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * 
 * @param sdata (realptr): A shared array in which all of the aceleration values of this block are stored
 * @param x (int): the x position in the block
 * @param y (int): the y position in the block
 */
{   
    if (blockSize >= 64) 
    {
        sdata[baseIndex0] += sdata[baseIndex0 + 32];
        sdata[baseIndex1] += sdata[baseIndex1 + 32];
        sdata[baseIndex2] += sdata[baseIndex2 + 32];
    }

    if (blockSize >= 32) 
    {
        sdata[baseIndex0] += sdata[baseIndex0 + 16];
        sdata[baseIndex1] += sdata[baseIndex1 + 16];
        sdata[baseIndex2] += sdata[baseIndex2 + 16];
    }

    if (blockSize >= 16)
    {
        sdata[baseIndex0] += sdata[baseIndex0 + 8];
        sdata[baseIndex1] += sdata[baseIndex1 + 8];
        sdata[baseIndex2] += sdata[baseIndex2 + 8];
    }   

    if (blockSize >= 8) 
    {
        sdata[baseIndex0] += sdata[baseIndex0 + 4];
        sdata[baseIndex1] += sdata[baseIndex1 + 4];
        sdata[baseIndex2] += sdata[baseIndex2 + 4];
    }

    if (blockSize >= 4) 
    {
        sdata[baseIndex0] += sdata[baseIndex0 + 2];
        sdata[baseIndex1] += sdata[baseIndex1 + 2];
        sdata[baseIndex2] += sdata[baseIndex2 + 2];
    }
    if (blockSize >= 2) 
    {
        sdata[baseIndex0] += sdata[baseIndex0 + 1];
        sdata[baseIndex1] += sdata[baseIndex1 + 1];
        sdata[baseIndex2] += sdata[baseIndex2 + 1];
    }
}

__global__ void finish_block_reduce (realptr acceleration, realptr d_block_holder, int n, unsigned int number_of_blocks_j)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) 
    {
        // Use registers to store intermediate sums
        real sum_x = 0.0;
        real sum_y = 0.0;
        real sum_z = 0.0;

        // Precompute base index part to avoid redundant calculation
        int baseIndex0 = number_of_blocks_j * i;  
        int baseIndex1 = n * number_of_blocks_j + number_of_blocks_j * i;  
        int baseIndex2 = n * number_of_blocks_j * 2 + number_of_blocks_j * i;  

        // Perform the reduction in the loop
        // Unrolling if number_of_blocks_j is small, this case for 4
        for (int j = 0; j < number_of_blocks_j; j += 4) 
        {
            sum_x += d_block_holder[baseIndex0 + j];
            sum_y += d_block_holder[baseIndex1 + j];
            sum_z += d_block_holder[baseIndex2 + j];

            //All threads should enter or not enter this loop
            if (j + 1 < number_of_blocks_j) 
            {
                sum_x += d_block_holder[baseIndex0 + j + 1];
                sum_y += d_block_holder[baseIndex1 + j + 1];
                sum_z += d_block_holder[baseIndex2 + j + 1];
            }
            if (j + 2 < number_of_blocks_j) 
            {
                sum_x += d_block_holder[baseIndex0 + j + 2];
                sum_y += d_block_holder[baseIndex1 + j + 2];
                sum_z += d_block_holder[baseIndex2 + j + 2];
            }
            if (j + 3 < number_of_blocks_j) 
            {
                sum_x += d_block_holder[baseIndex0 + j + 3];
                sum_y += d_block_holder[baseIndex1 + j + 3];
                sum_z += d_block_holder[baseIndex2 + j + 3];
            }
        }


        // After the loop, store the results to global memory
        acceleration[i] += sum_x;
        acceleration[i + n] += sum_y;
        acceleration[i + 2 * n] += sum_z;
    }
}

