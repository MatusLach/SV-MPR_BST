/* BST.cu

GEOSTATISTICAL APPLICATION "v2_1" uses checkerboard spin-flip Metropolis simulation
of a two-dimensional ferromagnetic XY model with modified hamiltonian
H = -J sum_{ij} cos( Qfactor*(theta_i - theta_j) )
on graphics processing units (GPUs) using the NVIDIA CUDA framework.

Implements spatially variable MPR using double checkerboard decomposition (switchable to single chcekerboard) and block-specific temperatures. 
*/

#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL
#endif

#include <iostream>
#include <fstream>
//#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstdio>
#define _USE_MATH_DEFINES	// for pi constant
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <vector>
//#include <list>
#include <iterator>
#include <algorithm> 		// for sort operation
#include <limits>			// has own min(), max() ???? which behave correctly with NAN (ignoing NAN values)
#include <windows.h>
#include <random>

#include <cub/cub.cuh>

#undef min
#undef max

#include <chrono>	// high precision execution time measurment

/*#include <thread>*/
//#include <boost/timer/timer.hpp>

//using namespace std;

#define DIM 2

#define L 256    // minumum of L = 2*BLOCKL

#define Qfactor 0.5f	
#define BLOCKL 32
#define GRIDL (L/BLOCKL)
#define BLOCKS ((GRIDL*GRIDL)/2)
#define THREADS ((BLOCKL*BLOCKL)/2)
#define N (L*L)
#define Nbond (2*L*(L - 1))
//#define TOTTHREADS (BLOCKS*THREADS)

#define DOUBLE_CHECKERBOARD
#ifdef DOUBLE_CHECKERBOARD

#define DC_EQ
#ifdef DC_EQ
#define SWEEPS_LOCAL_EQ 1
#define SWEEPS_GLOBAL_EQ 0
#endif // DC_EQ

#define DC_SIM
#ifdef DC_SIM
#define SWEEPS_LOCAL_SIM 1
#define SWEEPS_GLOBAL_SIM 0
#endif // DC_SIM


#define SWEEPS_COMPLETE 100
#endif // DOUBLE_CHECKERBOARD


#ifndef DOUBLE_CHECKERBOARD
#define SWEEPS_LOCAL_EQ 1
#define SWEEPS_LOCAL_SIM 1
#define SWEEPS_GLOBAL 100
#endif

#ifndef DC_SIM
#define SWEEPS_GLOBAL 50
#endif // !DC_SIM

#define SWEEPS_EMPTY 1
#define CONFIG_SAMPLES 100		// M = 100


#define ACC_RATE_MIN_EQ 0.30		// A_targ = 0.3
#define ACC_RATE_MIN_SIM 0.30		


#define ACC_TEST_FREQUENCY_EQ 10	//ACC_TEST_FREQUENCY_EQ 
#define ACC_TEST_FREQUENCY_SIM 10	//ACC_TEST_FREQUENCY_SIM
#define EQUI_TEST_FREQUENCY 5   // n_f = 5
#define EQUI_TEST_SAMPLES 20    // n_fit 



#define SWEEPS_EQUI_MAX 300		// upper limit for equilibration hybrid sweeps; probably not necessary
#define SLOPE_RESTR_FACTOR 3.0	// k_a = 3; for a = 1 + i/k_a (SLOPE_RESTR = k_a)

#define RemovedDataRatio 0.80f


//#define SOURCE_DATA_PATH "zo_L2048_ka02_nu05.bin"
#define SOURCE_DATA_PATH "walker_lake.bin"
//#define SOURCE_DATA_PATH "wall_3_L2048.bin"
#define SOURCE_DATA_NAME "walker"

/*#define RNG_SEED_DILUTION 1564564ULL
#define RNG_SEED_FILL 25756655ULL
#define RNG_SEED_SIMULATION 3456454624ULL*/
#define RNG_SEED_DILUTION 842301111UL
#define RNG_SEED_FILL 5451UL
#define RNG_SEED_SIMULATION 3645445443UL

//#define DOUBLE_PRECISION
#ifndef DOUBLE_PRECISION
#define INTRINSIC_FLOAT
#endif

#define OVER_RELAXATION_EQ
//#define OVER_RELAXATION_SIM

//comment these for time measurements
#define ENERGIES_PRINT
//#define CONFIGURATION_PRINT
//#define RECONSTRUCTION_PRINT
//#define ERROR_PRINT			
//#define RANDOM_INIT
//#define DIL_ENERGIES_PRINT	// not working yet
//#define SOURCE_MAPPING

//#define COLD_START			// not working yet

// other macros
// linear congruential generator
#define AA 1664525
#define CC 1013904223
#define RAN(n) (n = AA*n + CC)
#define MULT 2.328306437080797e-10f
/*
#define MULT2 4.6566128752457969e-10f
*/
#define sS(x,y) sS[(y+1)*(BLOCKL+2)+x+1]

typedef double source_t;
#ifdef DOUBLE_PRECISION
typedef double spin_t;
typedef double energy_t;
#else
typedef float spin_t;
typedef float energy_t;
#endif

// GPU processing partition
const dim3 gridLinearLattice((int)ceil(N / 256.0));
const dim3 gridLinearLatticeHalf((int)ceil(N / 2.0 / 256.0));
const dim3 blockLinearLattice(256);

// For double checkerboard
const dim3 grid(GRIDL, GRIDL / 2);
const dim3 block(BLOCKL, BLOCKL / 2);
// For single checkerboard
const dim3 grid_check(GRIDL, GRIDL);
const dim3 block_check(BLOCKL, BLOCKL / 2);

const dim3 gridAcc((int)ceil(BLOCKS * 2 / 128.0));
const dim3 blockAcc(128);

const dim3 gridEn(GRIDL, GRIDL);
const dim3 blockEn(BLOCKL, BLOCKL);


// CUDA error checking macro
#define CUDAErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s ; %s ; line %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// cuRAND error checking macro
#define cuRAND_ErrChk(err) { if (err != CURAND_STATUS_SUCCESS) std::cout << curandGetErrorString(err) << "\n"; }

// cuRAND errors
char* curandGetErrorString(curandStatus_t);
const char* curanderr[14] = {
    "No errors", "Header file and linked library version do not match",
    "Generator not initialized", "Memory allocation failed",
    "Generator is wrong type", "Argument out of range",
    "Length requested is not a multiple of dimension",
    "GPU does not have double precision required by MRG32k3a",
    "Kernel launch failure", "Pre-existing failure on library entry",
    "Initialization of CUDA failed", "Architecture mismatch, GPU does not support requested feature",
    "Internal library error", "Unknown error"
};



// function declarations
//double checkerboard
#ifdef DOUBLE_CHECKERBOARD
//specific temperatures
__global__ void metro_conditioned_equil_sublattice_shared_k(spin_t*, spin_t*, float*, unsigned int, energy_t*, energy_t*, energy_t*, unsigned int*);
__global__ void metro_conditioned_sublattice_shared_k(spin_t*, spin_t*, float*, unsigned int, energy_t*, energy_t*, energy_t*, unsigned int*);
__global__ void setInitialAlphas(spin_t*, spin_t*, spin_t*);
__global__ void setAlphas(energy_t*, spin_t*, int, energy_t, unsigned int*, spin_t*, spin_t*);
__global__ void correctTemps(energy_t*, energy_t, unsigned int*);
__global__ void resetAccD_k(energy_t*, unsigned int*);
#endif // DOUBLE_CHECKERBOARD

//single checkerboard
__global__ void metro_conditioned_equil_sublattice_k(spin_t*, spin_t*, float*, unsigned int, energy_t, energy_t*, float);
__global__ void metro_conditioned_sublattice_k(spin_t*, spin_t*, float*, unsigned int, energy_t, energy_t*, float);

__global__ void spin_mult(spin_t*, spin_t);
__global__ void over_relaxation_k(spin_t*, spin_t*, int);
__global__ void energyCalc_k(spin_t*, energy_t*);
__global__ void energyCalcDiluted_k(spin_t*, energy_t*);
__global__ void energyCalcDiluted_per_block(energy_t*, unsigned int*, unsigned int);
__global__ void min_max_avg_block(spin_t*, spin_t*, spin_t*, spin_t*);
__global__ void resetAccD_k(energy_t*);
__global__ void min_max_k(source_t*, source_t*, source_t*, bool, spin_t*);
__global__ void XY_mapping_k(source_t*, spin_t*, source_t, source_t, bool, spin_t*);
__global__ void create_dilution_mask_k(spin_t*, float*, unsigned int*);
__global__ void fill_lattice_nans_averaged(spin_t*, spin_t*);
__global__ void fill_lattice_nans_averaged_global(spin_t*, spin_t);
__global__ void fill_lattice_nans_random(spin_t*, float*);
__global__ void data_reconstruction_k(source_t*, spin_t*, source_t, source_t, source_t*, source_t*);
__global__ void mean_stdDev_reconstructed_k(source_t*, source_t*, unsigned int);
__global__ void sum_prediction_errors_k(source_t*, source_t*, spin_t*, source_t*, source_t*, source_t*, source_t*, source_t*, source_t*);
__global__ void sum_prediction_errors_k(source_t*, source_t*, spin_t*, source_t*, source_t*, source_t*, source_t*);
__global__ void bondCount_k(spin_t*, unsigned int*, unsigned int *);
__global__ void find_temperature_gpu(energy_t*, double*, double*, energy_t*, int, int);
energy_t cpu_energy(spin_t*);
double find_temperature(energy_t, std::vector<double>, std::vector<double>);

template <class T> T sumPartialSums(T *, int);
template <class T> std::vector<T> findMinMax(T *, T *, int);
template <class T> T find_median(T *, int);
template <int BLOCK_THREADS, int ITEMS_PER_THREAD> __global__ void BlockSortKernel(energy_t *d_in, energy_t *d_out);


int main()
{
#ifdef DOUBLE_CHECKERBOARD
    std::cout << "---Double checkerboard algorithm with block-specific temperatures---\n"
#ifdef DC_SIM
        << "Double checkerboard used in SIMULATION with " << SWEEPS_LOCAL_SIM << " local sweeps " << "and " << SWEEPS_GLOBAL_SIM << " global sweeps" << "\n"
#endif
#ifdef DC_EQ
        << "Double checkerboard used in EQUILIBRATION with " << SWEEPS_LOCAL_EQ << " local sweeps " << "and " << SWEEPS_GLOBAL_EQ << " global sweeps" << "\n"
#   endif
#else
    std::cout << "---Standard checkerboard algorithm---\n"
#endif

        << "\nRECONSTRUCTION SIMULATION CONFIGURATION:\n"
        << "L = " << L << ",\tQfactor = " << Qfactor << "\n"
        << "BLOCKL = " << BLOCKL << "\n"
        << "Missing data = " << RemovedDataRatio * 100 << "%\n"
        << "Equilibration samples for convergence testing = " << EQUI_TEST_SAMPLES << "\n"
#ifdef DC_SIM
        << "Reconstruction samples = " << (SWEEPS_GLOBAL_SIM + 1) * SWEEPS_COMPLETE << "\n"
#else
        << "Reconstruction samples = " << SWEEPS_GLOBAL << "\n"
#endif
        << "Configuration samples = " << CONFIG_SAMPLES << "\n" << "Active macros: ";
#ifdef DOUBLE_PRECISION
    std::cout << " DOUBLE_PRECISION,";
#else
    std::cout << " SINGLE_PRECISION,";
#ifdef INTRINSIC_FLOAT
    std::cout << " INTRINSIC_FLOAT,";
#endif
#endif
#ifdef ENERGIES_PRINT
    std::cout << " ENERGIES_PRINT,";
#endif
#ifdef CONFIGURATION_PRINT
    std::cout << " CONFIGURATION_PRINT,";
#endif
#ifdef RECONSTRUCTION_PRINT
    std::cout << " RECONSTRUCTION_PRINT,";
#endif
#ifdef ERROR_PRINT
    std::cout << " ERROR_PRINT,";
#endif
#ifdef OVER_RELAXATION_EQ
    std::cout << " OVER_RELAXATION_EQ,";
#endif
#ifdef OVER_RELAXATION_SIM
    std::cout << " OVER_RELAXATION_SIM,";
#endif
#ifdef SOURCE_MAPPING
    std::cout << " SOURCE_MAPPING,";
#endif
#ifdef RANDOM_INIT
    std::cout << " RANDOM_INIT,";
#endif

    std::cout << "\n";

    // time measurement - entire process
    std::chrono::high_resolution_clock::time_point t_sim_begin = std::chrono::high_resolution_clock::now();

    /* time measurement - relevant parts for geostatistical calulation
    (loading reference E = E(T), loading source, mapping to XY model, equilibration and reconstruction sample collection)
    */
    std::chrono::high_resolution_clock::time_point t_geo_begin;
    std::chrono::high_resolution_clock::time_point t_geo_end;

    t_geo_begin = std::chrono::high_resolution_clock::now();

    //std::cout << "------ LOADING REFERENCES AND SOURCE DATA ------\n";

    // read reference energies and temperatures
    char *buffer;
    const int ref_size = 1100;
    std::vector<double> T_ref;
    std::ifstream fileT("./reference/reference_T.bin", std::ios::in | std::ios::binary);
    buffer = (char*)malloc(ref_size * sizeof(double));
    fileT.read(buffer, ref_size * sizeof(double));
    T_ref.assign(reinterpret_cast<double*>(buffer), reinterpret_cast<double*>(buffer) + ref_size);
    fileT.close();

    std::vector<double> E_ref;
    std::ifstream fileE("./reference/reference_E.bin", std::ios::in | std::ios::binary);
    fileE.read(buffer, ref_size * sizeof(double));
    E_ref.assign(reinterpret_cast<double*>(buffer), reinterpret_cast<double*>(buffer) + ref_size);
    fileE.close();

    free(buffer);


#ifdef DOUBLE_CHECKERBOARD
    //allocate memory and copy reference energies and temperatures to the GPU
    double *T_ref_d, *E_ref_d;
    CUDAErrChk(cudaMalloc((void**)&T_ref_d, ref_size * sizeof(double)));
    CUDAErrChk(cudaMalloc((void**)&E_ref_d, ref_size * sizeof(double)));
    CUDAErrChk(cudaMemcpy(T_ref_d, T_ref.data(), ref_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDAErrChk(cudaMemcpy(E_ref_d, E_ref.data(), ref_size * sizeof(double), cudaMemcpyHostToDevice));
#endif // DOUBLE_CHECKERBOARD


    /*
    std::cout << "Number of temperature points: " << T_ref.size() << "\n";
    std::cout << "Temperatures:\n";
    for (auto it = T_ref.begin(); it != T_ref.end(); ++it)
    std::cout << *it << " ";
    std::cout << "\n";
    std::cout << "Energies:\n";
    for (auto it = E_ref.begin(); it != E_ref.end(); ++it)
    std::cout << *it << " ";
    std::cout << "\n";
    */



    // read data source
#ifdef SOURCE_DATA_PATH
    std::cout << "Source data: " << SOURCE_DATA_PATH << "\n";
    std::ifstream fileSource(SOURCE_DATA_PATH, std::ios::in | std::ios::binary);
    std::vector<source_t> complete_source;
    buffer = (char*)malloc(N * sizeof(source_t));
    fileSource.read(buffer, N * sizeof(source_t));
    complete_source.assign(reinterpret_cast<source_t*>(buffer), reinterpret_cast<source_t*>(buffer) + N);
    fileSource.close();
    free(buffer);
#else
    std::cout << "Source data path not specified!";
    return 0;
#endif
    std::cout << "Source size: " << complete_source.size() << "\n";


    //cudaSetDevice(0);


    // allocate GPU memory for source data, mapped data (XY model) and dilution mask (array of ones and NANs) & other variables
    source_t *source_d, *reconstructed_d, *mean_recons_d, *stdDev_recons_d, *AAE_d, *ARE_d, *AARE_d, *RASE_d;
    spin_t *XY_mapped_d, *dilution_mask_d;
    energy_t *E_d;

#ifdef ERROR_PRINT
    source_t *error_map_d, *error_map_block_d;
    CUDAErrChk(cudaMalloc((void**)&error_map_d, N * sizeof(source_t)));
    CUDAErrChk(cudaMemset(error_map_d, 0.0, N * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&error_map_block_d, GRIDL * GRIDL * sizeof(source_t)));
    CUDAErrChk(cudaMemset(error_map_block_d, 0.0, GRIDL * GRIDL * sizeof(source_t)));
#endif

#ifdef DOUBLE_CHECKERBOARD
    spin_t *alphas_per_block_d, *block_min_d, *block_max_d, *avg_per_block_d;
    energy_t *T_diluted_per_block_d;
    CUDAErrChk(cudaMalloc((void **)&T_diluted_per_block_d, GRIDL * GRIDL * sizeof(energy_t)));
    CUDAErrChk(cudaMalloc((void **)&alphas_per_block_d, GRIDL * GRIDL * sizeof(spin_t)));
    CUDAErrChk(cudaMalloc((void **)&block_min_d, GRIDL * GRIDL * sizeof(spin_t)));
    CUDAErrChk(cudaMalloc((void **)&block_max_d, GRIDL * GRIDL * sizeof(spin_t)));
    CUDAErrChk(cudaMalloc((void **)&avg_per_block_d, GRIDL * GRIDL * sizeof(spin_t)));
#endif // DOUBLE_CHECKERBOARD


    energy_t *AccD;
    unsigned int* tryD;

    CUDAErrChk(cudaMalloc((void**)&source_d, N * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&reconstructed_d, N * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&XY_mapped_d, N * sizeof(spin_t)));
    CUDAErrChk(cudaMalloc((void**)&dilution_mask_d, N * sizeof(spin_t)));

    CUDAErrChk(cudaMalloc((void**)&mean_recons_d, N * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&stdDev_recons_d, N * sizeof(source_t)));

    CUDAErrChk(cudaMalloc((void**)&AAE_d, (int)ceil(N / 256.0) * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&ARE_d, (int)ceil(N / 256.0) * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&AARE_d, (int)ceil(N / 256.0) * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&RASE_d, (int)ceil(N / 256.0) * sizeof(source_t)));

    CUDAErrChk(cudaMalloc((void **)&E_d, GRIDL * GRIDL * sizeof(energy_t)));

    CUDAErrChk(cudaMalloc((void**)&AccD, GRIDL * GRIDL * sizeof(energy_t)));
    CUDAErrChk(cudaMalloc((void**)&tryD, GRIDL * GRIDL * sizeof(unsigned int)));

    // for calculating maximum and minimum of data
    source_t *min_d, *max_d;
    CUDAErrChk(cudaMalloc((void**)&min_d, (int)ceil(N / 2.0 / 256.0) * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&max_d, (int)ceil(N / 2.0 / 256.0) * sizeof(source_t)));

    std::vector<source_t> min_max;

    // copy source data to GPU memory
    CUDAErrChk(cudaMemcpy(source_d, complete_source.data(), N * sizeof(source_t), cudaMemcpyHostToDevice));


#ifdef SOURCE_MAPPING
    // ----- MAPPING PROCESS -----
    std::cout << "------ SOURCE MAPPING PROCESS ------\n";

    min_max_k << < gridLinearLatticeHalf, blockLinearLattice >> > (source_d, min_d, max_d, false, dilution_mask_d);
    CUDAErrChk(cudaPeekAtLastError());

    min_max = findMinMax(min_d, max_d, (int)ceil(N / 2.0 / 256.0));

    std::cout.precision(17);
    std::cout << "from GPU:  min(source) = " << min_max.at(0)
        << " ; max(source) = " << min_max.at(1) << "\n";
    std::cout.precision(6);

    // mapping to XY model based on max and min
    XY_mapping_k << < gridLinearLattice, blockLinearLattice >> > (source_d, XY_mapped_d, min_max.at(0), min_max.at(1), false, dilution_mask_d);
    CUDAErrChk(cudaPeekAtLastError());

    // calculate energy
    energyCalc_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
    CUDAErrChk(cudaPeekAtLastError());
    energy_t E_source = sumPartialSums(E_d, (int)GRIDL * GRIDL) / Nbond;

    // assign temperature
    energy_t T_source = find_temperature(E_source, T_ref, E_ref);
    std::cout << "Source energy per bond: " << E_source << "\n";
    std::cout << "Source temperature: " << T_source << "\n";
#endif

    // print energies
#ifdef ENERGIES_PRINT
    // energies file name + create
    char fileGpuEn[100];
    char fileGpuEnEQ[100];

#ifdef DOUBLE_PRECISION
    sprintf(fileGpuEn, "./data/gpuEn_DP_removed%0.3f_Q%0.3f_L%d_ConfSamples%d_SwGlob%d.dat",
        RemovedDataRatio, (double)Qfactor, L, CONFIG_SAMPLES, SWEEPS_GLOBAL);
    sprintf(fileGpuEnEQ, "./data/gpuEnEQ_DP_removed%0.3f_Q%0.3f_L%d_ConfSamples%d_SwGlob%d.dat",
        RemovedDataRatio, (double)Qfactor, L, CONFIG_SAMPLES, SWEEPS_GLOBAL);
#else
#ifdef INTRINSIC_FLOAT
    sprintf(fileGpuEn, "./data/Energy_SIM_DC_p%0.2f_M%d_%s.dat",
        RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
    sprintf(fileGpuEnEQ, "./data/Energy_EQ_DC_p%0.2f_M%d_%s.dat",
        RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
#else
    sprintf(fileGpuEn, "./data/gpuEn_SP_removed%0.3f_Q%0.2f_L%d_ConfSamples%d_SwGlob%d.dat",
        RemovedDataRatio, (double)Qfactor, L, CONFIG_SAMPLES, SWEEPS_GLOBAL);
    sprintf(fileGpuEnEQ, "./data/gpuEnEQ_removed%0.3f_SP_Q%0.2f_L%d_ConfSamples%d_SwGlob%d.dat",
        RemovedDataRatio, (double)Qfactor, L, CONFIG_SAMPLES, SWEEPS_GLOBAL);
#endif
#endif

    FILE *fp = fopen(fileGpuEn, "wb");
    FILE *fpEQ = fopen(fileGpuEnEQ, "wb");
#endif

    // store output data
#ifdef RECONSTRUCTION_PRINT
    char fileMean[100];
    //char fileStdDev[100];
#ifdef DOUBLE_CHECKERBOARD

#if !defined DC_EQ && defined DC_SIM
    sprintf(fileMean, "./data/Recons_p%0.2f_L%d_M%d_SwGlob_SIM%d_SwGlobEQ%d_SwLocSIM%d_SwLocEQ%d_Sw_Comp%d.dat",
        RemovedDataRatio, L, CONFIG_SAMPLES, SWEEPS_GLOBAL_SIM, 1, SWEEPS_LOCAL_SIM, 0, SWEEPS_COMPLETE);

#elif defined DC_EQ && !defined DC_SIM
    sprintf(fileMean, "./data/Recons_p%0.2f_L%d_M%d_SwGlob_SIM%d_SwGlobEQ%d_SwLocSIM%d_SwLocEQ%d_Sw_Comp%d.dat",
        RemovedDataRatio, L, CONFIG_SAMPLES, 1, SWEEPS_GLOBAL_EQ, 0, SWEEPS_LOCAL_EQ, SWEEPS_COMPLETE);

#else
    sprintf(fileMean, "./data/Recons_DC_p%0.2f_M%d_%s.dat",
        RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
    //sprintf(fileStdDev, "./data/stdDev_recons_DP_removed%0.2f_Q%0.1f_L%d_ConfSamples%d_SwGlob_SIM%d_SwGlob_EQ%d_SwLoc%d_Sw_Comp%d.dat",
    //RemovedDataRatio, (double)Qfactor, L, CONFIG_SAMPLES, SWEEPS_GLOBAL_SIM, SWEEPS_GLOBAL_EQ, SWEEPS_LOCAL_SIM, SWEEPS_COMPLETE);
#endif
#else
    sprintf(fileMean, "./data/mean_recons_DP_removed%0.3f_Q%0.3f_L%d_ConfSamples%d_SwGlob%d.dat",
        RemovedDataRatio, (double)Qfactor, L, CONFIG_SAMPLES, SWEEPS_GLOBAL);
    //sprintf(fileStdDev, "./data/stdDev_recons_DP_removed%0.3f_Q%0.3f_L%d_ConfSamples%d_SwGlob%d.dat",
    // RemovedDataRatio, (double)Qfactor, L, CONFIG_SAMPLES, SWEEPS_GLOBAL);
#endif DOUBLE_CHECKERBOARD

    FILE *fpMean = fopen(fileMean, "wb");
    //FILE *fpStdDev = fopen(fileStdDev, "wb");
#endif

#ifdef CONFIGURATION_PRINT
    // print diluted data into file
    spin_t *mask;
    mask = (spin_t*)malloc(N * sizeof(spin_t));
    CUDAErrChk(cudaMemcpy(mask, dilution_mask_d, N * sizeof(spin_t), cudaMemcpyDeviceToHost));
    char strConf[100];
    sprintf(strConf, "./data/conf_removed%1.3f_Q%0.3f_L%d_ConfSamples%d_SwGlob%d.dat",
        RemovedDataRatio, (double)Qfactor, L, CONFIG_SAMPLES, SWEEPS_GLOBAL);

    FILE *f_conf = fopen(strConf, "wb");
#endif

#ifdef ERROR_PRINT
    char fileError[100];
    char fileErrorBlock[100];
    sprintf(fileError, "./data/Error_DC_p%0.2f_M%d_%s.dat",
        RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
    sprintf(fileErrorBlock, "./data/Error_DC_Block_p%0.2f_M%d_%s.dat",
        RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
    FILE *fpError = fopen(fileError, "wb");
    FILE *fpErrorBlock = fopen(fileErrorBlock, "wb");
#endif

    // SEEDS
    unsigned long long seed_dilution;
    unsigned long long seed_fill;
    unsigned long long seed_simulation;

    // calculation of configurational means
    source_t MAAE = 0.0, MARE = 0.0, MAARE = 0.0, MRASE = 0.0,
        M_timeEQ = 0.0, M_timeSamples = 0.0;
    int sum_eqSw = 0;

    t_geo_end = std::chrono::high_resolution_clock::now();
    long long duration_initial = std::chrono::duration_cast<std::chrono::microseconds>(t_geo_end - t_geo_begin).count();
    long long duration_mapping_EQ_sampling = 0;

    /*
    --------------------------------------------------------------
    --------------- LOOP FOR CONFIGURATION SAMPLES ---------------
    --------------------------------------------------------------
    */
    for (int n = 0; n < CONFIG_SAMPLES; ++n)
    {
        // ----- GPU DILUTION ------
        //std::cout << "------ GPU DILUTION ------\n";
        // creating RN generator for dilution
        float *devRand_dil;
        unsigned int *remSum_d;
        CUDAErrChk(cudaMalloc((void **)&devRand_dil, N * sizeof(float)));
        CUDAErrChk(cudaMalloc((void **)&remSum_d, (int)ceil(N / 256.0) * sizeof(unsigned int)));

        curandGenerator_t RNgen_dil;
        curandStatus_t err; // curand errors
        err = curandCreateGenerator(&RNgen_dil, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        cuRAND_ErrChk(err);

        // setting seed
        seed_dilution = (n == 0) ?
#ifdef RNG_SEED_DILUTION 
            RNG_SEED_DILUTION
#else
            time(NULL)
#endif
            : RAN(seed_dilution);

        err = curandSetPseudoRandomGeneratorSeed(RNgen_dil, seed_dilution);
        cuRAND_ErrChk(err);
        // generate random floats on device - for every spin in the lattice and for every local sweep
        err = curandGenerateUniform(RNgen_dil, devRand_dil, N);
        cuRAND_ErrChk(err);


        create_dilution_mask_k << < gridLinearLattice, blockLinearLattice >> > (dilution_mask_d, devRand_dil, remSum_d);
        CUDAErrChk(cudaPeekAtLastError());
        int removedTotal = sumPartialSums(remSum_d, (int)ceil(N / 256.0));

        // std::cout << "Removed = " << removedTotal << " , Removed data ratio = " << removedTotal / (double)N << "\n";

        // RNG cease activity here
        curandDestroyGenerator(RNgen_dil);
        CUDAErrChk(cudaFree(devRand_dil));
        CUDAErrChk(cudaFree(remSum_d));

        // time measurement - relevant part for geostatistical application
        t_geo_begin = std::chrono::high_resolution_clock::now();

        // calculate number of bonds in diluted system
        unsigned int *bondCount_d, *sparse_blocks_d;
        //std::vector<unsigned int> sparse_blocks_h;
        // sparse_blocks_h.resize(1);
        //sparse_blocks_h[0] = 0;
        CUDAErrChk(cudaMalloc((void **)&bondCount_d, GRIDL * GRIDL * sizeof(unsigned int)));
        //CUDAErrChk(cudaMalloc((void **)&sparse_blocks_d, sizeof(unsigned int)));
        //CUDAErrChk(cudaMemcpy(sparse_blocks_d, sparse_blocks_h.data(), sizeof(unsigned int), cudaMemcpyHostToDevice));
        bondCount_k << < gridEn, blockEn >> > (dilution_mask_d, bondCount_d, sparse_blocks_d);
        CUDAErrChk(cudaPeekAtLastError());
        //CUDAErrChk(cudaMemcpy(sparse_blocks_h.data(), sparse_blocks_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        int Nbonds_dil = sumPartialSums(bondCount_d, GRIDL * GRIDL);

        // std::cout << "Number of bonds in diluted system = " << Nbonds_dil << "\n";
        //std::cout << "Number of sparse blocks = " << sparse_blocks_h[0] << "\t";
        //CUDAErrChk(cudaFree(sparse_blocks_d));
        //in double checkerboad we still need this
#ifndef DOUBLE_CHECKERBOARD
        CUDAErrChk(cudaFree(bondCount_d));
#endif // DOUBLE_CHECKERBOARD

        // mapping diluted system to XY model
        min_max_k << < gridLinearLatticeHalf, blockLinearLattice >> > (source_d, min_d, max_d, true, dilution_mask_d);
        CUDAErrChk(cudaPeekAtLastError());

        min_max = findMinMax(min_d, max_d, (int)ceil(N / 2.0 / 256.0));

        /*
        std::cout.precision(17);
        std::cout << "from GPU:  min(diluted) = " << min_max.at(0)
        << " ; max(diluted) = " << min_max.at(1) << "\n";
        std::cout.precision(6);
        */

        // mapping to XY model based on max and min
        XY_mapping_k << < gridLinearLattice, blockLinearLattice >> > (source_d, XY_mapped_d, min_max.at(0), min_max.at(1), true, dilution_mask_d);
        CUDAErrChk(cudaPeekAtLastError());

        // calculate energy
        energyCalcDiluted_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
        CUDAErrChk(cudaPeekAtLastError());
        energy_t E_diluted = sumPartialSums(E_d, (int)GRIDL * GRIDL) / (energy_t)Nbonds_dil;
        // assign a single temperature
        energy_t T_avg = find_temperature(E_diluted, T_ref, E_ref);
        //std::cout << "Diluted - energy per bond: " << E_diluted << "\n";
        //std::cout << "Diluted - temperature: " << T_diluted << "\n";
        //calculate the energy per bond separately in each block
#ifdef DOUBLE_CHECKERBOARD
        //number of per-block results
        int size = GRIDL * GRIDL;
        int n_blocks = std::min((size + 1024 - 1) / 1024, 1024);
        energyCalcDiluted_per_block << <n_blocks, 1024 >> > (E_d, bondCount_d, size);
        //std::cout << "avg bonds per block = " << (energy_t)Nbonds_dil / size << "\n";
       

        min_max_avg_block << < gridEn, blockEn >> > (XY_mapped_d, block_min_d, block_max_d, avg_per_block_d);//
        // assign a different temperature to each block
        find_temperature_gpu << <n_blocks, 1024 >> > (E_d, T_ref_d, E_ref_d, T_diluted_per_block_d, size, ref_size);
        energy_t T_diluted;
        T_diluted = find_median(T_diluted_per_block_d, size);
        //correcting block temperatures in case there were no bond in some blocks, these blocks get the median temperature instead
        correctTemps << <n_blocks, 1024 >> > (T_diluted_per_block_d, T_diluted, bondCount_d);
        CUDAErrChk(cudaFree(bondCount_d));
        /*
        energy_t* d_sorted_temps;
        CUDAErrChk(cudaMalloc((void **)&d_sorted_temps, size * sizeof(energy_t)));
        const int num_threads = 64;
        const int items_per_thread = 1;
        BlockSortKernel<num_threads, items_per_thread> << <1, num_threads >> >(T_diluted_per_block_d, d_sorted_temps);
        energy_t value1, value2;
        cudaMemcpy(&value1, &d_sorted_temps[(size - 1) / 2], sizeof(energy_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&value2, &d_sorted_temps[size / 2], sizeof(energy_t), cudaMemcpyDeviceToHost);
        */
        //std::cout << "median temperature CPU = " << T_diluted << " median temperature GPU = " << (value1 + value2) / 2 << "\n\n";
        //std::cout << " median temperature GPU = " << (value1 + value2) / 2 << "\n";
#endif // DOUBLE_CHECKERBOARD

        //std::cout << "Diluted - temperature: " << T_avg << "\t" << "median temperature = " << T_diluted << "\n";

        if (n == 0)
        {
            /*
            energy_t* h_sorted_temps;
            cudaMemcpy(h_sorted_temps, d_sorted_temps, size * sizeof(energy_t), cudaMemcpyDeviceToHost);
            CUDAErrChk(cudaFree(d_sorted_temps));
            std::cout << "median temperature CPU = " << T_diluted << " median temperature GPU = " << (value1 + value2) / 2 << "\n\n";
            //std::cout << " median temperature GPU = " << (value1 + value2) / 2 << "\n";

            for (int i = 0; i < size; i++)
            {
            std::cout << h_sorted_temps[i] << "\t";
            }
            std::cout << "\n\n";
            */
            std::cout << "Diluted - energy per bond: " << E_diluted << "\n";
            //std::cout << "Diluted - temperature: " << T_avg << "\n";
            std::cout << "Median temperature = " << T_diluted << "\n";

            //std::cout << "Max - temperature: " << min_max_T[1] << "\n";
            //std::cout << "Min - temperature: " << min_max_T[0] << "\n";
            
            std::vector<energy_t> teploty;
            teploty.resize(GRIDL * GRIDL);
            //CUDAErrChk(cudaMemcpy(source_d, complete_source.data(), N * sizeof(source_t), cudaMemcpyHostToDevice));
            CUDAErrChk(cudaMemcpy(teploty.data(), T_diluted_per_block_d, sizeof(energy_t) * GRIDL * GRIDL, cudaMemcpyDeviceToHost));
            std::ofstream myfile;
            myfile.open("per_block_temps.txt");
            for (int i = 0; i < GRIDL * GRIDL; i++)
            {
                myfile << teploty[i] << "\n";
                //if(teploty[i] > 10.0)
                    //std::cout << "Temp in block " << i << " is " << teploty[i] << " iteracia " << n << "\n";
            }
            myfile.close();
            
        }


#ifdef CONFIGURATION_PRINT
        // print diluted data into file
        spin_t *mask;
        mask = (spin_t*)malloc(N * sizeof(spin_t));
        CUDAErrChk(cudaMemcpy(mask, dilution_mask_d, N * sizeof(spin_t), cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i)
        {
            source_t temp = complete_source.at(i) * mask[i];
            fwrite(&temp, sizeof(source_t), 1, f_conf);
        }

#endif
#ifdef RANDOM_INIT
        // ------ FILLING NAN VALUES WITH RANDOM SPINS ------
        //std::cout << "------ FILLING NAN VALUES WITH RANDOM SPINS ------\n";
        // creating RN generator for dilution
        float *devRand_fill;
        CUDAErrChk(cudaMalloc((void **)&devRand_fill, N * sizeof(float)));

        curandGenerator_t RNgen_fill;
        err = curandCreateGenerator(&RNgen_fill, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        cuRAND_ErrChk(err);

        // setting seed
        seed_fill = (n == 0) ?
#ifdef RNG_SEED_FILL 
            RNG_SEED_FILL
#else
            time(NULL)
#endif
            : RAN(seed_fill);

        err = curandSetPseudoRandomGeneratorSeed(RNgen_fill, seed_fill);
        cuRAND_ErrChk(err);
        // generate random floats on device - for every spin site
        err = curandGenerateUniform(RNgen_fill, devRand_fill, N);
        cuRAND_ErrChk(err);


        fill_lattice_nans_random << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, devRand_fill);
        // RNG cease activity here
        curandDestroyGenerator(RNgen_fill);
        CUDAErrChk(cudaFree(devRand_fill));
#else
        //spin_t global_average = sumPartialSums(avg_per_block_d, (int)GRIDL * GRIDL) / (GRIDL * GRIDL);
        //fill_lattice_nans_averaged_global << < gridEn, blockEn >> > (XY_mapped_d, global_average);
        fill_lattice_nans_averaged << < gridEn, blockEn >> > (XY_mapped_d, avg_per_block_d);
#endif 
        CUDAErrChk(cudaPeekAtLastError());
        

        // ------ CONDITIONED MC SIMULATION -----
        //std::cout << "------ GPU CONDITIONED MC SIMULATION ------\n";
        // create data arrays for thermodynamic variables
        std::vector<energy_t> EnergiesEq;
        std::vector<energy_t> Energies(SWEEPS_COMPLETE);

        // creating RN generator for equilibration and simulation
        // setting seed
        seed_simulation = (n == 0) ?
#ifdef RNG_SEED_SIMULATION 
            RNG_SEED_SIMULATION
#else
            time(NULL)
#endif
            : RAN(seed_simulation);

        // creating RN generator for equilibration and simulation
        float* devRand;
        energy_t alpha = (energy_t)2 * M_PI;
#ifdef DC_EQ 
#ifdef DC_SIM 
        CUDAErrChk(cudaMalloc((void **)&devRand, 2 * N * SWEEPS_EMPTY * max(SWEEPS_LOCAL_EQ + SWEEPS_GLOBAL_EQ, SWEEPS_LOCAL_SIM + SWEEPS_GLOBAL_SIM) * sizeof(float)));
#endif

#ifndef DC_SIM 
        CUDAErrChk(cudaMalloc((void **)&devRand, 2 * N * (SWEEPS_LOCAL_EQ + SWEEPS_GLOBAL_EQ) * sizeof(float)));
#endif
#elif defined DC_SIM
        CUDAErrChk(cudaMalloc((void **)&devRand, 2 * N * (SWEEPS_LOCAL_SIM + SWEEPS_GLOBAL_SIM) * sizeof(float)));

#else
        CUDAErrChk(cudaMalloc((void **)&devRand, 2 * N * SWEEPS_EMPTY * sizeof(float)));
#endif // DOUBLE_CHECKERBOARD

        curandGenerator_t RNgen;
        err = curandCreateGenerator(&RNgen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        cuRAND_ErrChk(err);
        err = curandSetPseudoRandomGeneratorSeed(RNgen, seed_simulation);
        cuRAND_ErrChk(err);

        // summation of reconstructed data for means and standard deviations
        std::vector<source_t> mean_reconstructed(N, 0.0);
        std::vector<source_t> stdDev_reconstructed(N, 0.0);
        CUDAErrChk(cudaMemcpy(mean_recons_d, mean_reconstructed.data(), N * sizeof(source_t), cudaMemcpyHostToDevice));
        CUDAErrChk(cudaMemcpy(stdDev_recons_d, stdDev_reconstructed.data(), N * sizeof(source_t), cudaMemcpyHostToDevice));


#ifdef ENERGIES_PRINT
        // Calculate initial energy and write it into file
        energyCalc_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
        CUDAErrChk(cudaPeekAtLastError());
        energy_t energy = sumPartialSums(E_d, (int)GRIDL * GRIDL) / Nbond;
        EnergiesEq.push_back(energy);

        fwrite(&(EnergiesEq.back()), sizeof(energy_t), 1, fpEQ);
#endif

        // event creation
        cudaEvent_t start, stop, startEq, stopEq;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&startEq);
        cudaEventCreate(&stopEq);
        float Etime;
        float EtimeEq;

        // start measurment

        cudaEventRecord(startEq, 0);

        // ------ EQUILIBRATION ------
        //std::cout << "------ EQUILIBRATION ------\n";
        // acceptance rate + adjustment of spin-perturbation interval parameter "alpha"
        //float alpha = (float)(2.0*M_PI);
        //double AccRate;
#ifdef DC_EQ
        std::vector<energy_t> AccH(GRIDL * GRIDL, 0.0);
        std::vector<unsigned int> tryH(GRIDL * GRIDL, 0);
        setInitialAlphas << < gridAcc, blockAcc >> > (alphas_per_block_d, block_min_d, block_max_d);
        //std::vector<energy_t> alphas_H(GRIDL * GRIDL, (energy_t)(2.0*M_PI));
        CUDAErrChk(cudaMemcpy(AccD, AccH.data(), GRIDL * GRIDL * sizeof(energy_t), cudaMemcpyHostToDevice));
        CUDAErrChk(cudaMemcpy(tryD, tryH.data(), GRIDL * GRIDL * sizeof(unsigned int), cudaMemcpyHostToDevice));
        //CUDAErrChk(cudaMemcpy(alphas_per_block_d, alphas_H.data(), GRIDL * GRIDL * sizeof(energy_t), cudaMemcpyHostToDevice));
#else
        std::vector<double> AccH(2 * BLOCKS, 0.0);
        CUDAErrChk(cudaMemcpy(AccD, AccH.data(), 2 * BLOCKS * sizeof(energy_t), cudaMemcpyHostToDevice));
#endif // DOUBLE_CHECKERBOARD

        // slope of simple linear regression
        energy_t Slope = -1;
        int it_EQ = 0;
        energy_t meanX = EQUI_TEST_SAMPLES / (energy_t)2.0;
        energy_t varX = 0.0;
        std::vector<energy_t> Xdiff;
        for (int i = 0; i < EQUI_TEST_SAMPLES; ++i)
        {
            Xdiff.push_back(i - meanX);
            varX += Xdiff.at(i) * Xdiff.at(i);
        }

        while ((Slope < 0) && (it_EQ <= SWEEPS_EQUI_MAX))
            //while (abs(Slope) > 1e-7)
        {
#ifdef OVER_RELAXATION_EQ
            // over-relaxation algorithm
            spin_mult << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, Qfactor);
            CUDAErrChk(cudaPeekAtLastError());
            over_relaxation_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, 0);
            CUDAErrChk(cudaPeekAtLastError());
            over_relaxation_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, 1);
            CUDAErrChk(cudaPeekAtLastError());
            spin_mult << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, 1 / Qfactor);
            CUDAErrChk(cudaPeekAtLastError());
#endif
            // restricted Metropolis update
            // generate random floats on device - for every spin in the lattice and for every local sweep
#ifdef DC_EQ
            err = curandGenerateUniform(RNgen, devRand, 2 * N * (SWEEPS_GLOBAL_EQ + SWEEPS_LOCAL_EQ));
#else
            err = curandGenerateUniform(RNgen, devRand, 2 * N * SWEEPS_EMPTY);
#endif // DOUBLE_CHECKERBOARD
            cuRAND_ErrChk(err);

#ifndef DC_EQ
            for (int j = 0; j < SWEEPS_EMPTY; ++j)
            {
                metro_conditioned_equil_sublattice_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, devRand + j*N, 0, 1.0 / T_diluted, AccD, alpha);
                CUDAErrChk(cudaPeekAtLastError());
                metro_conditioned_equil_sublattice_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, devRand + j*N, 1, 1.0 / T_diluted, AccD, alpha);
                CUDAErrChk(cudaPeekAtLastError());
            }

            // energy calculation and sample filling
            energyCalc_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
            CUDAErrChk(cudaPeekAtLastError());
            EnergiesEq.push_back(sumPartialSums(E_d, (int)GRIDL * GRIDL) / Nbond);

#ifdef ENERGIES_PRINT
            fwrite(&(EnergiesEq.back()), sizeof(energy_t), 1, fpEQ);
#endif

            // keeps the number of energy samples stable (= EQUI_TEST_SAMPLES)
            if (EnergiesEq.size() > EQUI_TEST_SAMPLES)
                EnergiesEq.erase(EnergiesEq.begin());

            ++it_EQ;	// iterator update ("it_EQ = 1" for 1st hybrid sweep)

                        // Acceptance Rate measurment and modification of "alpha" for the restriction of spin states
                        // Acceptance rate calculation
            if ((it_EQ % ACC_TEST_FREQUENCY_EQ) == 0)
            {
                AccRate = sumPartialSums(AccD, 2 * BLOCKS) / (double)(removedTotal*SWEEPS_EMPTY*ACC_TEST_FREQUENCY_EQ);
                resetAccD_k << < gridAcc, blockAcc >> > (AccD);
                CUDAErrChk(cudaPeekAtLastError());
                //std::cout << "AccRate = " <<  AccRate << "\n";
                // "alpha" update
                if (alpha > 0)
                {
                    if (AccRate < ACC_RATE_MIN_EQ)
                        alpha = (float)(2.0 * M_PI / (1 + it_EQ / (double)SLOPE_RESTR_FACTOR));
                }
            }
            // Slope update
            if ((it_EQ % EQUI_TEST_FREQUENCY) == 0)
            {
                // testing equilibration condition - claculation of linear regression slope from stored energies
                if (EnergiesEq.size() == EQUI_TEST_SAMPLES)
                {
                    energy_t sumEn = 0.0;
                    for (auto n : EnergiesEq) sumEn += n;
                    energy_t meanEn = sumEn / EQUI_TEST_SAMPLES;
                    sumEn = 0.0;
                    for (int k = 0; k < EQUI_TEST_SAMPLES; ++k)
                        sumEn += (EnergiesEq.at(k) - meanEn) * Xdiff.at(k);
                    Slope = sumEn / varX;
                }
            }
        }
#endif // SIngle_Checkerboard

#ifdef DC_EQ
        metro_conditioned_equil_sublattice_shared_k << < grid, block >> > (XY_mapped_d, dilution_mask_d, devRand, 0, T_diluted_per_block_d, AccD, alphas_per_block_d, tryD);
        CUDAErrChk(cudaPeekAtLastError());
        metro_conditioned_equil_sublattice_shared_k << < grid, block >> > (XY_mapped_d, dilution_mask_d, devRand, 1, T_diluted_per_block_d, AccD, alphas_per_block_d, tryD);
        CUDAErrChk(cudaPeekAtLastError());

        // energy calculation and sample filling
        energyCalc_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
        CUDAErrChk(cudaPeekAtLastError());
        EnergiesEq.push_back(sumPartialSums(E_d, (int)GRIDL * GRIDL) / Nbond);

#ifdef ENERGIES_PRINT
        fwrite(&(EnergiesEq.back()), sizeof(energy_t), 1, fpEQ);
#endif

        // keeps the number of energy samples stable (= EQUI_TEST_SAMPLES)
        if (EnergiesEq.size() > EQUI_TEST_SAMPLES)
            EnergiesEq.erase(EnergiesEq.begin());

        ++it_EQ;	// iterator update ("it_EQ = 1" for 1st hybrid sweep)

                    // Acceptance Rate measurment and modification of "alpha" for the restriction of spin states
                    // Acceptance rate calculation

        setAlphas << < gridAcc, blockAcc >> > (AccD, alphas_per_block_d, it_EQ, ACC_RATE_MIN_EQ, tryD, block_min_d, block_max_d);
        resetAccD_k << < gridAcc, blockAcc >> > (AccD, tryD);

        /*
        if (n == 0)
        {
            std::vector<energy_t> alphas;
            alphas.resize(GRIDL * GRIDL);
            //CUDAErrChk(cudaMemcpy(source_d, complete_source.data(), N * sizeof(source_t), cudaMemcpyHostToDevice));
            CUDAErrChk(cudaMemcpy(alphas.data(), alphas_per_block_d, sizeof(energy_t) * GRIDL * GRIDL, cudaMemcpyDeviceToHost));
            std::ofstream myfile;
            myfile.open("per_block_alphas_equi.txt");
            for (int i = 0; i < GRIDL * GRIDL; i++)
            {
                myfile << alphas[i] << "\n";
                //std::cout << "Temp in block " << i << " is " << teploty[i] << "\n";
            }
            myfile.close();
        }
        */

        // Slope update
        if ((it_EQ % EQUI_TEST_FREQUENCY) == 0)
        {
            // testing equilibration condition - claculation of linear regression slope from stored energies
            if (EnergiesEq.size() == EQUI_TEST_SAMPLES)
            {
                energy_t sumEn = 0.0;
                for (auto n : EnergiesEq) sumEn += n;
                energy_t meanEn = sumEn / EQUI_TEST_SAMPLES;
                sumEn = 0.0;
                for (int k = 0; k < EQUI_TEST_SAMPLES; ++k)
                    sumEn += (EnergiesEq.at(k) - meanEn) * Xdiff.at(k);
                Slope = sumEn / varX;
            }
        }
        for (int j = 0; j < SWEEPS_GLOBAL_EQ; ++j)
        {
            metro_conditioned_equil_sublattice_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, devRand + 2 * N*SWEEPS_LOCAL_EQ + 2 * j*N, 0, 1.0 / T_diluted, AccD, alpha);
            CUDAErrChk(cudaPeekAtLastError());
            metro_conditioned_equil_sublattice_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, devRand + 2 * N*SWEEPS_LOCAL_EQ + 2 * j*N, 1, 1.0 / T_diluted, AccD, alpha);
            CUDAErrChk(cudaPeekAtLastError());


            // energy calculation and sample filling
            energyCalc_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
            CUDAErrChk(cudaPeekAtLastError());
            EnergiesEq.push_back(sumPartialSums(E_d, (int)GRIDL * GRIDL) / Nbond);

#ifdef ENERGIES_PRINT
            fwrite(&(EnergiesEq.back()), sizeof(energy_t), 1, fpEQ);
#endif

            // keeps the number of energy samples stable (= EQUI_TEST_SAMPLES)
            if (EnergiesEq.size() > EQUI_TEST_SAMPLES)
                EnergiesEq.erase(EnergiesEq.begin());

            ++it_EQ;	// iterator update ("it_EQ = 1" for 1st hybrid sweep)

            // Slope update
            if ((it_EQ % EQUI_TEST_FREQUENCY) == 0)
            {
                // testing equilibration condition - claculation of linear regression slope from stored energies
                if (EnergiesEq.size() == EQUI_TEST_SAMPLES)
                {
                    energy_t sumEn = 0.0;
                    for (auto n : EnergiesEq) sumEn += n;
                    energy_t meanEn = sumEn / EQUI_TEST_SAMPLES;
                    sumEn = 0.0;
                    for (int k = 0; k < EQUI_TEST_SAMPLES; ++k)
                        sumEn += (EnergiesEq.at(k) - meanEn) * Xdiff.at(k);
                    Slope = sumEn / varX;
                }
            }
        }
    }
#endif // Double_Checkerboard

    // end measurment
    CUDAErrChk(cudaEventRecord(stopEq, 0));
    CUDAErrChk(cudaEventSynchronize(stopEq));
    CUDAErrChk(cudaEventElapsedTime(&EtimeEq, startEq, stopEq));

#ifdef ENERGIES_PRINT
    for (int i = it_EQ; i < SWEEPS_EQUI_MAX; i++)
    {
        int k = 0;
        fwrite(&k, sizeof(int), 1, fpEQ);
    }
#endif
    // start measurment

    cudaEventRecord(start, 0);

    // ------ GENERATING SAMPLES ------
    //single checkerboard version
#ifndef DC_SIM
    for (int i = 0; i < SWEEPS_GLOBAL; ++i)
    {

#ifdef OVER_RELAXATION_SIM
        // over-relaxation algorithm
        spin_mult << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, Qfactor);
        CUDAErrChk(cudaPeekAtLastError());
        over_relaxation_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, 0);
        CUDAErrChk(cudaPeekAtLastError());
        over_relaxation_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, 1);
        CUDAErrChk(cudaPeekAtLastError());
        spin_mult << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, 1 / Qfactor);
        CUDAErrChk(cudaPeekAtLastError());
#endif
        // generate random floats on device - for every spin in the lattice and for every empty sweep
        err = curandGenerateUniform(RNgen, devRand, 2 * N * SWEEPS_EMPTY);
        cuRAND_ErrChk(err);

        for (int j = 0; j < SWEEPS_EMPTY; ++j)
        {
            metro_conditioned_sublattice_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, devRand + j*N, 0, 1.0 / T_diluted, AccD, alpha);
            CUDAErrChk(cudaPeekAtLastError());
            metro_conditioned_sublattice_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, devRand + j*N, 1, 1.0 / T_diluted, AccD, alpha);
            CUDAErrChk(cudaPeekAtLastError());
        }
#ifdef ENERGIES_PRINT
        // energy calculation
        energyCalc_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
        CUDAErrChk(cudaPeekAtLastError());
        Energies.at(i) = sumPartialSums(E_d, (int)GRIDL * GRIDL) / Nbond;
        fwrite(&(Energies.at(i)), sizeof(energy_t), 1, fp);
#endif

        // data reconstruction + summation for mean and standard deviation
        data_reconstruction_k << < gridLinearLattice, blockLinearLattice >> > (reconstructed_d, XY_mapped_d, min_max.at(0), min_max.at(1), mean_recons_d, stdDev_recons_d);
        CUDAErrChk(cudaPeekAtLastError());
    }
#endif // !DC_SIM
    int it_SIM = it_EQ;
    //int it_SIM = 0;
    //alpha = (float)(2.0*M_PI);
    //Double checkerboard with ability to mix local and global sweeps
#ifdef DC_SIM
    for (int i = 0; i < SWEEPS_COMPLETE; ++i)
    {
        //std::cout << "alpha_sim = " << alpha << "\n";
#ifdef OVER_RELAXATION_SIM
        // over-relaxation algorithm
        spin_mult << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, Qfactor);
        CUDAErrChk(cudaPeekAtLastError());
        over_relaxation_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, 0);
        CUDAErrChk(cudaPeekAtLastError());
        over_relaxation_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, 1);
        CUDAErrChk(cudaPeekAtLastError());
        spin_mult << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, 1 / Qfactor);
        CUDAErrChk(cudaPeekAtLastError());
#endif
        // generate random floats on device - for every spin in the lattice, for every empty sweep and for every local sweep
        err = curandGenerateUniform(RNgen, devRand, 2 * N * (SWEEPS_GLOBAL_SIM + SWEEPS_LOCAL_SIM));
        cuRAND_ErrChk(err);
        //alpha for restricted metropolis
        //alpha = 0.01f;
        metro_conditioned_sublattice_shared_k << < grid, block >> > (XY_mapped_d, dilution_mask_d, devRand, 0, T_diluted_per_block_d, AccD, alphas_per_block_d, tryD);
        CUDAErrChk(cudaPeekAtLastError());
        metro_conditioned_sublattice_shared_k << < grid, block >> > (XY_mapped_d, dilution_mask_d, devRand, 1, T_diluted_per_block_d, AccD, alphas_per_block_d, tryD);
        CUDAErrChk(cudaPeekAtLastError());
#ifdef ENERGIES_PRINT
        // energy calculation
        energyCalc_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
        CUDAErrChk(cudaPeekAtLastError());
        Energies.at(i) = sumPartialSums(E_d, (int)GRIDL * GRIDL) / Nbond;
        fwrite(&(Energies.at(i)), sizeof(energy_t), 1, fp);
#endif
        // data reconstruction + summation for mean and standard deviation
        data_reconstruction_k << < gridLinearLattice, blockLinearLattice >> > (reconstructed_d, XY_mapped_d, min_max.at(0), min_max.at(1), mean_recons_d, stdDev_recons_d);
        CUDAErrChk(cudaPeekAtLastError());

        ++it_SIM;	// iterator update ("it_SIM = 1" for 1st hybrid sweep)

                    // Acceptance Rate measurment and modification of "alpha" for the restriction of spin states
        setAlphas << < gridAcc, blockAcc >> > (AccD, alphas_per_block_d, it_SIM, ACC_RATE_MIN_SIM, tryD, block_min_d, block_max_d);
        resetAccD_k << < gridAcc, blockAcc >> > (AccD, tryD);
        
        /*
        if (n == 0)
        {
            std::vector<energy_t> alphas;
            alphas.resize(GRIDL * GRIDL);
            //CUDAErrChk(cudaMemcpy(source_d, complete_source.data(), N * sizeof(source_t), cudaMemcpyHostToDevice));
            CUDAErrChk(cudaMemcpy(alphas.data(), alphas_per_block_d, sizeof(energy_t) * GRIDL * GRIDL, cudaMemcpyDeviceToHost));
            std::ofstream myfile;
            std::string nazov = "per_block_alphas_sim_k_a = " + std::to_string(SLOPE_RESTR_FACTOR) + ".txt";
            myfile.open(nazov);
            for (int i = 0; i < GRIDL * GRIDL; i++)
            {
                myfile << alphas[i] << "\n";
                //std::cout << "Temp in block " << i << " is " << teploty[i] << "\n";
            }
            myfile.close();
        }
        */

        for (int j = 0; j < SWEEPS_GLOBAL_SIM; ++j)
        {
#ifdef OVER_RELAXATION_SIM
            // over-relaxation algorithm
            spin_mult << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, Qfactor);
            CUDAErrChk(cudaPeekAtLastError());
            over_relaxation_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, 0);
            CUDAErrChk(cudaPeekAtLastError());
            over_relaxation_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, 1);
            CUDAErrChk(cudaPeekAtLastError());
            spin_mult << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, 1 / Qfactor);
            CUDAErrChk(cudaPeekAtLastError());
#endif

            metro_conditioned_sublattice_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, devRand + 2 * N*SWEEPS_LOCAL_SIM + 2 * j*N, 0, 1.0 / T_diluted, AccD, alpha);
            CUDAErrChk(cudaPeekAtLastError());
            metro_conditioned_sublattice_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, devRand + 2 * N*SWEEPS_LOCAL_SIM + 2 * j*N, 1, 1.0 / T_diluted, AccD, alpha);
            CUDAErrChk(cudaPeekAtLastError());

#ifdef ENERGIES_PRINT
            // energy calculation
            energyCalc_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
            CUDAErrChk(cudaPeekAtLastError());
            Energies.at(i) = sumPartialSums(E_d, (int)GRIDL * GRIDL) / Nbond;
            fwrite(&(Energies.at(i)), sizeof(energy_t), 1, fp);
#endif

            // data reconstruction + summation for mean and standard deviation
            data_reconstruction_k << < gridLinearLattice, blockLinearLattice >> > (reconstructed_d, XY_mapped_d, min_max.at(0), min_max.at(1), mean_recons_d, stdDev_recons_d);
            CUDAErrChk(cudaPeekAtLastError());
        }

    }
#endif
    // end measurment
    CUDAErrChk(cudaEventRecord(stop, 0));
    CUDAErrChk(cudaEventSynchronize(stop));
    CUDAErrChk(cudaEventElapsedTime(&Etime, start, stop));

    // GPU time
    M_timeEQ += EtimeEq / 1000;
    M_timeSamples += Etime / 1000;

    // prediction averages and standard deviations
#ifdef DC_SIM
    mean_stdDev_reconstructed_k << < gridLinearLattice, blockLinearLattice >> > (mean_recons_d, stdDev_recons_d, SWEEPS_COMPLETE * (1 + SWEEPS_GLOBAL_SIM));

    CUDAErrChk(cudaPeekAtLastError());
#else
    mean_stdDev_reconstructed_k << < gridLinearLattice, blockLinearLattice >> > (mean_recons_d, stdDev_recons_d, SWEEPS_GLOBAL);
    CUDAErrChk(cudaPeekAtLastError());
#endif // DOUBLE_CHECKERBOARD

    t_geo_end = std::chrono::high_resolution_clock::now();
    duration_mapping_EQ_sampling += std::chrono::duration_cast<std::chrono::microseconds>(t_geo_end - t_geo_begin).count();

#ifdef RECONSTRUCTION_PRINT
    CUDAErrChk(cudaMemcpy(mean_reconstructed.data(), mean_recons_d, N * sizeof(source_t), cudaMemcpyDeviceToHost));
    if (n == 0)
    {
        for (int k = 0; k < N; ++k)
        {
            fwrite(&(mean_reconstructed.at(k)), sizeof(source_t), 1, fpMean);
            //fwrite(&(stdDev_reconstructed.at(k)), sizeof(source_t), 1, fpStdDev);
        }
    }
#endif
    

    // does not work with blockL 8
#ifdef ERROR_PRINT
    // prediction errors
    sum_prediction_errors_k << < gridEn, blockEn >> > (source_d, mean_recons_d, dilution_mask_d, AAE_d, ARE_d, AARE_d, RASE_d, error_map_d, error_map_block_d);
    CUDAErrChk(cudaPeekAtLastError());
    MAAE += sumPartialSums(AAE_d, (int)GRIDL * GRIDL) / (source_t)removedTotal;
    MARE += sumPartialSums(ARE_d, (int)GRIDL * GRIDL) / (source_t)removedTotal;
    MAARE += sumPartialSums(AARE_d, (int)GRIDL * GRIDL) / (source_t)removedTotal;
    MRASE += sqrt(sumPartialSums(RASE_d, (int)GRIDL * GRIDL) / (source_t)removedTotal);
    
#else
    // works with BlockL 8
    // prediction errors
    sum_prediction_errors_k << < gridLinearLattice, blockLinearLattice >> > (source_d, mean_recons_d, dilution_mask_d, AAE_d, ARE_d, AARE_d, RASE_d);
    CUDAErrChk(cudaPeekAtLastError());
    MAAE += sumPartialSums(AAE_d, (int)ceil(N / 256.0)) / (source_t)removedTotal;
    MARE += sumPartialSums(ARE_d, (int)ceil(N / 256.0)) / (source_t)removedTotal;
    MAARE += sumPartialSums(AARE_d, (int)ceil(N / 256.0)) / (source_t)removedTotal;
    MRASE += sqrt(sumPartialSums(RASE_d, (int)ceil(N / 256.0)) / (source_t)removedTotal);
#endif
    // Number of equilibration sweeps
    sum_eqSw += it_EQ;

    // cudaFree after equilibration
    curandDestroyGenerator(RNgen);
    CUDAErrChk(cudaFree(devRand));

    if (n == 0) std::cout << "Seeds[configurations, filling, simulation] = " << "["
        << seed_dilution << ", " << seed_fill << ", " << seed_simulation << "]\n";

}

std::cout.precision(8);
std::cout << "Mean elapsed time (equilibration for average " << sum_eqSw / (source_t)CONFIG_SAMPLES << " sweeps) = " << M_timeEQ / CONFIG_SAMPLES << " s\n";
std::cout << "Mean elapsed time (collection of " << CONFIG_SAMPLES << " samples) = " << M_timeSamples / CONFIG_SAMPLES << " s\n";

// prediction errors
std::cout << "MAAE = " << MAAE / CONFIG_SAMPLES << "\n";
std::cout << "MARE = " << MARE * 100 / CONFIG_SAMPLES << " %\n";
std::cout << "MAARE = " << MAARE * 100 / CONFIG_SAMPLES << " %\n";
std::cout << "MRASE = " << MRASE / CONFIG_SAMPLES << "\n";

#ifdef ERROR_PRINT
std::vector<source_t> error_map_h(N);
std::vector<source_t> error_map_block_h(GRIDL * GRIDL);
CUDAErrChk(cudaMemcpy(error_map_h.data(), error_map_d, N * sizeof(source_t), cudaMemcpyDeviceToHost));
CUDAErrChk(cudaMemcpy(error_map_block_h.data(), error_map_block_d, GRIDL * GRIDL * sizeof(source_t), cudaMemcpyDeviceToHost));

for (int k = 0; k < N; ++k)
{
    error_map_h[k] = error_map_h[k] / (source_t)CONFIG_SAMPLES;
    fwrite(&(error_map_h.at(k)), sizeof(source_t), 1, fpError);
}

for (int k = 0; k < GRIDL * GRIDL; k++)
{
    error_map_block_h[k] = error_map_block_h[k] / (source_t)CONFIG_SAMPLES;
    fwrite(&(error_map_block_h.at(k)), sizeof(source_t), 1, fpErrorBlock);
}
#endif

// closing time series storage
#ifdef ENERGIES_PRINT  
fclose(fp);
fclose(fpEQ);
#endif
#ifdef RECONSTRUCTION_PRINT
fclose(fpMean);
//fclose(fpStdDev);
#endif
#ifdef CONFIGURATION_PRINT
fclose(f_conf);
#endif
#ifdef ERROR_PRINT
CUDAErrChk(cudaFree(error_map_d));
fclose(fpError);
CUDAErrChk(cudaFree(error_map_block_d));
fclose(fpErrorBlock);
#endif
// free CUDA variable
CUDAErrChk(cudaFree(source_d));
CUDAErrChk(cudaFree(XY_mapped_d));
CUDAErrChk(cudaFree(dilution_mask_d));
CUDAErrChk(cudaFree(reconstructed_d));
CUDAErrChk(cudaFree(min_d));
CUDAErrChk(cudaFree(max_d));
CUDAErrChk(cudaFree(E_d));

#ifdef DOUBLE_CHECKERBOARD
CUDAErrChk(cudaFree(T_diluted_per_block_d));
CUDAErrChk(cudaFree(alphas_per_block_d));
CUDAErrChk(cudaFree(T_ref_d));
CUDAErrChk(cudaFree(E_ref_d));
CUDAErrChk(cudaFree(tryD));
CUDAErrChk(cudaFree(block_min_d));
CUDAErrChk(cudaFree(block_max_d));
CUDAErrChk(cudaFree(avg_per_block_d));
#endif // DOUBLE_CHECKERBOARD

CUDAErrChk(cudaFree(AccD));
CUDAErrChk(cudaFree(mean_recons_d));
CUDAErrChk(cudaFree(stdDev_recons_d));
CUDAErrChk(cudaFree(AAE_d));
CUDAErrChk(cudaFree(ARE_d));
CUDAErrChk(cudaFree(AARE_d));
CUDAErrChk(cudaFree(RASE_d));

// time measurement - entire process
std::chrono::high_resolution_clock::time_point t_sim_end = std::chrono::high_resolution_clock::now();
auto tot_duration = std::chrono::duration_cast<std::chrono::microseconds>(t_sim_end - t_sim_begin).count();
std::cout << "Total duration = " << (double)tot_duration / 1e6 << " s\n";
std::cout << "Total duration per configuration sample = " << (double)tot_duration / 1e6 / CONFIG_SAMPLES << " s\n";
// time measurement - relevant part for geostatistical application
//(loading reference E = E(T), loading source, mapping to XY model, equilibration and reconstruction sample collection)
std::cout << "------DURATION OF GEOSTATISTICAL APPLICATION------\n"
//<< "Inicialization processes (loading reference E=E(T), loading source data, GPU memory allocation and copying):\n"
<< "t_initialization = " << (double)duration_initial / 1e6 << " s\n"
//<< "Mapping to XY model, equilibration and reconstruction sample collection (per configuration sample):\n"
<< "t_reconstruction = " << (double)duration_mapping_EQ_sampling / 1e6 / CONFIG_SAMPLES << " s\n"
//<< "Mapping to XY model, equilibration and reconstruction sample collection:\n"
<< "t_TOTAL = " << ((double)duration_initial / 1e6 + (double)duration_mapping_EQ_sampling / 1e6 / CONFIG_SAMPLES) << " s\n";

return 0;
}

//single checkerboard versions
__global__ void metro_conditioned_equil_sublattice_k(spin_t *s, spin_t *dilution_mask_d, float *devRand, unsigned int offset, energy_t BETA, energy_t *AccD, float alpha)
{
    // int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = 2 * threadIdx.y + (threadIdx.x + offset) % 2 + BLOCKL*blockIdx.y;

    unsigned int n = threadIdx.x + threadIdx.y*BLOCKL;
    unsigned int idx = n + THREADS * (blockIdx.x + gridDim.x*blockIdx.y);

    // Acceptance rate measurement
    unsigned int Acc = 0;

    if (isnan(dilution_mask_d[x + L*y]))
    {
        spin_t S_old = s[x + L*y];
        spin_t S_new = S_old + alpha * (devRand[idx + offset*N / 2 + N*SWEEPS_EMPTY] - 0.5f);
        S_new = (S_new < 0.0f) ? 0.0f : S_new;
        S_new = (S_new > 2.0f * M_PI) ? 2.0f * M_PI : S_new;

        energy_t E1 = 0.0, E2 = 0.0;

        // NOTE: open boundary conditions -> energy contribution on boundary always results in -cos(S(x,y) - S(x,y)) = -1 
#ifdef DOUBLE_PRECISION
        E1 -= (x == 0) ? 1 : cos(Qfactor * (S_old - s[x - 1 + L*y]));		// from s(x-1,y)
        E2 -= (x == 0) ? 1 : cos(Qfactor * (S_new - s[x - 1 + L*y]));
        E1 -= (x == L - 1) ? 1 : cos(Qfactor * (S_old - s[x + 1 + L*y]));	// from s(x+1,y)
        E2 -= (x == L - 1) ? 1 : cos(Qfactor * (S_new - s[x + 1 + L*y]));
        E1 -= (y == 0) ? 1 : cos(Qfactor * (S_old - s[x + L*(y - 1)]));		// from s(x,y-1)
        E2 -= (y == 0) ? 1 : cos(Qfactor * (S_new - s[x + L*(y - 1)]));
        E1 -= (y == L - 1) ? 1 : cos(Qfactor * (S_old - s[x + L*(y + 1)]));	// from s(x,y+1)
        E2 -= (y == L - 1) ? 1 : cos(Qfactor * (S_new - s[x + L*(y + 1)]));

        if (devRand[idx + offset*N / 2] < exp(-BETA * (E2 - E1)))
        {
            s[x + L*y] = S_new;
            ++Acc;
        }
#else
#ifdef INTRINSIC_FLOAT
        E1 -= (x == 0) ? 1 : __cosf(Qfactor * (S_old - s[x - 1 + L*y]));		// from s(x-1,y)
        E2 -= (x == 0) ? 1 : __cosf(Qfactor * (S_new - s[x - 1 + L*y]));
        E1 -= (x == L - 1) ? 1 : __cosf(Qfactor * (S_old - s[x + 1 + L*y]));	// from s(x+1,y)
        E2 -= (x == L - 1) ? 1 : __cosf(Qfactor * (S_new - s[x + 1 + L*y]));
        E1 -= (y == 0) ? 1 : __cosf(Qfactor * (S_old - s[x + L*(y - 1)]));		// from s(x,y-1)
        E2 -= (y == 0) ? 1 : __cosf(Qfactor * (S_new - s[x + L*(y - 1)]));
        E1 -= (y == L - 1) ? 1 : __cosf(Qfactor * (S_old - s[x + L*(y + 1)]));	// from s(x,y+1)
        E2 -= (y == L - 1) ? 1 : __cosf(Qfactor * (S_new - s[x + L*(y + 1)]));

        if (devRand[idx + offset*N / 2] < __expf(-BETA * (E2 - E1)))
        {
            s[x + L*y] = S_new;
            ++Acc;
        }
#else
        E1 -= (x == 0) ? 1 : cosf(Qfactor * (S_old - s[x - 1 + L*y]));			// from s(x-1,y)
        E2 -= (x == 0) ? 1 : cosf(Qfactor * (S_new - s[x - 1 + L*y]));
        E1 -= (x == L - 1) ? 1 : cosf(Qfactor * (S_old - s[x + 1 + L*y]));		// from s(x+1,y)
        E2 -= (x == L - 1) ? 1 : cosf(Qfactor * (S_new - s[x + 1 + L*y]));
        E1 -= (y == 0) ? 1 : cosf(Qfactor * (S_old - s[x + L*(y - 1)]));		// from s(x,y-1)
        E2 -= (y == 0) ? 1 : cosf(Qfactor * (S_new - s[x + L*(y - 1)]));
        E1 -= (y == L - 1) ? 1 : cosf(Qfactor * (S_old - s[x + L*(y + 1)]));	// from s(x,y+1)
        E2 -= (y == L - 1) ? 1 : cosf(Qfactor * (S_new - s[x + L*(y + 1)]));

        if (devRand[idx + offset*N / 2] < expf(-BETA * (E2 - E1)))
        {
            s[x + L*y] = S_new;
            ++Acc;
        }
#endif
#endif

    }

    __shared__ unsigned int AccSum[THREADS];

    AccSum[n] = Acc;

    for (int stride = THREADS >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (n < stride) {
            AccSum[n] += AccSum[n + stride];
        }
    }

    if (n == 0) AccD[blockIdx.y*GRIDL + blockIdx.x] += AccSum[0];
}

__global__ void metro_conditioned_sublattice_k(spin_t *s, spin_t *dilution_mask_d, float *devRand, unsigned int offset, energy_t BETA, energy_t *AccD, float alpha)
{
    // int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = 2 * threadIdx.y + (threadIdx.x + offset) % 2 + BLOCKL*blockIdx.y;

    unsigned int n = threadIdx.x + threadIdx.y*BLOCKL;
    unsigned int idx = n + THREADS * (blockIdx.x + gridDim.x*blockIdx.y);

    // Acceptance rate measurement
    unsigned int Acc = 0;

    if (isnan(dilution_mask_d[x + L*y]))
    {
        spin_t S_old = s[x + L*y];
        spin_t S_new = S_old + alpha * (devRand[idx + offset*N / 2 + N*SWEEPS_EMPTY] - 0.5f);
        S_new = (S_new < 0.0f) ? 0.0f : S_new;
        S_new = (S_new > 2.0f * M_PI) ? 2.0f * M_PI : S_new;

        energy_t E1 = 0.0, E2 = 0.0;

        // NOTE: open boundary conditions -> energy contribution on boundary always results in -cos(S(x,y) - S(x,y)) = -1 
#ifdef DOUBLE_PRECISION
        E1 -= (x == 0) ? 1 : cos(Qfactor * (S_old - s[x - 1 + L*y]));		// from s(x-1,y)
        E2 -= (x == 0) ? 1 : cos(Qfactor * (S_new - s[x - 1 + L*y]));
        E1 -= (x == L - 1) ? 1 : cos(Qfactor * (S_old - s[x + 1 + L*y]));	// from s(x+1,y)
        E2 -= (x == L - 1) ? 1 : cos(Qfactor * (S_new - s[x + 1 + L*y]));
        E1 -= (y == 0) ? 1 : cos(Qfactor * (S_old - s[x + L*(y - 1)]));		// from s(x,y-1)
        E2 -= (y == 0) ? 1 : cos(Qfactor * (S_new - s[x + L*(y - 1)]));
        E1 -= (y == L - 1) ? 1 : cos(Qfactor * (S_old - s[x + L*(y + 1)]));	// from s(x,y+1)
        E2 -= (y == L - 1) ? 1 : cos(Qfactor * (S_new - s[x + L*(y + 1)]));

        if (devRand[idx + offset*N / 2] < exp(-BETA * (E2 - E1)))
        {
            s[x + L*y] = S_new;
            ++Acc;
        }
#else
#ifdef INTRINSIC_FLOAT
        E1 -= (x == 0) ? 1 : __cosf(Qfactor * (S_old - s[x - 1 + L*y]));		// from s(x-1,y)
        E2 -= (x == 0) ? 1 : __cosf(Qfactor * (S_new - s[x - 1 + L*y]));
        E1 -= (x == L - 1) ? 1 : __cosf(Qfactor * (S_old - s[x + 1 + L*y]));	// from s(x+1,y)
        E2 -= (x == L - 1) ? 1 : __cosf(Qfactor * (S_new - s[x + 1 + L*y]));
        E1 -= (y == 0) ? 1 : __cosf(Qfactor * (S_old - s[x + L*(y - 1)]));		// from s(x,y-1)
        E2 -= (y == 0) ? 1 : __cosf(Qfactor * (S_new - s[x + L*(y - 1)]));
        E1 -= (y == L - 1) ? 1 : __cosf(Qfactor * (S_old - s[x + L*(y + 1)]));	// from s(x,y+1)
        E2 -= (y == L - 1) ? 1 : __cosf(Qfactor * (S_new - s[x + L*(y + 1)]));

        if (devRand[idx + offset*N / 2] < __expf(-BETA * (E2 - E1)))
        {
            s[x + L*y] = S_new;
            ++Acc;
        }
#else
        E1 -= (x == 0) ? 1 : cosf(Qfactor * (S_old - s[x - 1 + L*y]));			// from s(x-1,y)
        E2 -= (x == 0) ? 1 : cosf(Qfactor * (S_new - s[x - 1 + L*y]));
        E1 -= (x == L - 1) ? 1 : cosf(Qfactor * (S_old - s[x + 1 + L*y]));		// from s(x+1,y)
        E2 -= (x == L - 1) ? 1 : cosf(Qfactor * (S_new - s[x + 1 + L*y]));
        E1 -= (y == 0) ? 1 : cosf(Qfactor * (S_old - s[x + L*(y - 1)]));		// from s(x,y-1)
        E2 -= (y == 0) ? 1 : cosf(Qfactor * (S_new - s[x + L*(y - 1)]));
        E1 -= (y == L - 1) ? 1 : cosf(Qfactor * (S_old - s[x + L*(y + 1)]));	// from s(x,y+1)
        E2 -= (y == L - 1) ? 1 : cosf(Qfactor * (S_new - s[x + L*(y + 1)]));

        if (devRand[idx + offset*N / 2] < expf(-BETA * (E2 - E1)))
        {
            s[x + L*y] = S_new;
            ++Acc;
        }
#endif
#endif

    }

    __shared__ unsigned int AccSum[THREADS];

    AccSum[n] = Acc;

    for (int stride = THREADS >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (n < stride) {
            AccSum[n] += AccSum[n + stride];
        }
    }

    if (n == 0) AccD[blockIdx.y*GRIDL + blockIdx.x] += AccSum[0];
    //if (n == 0) printf("block %d ma alfu %1.2f a acc. pomer %1.5f\n", blockIdx.y*GRIDL + blockIdx.x, alpha, float(AccSum[0]) / THREADS);
}

//double checkerboard versions
#ifdef DC_EQ

__global__ void metro_conditioned_equil_sublattice_shared_k(spin_t *s, spin_t *dilution_mask_d, float *devRand, unsigned int offset, energy_t *T_diluted_per_block_d, energy_t *AccD, energy_t* alphas_per_block_d, unsigned int* Tried)
{
    unsigned int n = threadIdx.y*BLOCKL + threadIdx.x;

    unsigned int xoffset = blockIdx.x*BLOCKL;
    unsigned int yoffset = (2 * blockIdx.y + (blockIdx.x + offset) % 2)*BLOCKL;
    unsigned int blockIndex = (2 * blockIdx.y + (blockIdx.x + offset) % 2) * gridDim.x + blockIdx.x;
    //unique temperature for each block
    energy_t BETA = 1.0 / T_diluted_per_block_d[blockIndex];
    energy_t alpha = alphas_per_block_d[blockIndex];
    //if(alpha > 6.0)
    //if (blockIndex == 56 && n == 0)
    //{
    //alpha = alpha / 100;
    //printf("block index = %d, T = %1.7f, alpha = %1.7f\n", blockIndex, T_diluted_per_block_d[blockIndex], alpha);
    //}
    //printf("alpha = %1.4f\n", alpha);
    //energy_t alpha = 2 * M_PI;
    __shared__ spin_t sS[(BLOCKL + 2)*(BLOCKL + 2)];

    // central spins of the tile
    sS[(2 * threadIdx.y + 1)*(BLOCKL + 2) + threadIdx.x + 1] = *(s + (yoffset + 2 * threadIdx.y)*L + xoffset + threadIdx.x);
    sS[(2 * threadIdx.y + 2)*(BLOCKL + 2) + threadIdx.x + 1] = *(s + (yoffset + 2 * threadIdx.y + 1)*L + xoffset + threadIdx.x);
    // lower-boundary spins of the tile; if on the edge of the system -> load edge values as boundary
    if (threadIdx.y == 0)
        sS[threadIdx.x + 1] = (yoffset == 0) ? *(s + xoffset + threadIdx.x) : *(s + (yoffset - 1)*L + xoffset + threadIdx.x);
    // upper-boundary spins of the tile; if on the edge -> load edge values as boundary
    if (threadIdx.y == BLOCKL / 2 - 1)
        sS[(BLOCKL + 1)*(BLOCKL + 2) + threadIdx.x + 1] = (yoffset == L - BLOCKL) ? *(s + (L - 1)*L + xoffset + threadIdx.x) :
        *(s + (yoffset + BLOCKL)*L + xoffset + threadIdx.x);
    // left-boundary spins of the tile; if on the edge of the system -> load edge values as boundary
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) {
            sS[(2 * threadIdx.y + 1)*(BLOCKL + 2)] = *(s + (yoffset + 2 * threadIdx.y)*L);
            sS[(2 * threadIdx.y + 2)*(BLOCKL + 2)] = *(s + (yoffset + 2 * threadIdx.y + 1)*L);
        }
        else {
            sS[(2 * threadIdx.y + 1)*(BLOCKL + 2)] = *(s + (yoffset + 2 * threadIdx.y)*L + xoffset - 1);
            sS[(2 * threadIdx.y + 2)*(BLOCKL + 2)] = *(s + (yoffset + 2 * threadIdx.y + 1)*L + xoffset - 1);
        }
    }
    // right-boundary spins of the tile; if on the edge of the system -> load edge values as boundary
    if (threadIdx.x == BLOCKL - 1) {
        if (blockIdx.x == GRIDL - 1) {
            sS[(2 * threadIdx.y + 1)*(BLOCKL + 2) + BLOCKL + 1] = *(s + (yoffset + 2 * threadIdx.y)*L + (L - 1));
            sS[(2 * threadIdx.y + 2)*(BLOCKL + 2) + BLOCKL + 1] = *(s + (yoffset + 2 * threadIdx.y + 1)*L + (L - 1));
        }
        else {
            sS[(2 * threadIdx.y + 2)*(BLOCKL + 2) + BLOCKL + 1] = *(s + (yoffset + 2 * threadIdx.y + 1)*L + xoffset + BLOCKL);
            sS[(2 * threadIdx.y + 1)*(BLOCKL + 2) + BLOCKL + 1] = *(s + (yoffset + 2 * threadIdx.y)*L + xoffset + BLOCKL);
        }
    }

    __syncthreads();

    // Acceptance rate measurement
    unsigned int Acc = 0;
    //measures tried spins for correct acceptance rate calculation
    unsigned int tried = 0;

    // unsigned int ran = ranvec[(blockIdx.y*GRIDL + blockIdx.x)*THREADS + n];

    unsigned int x = threadIdx.x;
    unsigned int y1 = (threadIdx.x % 2) + 2 * threadIdx.y;
    unsigned int y2 = ((threadIdx.x + 1) % 2) + 2 * threadIdx.y;

    unsigned int idx = (blockIdx.y*gridDim.x + blockIdx.x)*THREADS + n;

    spin_t S_new;
    energy_t E1, E2;

    for (int i = 0; i < SWEEPS_LOCAL_EQ; ++i)
    {

        if (isnan(dilution_mask_d[(yoffset + y1)*L + xoffset + x]))
        {
            ++tried;
            S_new = sS(x, y1) + alpha * (devRand[idx + offset*N / 2 + N*i + N*SWEEPS_EMPTY*SWEEPS_LOCAL_EQ] - 0.5f);

            S_new = (S_new < 0.0f) ? 0.0f : S_new;
            //S_new = (S_new < 0.0f) ? -S_new : S_new;
            S_new = (S_new > 2.0f * M_PI) ? 2.0f * M_PI : S_new;

#ifdef DOUBLE_PRECISION
            E1 = 0.0 - cos(Qfactor * (sS(x, y1) - sS(x - 1, y1))) - cos(Qfactor * (sS(x, y1) - sS(x, y1 - 1)))
                - cos(Qfactor * (sS(x, y1) - sS(x + 1, y1))) - cos(Qfactor * (sS(x, y1) - sS(x, y1 + 1))); /*E1*/
            E2 = 0.0 - cos(Qfactor * (S_new - sS(x - 1, y1))) - cos(Qfactor * (S_new - sS(x, y1 - 1)))
                - cos(Qfactor * (S_new - sS(x + 1, y1))) - cos(Qfactor * (S_new - sS(x, y1 + 1)));  /*E2*/

            if (devRand[idx + offset*N / 2 + N*i] < exp(-BETA * (E2 - E1)))
            {
#else
#ifdef INTRINSIC_FLOAT
            E1 = 0.0 - __cosf(Qfactor * (sS(x, y1) - sS(x - 1, y1))) - __cosf(Qfactor * (sS(x, y1) - sS(x, y1 - 1)))
                - __cosf(Qfactor * (sS(x, y1) - sS(x + 1, y1))) - __cosf(Qfactor * (sS(x, y1) - sS(x, y1 + 1))); /*E1*/
            E2 = 0.0 - __cosf(Qfactor * (S_new - sS(x - 1, y1))) - __cosf(Qfactor * (S_new - sS(x, y1 - 1)))
                - __cosf(Qfactor * (S_new - sS(x + 1, y1))) - __cosf(Qfactor * (S_new - sS(x, y1 + 1)));  /*E2*/

            if (devRand[idx + offset*N / 2 + N*i] < __expf(-BETA * (E2 - E1)))
            {
#else
            E1 = 0.0 - cosf(Qfactor * (sS(x, y1) - sS(x - 1, y1))) - cosf(Qfactor * (sS(x, y1) - sS(x, y1 - 1)))
                - cosf(Qfactor * (sS(x, y1) - sS(x + 1, y1))) - cosf(Qfactor * (sS(x, y1) - sS(x, y1 + 1))); /*E1*/
            E2 = 0.0 - cosf(Qfactor * (S_new - sS(x - 1, y1))) - cos(Qfactor * (S_new - sS(x, y1 - 1)))
                - cosf(Qfactor * (S_new - sS(x + 1, y1))) - cosf(Qfactor * (S_new - sS(x, y1 + 1)));  /*E2*/

            if (devRand[idx + offset*N / 2 + N*i] < expf(-BETA * (E2 - E1)))
            {
#endif
#endif
                //if (blockIndex == 35)
                //printf("block %d beta = %1.4f, exp = %1.4f, old = %1.4f, new = %1.4f \n", blockIndex, BETA, __expf(-BETA * (E2 - E1)), sS(x, y1), S_new);

                // if (blockIndex == 35)
                // printf("energy = %1.7f, factor = %1.7f devrand = %1.7f, prob = %1.7f, S_old = %1.7f, S_new = %1.7f\n", (E2 - E1), -BETA * (E2 - E1), devRand[idx + offset*N / 2 + N*i], __expf(-BETA * (E2 - E1)), sS(x, y1), S_new);
                sS(x, y1) = S_new;
                // update of spins on the system boundary means that the spins behind open boundary must be updated too
                if (xoffset + x == 0) sS(x - 1, y1) = S_new;
                if (xoffset + x == L - 1) sS(x + 1, y1) = S_new;
                if (yoffset + y1 == 0) sS(x, y1 - 1) = S_new;
                if (yoffset + y1 == L - 1) sS(x, y1 + 1) = S_new;
                ++Acc;
            }
            }

        __syncthreads();

        if (isnan(dilution_mask_d[(yoffset + y2)*L + xoffset + x]))
        {
            ++tried;
            S_new = sS(x, y2) + alpha * (devRand[idx + offset*N / 2 + N / 4 + N*i + N*SWEEPS_EMPTY*SWEEPS_LOCAL_EQ] - 0.5f);

            S_new = (S_new < 0.0f) ? 0.0f : S_new;
            S_new = (S_new > 2.0f * M_PI) ? 2.0f * M_PI : S_new;

#ifdef DOUBLE_PRECISION
            E1 = 0 - cos(Qfactor * (sS(x, y2) - sS(x - 1, y2))) - cos(Qfactor * (sS(x, y2) - sS(x, y2 - 1)))
                - cos(Qfactor * (sS(x, y2) - sS(x + 1, y2))) - cos(Qfactor * (sS(x, y2) - sS(x, y2 + 1))); /*E1*/
            E2 = 0 - cos(Qfactor * (S_new - sS(x - 1, y2))) - cos(Qfactor * (S_new - sS(x, y2 - 1)))
                - cos(Qfactor * (S_new - sS(x + 1, y2))) - cos(Qfactor * (S_new - sS(x, y2 + 1)));  /*E2*/

            if (devRand[idx + offset*N / 2 + N / 4 + N*i] < exp(-BETA * (E2 - E1)))
            {
#else

#ifdef INTRINSIC_FLOAT
            E1 = 0 - __cosf(Qfactor * (sS(x, y2) - sS(x - 1, y2))) - __cosf(Qfactor * (sS(x, y2) - sS(x, y2 - 1)))
                - __cosf(Qfactor * (sS(x, y2) - sS(x + 1, y2))) - __cosf(Qfactor * (sS(x, y2) - sS(x, y2 + 1))); /*E1*/
            E2 = 0 - __cosf(Qfactor * (S_new - sS(x - 1, y2))) - __cosf(Qfactor * (S_new - sS(x, y2 - 1)))
                - __cosf(Qfactor * (S_new - sS(x + 1, y2))) - __cosf(Qfactor * (S_new - sS(x, y2 + 1)));  /*E2*/

            if (devRand[idx + offset*N / 2 + N / 4 + N*i] < __expf(-BETA * (E2 - E1)))
            {
#else
            E1 = 0 - cosf(Qfactor * (sS(x, y2) - sS(x - 1, y2))) - cosf(Qfactor * (sS(x, y2) - sS(x, y2 - 1)))
                - cosf(Qfactor * (sS(x, y2) - sS(x + 1, y2))) - cosf(Qfactor * (sS(x, y2) - sS(x, y2 + 1))); /*E1*/
            E2 = 0 - cosf(Qfactor * (S_new - sS(x - 1, y2))) - cosf(Qfactor * (S_new - sS(x, y2 - 1)))
                - cosf(Qfactor * (S_new - sS(x + 1, y2))) - cosf(Qfactor * (S_new - sS(x, y2 + 1)));  /*E2*/

            if (devRand[idx + offset*N / 2 + N / 4 + N*i] < expf(-BETA * (E2 - E1)))
            {
#endif

#endif
                sS(x, y2) = S_new;
                // update of spins on the system boundary means that the spins behind open boundary must be updated too
                if (xoffset + x == 0) sS(x - 1, y2) = S_new;
                if (xoffset + x == L - 1) sS(x + 1, y2) = S_new;
                if (yoffset + y2 == 0) sS(x, y2 - 1) = S_new;
                if (yoffset + y2 == L - 1) sS(x, y2 + 1) = S_new;
                ++Acc;
            }
            }

        __syncthreads();

            }

    s[(yoffset + 2 * threadIdx.y)*L + xoffset + threadIdx.x] = sS[(2 * threadIdx.y + 1)*(BLOCKL + 2) + threadIdx.x + 1];
    s[(yoffset + 2 * threadIdx.y + 1)*L + xoffset + threadIdx.x] = sS[(2 * threadIdx.y + 2)*(BLOCKL + 2) + threadIdx.x + 1];
    //ranvec[(blockIdx.y*GRIDL + blockIdx.x)*THREADS + n] = ran;

    __shared__ unsigned int AccSum[THREADS];
    __shared__ unsigned int triedSum[THREADS];
    AccSum[n] = Acc;
    triedSum[n] = tried;
    for (unsigned int stride = THREADS >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (n < stride) {
            AccSum[n] += AccSum[n + stride];
            triedSum[n] += triedSum[n + stride];
        }
    }

    if (n == 0)
    {
        AccD[blockIndex] += AccSum[0];
        Tried[blockIndex] += triedSum[0];
        //if(blockIndex == 35)
        //printf("block index = %d, ACC_rate = %1.7f, accepted = %d, tried  = %d\n", blockIndex, AccSum[0] / (float)triedSum[0], AccSum[0], triedSum[0]);

    }
        }


#endif

#ifdef DC_SIM
__global__ void metro_conditioned_sublattice_shared_k(spin_t *s, spin_t *dilution_mask_d, float *devRand, unsigned int offset, energy_t *T_diluted_per_block_d, energy_t *AccD, energy_t* alphas_per_block_d, unsigned int* Tried)
{
    unsigned int n = threadIdx.y*BLOCKL + threadIdx.x;

    unsigned int xoffset = blockIdx.x*BLOCKL;
    unsigned int yoffset = (2 * blockIdx.y + (blockIdx.x + offset) % 2)*BLOCKL;
    //unique temperature for each block
    energy_t BETA = 1.0 / T_diluted_per_block_d[(2 * blockIdx.y + (blockIdx.x + offset) % 2) * gridDim.x + blockIdx.x];
    energy_t alpha = alphas_per_block_d[(2 * blockIdx.y + (blockIdx.x + offset) % 2) * gridDim.x + blockIdx.x];
    //energy_t alpha = 2 * M_PI;
    //if((2 * blockIdx.y + (blockIdx.x + offset) % 2) * gridDim.x + blockIdx.x == 16)
    //printf("teplota %1.6f alpha = %1.4f\n", T_diluted_per_block_d[(2 * blockIdx.y + (blockIdx.x + offset) % 2) * gridDim.x + blockIdx.x], alpha);

    __shared__ spin_t sS[(BLOCKL + 2)*(BLOCKL + 2)];

    // central spins of the tile
    sS[(2 * threadIdx.y + 1)*(BLOCKL + 2) + threadIdx.x + 1] = *(s + (yoffset + 2 * threadIdx.y)*L + xoffset + threadIdx.x);
    sS[(2 * threadIdx.y + 2)*(BLOCKL + 2) + threadIdx.x + 1] = *(s + (yoffset + 2 * threadIdx.y + 1)*L + xoffset + threadIdx.x);
    // lower-boundary spins of the tile; if on the edge of the system -> load edge values as boundary
    if (threadIdx.y == 0)
        sS[threadIdx.x + 1] = (yoffset == 0) ? *(s + xoffset + threadIdx.x) : *(s + (yoffset - 1)*L + xoffset + threadIdx.x);
    // upper-boundary spins of the tile; if on the edge -> load edge values as boundary
    if (threadIdx.y == BLOCKL / 2 - 1)
        sS[(BLOCKL + 1)*(BLOCKL + 2) + threadIdx.x + 1] = (yoffset == L - BLOCKL) ? *(s + (L - 1)*L + xoffset + threadIdx.x) :
        *(s + (yoffset + BLOCKL)*L + xoffset + threadIdx.x);
    // left-boundary spins of the tile; if on the edge of the system -> load edge values as boundary
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) {
            sS[(2 * threadIdx.y + 1)*(BLOCKL + 2)] = *(s + (yoffset + 2 * threadIdx.y)*L);
            sS[(2 * threadIdx.y + 2)*(BLOCKL + 2)] = *(s + (yoffset + 2 * threadIdx.y + 1)*L);
        }
        else {
            sS[(2 * threadIdx.y + 1)*(BLOCKL + 2)] = *(s + (yoffset + 2 * threadIdx.y)*L + xoffset - 1);
            sS[(2 * threadIdx.y + 2)*(BLOCKL + 2)] = *(s + (yoffset + 2 * threadIdx.y + 1)*L + xoffset - 1);
        }
    }
    // right-boundary spins of the tile; if on the edge of the system -> load edge values as boundary
    if (threadIdx.x == BLOCKL - 1) {
        if (blockIdx.x == GRIDL - 1) {
            sS[(2 * threadIdx.y + 1)*(BLOCKL + 2) + BLOCKL + 1] = *(s + (yoffset + 2 * threadIdx.y)*L + (L - 1));
            sS[(2 * threadIdx.y + 2)*(BLOCKL + 2) + BLOCKL + 1] = *(s + (yoffset + 2 * threadIdx.y + 1)*L + (L - 1));
        }
        else {
            sS[(2 * threadIdx.y + 2)*(BLOCKL + 2) + BLOCKL + 1] = *(s + (yoffset + 2 * threadIdx.y + 1)*L + xoffset + BLOCKL);
            sS[(2 * threadIdx.y + 1)*(BLOCKL + 2) + BLOCKL + 1] = *(s + (yoffset + 2 * threadIdx.y)*L + xoffset + BLOCKL);
        }
    }

    __syncthreads();

    // Acceptance rate measurement
    unsigned int Acc = 0;
    //measures tried spins for correct acceptance rate calculation
    unsigned int tried = 0;

    // unsigned int ran = ranvec[(blockIdx.y*GRIDL + blockIdx.x)*THREADS + n];

    unsigned int x = threadIdx.x;
    unsigned int y1 = (threadIdx.x % 2) + 2 * threadIdx.y;
    unsigned int y2 = ((threadIdx.x + 1) % 2) + 2 * threadIdx.y;

    unsigned int idx = (blockIdx.y*gridDim.x + blockIdx.x)*THREADS + n;

    spin_t S_new;
    energy_t E1, E2;

    for (int i = 0; i < SWEEPS_LOCAL_SIM; ++i)
    {

        if (isnan(dilution_mask_d[(yoffset + y1)*L + xoffset + x]))
        {
            ++tried;
            S_new = sS(x, y1) + alpha * (devRand[idx + offset*N / 2 + N*i + N*SWEEPS_EMPTY*SWEEPS_LOCAL_SIM] - 0.5f);

            S_new = (S_new < 0.0f) ? 0.0f : S_new;
            S_new = (S_new > 2.0f * M_PI) ? 2.0f * M_PI : S_new;

#ifdef DOUBLE_PRECISION
            E1 = 0.0 - cos(Qfactor * (sS(x, y1) - sS(x - 1, y1))) - cos(Qfactor * (sS(x, y1) - sS(x, y1 - 1)))
                - cos(Qfactor * (sS(x, y1) - sS(x + 1, y1))) - cos(Qfactor * (sS(x, y1) - sS(x, y1 + 1))); /*E1*/
            E2 = 0.0 - cos(Qfactor * (S_new - sS(x - 1, y1))) - cos(Qfactor * (S_new - sS(x, y1 - 1)))
                - cos(Qfactor * (S_new - sS(x + 1, y1))) - cos(Qfactor * (S_new - sS(x, y1 + 1)));  /*E2*/

            if (devRand[idx + offset*N / 2 + N*i] < exp(-BETA * (E2 - E1)))
            {
#else
#ifdef INTRINSIC_FLOAT
            E1 = 0.0 - __cosf(Qfactor * (sS(x, y1) - sS(x - 1, y1))) - __cosf(Qfactor * (sS(x, y1) - sS(x, y1 - 1)))
                - __cosf(Qfactor * (sS(x, y1) - sS(x + 1, y1))) - __cosf(Qfactor * (sS(x, y1) - sS(x, y1 + 1))); /*E1*/
            E2 = 0.0 - __cosf(Qfactor * (S_new - sS(x - 1, y1))) - __cosf(Qfactor * (S_new - sS(x, y1 - 1)))
                - __cosf(Qfactor * (S_new - sS(x + 1, y1))) - __cosf(Qfactor * (S_new - sS(x, y1 + 1)));  /*E2*/

            if (devRand[idx + offset*N / 2 + N*i] < __expf(-BETA * (E2 - E1)))
            {
#else
            E1 = 0.0 - cosf(Qfactor * (sS(x, y1) - sS(x - 1, y1))) - cosf(Qfactor * (sS(x, y1) - sS(x, y1 - 1)))
                - cosf(Qfactor * (sS(x, y1) - sS(x + 1, y1))) - cosf(Qfactor * (sS(x, y1) - sS(x, y1 + 1))); /*E1*/
            E2 = 0.0 - cosf(Qfactor * (S_new - sS(x - 1, y1))) - cos(Qfactor * (S_new - sS(x, y1 - 1)))
                - cosf(Qfactor * (S_new - sS(x + 1, y1))) - cosf(Qfactor * (S_new - sS(x, y1 + 1)));  /*E2*/

            if (devRand[idx + offset*N / 2 + N*i] < expf(-BETA * (E2 - E1)))
            {
#endif
#endif
                sS(x, y1) = S_new;
                // update of spins on the system boundary means that the spins behind open boundary must be updated too
                if (xoffset + x == 0) sS(x - 1, y1) = S_new;
                if (xoffset + x == L - 1) sS(x + 1, y1) = S_new;
                if (yoffset + y1 == 0) sS(x, y1 - 1) = S_new;
                if (yoffset + y1 == L - 1) sS(x, y1 + 1) = S_new;
                ++Acc;
            }
            }

        __syncthreads();

        if (isnan(dilution_mask_d[(yoffset + y2)*L + xoffset + x]))
        {
            ++tried;
            S_new = sS(x, y2) + alpha * (devRand[idx + offset*N / 2 + N / 4 + N*i + N*SWEEPS_EMPTY*SWEEPS_LOCAL_EQ] - 0.5f);

            S_new = (S_new < 0.0f) ? 0.0f : S_new;
            S_new = (S_new > 2.0f * M_PI) ? 2.0f * M_PI : S_new;

#ifdef DOUBLE_PRECISION
            E1 = 0 - cos(Qfactor * (sS(x, y2) - sS(x - 1, y2))) - cos(Qfactor * (sS(x, y2) - sS(x, y2 - 1)))
                - cos(Qfactor * (sS(x, y2) - sS(x + 1, y2))) - cos(Qfactor * (sS(x, y2) - sS(x, y2 + 1))); /*E1*/
            E2 = 0 - cos(Qfactor * (S_new - sS(x - 1, y2))) - cos(Qfactor * (S_new - sS(x, y2 - 1)))
                - cos(Qfactor * (S_new - sS(x + 1, y2))) - cos(Qfactor * (S_new - sS(x, y2 + 1)));  /*E2*/

            if (devRand[idx + offset*N / 2 + N / 4 + N*i] < exp(-BETA * (E2 - E1)))
            {
#else

#ifdef INTRINSIC_FLOAT
            E1 = 0 - __cosf(Qfactor * (sS(x, y2) - sS(x - 1, y2))) - __cosf(Qfactor * (sS(x, y2) - sS(x, y2 - 1)))
                - __cosf(Qfactor * (sS(x, y2) - sS(x + 1, y2))) - __cosf(Qfactor * (sS(x, y2) - sS(x, y2 + 1))); /*E1*/
            E2 = 0 - __cosf(Qfactor * (S_new - sS(x - 1, y2))) - __cosf(Qfactor * (S_new - sS(x, y2 - 1)))
                - __cosf(Qfactor * (S_new - sS(x + 1, y2))) - __cosf(Qfactor * (S_new - sS(x, y2 + 1)));  /*E2*/

            if (devRand[idx + offset*N / 2 + N / 4 + N*i] < __expf(-BETA * (E2 - E1)))
            {
#else
            E1 = 0 - cosf(Qfactor * (sS(x, y2) - sS(x - 1, y2))) - cosf(Qfactor * (sS(x, y2) - sS(x, y2 - 1)))
                - cosf(Qfactor * (sS(x, y2) - sS(x + 1, y2))) - cosf(Qfactor * (sS(x, y2) - sS(x, y2 + 1))); /*E1*/
            E2 = 0 - cosf(Qfactor * (S_new - sS(x - 1, y2))) - cosf(Qfactor * (S_new - sS(x, y2 - 1)))
                - cosf(Qfactor * (S_new - sS(x + 1, y2))) - cosf(Qfactor * (S_new - sS(x, y2 + 1)));  /*E2*/

            if (devRand[idx + offset*N / 2 + N / 4 + N*i] < expf(-BETA * (E2 - E1)))
            {
#endif

#endif
                sS(x, y2) = S_new;
                // update of spins on the system boundary means that the spins behind open boundary must be updated too
                if (xoffset + x == 0) sS(x - 1, y2) = S_new;
                if (xoffset + x == L - 1) sS(x + 1, y2) = S_new;
                if (yoffset + y2 == 0) sS(x, y2 - 1) = S_new;
                if (yoffset + y2 == L - 1) sS(x, y2 + 1) = S_new;
                ++Acc;
            }
            }

        __syncthreads();

            }

    s[(yoffset + 2 * threadIdx.y)*L + xoffset + threadIdx.x] = sS[(2 * threadIdx.y + 1)*(BLOCKL + 2) + threadIdx.x + 1];
    s[(yoffset + 2 * threadIdx.y + 1)*L + xoffset + threadIdx.x] = sS[(2 * threadIdx.y + 2)*(BLOCKL + 2) + threadIdx.x + 1];
    //ranvec[(blockIdx.y*GRIDL + blockIdx.x)*THREADS + n] = ran;

    __shared__ unsigned int AccSum[THREADS];
    __shared__ unsigned int triedSum[THREADS];

    AccSum[n] = Acc;
    triedSum[n] = tried;


    for (unsigned int stride = THREADS >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (n < stride) {
            AccSum[n] += AccSum[n + stride];
            triedSum[n] += triedSum[n + stride];
        }
    }

    if (n == 0)
    {
        AccD[(2 * blockIdx.y + (blockIdx.x + offset) % 2) * gridDim.x + blockIdx.x] += AccSum[0];
        Tried[(2 * blockIdx.y + (blockIdx.x + offset) % 2) * gridDim.x + blockIdx.x] += triedSum[0];
    }
        }
#endif

__global__ void spin_mult(spin_t *s, spin_t mult_factor)
{
    unsigned int t = threadIdx.x;
    unsigned int b = blockIdx.x;
    unsigned int idx = t + blockDim.x * b;

    s[idx] = s[idx] * mult_factor;
}

__global__ void over_relaxation_k(spin_t *s, spin_t *dilution_mask_d, int offset)
{
    // int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = 2 * threadIdx.y + (threadIdx.x + offset) % 2 + BLOCKL*blockIdx.y;

    energy_t sumSin = 0.0, sumCos = 0.0;
    spin_t s_new;
    /*
    if (offset == 0)
    {
    s[x + L * y] = s[x + L * y] * Qfactor;
    offset = 1;
    y = 2 * threadIdx.y + (threadIdx.x + offset) % 2 + BLOCKL*blockIdx.y;
    s[x + L * y] = s[x + L * y] * Qfactor;
    offset = 0;
    }
    */

    // checkerboard update
    // not updating spins on the edge of the system
    if (isnan(dilution_mask_d[x + L*y]) && (x > 0) && (x < L - 1) && (y > 0) && (y < L - 1))
    {
        //summation of sin and cos from neighbouring spins
#ifdef DOUBLE_PRECISION
        sumSin += sin(s[x - 1 + L*y]);
        sumCos += cos(s[x - 1 + L*y]);
        sumSin += sin(s[x + 1 + L*y]);
        sumCos += cos(s[x + 1 + L*y]);
        sumSin += sin(s[x + L*(y - 1)]);
        sumCos += cos(s[x + L*(y - 1)]);
        sumSin += sin(s[x + L*(y + 1)]);
        sumCos += cos(s[x + L*(y + 1)]);
#else
#ifdef INTRINSIC_FLOAT
        sumSin += __sinf(s[x - 1 + L*y]);
        sumCos += __cosf(s[x - 1 + L*y]);
        sumSin += __sinf(s[x + 1 + L*y]);
        sumCos += __cosf(s[x + 1 + L*y]);
        sumSin += __sinf(s[x + L*(y - 1)]);
        sumCos += __cosf(s[x + L*(y - 1)]);
        sumSin += __sinf(s[x + L*(y + 1)]);
        sumCos += __cosf(s[x + L*(y + 1)]);
#else
        sumSin += sinf(s[x - 1 + L*y]);
        sumCos += cosf(s[x - 1 + L*y]);
        sumSin += sinf(s[x + 1 + L*y]);
        sumCos += cosf(s[x + 1 + L*y]);
        sumSin += sinf(s[x + L*(y - 1)]);
        sumCos += cosf(s[x + L*(y - 1)]);
        sumSin += sinf(s[x + L*(y + 1)]);
        sumCos += cosf(s[x + L*(y + 1)]);
#endif
#endif
        s_new = (spin_t)(fmod(2.0 * atan2(sumSin, sumCos) - s[x + L*y], 2.0 * M_PI));
        if ((s_new >= 0.0) && (s_new <= Qfactor * 2 * M_PI))
            s[x + L*y] = s_new;
    }
    /*
    if (offset == 1)
    {
    s[x + L * y] = s[x + L * y] / Qfactor;
    offset = 0;
    y = 2 * threadIdx.y + (threadIdx.x + offset) % 2 + BLOCKL*blockIdx.y;
    s[x + L * y] = s[x + L * y] / Qfactor;
    }
    */
}

__global__ void energyCalc_k(spin_t *s, energy_t *Ed) {

    unsigned int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;

    energy_t partE = 0;

    // (x,y < L - 1) conditions prevent from accounting bonds outside system boundaries 
#ifdef DOUBLE_PRECISION
    // if (x < L - 1) partE -= cos((energy_t)(Qfactor * (s[x + L*y] - s[x + 1 + L*y])));
    // if (y < L - 1) partE -= cos((energy_t)(Qfactor * (s[x + L*y] - s[x + L*(y + 1)])));
    if (x < L - 1) partE -= cos(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
    if (y < L - 1) partE -= cos(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
#else

#ifdef INTRINSIC_FLOAT
    if (x < L - 1) partE -= __cosf(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
    if (y < L - 1) partE -= __cosf(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
#else
    if (x < L - 1) partE -= cosf(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
    if (y < L - 1) partE -= cosf(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
#endif

#endif

    __shared__ energy_t EnSum[BLOCKL*BLOCKL];
    EnSum[t] = partE;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) EnSum[t] += EnSum[t + stride];
    }

    if (t == 0) Ed[blockIdx.x + gridDim.x*blockIdx.y] = EnSum[0];

}

__global__ void energyCalcDiluted_k(spin_t *s, energy_t *Ed)
{
    unsigned int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;

    energy_t partE = 0;
    energy_t tryLocalE;


    // (x,y < L - 1) conditions prevent from accounting bonds outside system boundaries 
#ifdef DOUBLE_PRECISION	
    if (x < L - 1)
    {
        tryLocalE = cos(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
        partE -= isnan(tryLocalE) ? 0 : tryLocalE;
    }
    if (y < L - 1)
    {
        tryLocalE = cos(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
        partE -= isnan(tryLocalE) ? 0 : tryLocalE;
    }
#else

#ifdef INTRINSIC_FLOAT
    if (x < L - 1)
    {
        tryLocalE = __cosf(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
        partE -= isnan(tryLocalE) ? 0 : tryLocalE;
    }
    if (y < L - 1)
    {
        tryLocalE = __cosf(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
        partE -= isnan(tryLocalE) ? 0 : tryLocalE;
    }
#else
    if (x < L - 1)
    {
        tryLocalE = cosf(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
        partE -= isnan(tryLocalE) ? 0 : tryLocalE;
    }
    if (y < L - 1)
    {
        tryLocalE = cosf(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
        partE -= isnan(tryLocalE) ? 0 : tryLocalE;
    }
#endif

#endif

    __shared__ energy_t EnSum[BLOCKL*BLOCKL];
    EnSum[t] = partE;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) EnSum[t] += EnSum[t + stride];
    }

    if (t == 0) Ed[blockIdx.x + gridDim.x*blockIdx.y] = EnSum[0];
    //uncomment for verifying the correct simulation temperatures per block
    //verification[x + L*y] = blockIdx.x + gridDim.x*blockIdx.y;
}

__global__ void energyCalcDiluted_per_block(energy_t *Ed, unsigned int *bondCount_d, unsigned int size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
    {
        Ed[i] = Ed[i] / bondCount_d[i];
    }
}

__global__ void min_max_avg_block(spin_t *d_s, spin_t *d_min, spin_t *d_max, spin_t *d_avg)
{
    unsigned int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;


    //stores values needed to compute min, max, sum and number of non-NaN values
    __shared__ spin_t min_max_avg_s[4 * BLOCKL*BLOCKL];
    spin_t spin = d_s[x + L*y];
    //if(t == 0)
    //printf("block %d has number = %1.7f\n", blockIdx.x + gridDim.x*blockIdx.y, d_s[x + L*y]);

    min_max_avg_s[t] = spin;
    min_max_avg_s[t + BLOCKL*BLOCKL] = spin;
    min_max_avg_s[t + 2 * BLOCKL*BLOCKL] = isnan(spin) ? 0 : spin;
    min_max_avg_s[t + 3 * BLOCKL*BLOCKL] = isnan(spin) ? 0 : 1;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride)
        {
            min_max_avg_s[t] = fmin(min_max_avg_s[t], min_max_avg_s[t + stride]);				// minimum search
            min_max_avg_s[t + BLOCKL*BLOCKL] = fmax(min_max_avg_s[t + BLOCKL*BLOCKL], min_max_avg_s[t + BLOCKL*BLOCKL + stride]);	// maximum search
            min_max_avg_s[t + 2 * BLOCKL*BLOCKL] += min_max_avg_s[t + 2 * BLOCKL*BLOCKL + stride];
            min_max_avg_s[t + 3 * BLOCKL*BLOCKL] += min_max_avg_s[t + 3 * BLOCKL*BLOCKL + stride];
        }
    }

    if (t == 0)
    {
        d_min[blockIdx.x + gridDim.x*blockIdx.y] = min_max_avg_s[0];
        d_max[blockIdx.x + gridDim.x*blockIdx.y] = min_max_avg_s[BLOCKL*BLOCKL];
        d_avg[blockIdx.x + gridDim.x*blockIdx.y] = min_max_avg_s[2 * BLOCKL*BLOCKL] / min_max_avg_s[3 * BLOCKL*BLOCKL];
        //printf("block %d has number = %1.7f\n", blockIdx.x + gridDim.x*blockIdx.y, d_pointers_to_blocks[blockIdx.x + gridDim.x*blockIdx.y][t]);

        //uncomment for verification
        //if(min_max_avg_s[BLOCKL*BLOCKL] > 6.0)
        //printf("block %d has min = %1.7f and max = %1.7f\n", blockIdx.x + gridDim.x*blockIdx.y, min_max_avg_s[0], min_max_avg_s[BLOCKL*BLOCKL]);
        //printf("block %d has avg = %1.7f\n", blockIdx.x + gridDim.x*blockIdx.y, avg[blockIdx.x + gridDim.x*blockIdx.y]);

    }
    //uncomment for verification
    /*
    __syncthreads();
    if (blockIdx.x + gridDim.x*blockIdx.y == 179)
    {
    for (int i = 0; i < BLOCKL*BLOCKL; i++)
    {
    __syncthreads();
    if(t == i)
    printf("thread %d has a low = %1.7f and high = %1.7f\n", t, min_max_avg_s[t], min_max_avg_s[t_off]);
    }

    }
    */
}

__global__ void resetAccD_k(energy_t *AccD) {

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < BLOCKS) AccD[idx] = 0;

}

__global__ void resetAccD_k(energy_t *AccD, unsigned int *Tried)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < BLOCKS * 2)
    {
        AccD[idx] = 0;
        Tried[idx] = 0;
    }
}

__global__ void setInitialAlphas(spin_t *alphas_per_block_d, spin_t* block_min, spin_t* block_max)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BLOCKS * 2)
    {

        //if(idx >= 32 )
        //printf("block %d has AccD = %1.4f and tryD = %df\n", idx, AccD[idx], tryD[idx]);
        //printf("accRate = %1.4f\t", acc_rate);

        //printf("zmena alfy block %d z %1.4f ", idx, alphas_per_block_d[idx]);
        //spin_t newAlpha = block_max[idx] - block_min[idx];
        //alphas_per_block_d[idx] = (newAlpha > 2.0 * M_PI) ? 2.0 * M_PI : ((newAlpha < 2.0 * M_PI / 100) ? 2.0 * M_PI / 100 : newAlpha);
        alphas_per_block_d[idx] = block_max[idx] - block_min[idx];
        //if(alphas_per_block_d[idx] > 6.0f)
        //if(threadIdx.x == 0)
        //printf("alfa block %d = %1.4f \n", idx, alphas_per_block_d[idx]);

        //if (idx == 35)
        //printf("temp_ratio = %1.5f, citatel = %1.5f, menovatel = %1.5f, teplota = %1.7f, logTeplota = %1.7f \n", temp_ratio, log(min_T - per_block_temps[idx] + 1), log(min_T - max_T), per_block_temps[idx], log(per_block_temps[idx]));
    }



}

__global__ void setAlphas(energy_t *AccD, spin_t *alphas_per_block_d, int iterations, energy_t acc_rate_min, unsigned int* tryD, spin_t* block_min, spin_t* block_max)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BLOCKS * 2)
    {
        energy_t acc_rate = AccD[idx] / (energy_t)tryD[idx];

        if (alphas_per_block_d[idx] > 0)
        {
            if (acc_rate < acc_rate_min)
            {
                //printf("zmena alfy block %d z %1.4f ", idx, alphas_per_block_d[idx]);
                //spin_t newAlpha = (block_max[idx] - block_min[idx]) / (1 + iterations / (spin_t)SLOPE_RESTR_FACTOR);
                //alphas_per_block_d[idx] = (newAlpha > 2.0 * M_PI) ? 2.0 * M_PI : ((newAlpha < 2.0 * M_PI / 100) ? 2.0 * M_PI / 100 : newAlpha);
                //alphas_per_block_d[idx] = (newAlpha > 2.0 * M_PI) ? 2.0 * M_PI : ((newAlpha < 2.0 * M_PI / 100) ? 2.0 * M_PI / 100 : newAlpha);
                alphas_per_block_d[idx] = (block_max[idx] - block_min[idx]) / (1 + iterations / (spin_t)SLOPE_RESTR_FACTOR);
                //alphas_per_block_d[idx] = (energy_t)(2.0 * M_PI * temp_ratio / (1 + iterations / (energy_t)SLOPE_RESTR_FACTOR));
                //if(idx == 35)
                //printf("block %d per block alpha = % 1.5f accRate = %1.4f\n", idx, alphas_per_block_d[idx], acc_rate);
            }
        }

    }
}

__global__ void correctTemps(energy_t *T_diluted_per_block_d, energy_t medianValue, unsigned int* bondCount_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BLOCKS * 2)
    {
        if (bondCount_d[idx] < 3)
            T_diluted_per_block_d[idx] = medianValue;

        //if (T_diluted_per_block_d[idx] < 0 || T_diluted_per_block_d[idx] > 900.0)
            //T_diluted_per_block_d[idx] = medianValue;

    }
}

__global__ void min_max_k(source_t *source_d, source_t *min_d, source_t *max_d, bool isDiluted, spin_t *diluted_mask_d)
{
    unsigned int t = threadIdx.x;
    unsigned int b = blockIdx.x;
    unsigned int idx = t + (blockDim.x * 2) * b;

    unsigned int t_off = t + 256;	// shared memory access with offset - it was calculated too many times

                                    /* By declaring the shared memory buffer as "volatile", the compiler is forced to enforce
                                    the shared memory write after each stage of the reduction,
                                    and the implicit data synchronisation between threads within the warp is restored */
    __shared__ volatile source_t min_max_s[512];
    if (isDiluted)
    {
        min_max_s[t] = source_d[idx] * diluted_mask_d[idx];
        min_max_s[t_off] = source_d[idx + 256] * diluted_mask_d[idx + 256];
        //if (idx + 256 == 46909)
        //printf("block %d and thread %d has source = %1.7f and source*dil = %1.7f\n", b, t, source_d[idx + 256], source_d[idx + 256] * diluted_mask_d[idx]);

    }
    else
    {
        min_max_s[t] = source_d[idx];
        min_max_s[t_off] = source_d[idx + 256];
        //if (idx + 256 == 46909)
        //printf("block %d and thread %d has source = %1.7f and \n", b, t, source_d[idx + 256]);
    }

    __syncthreads();

    // divide min_max_avg_s araray to "min" part (indices 0 ... 255) and "max" part (256 ... 511)
    // macros min(a,b) (and max(a,b)) from math.h are equivalent to conditional ((a < b) ? (a) : (b)) -> will be added in preprocessing
    source_t temp = fmax(min_max_s[t], min_max_s[t_off]);
    min_max_s[t] = fmin(min_max_s[t], min_max_s[t_off]);
    min_max_s[t_off] = temp;

    // unrolling for loop -> to remove instrunction overhead
    __syncthreads();
    if (t < 128)
    {
        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 128]);				// minimum search
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 128]);	// maximum search
    }

    __syncthreads();
    if (t < 64)
    {
        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 64]);				// minimum search
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 64]);	// maximum search
    }

    /* when we have one warp left ->
    no need for "if(t<stride)" and "__syncthreads"
    (no extra work is saved and because instructions are SIMD synchronous within a warp)	*/
    __syncthreads();
    if (t < 32)
    {
        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 32]);
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 32]);

        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 16]);
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 16]);

        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 8]);
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 8]);

        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 4]);
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 4]);

        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 2]);
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 2]);

        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 1]);
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 1]);
    }

    // per block results are stored to global memory
    if (t == 0)
    {
        min_d[b] = min_max_s[0];
        max_d[b] = min_max_s[256];
        //if(min_max_avg_s[BLOCKL*BLOCKL] > 6.0)
        //printf("block %d has min = %1.7f and max = %1.7f\n", b, min_d[b], max_d[b]);
    }
    /*
    __syncthreads();
    if(min_max_s[t] > max_d[b])
    printf("block %d has min = %1.7f and max = %1.7f, but thread %d has min = %1.7f and max = %1.7f\n", b, min_d[b], max_d[b], t, min_max_s[t], min_max_s[t_off]);
    if (idx + 256 == 46909)
    printf("block %d has min = %1.7f and max = %1.7f, but thread %d has min = %1.7f and max = %1.7f, source = %1.7f\n", b, min_d[b], max_d[b], t, min_max_s[t], min_max_s[t_off], source_d[idx+256]);
    */

    /*
    __syncthreads();
    if (b == 91)
    {
    if(t == 0)
    printf("block %d has a low = %1.7f and high = %1.7f\n", b, min_d[b], max_d[b]);
    for (int i = 0; i < 256; i++)
    {
    __syncthreads();
    //if (t == i)
    //printf("thread %d has a low = %1.7f and high = %1.7f\n", t, min_max_s[t], min_max_s[t_off]);
    }

    }
    */
}

__global__ void XY_mapping_k(source_t *source_d, spin_t *XY_mapped_d, source_t minSource, source_t maxSource, bool isDiluted, spin_t *diluted_mask_d)
{
    unsigned int t = threadIdx.x;
    unsigned int b = blockIdx.x;
    unsigned int idx = t + blockDim.x * b;

    XY_mapped_d[idx] = (isDiluted) ? (spin_t)(2 * M_PI * (source_d[idx] * diluted_mask_d[idx] - minSource) / (maxSource - minSource)) :
        (spin_t)(2 * M_PI * (source_d[idx] - minSource) / (maxSource - minSource));
    //if (XY_mapped_d[idx]  > 6.29)
    //printf("block %d has value = %1.7f, maxSource = %1.7f, source_d[%d] = %1.7f\n", b, XY_mapped_d[idx], maxSource, idx, source_d[idx]);

}

__global__ void create_dilution_mask_k(spin_t *dilution_mask_d, float* devRandDil, unsigned int* remSum_d)
{
    unsigned int t = threadIdx.x;
    unsigned int b = blockIdx.x;
    unsigned int idx = t + blockDim.x * b;
    unsigned int rem;
    if (devRandDil[idx] < RemovedDataRatio)
    {
#ifdef DOUBLE_PRECISION
        dilution_mask_d[idx] = nan("");
#else
        dilution_mask_d[idx] = nanf("");
#endif
        rem = 1;
    }
    else
    {
        dilution_mask_d[idx] = 1;
        rem = 0;
    }
    volatile __shared__ unsigned int removed_Sum[256];
    removed_Sum[t] = rem;
    // unrolling for loop -> to remove instrunction overhead
    __syncthreads();
    if (t < 128) removed_Sum[t] += removed_Sum[t + 128];

    __syncthreads();
    if (t < 64) removed_Sum[t] += removed_Sum[t + 64];

    // reduction for last warp
    __syncthreads();
    if (t < 32)
    {
        removed_Sum[t] += removed_Sum[t + 32];
        removed_Sum[t] += removed_Sum[t + 16];
        removed_Sum[t] += removed_Sum[t + 8];
        removed_Sum[t] += removed_Sum[t + 4];
        removed_Sum[t] += removed_Sum[t + 2];
        removed_Sum[t] += removed_Sum[t + 1];
    }

    if (t == 0) remSum_d[b] = removed_Sum[0];
}

__global__ void fill_lattice_nans_averaged(spin_t *XY_mapped_d, spin_t *avg)
{
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;
    unsigned int idx = x + L*y;
    if (isnan(XY_mapped_d[idx])) XY_mapped_d[idx] = avg[blockIdx.x + gridDim.x*blockIdx.y];
}

__global__ void fill_lattice_nans_averaged_global(spin_t *XY_mapped_d, spin_t avg)
{
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;
    unsigned int idx = x + L*y;
    if (isnan(XY_mapped_d[idx])) XY_mapped_d[idx] = avg;
}

__global__ void fill_lattice_nans_random(spin_t *XY_mapped_d, float*devRand_fill)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (isnan(XY_mapped_d[idx])) XY_mapped_d[idx] = 2 * M_PI * devRand_fill[idx];
}

__global__ void data_reconstruction_k(source_t *reconstructed_d, spin_t *XY_mapped_d, source_t minSource, source_t maxSource, source_t *sum_d, source_t *sumSqr_d)
{
    unsigned int t = threadIdx.x;
    unsigned int b = blockIdx.x;
    unsigned int idx = t + blockDim.x * b;

    reconstructed_d[idx] = ((source_t)XY_mapped_d[idx])*(maxSource - minSource) / (2 * M_PI) + minSource;
    sum_d[idx] += reconstructed_d[idx];
    sumSqr_d[idx] += reconstructed_d[idx] * reconstructed_d[idx];
}

__global__ void bondCount_k(spin_t *mask_d, unsigned int *bondCount_d, unsigned int *sparse_blocks)
{
    unsigned int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;

    unsigned int bondCount = 0;
    bool isNotCentralNAN = !isnan(mask_d[x + L*y]);

    if (x < L - 1)
        bondCount += isNotCentralNAN && (!isnan(mask_d[x + 1 + L*y]));
    if (y < L - 1)
        bondCount += isNotCentralNAN && (!isnan(mask_d[x + L*(y + 1)]));

    __shared__ unsigned int bondSum[BLOCKL*BLOCKL];
    bondSum[t] = bondCount;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) bondSum[t] += bondSum[t + stride];
    }
    //sparse_blocks[0] = 0;
    if (t == 0) bondCount_d[blockIdx.x + gridDim.x*blockIdx.y] = bondSum[0];
    //if (t == 0 && bondSum[0] < 10) printf("bond count in block %d is %d\n", blockIdx.x + gridDim.x*blockIdx.y, bondSum[0]);
    //if (t == 0 && bondSum[0] < 40) atomicAdd(&sparse_blocks[0], 1);
    //if (t == 0 ) printf(" sparse_blocks %d \n", sparse_blocks[0]);
}

__global__ void mean_stdDev_reconstructed_k(source_t *mean_d, source_t *stdDev_d, unsigned int size)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    mean_d[idx] /= size;
    stdDev_d[idx] = sqrt(stdDev_d[idx] / size + mean_d[idx] * mean_d[idx]);
}

__global__ void sum_prediction_errors_k(source_t *source_d, source_t * mean_d, spin_t *dilution_mask_d,
    source_t *AAE_d, source_t *ARE_d, source_t *AARE_d, source_t *RASE_d, source_t* error_map_d, source_t* error_map_block_d)
{
    unsigned int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;
    unsigned int idx = x + L*y;

    source_t source = source_d[idx];
    source_t est_error = (source - mean_d[idx]);
    bool isnan_site = isnan(dilution_mask_d[idx]);

    //get error for each particular spin for error map
    error_map_d[idx] += isnan_site * fabs(est_error);

    volatile __shared__ source_t sum_err[BLOCKL*BLOCKL];
    volatile __shared__ unsigned int validSpins[BLOCKL*BLOCKL];
    // AVERAGE ABSOLUTE ERROR
    sum_err[t] = isnan_site * fabs(est_error);
    validSpins[t] = (unsigned int)isnan_site;
    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride)
        {
            sum_err[t] += sum_err[t + stride];
            validSpins[t] += validSpins[t + stride];
        }
    }

    if (t == 0)
    {
        AAE_d[blockIdx.x + gridDim.x*blockIdx.y] = sum_err[0];
        error_map_block_d[blockIdx.x + gridDim.x*blockIdx.y] += sum_err[0] / (source_t)validSpins[0];
    }
    // AVERAGE RELAITVE ERROR
    __syncthreads();
    sum_err[t] = isnan_site * est_error / source;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) sum_err[t] += sum_err[t + stride];
    }
    if (t == 0) ARE_d[blockIdx.x + gridDim.x*blockIdx.y] = sum_err[0];
    // AVERAGE ABSOLUTE RELATIVE ERROR
    __syncthreads();
    sum_err[t] = isnan_site * fabs(est_error) / source;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) sum_err[t] += sum_err[t + stride];
    }
    if (t == 0) AARE_d[blockIdx.x + gridDim.x*blockIdx.y] = sum_err[0];

    // summation for ROOT AVERAGE SQUARED ROOT
    __syncthreads();
    sum_err[t] = isnan_site * est_error * est_error;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) sum_err[t] += sum_err[t + stride];
    }
    if (t == 0) RASE_d[blockIdx.x + gridDim.x*blockIdx.y] = sum_err[0];
}


__global__ void sum_prediction_errors_k(source_t *source_d, source_t * mean_d, spin_t *dilution_mask_d,
    source_t *AAE_d, source_t *ARE_d, source_t *AARE_d, source_t *RASE_d)
{
    unsigned int t = threadIdx.x;
    unsigned int idx = t + blockDim.x * blockIdx.x;
    source_t source = source_d[idx];
    source_t est_error = (source - mean_d[idx]);
    bool isnan_site = isnan(dilution_mask_d[idx]);

    volatile __shared__ source_t sum_err[256];

    // AVERAGE ABSOLUTE ERROR
    sum_err[t] = isnan_site * fabs(est_error);

    // unrolling for loop -> to remove instrunction overhead
    __syncthreads();
    if (t < 128) sum_err[t] += sum_err[t + 128];
    __syncthreads();
    if (t < 64) sum_err[t] += sum_err[t + 64];
    __syncthreads();
    if (t < 32)
    {
        sum_err[t] += sum_err[t + 32];
        sum_err[t] += sum_err[t + 16];
        sum_err[t] += sum_err[t + 8];
        sum_err[t] += sum_err[t + 4];
        sum_err[t] += sum_err[t + 2];
        sum_err[t] += sum_err[t + 1];
    }
    if (t == 0) AAE_d[blockIdx.x] = sum_err[0];

    // AVERAGE RELAITVE ERROR
    __syncthreads();
    sum_err[t] = isnan_site * est_error / source;

    // unrolling for loop -> to remove instrunction overhead
    __syncthreads();
    if (t < 128) sum_err[t] += sum_err[t + 128];
    __syncthreads();
    if (t < 64) sum_err[t] += sum_err[t + 64];
    __syncthreads();
    if (t < 32)
    {
        sum_err[t] += sum_err[t + 32];
        sum_err[t] += sum_err[t + 16];
        sum_err[t] += sum_err[t + 8];
        sum_err[t] += sum_err[t + 4];
        sum_err[t] += sum_err[t + 2];
        sum_err[t] += sum_err[t + 1];
    }
    if (t == 0) ARE_d[blockIdx.x] = sum_err[0];

    // AVERAGE ABSOLUTE RELATIVE ERROR
    __syncthreads();
    sum_err[t] = isnan_site * fabs(est_error) / source;

    // unrolling for loop -> to remove instrunction overhead
    __syncthreads();
    if (t < 128) sum_err[t] += sum_err[t + 128];
    __syncthreads();
    if (t < 64) sum_err[t] += sum_err[t + 64];
    __syncthreads();
    if (t < 32)
    {
        sum_err[t] += sum_err[t + 32];
        sum_err[t] += sum_err[t + 16];
        sum_err[t] += sum_err[t + 8];
        sum_err[t] += sum_err[t + 4];
        sum_err[t] += sum_err[t + 2];
        sum_err[t] += sum_err[t + 1];
    }
    if (t == 0) AARE_d[blockIdx.x] = sum_err[0];

    // summation for ROOT AVERAGE SQUARED ROOT
    __syncthreads();
    sum_err[t] = isnan_site * est_error * est_error;

    // unrolling for loop -> to remove instrunction overhead
    __syncthreads();
    if (t < 128) sum_err[t] += sum_err[t + 128];
    __syncthreads();
    if (t < 64) sum_err[t] += sum_err[t + 64];
    __syncthreads();
    if (t < 32)
    {
        sum_err[t] += sum_err[t + 32];
        sum_err[t] += sum_err[t + 16];
        sum_err[t] += sum_err[t + 8];
        sum_err[t] += sum_err[t + 4];
        sum_err[t] += sum_err[t + 2];
        sum_err[t] += sum_err[t + 1];
    }
    if (t == 0) RASE_d[blockIdx.x] = sum_err[0];
}

__global__ void find_temperature_gpu(energy_t* E_source, double* T_ref, double* E_ref, energy_t* E_result, int size_source, int size_refs)
{
    for (int myIdx = blockIdx.x * blockDim.x + threadIdx.x; myIdx < size_source; myIdx += blockDim.x * gridDim.x)
    {
        energy_t myEnergy = E_source[myIdx];
        int it_E = 0;
        int it_T = 0;

        while ((it_E != size_refs - 1) && (myEnergy < E_ref[it_E]))
        {
            ++it_E;
            ++it_T;
        }
        // linear interpolation
        E_result[myIdx] = (it_E == 0) ? (T_ref[it_T]) : ((T_ref[it_T] - T_ref[it_T - 1]) * (myEnergy - E_ref[it_E]) / (E_ref[it_E] - E_ref[it_E - 1]) + T_ref[it_T]);
        //E_result[myIdx] = 0.000001;
        //printf("v blocku %d teplota je %1.4f \t energia je %1.4f\n", myIdx, E_result[myIdx], myEnergy);
    }
}

energy_t cpu_energy(spin_t *s)
{
    // double ie = 0;
    energy_t partE = 0;
    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
#ifdef DOUBLE_PRECISION
            if (x < L - 1) partE -= cos(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
            if (y < L - 1) partE -= cos(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
#else
            if (x < L - 1) partE -= cosf(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
            if (y < L - 1) partE -= cosf(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
#endif
        }
    }
    return partE / Nbond;
}

double find_temperature(energy_t E_source, std::vector<double> T_ref, std::vector<double> E_ref)
{
    auto it_E = E_ref.begin();
    auto it_T = T_ref.begin();

    while ((it_E != E_ref.end()) && (E_source < *it_E))
    {
        ++it_E;
        ++it_T;
    }
    // linear interpolation
    return (it_E == E_ref.begin()) ? (*it_T) : ((*it_T - *(it_T - 1)) * (E_source - *it_E) / (*it_E - *(it_E - 1)) + *it_T);
}

// templates
template <class T> T sumPartialSums(T *parSums_d, int length)
{
    std::vector<T> parSums(length);
    CUDAErrChk(cudaMemcpy(parSums.data(), parSums_d, length * sizeof(T), cudaMemcpyDeviceToHost));
    T sum = 0;
    for (auto i : parSums) sum += i;
    return sum;
}

template <class T> std::vector<T> findMinMax(T *min_d, T *max_d, int length)
{
    std::vector<T> min_h(length);
    std::vector<T> max_h(length);
    CUDAErrChk(cudaMemcpy(min_h.data(), min_d, length * sizeof(T), cudaMemcpyDeviceToHost));
    CUDAErrChk(cudaMemcpy(max_h.data(), max_d, length * sizeof(T), cudaMemcpyDeviceToHost));
    /*T min_temp = *(std::min_element(min_h.begin(), min_h.end()));
    T max_temp = *(std::max_element(max_h.begin(), max_h.end()));
    std::vector<T> min_max = { min_temp, max_temp };*/
    std::vector<T> min_max = { min_h.at(0), max_h.at(0) };
    for (auto i : min_h) min_max.at(0) = std::fmin(min_max.at(0), i);
    for (auto i : max_h) min_max.at(1) = std::fmax(min_max.at(1), i);

    /* std::cout << "Block Minimum elements: ";
    for (auto i : min_h) std::cout << i << " ";
    std::cout << "\n"; */

    return min_max;
}

template <class T> T find_median(T *data_d, int length)
{
    std::vector<T> data_h(length);
    CUDAErrChk(cudaMemcpy(data_h.data(), data_d, length * sizeof(T), cudaMemcpyDeviceToHost));
    // First we sort the array 
    std::sort(data_h.begin(), data_h.end());
    // check for even case 
    if (length % 2 != 0)
        return data_h[length / 2];
    
    return (data_h[(length - 1) / 2] + data_h[length / 2]) / (T)2.0;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockSortKernel(energy_t *d_in, energy_t *d_out)
{
    // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
    typedef cub::BlockLoad<
        energy_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockLoadT;
    //typedef BlockLoad<Key, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;

    typedef cub::BlockStore<
        energy_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockStoreT;
    typedef cub::BlockRadixSort<
        energy_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
    // Allocate type-safe, repurposable shared memory for collectives
    __shared__ union {
        typename BlockLoadT::TempStorage       load;
        typename BlockStoreT::TempStorage      store;
        typename BlockRadixSortT::TempStorage  sort;
    } temp_storage;
    // Obtain this block's segment of consecutive keys (blocked across threads)
    energy_t thread_keys[ITEMS_PER_THREAD];
    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    BlockLoadT(temp_storage.load).Load(d_in + block_offset, thread_keys);

    __syncthreads();    // Barrier for smem reuse
                        // Collectively sort the keys
    BlockRadixSortT(temp_storage.sort).Sort(thread_keys);
    __syncthreads();    // Barrier for smem reuse
                        // Store the sorted segment 
    BlockStoreT(temp_storage.store).Store(d_out + block_offset, thread_keys);
   
}



// cuRAND errors
char* curandGetErrorString(curandStatus_t rc)
{
    switch (rc) {
    case CURAND_STATUS_SUCCESS:                   return (char*)curanderr[0];
    case CURAND_STATUS_VERSION_MISMATCH:          return (char*)curanderr[1];
    case CURAND_STATUS_NOT_INITIALIZED:           return (char*)curanderr[2];
    case CURAND_STATUS_ALLOCATION_FAILED:         return (char*)curanderr[3];
    case CURAND_STATUS_TYPE_ERROR:                return (char*)curanderr[4];
    case CURAND_STATUS_OUT_OF_RANGE:              return (char*)curanderr[5];
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:       return (char*)curanderr[6];
#if CUDART_VERSION >= 4010 
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return (char*)curanderr[7];
#endif
    case CURAND_STATUS_LAUNCH_FAILURE:            return (char*)curanderr[8];
    case CURAND_STATUS_PREEXISTING_FAILURE:       return (char*)curanderr[9];
    case CURAND_STATUS_INITIALIZATION_FAILED:     return (char*)curanderr[10];
    case CURAND_STATUS_ARCH_MISMATCH:             return (char*)curanderr[11];
    case CURAND_STATUS_INTERNAL_ERROR:            return (char*)curanderr[12];
    default:                                      return (char*)curanderr[13];
    }
}