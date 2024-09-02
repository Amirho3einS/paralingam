#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include "cuda_runtime.h"
#include "gpuerrors.h"
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <cuda.h>

#define data_type double


__global__ void normalize_all_device(data_type** X, int* U, int dims, int samples);
__global__ void normalize_all_device_base(data_type** out, data_type** X, int* U, int dims, int samples);
__global__ void calculate_cov_mat_device(data_type* cov_mat, data_type** all_data, int* U, int dims, int samples);
__global__ void update_cov_mat(data_type* cov_mat, int* U, int U_size, int dims, int root);
__global__ void regress_root_base(data_type** all_data, data_type* cov_mat, int root,
                                int* U, int U_size,
                                int dims, int samples);
__global__ void regress_root_base_V1(data_type** all_data, int root,
                                bool base,
                                int* U, int U_size,
                                int dims, int samples);


__global__ void calculate_cov_mat_base(data_type* cov_mat, data_type** all_data, int* U, int dims, int samples);


__global__ void regress_root(data_type** all_data, data_type* cov_mat, int root,
                                int* U, int U_size,
                                int dims, int samples);

__global__ void do_all_device_global(data_type** all_data, data_type* cov_mat,
                                data_type* entropy_result, data_type* M_list,
                                int* U, int U_size,
                                int* Uc, int Uc_size,
                                int* Vj, int Vj_size,
                                int dims, int samples,
                                data_type treshold,
                                bool* treshold_flag, bool* threshold_flag_block,
                                bool* entropy_flag, volatile int* entropy_Lock, int* checkpoint,
                                volatile bool* messages, volatile bool* done_dims, volatile data_type* M_all);

__global__ void do_all_device_Block_Worker_V1(data_type** all_data,
                                data_type** x_i, data_type** x_j,
                                data_type** ri_j, data_type** rj_i,
                                data_type* cov_mat,
                                data_type* M_list,
                                bool base,
                                int* U, int U_size,
                                int* Uc, int Uc_size,
                                int* Vj, int Vj_size,
                                int dims, int samples);

__global__ void do_all_device_Thread_Worker_V1(data_type** all_data,
                                            data_type* cov_mat,
                                            data_type* M_list,
                                            bool base,
                                            int* U, int U_size,
                                            int* Uc, int Uc_size,
                                            int* Vj, int Vj_size,
                                            int dims, int samples);

__global__ void do_all_device_Block_Compare_V1(data_type** all_data,
                                            data_type* cov_mat,
                                            data_type* M_all,
                                            bool base,
                                            int* U, int U_size,
                                            int* Uc, int Uc_size,
                                            int* Vj, int Vj_size,
                                            int dims, int samples);

__global__ void reduce_Block_Comapare (data_type* M_all, data_type* M_list,
                                        int* U, int U_size,
                                        int* Uc, int Uc_size,
                                        int dims, int samples);

__device__ bool find(int* list, int size, int id);

#endif