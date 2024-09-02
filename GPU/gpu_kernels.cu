#include "gpu_kernels.h"



namespace cg = cooperative_groups;

__device__ volatile int sem[3000 * 3000] = {};
__device__ void acquire_semaphore(volatile int *lock){
    while (atomicCAS((int *)lock, 0, 1) != 0);
}

__device__ void release_semaphore(volatile int *lock){
    *lock = 0;
    __threadfence();
}


template <unsigned int blockSize>
__device__ void reduceBlock(volatile data_type *sdata, data_type mySum, const unsigned int tid, cg::thread_block cta){
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    sdata[tid] = mySum;
    cg::sync(tile32);

    const int VEC = 32;
    const int vid = tid & (VEC-1);

    data_type beta = mySum;
    data_type temp;

    for (int i = VEC/2; i > 0; i>>=1){
        if (vid < i){
            temp      = sdata[tid+i];
            beta     += temp;
            sdata[tid]  = beta;
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (cta.thread_rank() == 0){
        beta = 0;
        for (int i = 0; i < blockDim.x; i += VEC){
            beta += sdata[i];
        }
        sdata[0] = beta;
    }
    cg::sync(cta);
}


template <unsigned int blockSize>
__device__ void covariance(volatile data_type* data_fir,volatile data_type* data_sec,
                            volatile data_type* sdata,
                            const bool base,
                            const unsigned int size,
                            const unsigned int id, 
                            volatile data_type* result, cg::thread_block cta){

    data_type temp = 0;
    data_type mean_fir = 0;
    data_type mean_sec = 0;
    data_type mySum = 0;
    int j = id;

    // calculating mean first
    while (j < size){
        mySum += data_fir[j];
        j += blockSize;
    }
    reduceBlock<512>(sdata, mySum, id, cta);
    mean_fir = sdata[0] / size;

    
    // calculating mean second
    j = id;
    mySum = 0;
    while (j < size){
        mySum += data_sec[j];
        j += blockSize;
    }
    reduceBlock<512>(sdata, mySum, id, cta);
    mean_sec = sdata[0] / size;

    j = id;
    while (j < size){
        temp += (data_fir[j] - mean_fir) * (data_sec[j] - mean_sec);
        j += blockSize;
    }
    reduceBlock<512>(sdata, temp, id, cta);
    result[0] = sdata[0] / (size - 1);
    if(base){
        if(result[0]> 0.99){
            result[0] = 0.99;
        }
        else if(result[0] < -0.99){
            result[0] = -0.99;
        }
    }
}

template <unsigned int blockSize>
__device__ void variance(volatile data_type* data,
                            volatile data_type* sdata,
                            const unsigned int size,
                            const unsigned int id, 
                            volatile data_type* result, cg::thread_block cta){

    data_type temp = 0;
    data_type mean = 0;
    data_type mySum = 0;
    int j = id;

    // calculating mean first
    while (j < size){
        mySum += data[j];
        j += blockSize;
    }
    reduceBlock<512>(sdata, mySum, id, cta);
    mean = sdata[0] / size;


    j = id;
    while (j < size){
        temp += pow(data[j] - mean, 2);
        j += blockSize;
    }
    reduceBlock<512>(sdata, temp, id, cta);
    
    result[0] = sdata[0] / (size - 1);
}


template <unsigned int blockSize>
__device__ void regress(data_type** all_data, volatile data_type* result,
                            data_type* cov_mat,
                            const unsigned int dims,
                            const unsigned int fir,
                            const unsigned int sec,
                            const unsigned int size,
                            const unsigned int id, cg::thread_block cta){

    data_type temp;

    int j = id;
    temp = pow((1. - pow(cov_mat[fir * dims + sec], 2)), 0.5);
    while (j < size){
        result[j] = (all_data[fir][j] - cov_mat[fir * dims + sec] * all_data[sec][j]) / temp;
        j += blockSize;
    }
}

template <unsigned int blockSize>
__device__ void regress_Norm_V1(data_type* x_i, data_type* x_j,
                            volatile data_type* result,
                            volatile data_type* sdata,
                            const bool base,
                            const unsigned int size,
                            const unsigned int id, cg::thread_block cta){

    data_type temp;
    data_type cov_value;
    data_type var_j;
    data_type var_value;
    int j = id;

    covariance<512>(x_i, x_j, sdata, base, size, id, &cov_value, cta);
    variance<512>(x_j, sdata, size, id, &var_j, cta);

    if (base){
        temp = pow((1. - pow(cov_value,2)), 0.5);
        while (j < size){
            result[j] = (x_i[j] - cov_value * x_j[j])/temp;
            j += blockSize;
        }
    }
    else{
        while (j < size){
            result[j] = (x_i[j] - cov_value/var_j * x_j[j]);
            j += blockSize;
        }
        cg::sync(cta);
        variance<512>(result, sdata, size, id, &var_value, cta);
        temp = pow(var_value, 0.5);
        j = id;
        while (j < size){
            result[j] /= temp;
            j += blockSize;
        }
    }
}


template <unsigned int blockSize>
__device__ void normalize(data_type* data,
                            volatile data_type* out,
                            volatile data_type* sdata,
                            const unsigned int size,
                            const unsigned int id, cg::thread_block cta){

    int i = id;

    data_type mean = 0;
    data_type std_cal;
    data_type mySum = 0;

    // calculating mean
    while (i < size){
        mySum += data[i];
        i += blockSize;
    }

    reduceBlock<512>(sdata, mySum, id, cta);

    mean = sdata[0] / size;
    i = id;
    mySum = 0;
    while (i < size){
        mySum += pow((data[i] - mean), 2);
        i += blockSize;
    }

    reduceBlock<512>(sdata, mySum, id, cta);
    std_cal = sqrt(sdata[0]/data_type(size-1));

    i = id;
    while (i < size){
        out[i] = (data[i] - mean) / std_cal;
        i += blockSize;
    }
}



template <unsigned int blockSize>
__device__ void entropy(volatile data_type *sdata, data_type* dim_data, volatile data_type* result,
                                const unsigned int size,
                                const unsigned int id, cg::thread_block cta){

    const data_type k1 = 79.047;
    const data_type k2 = 7.4129;
    const data_type gamma = 0.37457;
    data_type cal_1;
    data_type cal_2;
    int j = id;


    cal_1 = 0;
    cal_2 = 0;
    while (j < size){
        cal_1 += log(cosh(dim_data[j])) / size;
        cal_2 += dim_data[j] * (exp(-0.5* pow(dim_data[j], 2))) / size;
        j += blockSize;
    }
    reduceBlock<512>(sdata, cal_1, id, cta);
    if(id == 0){
        cal_1 = sdata[0];
    }
    cg::sync(cta);
    reduceBlock<512>(sdata, cal_2, id, cta);

    if(id == 0){
        cal_1 = cal_1;
        cal_2 = sdata[0];
        result[0] = (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
    }
}

__global__ void normalize_all_device(data_type** X, int* U, int dims, int samples){

    extern __shared__ data_type sdata[];
    cg::thread_block cta = cg::this_thread_block();
    int id  = threadIdx.x;
    int dim = U[blockIdx.x];
    int i = id;
    data_type* data = X[dim];
    data_type mean = 0;
    data_type std_cal;
    data_type mySum = 0;
    int blockSize = blockDim.x;

    // calculating mean
    while (i < samples){
        mySum += data[i];
        i += blockSize;
    }

    // do reduction in shared mem
    reduceBlock<512>(sdata, mySum, id, cta);

    mean = sdata[0] / samples;
    i = id;
    mySum = 0;
    while (i < samples){
        mySum += pow((data[i] - mean), 2);
        i += blockSize;
    }

    reduceBlock<512>(sdata, mySum, id, cta);
    std_cal = sqrt(sdata[0]/data_type(samples-1));

    i = id;
    while (i < samples){
        data[i] = (data[i] - mean) / std_cal;
        i += blockSize;
    }
}


__global__ void normalize_all_device_base(data_type** out, data_type** X, int* U, int dims, int samples){
    extern __shared__ data_type sdata[];
    cg::thread_block cta = cg::this_thread_block();
    int id  = threadIdx.x;
    int dim = U[blockIdx.x];
    int i = id;
    data_type* data = X[dim];
    data_type* dataOut = out[dim];
    data_type mean = 0;
    data_type std_cal;
    data_type mySum = 0;
    int blockSize = blockDim.x;

    // calculating mean
    while (i < samples){
        mySum += data[i];
        i += blockSize;
    }

    // do reduction in shared mem
    reduceBlock<512>(sdata, mySum, id, cta);
    mean = sdata[0] / samples;
    i = id;
    mySum = 0;
    while (i < samples){
        mySum += pow((data[i] - mean), 2);
        i += blockSize;
    }

    reduceBlock<512>(sdata, mySum, id, cta);
    std_cal = sqrt(sdata[0]/data_type(samples-1));

    i = id;
    while (i < samples){
        dataOut[i] = (data[i] - mean) / std_cal;
        i += blockSize;
    }
}

__global__ void calculate_cov_mat_device(data_type* cov_mat, data_type** all_data, int* U, int dims, int samples){
    extern __shared__ data_type sdata[];
    
    int i = 0;
    int j = 0;
    int id  = threadIdx.x;
    int dim = U[blockIdx.x];
    int blockSize = blockDim.x;
    cg::thread_block cta = cg::this_thread_block();
    data_type* data = all_data[dim];
    data_type* dest;
    data_type temp;

    cov_mat[dim * dims + dim] = 1;

    if(blockIdx.x == gridDim.x - 1){
        return;
    }

    for(i = blockIdx.x + 1; i < gridDim.x ; i++){
        dest = all_data[U[i]];
        temp = 0;
        j = id;
        while (j < samples){
            temp += data[j] * dest[j];
            j += blockSize;
        }
        reduceBlock<512>(sdata, temp, id, cta);
        if(id == 0){
            temp = sdata[0] / (samples - 1);
            cov_mat[dim * dims + U[i]] = temp;
            cov_mat[U[i] * dims + dim] = temp;
        }
    }
}
__global__ void update_cov_mat(data_type* cov_mat, int* U, int U_size, int dims, int root){
    cg::thread_block cta = cg::this_thread_block();
    data_type temp;
    data_type temp1;
    data_type cov_1;
    data_type cov_2;
    int dim = U[blockIdx.x];
    int j = U[threadIdx.x];
    if(j == dim){
        temp = 1;
    }
    else{
        cov_1 = cov_mat[dim * dims + root];
        cov_2 = cov_mat[j * dims + root];
        temp = cov_mat[dim * dims + j] - cov_1 * cov_2;
        temp1 = pow((1. - pow(cov_1, 2)), 0.5) * pow((1. - pow(cov_2, 2)), 0.5);
        temp = temp / temp1;
    }
    cg::sync(cta);
    cov_mat[dim * dims + j] = temp;
}

__global__ void regress_root_base(data_type** all_data, data_type* cov_mat, int root,
                                int* U, int U_size,
                                int dims, int samples){
    
    cg::thread_block cta = cg::this_thread_block();
    int dim = U[blockIdx.x];
    int j = 0;
    int id  = threadIdx.x;
    int block_size = blockDim.x;
    
    j = id;
    while (j < samples){
        all_data[dim][j] = (all_data[dim][j] - cov_mat[dim * dims + root] * all_data[root][j]);
        j += block_size;
    }
}

__global__ void regress_root_base_V1(data_type** all_data, int root,
                                bool base,
                                int* U, int U_size,
                                int dims, int samples){
    extern __shared__ data_type sdata[];
    cg::thread_block cta = cg::this_thread_block();
    int dim = U[blockIdx.x];
    int j = 0;
    int id  = threadIdx.x;
    int block_size = blockDim.x;
    data_type cov_value;
    data_type var_j;
 
    covariance<512>(all_data[dim], all_data[root], sdata, base, samples, id, &cov_value, cta);
    variance<512>(all_data[root], sdata, samples, id, &var_j, cta);

    j = id;
    if(base){
        while (j < samples){
            all_data[dim][j] = (all_data[dim][j] - cov_value * all_data[root][j]);
            j += block_size;
        }
    }
    else{
        while (j < samples){
            all_data[dim][j] = (all_data[dim][j] - cov_value/var_j * all_data[root][j]);
            j += block_size;
        }
    }
}


__global__ void calculate_cov_mat_base(data_type* cov_mat, data_type** all_data, int* U, int dims, int samples){
    extern __shared__ data_type sdata[];
    
    int i = 0;
    int j = 0;
    int id  = threadIdx.x;
    int dim = U[blockIdx.x];
    int blockSize = blockDim.x;
    cg::thread_block cta = cg::this_thread_block();
    data_type* data = all_data[dim];
    data_type* dest;
    data_type temp;
    data_type mean_fir;
    data_type mean_sec;

    cov_mat[dim * dims + dim] = 1;

    if(blockIdx.x == gridDim.x - 1){
        return;
    }

    for(i = blockIdx.x + 1; i < gridDim.x ; i++){
        dest = all_data[U[i]];
        temp = 0;
        // calculating mean first
        j = id;
        while (j < samples){
            temp += data[j];
            j += blockSize;
        }
        reduceBlock<512>(sdata, temp, id, cta);
        mean_fir = sdata[0] / samples;

        
        // calculating mean second
        j = id;
        temp = 0;
        while (j < samples){
            temp += dest[j];
            j += blockSize;
        }
        reduceBlock<512>(sdata, temp, id, cta);
        mean_sec = sdata[0] / samples;
        j = id;
        temp = 0;
        while (j < samples){
            temp += (data[j] - mean_fir) * (dest[j] - mean_sec);
            j += blockSize;
        }
        reduceBlock<512>(sdata, temp, id, cta);
        if(id == 0){
            temp = sdata[0] / (samples - 1);
            if(temp > 0.99){
                temp = 0.99;
            }
            if(temp < -0.99){
                temp = -0.99;
            }
            cov_mat[dim * dims + U[i]] = temp;
            cov_mat[U[i] * dims + dim] = temp;
        }
    }
}


__global__ void regress_root(data_type** all_data, data_type* cov_mat, int root,
                                int* U, int U_size,
                                int dims, int samples){
    
    cg::thread_block cta = cg::this_thread_block();
    int dim = U[blockIdx.x];
    int j = 0;
    int id  = threadIdx.x;
    int block_size = blockDim.x;
    data_type temp;
    

    j = id;
    temp = pow((1. - pow(cov_mat[dim * dims + root], 2)), 0.5);
    while (j < samples){
        all_data[dim][j] = (all_data[dim][j] - cov_mat[dim * dims + root] * all_data[root][j]) / temp;
        j += block_size;
    }
}

__global__ void do_all_device_global(data_type** all_data, data_type* cov_mat, data_type* entropy_result, data_type* M_list,
                                int* U, int U_size, int* Uc, int Uc_size, int* Vj, int Vj_size,int dims, int samples,
                                data_type threshold, bool* threshold_flag, bool* threshold_flag_block,
                                bool* entropy_flag, volatile int* entropy_Lock, int* checkpoint,
                                volatile bool* messages, volatile bool* done_dims, volatile data_type* M_all){
    
    __shared__ data_type sdata[512];
    __shared__ bool finish_flag;
    __shared__ bool calc_flag;
    __shared__ int counter;
    cg::thread_block cta = cg::this_thread_block();
    int dim = Uc[blockIdx.x];
    int i = 0;
    int j = 0;
    int id  = threadIdx.x;
    int block_size = blockDim.x;
    int temp_index;
    data_type temp;
    data_type temp_loop;
    data_type* dim_data;
    data_type* data;
    data_type M = 0;
    
    data_type k1 = 79.047;
    data_type k2 = 7.4129;
    data_type gamma = 0.37457;
    data_type cal_1;
    data_type cal_2;
    dim_data = all_data[dim];

    if(id == 0){
        M = -M_list[dim];
        if(M > threshold){
            finish_flag = true;
        }
        else{
            finish_flag = false;
        }
    }
    cg::sync(cta);
    if(finish_flag){
        return;
    }
    else{
        j = id;
        temp = 0;
        while (j < U_size){
            if(messages[blockIdx.x * dims + j]){
                temp += M_all[blockIdx.x * dims + j];
                messages[blockIdx.x * dims + j] = false;
                done_dims[blockIdx.x * dims + j] = true;
            }
            j += block_size;
        }
        reduceBlock<512>(sdata, temp, id, cta);
        if(id == 0){
            M += sdata[0];
        }
    }
    cg::sync(cta);

    if(U[checkpoint[blockIdx.x]] == dim){
        if(id == 0){
            acquire_semaphore(&sem[dim * dims + dim]);
            if (entropy_flag[dim * dims + dim] == false){
                calc_flag = true;
            }
            else{
                calc_flag = false;
                release_semaphore(&sem[dim * dims + dim]);
            }
        }
        cg::sync(cta);
        if (calc_flag){
            i = id;
            cal_1 = 0;
            cal_2 = 0;
            while (i < samples){
                cal_1 += log(cosh(dim_data[i])) / samples;
                cal_2 += dim_data[i] * (exp(-0.5* pow(dim_data[i], 2))) / samples;
                i += block_size;
            }
            reduceBlock<512>(sdata, cal_1, id, cta);
            if(id == 0){
                cal_1 = sdata[0];
            }
            cg::sync(cta);
            reduceBlock<512>(sdata, cal_2, id, cta);
            if(id == 0){
                cal_1 = cal_1;
                cal_2 = sdata[0];
                entropy_result[dim * dims + dim] = (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
                entropy_flag[dim * dims + dim] = true;
                release_semaphore(&sem[dim * dims + dim]);
            }
            __threadfence();
        }
    }
    if(id == 0){
        if(checkpoint[blockIdx.x] == U_size - 1){
            counter = 0;
        }
        else{
            counter = checkpoint[blockIdx.x] + 1;
        }
        while(done_dims[blockIdx.x * dims + counter] && U[counter] != dim){
            if(counter == U_size - 1){
                counter = 0;
            }
            else{
                counter += 1;
            }
        }
    }
    cg::sync(cta);
    while(U[counter] != dim){
        i = counter;
        if(id == 0){
            if(U[i] < dim){
                temp_index = U[i] * dims + dim;
            }
            else{
                temp_index = dim * dims + U[i];
            }
            if(atomicCAS((int*)(&sem[temp_index]), 0, 1) == 0){
                if (entropy_flag[temp_index] == false){
                    calc_flag = true;
                }
                else{
                    calc_flag = false;
                    release_semaphore(&sem[temp_index]);
                }
            }
            else{
                calc_flag = false;
            }
        }
        cg::sync(cta);
        if (calc_flag){
            cal_1 = 0;
            cal_2 = 0;
            if(find(Vj, Vj_size, dim) && find(Vj, Vj_size, U[i])){
                j = id;
                while (j < samples){
                    temp_loop = dim_data[j];
                    all_data[dim * dims + U[i]][j] = temp_loop;
                    cal_1 += log(cosh(temp_loop)) / samples;
                    cal_2 += temp_loop * (exp(-0.5* pow(temp_loop, 2))) / samples;
                    j += block_size;
                }
           }
            else{
                temp = pow((1. - pow(cov_mat[dim * dims + U[i]], 2)), 0.5);
                j = id;
                while (j < samples){
                    temp_loop = (dim_data[j] - cov_mat[dim * dims + U[i]] * all_data[U[i]][j]) / temp;
                    cal_1 += log(cosh(temp_loop)) / samples;
                    cal_2 += temp_loop * (exp(-0.5* pow(temp_loop, 2))) / samples;
                    j += block_size;
                }
            }
            reduceBlock<512>(sdata, cal_1, id, cta);
            if(id == 0){
                cal_1 = sdata[0];
            }
            cg::sync(cta);
            reduceBlock<512>(sdata, cal_2, id, cta);
            if(id == 0){
                cal_1 = cal_1;
                cal_2 = sdata[0];
                entropy_result[dim * dims + U[i]] = (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
                entropy_flag[dim * dims + U[i]] = true;
            }
            __threadfence();

            cal_1 = 0;
            cal_2 = 0;
            if(find(Vj, Vj_size, U[i]) && find(Vj, Vj_size, dim)){
                j = id;
                while (j < samples){
                    temp_loop = all_data[U[i] * dims + U[i]][j];
                    all_data[U[i] * dims + dim][j] = temp_loop;
                    cal_1 += log(cosh(temp_loop)) / samples;
                    cal_2 += temp_loop * (exp(-0.5* pow(temp_loop, 2))) / samples;
                    j += block_size;
                }
            }
            else{
                temp = pow((1. - pow(cov_mat[U[i] * dims + dim], 2)), 0.5);
                j = id;
                while (j < samples){
                    temp_loop = (all_data[U[i]][j] - cov_mat[U[i] * dims +dim] * dim_data[j]) / temp;
                    cal_1 += log(cosh(temp_loop)) / samples;
                    cal_2 += temp_loop * (exp(-0.5* pow(temp_loop, 2))) / samples;
                    j += block_size;
                }
            }
            reduceBlock<512>(sdata, cal_1, id, cta);
            if(id == 0){
                cal_1 = sdata[0];
            }
            cg::sync(cta);
            reduceBlock<512>(sdata, cal_2, id, cta);
            if(id == 0){
                cal_1 = cal_1;
                cal_2 = sdata[0];
                entropy_result[U[i] * dims + dim] = (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
                entropy_flag[U[i] * dims + dim] = true;
                release_semaphore(&sem[temp_index]);
            }
            __threadfence();

            if(id == 0){
                acquire_semaphore(&sem[U[i]* dims + U[i]]);
                if (entropy_flag[U[i]* dims + U[i]] == false){
                    calc_flag = true;
                }
                else{
                    calc_flag = false;
                    release_semaphore(&sem[U[i]* dims + U[i]]);
                }
            }
            cg::sync(cta);
            if(calc_flag){
                cal_1 = 0;
                cal_2 = 0;
                data = all_data[U[i]];
                j = id;
                while (j < samples){
                    cal_1 += log(cosh(data[j]))/ samples;
                    cal_2 += data[j] * (exp(-0.5* pow(data[j], 2)))/ samples;
                    j += block_size;
                }
                reduceBlock<512>(sdata, cal_1, id, cta);
                if(id == 0){
                    cal_1 = sdata[0];
                }
                cg::sync(cta);
                reduceBlock<512>(sdata, cal_2, id, cta);
                if(id == 0){
                    cal_1 = cal_1;
                    cal_2 = sdata[0];
                    entropy_result[U[i] * dims + U[i]] = (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
                    entropy_flag[U[i] * dims + U[i]] = true;
                    release_semaphore(&sem[U[i]* dims + U[i]]);
                }
                __threadfence();
           }
            if(id == 0){
                temp = entropy_result[U[i] * dims + U[i]] + entropy_result[dim * dims + U[i]] -
                            (entropy_result[dim * dims + dim] + entropy_result[U[i]* dims + dim]);
                //M_all is zero by default
                if(temp < 0){
                    M += pow(temp,2);
                }
                else{
                    M_all[i * dims + blockIdx.x] = pow(temp,2);
                }

                done_dims[blockIdx.x * dims + i] = true;
                messages[i * dims + blockIdx.x] = true;
                if(M > threshold){
                    finish_flag = true;
                }
            }
            __threadfence();
            cg::sync(cta);
            if(finish_flag){
                break;
            }
        }

        j = id;
        temp = 0;
        while (j < U_size){
            if(messages[blockIdx.x * dims + j]){
                temp += M_all[blockIdx.x * dims + j];
                messages[blockIdx.x * dims + j] = false;
                done_dims[blockIdx.x * dims + j] = true;
            }
            j += block_size;
        }
        reduceBlock<512>(sdata, temp, id, cta);
        if(id == 0){
            M += sdata[0];
        }
        if(id == 0){
            do{
                if(counter == U_size - 1){
                    counter = 0;
                }
                else{
                    counter += 1;
                }
            } while(done_dims[blockIdx.x * dims + counter] && U[counter] != dim);
        }
        cg::sync(cta);

    }
    if(id == 0){
        if(finish_flag){
            checkpoint[blockIdx.x] = counter;
        }
        else{
            for(i = 0; i < U_size; i++){
                if(done_dims[blockIdx.x * dims + i]== false && i != blockIdx.x){
                    while(!messages[blockIdx.x * dims + i]);
                    M += M_all[blockIdx.x * dims + i];
                    done_dims[blockIdx.x * dims + i]== true;
                }
            }
            if(M > threshold){
                finish_flag = true;
                checkpoint[blockIdx.x] = counter;
            }
            else {
                threshold_flag[0] = 1;
            }
        }
        M_list[dim] = -1.0 * M;
    }
}

__global__ void do_all_device_Block_Worker_V1(data_type** all_data,
                                            data_type** x_i, data_type** x_j,
                                            data_type** ri_j, data_type** rj_i,
                                            data_type* cov_mat,
                                            data_type* M_list,
                                            bool base,
                                            int* U, int U_size,
                                            int* Uc, int Uc_size,
                                            int* Vj, int Vj_size,
                                            int dims, int samples) {
    __shared__ data_type sdata[512];
    __shared__ bool skip_flag;
    cg::thread_block cta = cg::this_thread_block();
    int dim = Uc[blockIdx.x];
    int i = 0;
    int id  = threadIdx.x;

    data_type temp;
    data_type entropy_i = 0;
    data_type entropy_ij = 0;
    data_type entropy_ji = 0;
    data_type entropy_j = 0;
    data_type M = 0;

    for (i = 0; i < U_size; i++){
        
        if(id == 0){
            skip_flag = false;
            if(U[i] == dim){
                skip_flag = true;
            }
        }
        cg::sync(cta);
        
        if(skip_flag){
            continue;
        }
        normalize<512> (all_data[dim], x_i[dim], sdata, samples, id, cta);
        normalize<512> (all_data[U[i]], x_j[dim], sdata, samples, id, cta);

        regress_Norm_V1<512> (x_i[dim], x_j[dim], ri_j[dim], sdata, base, samples, id, cta);
        regress_Norm_V1<512> (x_j[dim], x_i[dim], rj_i[dim], sdata, base, samples, id, cta);

        entropy<512> (sdata, x_i[dim], &entropy_i, samples, id, cta);
        if(find(Vj, Vj_size, dim) && find(Vj, Vj_size, U[i])){
            entropy<512> (sdata, x_i[dim], &entropy_ij, samples, id, cta);
        }
        else{
            entropy<512> (sdata, ri_j[dim], &entropy_ij, samples, id, cta);
        }
        if(find(Vj, Vj_size, U[i]) && find(Vj, Vj_size, dim)){
            entropy<512> (sdata, x_j[dim], &entropy_ij, samples, id, cta);
        }
        else{
            entropy<512> (sdata, rj_i[dim], &entropy_ji, samples, id, cta);
        }
        entropy<512> (sdata, x_j[dim], &entropy_j, samples, id, cta);

        if(id == 0){       
            temp = entropy_j + entropy_ij - (entropy_i + entropy_ji);
            if(temp < 0){
                M += pow(temp,2);
            }
        }
    }
    if(id == 0){
        M_list[dim] = -1.0 * M;
    }
}

__global__ void do_all_device_Thread_Worker_V1(data_type** all_data,
                                            data_type* cov_mat,
                                            data_type* M_list,
                                            bool base,
                                            int* U, int U_size,
                                            int* Uc, int Uc_size,
                                            int* Vj, int Vj_size,
                                            int dims, int samples) {
    __shared__ data_type sdata[512];
    cg::thread_block cta = cg::this_thread_block();
    int dim = Uc[blockIdx.x];
    int id  = threadIdx.x;
    int i = id;
    int j = 0;

    data_type temp;
    data_type temp_loop;

    data_type sum_fir;
    data_type sum_sec;
    data_type mean_fir;
    data_type mean_sec;
    data_type std_fir;
    data_type std_sec;
    data_type cov_value;
    data_type temp_fir;
    data_type temp_sec;
    data_type mean_var;

    data_type entropy_i = 0;
    data_type entropy_ij = 0;
    data_type entropy_ji = 0;
    data_type entropy_j = 0;
    data_type M = 0;

    const data_type k1 = 79.047;
    const data_type k2 = 7.4129;
    const data_type gamma = 0.37457;
    data_type cal_1;
    data_type cal_2;
    

    while (i < U_size){
        if(i == blockIdx.x){
            i += 512;
            continue;
        }
        // we have to change calculation order to fit it in memory
        // calculating mean, variance, convariance
        sum_fir = 0;
        sum_sec = 0;
        for (j = 0; j < samples; j++){
            sum_fir += all_data[dim][j];
            sum_sec += all_data[U[i]][j];
        }
        mean_fir = sum_fir / samples;
        mean_sec = sum_sec / samples;

        sum_fir = 0;
        sum_sec = 0;
        for (j = 0; j < samples; j++){
            sum_fir += pow(all_data[dim][j] - mean_fir, 2);
            sum_sec += pow(all_data[U[i]][j] - mean_sec, 2);
        }
        std_fir = pow(sum_fir / (samples - 1), 0.5);
        std_sec = pow(sum_sec / (samples - 1), 0.5);
        
        sum_fir = 0;
        for (j = 0; j < samples; j++){
            sum_fir += ((all_data[dim][j] - mean_fir) / std_fir) * ((all_data[U[i]][j] - mean_sec) / std_sec);
        }
        
        cov_value = sum_fir / (samples - 1);
        if(base){
            if(cov_value > 0.99){
                cov_value = 0.99;
            }
            else if (cov_value < -0.99){
                cov_value = -0.99;
            }
        }

        // entropy i
        cal_1 = 0;
        cal_2 = 0;
        for (j = 0; j < samples; j++){
            temp_fir = (all_data[dim][j] - mean_fir) / std_fir;
            cal_1 += log(cosh(temp_fir)) / samples;
            cal_2 += temp_fir * (exp(-0.5* pow(temp_fir, 2))) / samples;
        }
        entropy_i = (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);

        // entropy ij
        cal_1 = 0;
        cal_2 = 0;
        if(find(Vj, Vj_size, dim) && find(Vj, Vj_size, U[i])){
            for (j = 0; j < samples; j++){
                temp_fir = (all_data[dim][j] - mean_fir) / std_fir;
                cal_1 += log(cosh(temp_fir)) / samples;
                cal_2 += temp_fir * (exp(-0.5* pow(temp_fir, 2))) / samples;
            }
        }
        else{
            // caluclating residual
            if(base){
                temp = pow((1. - pow(cov_value, 2)), 0.5);
                for (j = 0; j < samples; j++){
                    temp_fir = (all_data[dim][j] - mean_fir) / std_fir;
                    temp_sec = (all_data[U[i]][j] - mean_sec) / std_sec;

                    temp_loop = (temp_fir - cov_value * temp_sec) / temp;
                    cal_1 += log(cosh(temp_loop)) / samples;
                    cal_2 += temp_loop * (exp(-0.5* pow(temp_loop, 2))) / samples;
                }
            }
            else{
                sum_fir = 0;
                for (j = 0; j < samples; j++){
                    temp_fir = (all_data[dim][j] - mean_fir) / std_fir;
                    temp_sec = (all_data[U[i]][j] - mean_sec) / std_sec;
                    sum_fir += (temp_fir - cov_value * temp_sec);
                }
                mean_var = sum_fir / samples;

                sum_fir = 0;
                for (j = 0; j < samples; j++){
                    temp_fir = (all_data[dim][j] - mean_fir) / std_fir;
                    temp_sec = (all_data[U[i]][j] - mean_sec) / std_sec;
                    temp_loop = (temp_fir - cov_value * temp_sec);
                    sum_fir += pow(temp_loop - mean_var, 2);
                }
                temp = pow(sum_fir / (samples - 1), 0.5);

                for (j = 0; j < samples; j++){
                    temp_fir = (all_data[dim][j] - mean_fir) / std_fir;
                    temp_sec = (all_data[U[i]][j] - mean_sec) / std_sec;

                    temp_loop = (temp_fir - cov_value * temp_sec)/temp;
                    cal_1 += log(cosh(temp_loop)) / samples;
                    cal_2 += temp_loop * (exp(-0.5* pow(temp_loop, 2))) / samples;
                }
            }
        }
        entropy_ij = (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);

        // entropy ji
        cal_1 = 0;
        cal_2 = 0;
        if(find(Vj, Vj_size, U[i]) && find(Vj, Vj_size, dim)){
            for (j = 0; j < samples; j++){
                temp_fir = (all_data[U[i]][j] - mean_sec) / std_sec;
                cal_1 += log(cosh(temp_fir)) / samples;
                cal_2 += temp_fir * (exp(-0.5* pow(temp_fir, 2))) / samples;
            }
        }
        else{
            // caluclating residual
            if(base){
                temp = pow((1. - pow(cov_value, 2)), 0.5);
                for (j = 0; j < samples; j++){
                    temp_fir = (all_data[U[i]][j] - mean_sec) / std_sec;
                    temp_sec = (all_data[dim][j] - mean_fir) / std_fir;

                    temp_loop = (temp_fir - cov_value * temp_sec) / temp;
                    cal_1 += log(cosh(temp_loop)) / samples;
                    cal_2 += temp_loop * (exp(-0.5* pow(temp_loop, 2))) / samples;
                }
            }
            else{
                sum_fir = 0;
                for (j = 0; j < samples; j++){
                    temp_fir = (all_data[U[i]][j] - mean_sec) / std_sec;
                    temp_sec = (all_data[dim][j] - mean_fir) / std_fir;
                    sum_fir += (temp_fir - cov_value * temp_sec);
                }
                mean_var = sum_fir / samples;

                sum_fir = 0;
                for (j = 0; j < samples; j++){
                    temp_fir = (all_data[U[i]][j] - mean_sec) / std_sec;
                    temp_sec = (all_data[dim][j] - mean_fir) / std_fir;
                    temp_loop = (temp_fir - cov_value * temp_sec);
                    sum_fir += pow(temp_loop - mean_var, 2);
                }
                temp = pow(sum_fir / (samples - 1), 0.5);

                for (j = 0; j < samples; j++){
                    temp_fir = (all_data[U[i]][j] - mean_sec) / std_sec;
                    temp_sec = (all_data[dim][j] - mean_fir) / std_fir;

                    temp_loop = (temp_fir - cov_value * temp_sec)/temp;
                    cal_1 += log(cosh(temp_loop)) / samples;
                    cal_2 += temp_loop * (exp(-0.5* pow(temp_loop, 2))) / samples;
                }
            }
        }
        entropy_ji = (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
        
        // entropy j
        cal_1 = 0;
        cal_2 = 0;
        for (j = 0; j < samples; j++){
            temp_sec = (all_data[U[i]][j] - mean_sec) / std_sec;
            cal_1 += log(cosh(temp_sec)) / samples;
            cal_2 += temp_sec * (exp(-0.5* pow(temp_sec, 2))) / samples;
        }
        entropy_j = (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
        
        temp = entropy_j + entropy_ij - (entropy_i + entropy_ji);
        if(temp < 0){
            M += pow(temp,2);
        }
        i += 512;
    }
    cg::sync(cta);
    reduceBlock<512>(sdata, M, id, cta);
    if(id == 0){
        M_list[dim] = -1.0 * sdata[0];
    }
}

__global__ void do_all_device_Block_Compare_V1(data_type** all_data,
                                                data_type* cov_mat,
                                                data_type* M_all,
                                                bool base,
                                                int* U, int U_size,
                                                int* Uc, int Uc_size,
                                                int* Vj, int Vj_size,
                                                int dims, int samples){
    __shared__ data_type sdata[512];
    cg::thread_block cta = cg::this_thread_block();
    int fir = Uc[blockIdx.x];
    int sec = Uc[blockIdx.y];
    if (fir == sec){
        return;
    }

    int id  = threadIdx.x;
    int j = id;

    data_type temp;
    data_type temp_loop;

    data_type sum_fir;
    data_type sum_sec;
    data_type mean_fir;
    data_type mean_sec;
    data_type std_fir;
    data_type std_sec;
    data_type cov_value;
    data_type temp_fir;
    data_type temp_sec;
    data_type mean_var;

    data_type entropy_i = 0;
    data_type entropy_ij = 0;
    data_type entropy_ji = 0;
    data_type entropy_j = 0;

    const data_type k1 = 79.047;
    const data_type k2 = 7.4129;
    const data_type gamma = 0.37457;
    data_type cal_1;
    data_type cal_2;
    
   // calculating mean, variance, convariance
   sum_fir = 0;
   sum_sec = 0;
   for (j = id; j < samples; j+=512){
       sum_fir += all_data[fir][j];
       sum_sec += all_data[sec][j];
   }
   reduceBlock<512>(sdata, sum_fir, id, cta);
   mean_fir = sdata[0] / samples;
   cg::sync(cta);
   reduceBlock<512>(sdata, sum_sec, id, cta);
   mean_sec = sdata[0] / samples;

   sum_fir = 0;
   sum_sec = 0;
   for (j = id; j < samples; j+=512){
       sum_fir += pow(all_data[fir][j] - mean_fir, 2);
       sum_sec += pow(all_data[sec][j] - mean_sec, 2);
   }
   reduceBlock<512>(sdata, sum_fir, id, cta);
   std_fir = pow(sdata[0] / (samples - 1), 0.5);
   cg::sync(cta);
   reduceBlock<512>(sdata, sum_sec, id, cta);
   std_sec = pow(sdata[0] / (samples - 1), 0.5);
   
   sum_fir = 0;
   for (j = id; j < samples; j+=512){
       sum_fir += ((all_data[fir][j] - mean_fir) / std_fir) * ((all_data[sec][j] - mean_sec) / std_sec);
   }
   reduceBlock<512>(sdata, sum_fir, id, cta);
   cov_value = sdata[0] / (samples - 1);
   if(base){
       if(cov_value > 0.99){
           cov_value = 0.99;
       }
       else if (cov_value < -0.99){
           cov_value = -0.99;
       }
   }
    cal_1 = 0;
    cal_2 = 0;
    for (j = id; j < samples; j+=512){
        temp_fir = (all_data[fir][j] - mean_fir) / std_fir;
            cal_1 += log(cosh(temp_fir)) / samples;
            cal_2 += temp_fir * (exp(-0.5* pow(temp_fir, 2))) / samples;
    }
    reduceBlock<512>(sdata, cal_1, id, cta);
    if(id == 0){
        cal_1 = sdata[0];
    }
    cg::sync(cta);
    reduceBlock<512>(sdata, cal_2, id, cta);

    if(id == 0){
        cal_2 = sdata[0];
        entropy_i = (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
    }

    // entropy ij
    cal_1 = 0;
    cal_2 = 0;
    if(find(Vj, Vj_size, fir) && find(Vj, Vj_size, sec)){
        for (j = id; j < samples; j+=512){
            temp_fir = (all_data[fir][j] - mean_fir) / std_fir;
            cal_1 += log(cosh(temp_fir)) / samples;
            cal_2 += temp_fir * (exp(-0.5* pow(temp_fir, 2))) / samples;
        }
    }
    else{
        // caluclating residual
        if(base){
            temp = pow((1. - pow(cov_value, 2)), 0.5);
            for (j = id; j < samples; j+=512){
                temp_fir = (all_data[fir][j] - mean_fir) / std_fir;
                temp_sec = (all_data[sec][j] - mean_sec) / std_sec;

                temp_loop = (temp_fir - cov_value * temp_sec) / temp;
                cal_1 += log(cosh(temp_loop)) / samples;
                cal_2 += temp_loop * (exp(-0.5* pow(temp_loop, 2))) / samples;
            }
        }
        else{
            sum_fir = 0;
            for (j = id; j < samples; j+=512){
                temp_fir = (all_data[fir][j] - mean_fir) / std_fir;
                temp_sec = (all_data[sec][j] - mean_sec) / std_sec;
                sum_fir += (temp_fir - cov_value * temp_sec);
            }
            reduceBlock<512>(sdata, sum_fir, id, cta);
            mean_var = sdata[0] / samples;

            sum_fir = 0;
            for (j = id; j < samples; j+=512){
                temp_fir = (all_data[fir][j] - mean_fir) / std_fir;
                temp_sec = (all_data[sec][j] - mean_sec) / std_sec;
                temp_loop = (temp_fir - cov_value * temp_sec);
                sum_fir += pow(temp_loop - mean_var, 2);
            }
            reduceBlock<512>(sdata, sum_fir, id, cta);
            temp = pow(sdata[0] / (samples - 1), 0.5);

            for (j = id; j < samples; j+=512){
                temp_fir = (all_data[fir][j] - mean_fir) / std_fir;
                temp_sec = (all_data[sec][j] - mean_sec) / std_sec;

                temp_loop = (temp_fir - cov_value * temp_sec)/temp;
                cal_1 += log(cosh(temp_loop)) / samples;
                cal_2 += temp_loop * (exp(-0.5* pow(temp_loop, 2))) / samples;
            }
        }
    }
    reduceBlock<512>(sdata, cal_1, id, cta);
    if(id == 0){
        cal_1 = sdata[0];
    }
    cg::sync(cta);
    reduceBlock<512>(sdata, cal_2, id, cta);

    if(id == 0){
        cal_1 = cal_1;
        cal_2 = sdata[0];
        entropy_ij = (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
    }

    // entropy ji
    cal_1 = 0;
    cal_2 = 0;
    if(find(Vj, Vj_size, sec) && find(Vj, Vj_size, fir)){
        for (j = id; j < samples; j+=512){
            temp_fir = (all_data[sec][j] - mean_sec) / std_sec;
            cal_1 += log(cosh(temp_fir)) / samples;
            cal_2 += temp_fir * (exp(-0.5* pow(temp_fir, 2))) / samples;
        }
    }
    else{
        // caluclating residual
        if(base){
            temp = pow((1. - pow(cov_value, 2)), 0.5);
            for (j = id; j < samples; j+=512){
                temp_fir = (all_data[sec][j] - mean_sec) / std_sec;
                temp_sec = (all_data[fir][j] - mean_fir) / std_fir;

                temp_loop = (temp_fir - cov_value * temp_sec) / temp;
                cal_1 += log(cosh(temp_loop)) / samples;
                cal_2 += temp_loop * (exp(-0.5* pow(temp_loop, 2))) / samples;
            }
        }
        else{
            sum_fir = 0;
            for (j = id; j < samples; j+=512){
                temp_fir = (all_data[sec][j] - mean_sec) / std_sec;
                temp_sec = (all_data[fir][j] - mean_fir) / std_fir;
                sum_fir += (temp_fir - cov_value * temp_sec);
            }
            reduceBlock<512>(sdata, sum_fir, id, cta);
            mean_var = sdata[0] / samples;

            sum_fir = 0;
            for (j = id; j < samples; j+=512){
                temp_fir = (all_data[sec][j] - mean_sec) / std_sec;
                temp_sec = (all_data[fir][j] - mean_fir) / std_fir;
                temp_loop = (temp_fir - cov_value * temp_sec);
                sum_fir += pow(temp_loop - mean_var, 2);
            }
            reduceBlock<512>(sdata, sum_fir, id, cta);
            temp = pow(sdata[0] / (samples - 1), 0.5);

            for (j = id; j < samples; j+=512){
                temp_fir = (all_data[sec][j] - mean_sec) / std_sec;
                temp_sec = (all_data[fir][j] - mean_fir) / std_fir;

                temp_loop = (temp_fir - cov_value * temp_sec)/temp;
                cal_1 += log(cosh(temp_loop)) / samples;
                cal_2 += temp_loop * (exp(-0.5* pow(temp_loop, 2))) / samples;
            }
        }
    }
    reduceBlock<512>(sdata, cal_1, id, cta);
    if(id == 0){
        cal_1 = sdata[0];
    }
    cg::sync(cta);
    reduceBlock<512>(sdata, cal_2, id, cta);

    if(id == 0){
        cal_1 = cal_1;
        cal_2 = sdata[0];
        entropy_ji = (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
    }
    
    // entropy j
    cal_1 = 0;
    cal_2 = 0;
    for (j = id; j < samples; j+=512){
        temp_sec = (all_data[sec][j] - mean_sec) / std_sec;
        cal_1 += log(cosh(temp_sec)) / samples;
        cal_2 += temp_sec * (exp(-0.5* pow(temp_sec, 2))) / samples;
    }
    reduceBlock<512>(sdata, cal_1, id, cta);
    if(id == 0){
        cal_1 = sdata[0];
    }
    cg::sync(cta);
    reduceBlock<512>(sdata, cal_2, id, cta);

    if(id == 0){
        cal_1 = cal_1;
        cal_2 = sdata[0];
        entropy_j = (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
        temp = entropy_j + entropy_ij - (entropy_i + entropy_ji);
        if(temp < 0){
            M_all[fir*dims + sec] = pow(temp,2);
        }
        else{
            M_all[fir*dims + sec] = 0;
        }
    }
}


__global__ void reduce_Block_Comapare (data_type* M_all, data_type* M_list,
                                        int* U, int U_size,
                                        int* Uc, int Uc_size,
                                        int dims, int samples){
    
    __shared__ data_type sdata[512];
    cg::thread_block cta = cg::this_thread_block();

    int id = threadIdx.x;
    int j = id;
    data_type sum = 0;

    for (j = id; j < U_size; j+=512){
        sum += M_all[U[blockIdx.x] * dims + U[j]];
    }
    reduceBlock<512>(sdata, sum, id, cta);
    if(id == 0){
        M_list[U[blockIdx.x]] = -sdata[0];
    }
}

__device__ bool find(int* list, int size, int id){
    int i = 0;
    for(i = 0; i < size; i++){
        if( list[i] == id){
            return true;
        }
    }
    return false;
}
