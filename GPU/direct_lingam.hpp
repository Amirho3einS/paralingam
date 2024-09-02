#ifndef DIRECT_LINGAM_HPP
#define DIRECT_LINGAM_HPP

#include <iostream>
#include <vector>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <chrono>
#include "gpu_kernels.h"

#define data_type double

using namespace std;

class direct_lingam{
private:
    int dims = 0;
    int samples = 0;
    vector<int> causal_order;
    data_type threshold;
    bool base = false;
    bool verbose = true;
    vector<float> saved_comparisons;
    vector<float> runtimes;
    unsigned long long int num_compare = 0;
    unsigned long long int num_perform = 0;

    data_type update_rate = 2;

    data_type** allocate_all_data();
    vector<data_type**> allocate_all_data_device();
    void deallocate_all_data(data_type** all_data);

    data_type** allocate_dim_dim();
    data_type* allocate_dim_dim_device();
    void deallocate_dim_dim(data_type** cov_mat);
public:
    direct_lingam();
    ~direct_lingam();

    vector<int> fit(vector<vector<data_type>> X);
    vector<int> fit_Block_Worker_V1(vector<vector<data_type>> X);
    vector<int> fit_Thread_Worker_V1(vector<vector<data_type>> X);
    vector<int> fit_Block_Compare_V1(vector<vector<data_type>> X);
    vector<vector<int>> search_candidate(vector<int> U);
    int search_causal_order(data_type** &all_data, data_type** all_data_device,
                                data_type** &cov_mat, data_type* cov_mat_device,
                                data_type** &entropy_result, data_type* entropy_result_device,
                                vector<int> U, data_type* M_list_device,
                                int* U_device, int* Uc_device, int* Vj_device,
                                bool* threshold_flag, bool* threshold_flag_block,
                                bool* entropy_flag_device, volatile int* entropy_Lock,int* checkpoint,
                                bool* messages, bool* done_dims, data_type* M_all);

    int search_causal_order_Block_Worker_V1(data_type** &all_data, data_type** all_data_device,
                                data_type** &x_i_device, data_type** &x_j_device,
                                data_type** &ri_j_device, data_type** &rj_i_device,
                                data_type* cov_mat_device, 
                                vector<int> U, data_type* M_list_device,
                                int* U_device, int* Uc_device, int* Vj_device);

    int search_causal_order_Thread_Worker_V1(data_type** &all_data, data_type** all_data_device,
                                data_type* cov_mat_device,
                                vector<int> U, data_type* M_list_device,
                                int* U_device, int* Uc_device, int* Vj_device);

    int search_causal_order_Block_Compare_V1(data_type** &all_data, data_type** all_data_device,
                                        data_type* cov_mat_device,
                                        vector<int> U, data_type* M_list_device, int* U_device,
                                        int* Uc_device, int* Vj_device, data_type* M_all);
                                
    void make_checkpoint(int* result, vector<int> Uc, vector<int> U);
    void set_update_rate(data_type rate);
    void set_base_mode(bool base);
    void set_verbose_mode(bool verbose);
    void reset_saved_comparisons();
    float average_saved_comparisons();
    void reset_runtimes();
    float average_runtimes();

};

#endif