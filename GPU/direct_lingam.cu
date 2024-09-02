#include "direct_lingam.hpp"

direct_lingam::direct_lingam(/* args */){
}

direct_lingam::~direct_lingam(){
}


vector<int> direct_lingam::fit(vector<vector<data_type>> X){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    threshold = 1e-7; 
    dims = X.size();
    vector<data_type> temp = *X.begin();
    samples = temp.size();
    setbuf(stdout, NULL);
    causal_order.clear();
    num_compare = 0;
    num_perform = 0;
    data_type** cov_mat = NULL;
    data_type* cov_mat_device = allocate_dim_dim_device();
    data_type** all_data = allocate_all_data();
    vector<data_type**> device_temp = allocate_all_data_device();
    data_type** all_data_device = device_temp[0];
    data_type** all_data_device_copy = device_temp[1];
    data_type* all_device_main = (data_type*)device_temp[2];
    data_type** entropy_result = NULL;
    data_type** all_data_device_base = NULL;
    data_type* all_device_main_base = NULL;
    if(base){
        device_temp = allocate_all_data_device();
        all_data_device_base = device_temp[0];
        all_device_main_base = (data_type*)device_temp[2];
    }
    data_type* entropy_result_device = allocate_dim_dim_device();
    
    bool* entropy_flag_device;
    volatile int* entropy_Lock;
    data_type* M_list_device;
    int* U_device;
    int* Uc_device;
    int* Vj_device;
    bool* threshold_flag;
    bool* threshold_flag_block;
    int* checkpoint;
    bool* messages;
    bool* done_dims;
    data_type* M_all;
    HANDLE_ERROR(cudaMalloc((void**)&entropy_flag_device, dims * dims * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc((void**)&entropy_Lock, dims * dims * sizeof(volatile int)));
    HANDLE_ERROR(cudaMalloc((void**)&M_list_device, dims * sizeof(data_type)));
    HANDLE_ERROR(cudaMalloc((void**)&U_device, dims * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&Uc_device, dims * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&Vj_device, dims * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&threshold_flag, sizeof(bool)));
    HANDLE_ERROR(cudaMalloc((void**)&threshold_flag_block, dims * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc((void**)&checkpoint, dims * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&messages, dims * dims * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc((void**)&done_dims, dims * dims * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc((void**)&M_all, dims * dims * sizeof(data_type)));

    for(int i = 0; i < dims; i++){
        copy(X[i].begin(), X[i].end(), all_data[i]);
    }

    for(int i = 0; i < dims; i++){
        HANDLE_ERROR(cudaMemcpy(all_data_device_copy[i], all_data[i],
            samples * sizeof(data_type), cudaMemcpyHostToDevice));
    }

    vector<int> U(dims);
    std::iota(U.begin(), U.end(), 0);
    int grid_size = U.size();
    int block_size = 512;
    int smemSize = block_size * sizeof(data_type);
    int* temp_U = new int[dims];
    copy(U.begin(), U.end(), temp_U);
    HANDLE_ERROR(cudaMemcpy(U_device, temp_U, U.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    if(!base){
        normalize_all_device <<< grid_size, block_size, smemSize>>> (all_data_device, U_device, dims, samples);
        calculate_cov_mat_device <<<grid_size, block_size, smemSize>>>(cov_mat_device, all_data_device, U_device, dims, samples);
    }
    
    for (int dim = 0; dim < dims-1; dim++){
        threshold = 1e-7;
        int root = search_causal_order(all_data_device_base, all_data_device, cov_mat, cov_mat_device,
                                        entropy_result, entropy_result_device, U,
                                        M_list_device, U_device, Uc_device, Vj_device,
                                        threshold_flag, threshold_flag_block,
                                        entropy_flag_device, entropy_Lock, checkpoint,
                                        messages, done_dims, M_all);


        U.erase(std::remove(U.begin(), U.end(), root), U.end());
        copy(U.begin(), U.end(), temp_U);
        HANDLE_ERROR(cudaMemcpy(U_device, temp_U, U.size() * sizeof(int), cudaMemcpyHostToDevice));
        if(base){
            regress_root_base_V1<<<U.size(), block_size, smemSize>>>(all_data_device, root, base, U_device, U.size(), dims, samples);
        }
        else {
            regress_root<<<U.size(), block_size>>>(all_data_device, cov_mat_device, root, U_device, U.size(), dims, samples);
            update_cov_mat<<<U.size(), U.size()>>>(cov_mat_device, U_device, U.size(), dims, root);
        }
        causal_order.push_back(root);
    }
    causal_order.push_back(U[0]);

    deallocate_all_data(all_data);
    delete[] temp_U;
    HANDLE_ERROR(cudaFree(all_device_main));
    HANDLE_ERROR(cudaFree(all_data_device));
    HANDLE_ERROR(cudaFree(all_device_main_base));
    HANDLE_ERROR(cudaFree(all_data_device_base));
    HANDLE_ERROR(cudaFree(cov_mat_device));
    HANDLE_ERROR(cudaFree(M_list_device));
    HANDLE_ERROR(cudaFree(U_device));
    HANDLE_ERROR(cudaFree(Uc_device));
    HANDLE_ERROR(cudaFree(Vj_device));
    HANDLE_ERROR(cudaFree(threshold_flag));
    HANDLE_ERROR(cudaFree(threshold_flag_block));
    HANDLE_ERROR(cudaFree(checkpoint));
    HANDLE_ERROR(cudaFree(entropy_flag_device));
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Runtime: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << endl;
    cout << "Percentage of saved comparisons : " << (1 - float(num_perform)/float(num_compare)) * 100 << endl;
    saved_comparisons.push_back((1 - float(num_perform)/float(num_compare)) * 100);
    runtimes.push_back(float(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()));

    if (verbose){
        cout << "causal order : ";
        for (auto i = causal_order.begin(); i!= causal_order.end(); i++){
            std::cout << *i << ' ';
        }
        std::cout << endl;
    }
    return causal_order;
}


vector<int> direct_lingam::fit_Block_Worker_V1(vector<vector<data_type>> X){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    dims = X.size();
    vector<data_type> temp = *X.begin();
    samples = temp.size();
    setbuf(stdout, NULL);
    causal_order.clear();
    data_type* cov_mat_device = allocate_dim_dim_device();
    data_type** all_data = allocate_all_data();

    vector<data_type**> device_temp = allocate_all_data_device();
    data_type** all_data_device = device_temp[0];
    data_type** all_data_device_copy = device_temp[1];
    data_type* all_device_main = (data_type*)device_temp[2];

    device_temp = allocate_all_data_device();
    data_type** ri_j_device = device_temp[0];
    data_type* ri_j_device_main = (data_type*)device_temp[2];

    device_temp = allocate_all_data_device();
    data_type** rj_i_device = device_temp[0];
    data_type* rj_i_device_main = (data_type*)device_temp[2];
    
    device_temp = allocate_all_data_device();
    data_type** x_i_device = device_temp[0];
    data_type* x_i_device_main = (data_type*)device_temp[2];

    device_temp = allocate_all_data_device();
    data_type** x_j_device = device_temp[0];
    data_type* x_j_device_main = (data_type*)device_temp[2];

    data_type* M_list_device;
    int* U_device;
    int* Uc_device;
    int* Vj_device;
    HANDLE_ERROR(cudaMalloc((void**)&M_list_device, dims * sizeof(data_type)));
    HANDLE_ERROR(cudaMalloc((void**)&U_device, dims * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&Uc_device, dims * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&Vj_device, dims * sizeof(int)));


    for(int i = 0; i < dims; i++){
        copy(X[i].begin(), X[i].end(), all_data[i]);
    }

    for(int i = 0; i < dims; i++){
        HANDLE_ERROR(cudaMemcpy(all_data_device_copy[i], all_data[i],
            samples * sizeof(data_type), cudaMemcpyHostToDevice));
    }

    vector<int> U(dims);
    std::iota(U.begin(), U.end(), 0);
    int grid_size = U.size();
    int block_size = 512;
    int smemSize = block_size * sizeof(data_type);
    int* temp_U = new int[dims];
    copy(U.begin(), U.end(), temp_U);
    HANDLE_ERROR(cudaMemcpy(U_device, temp_U, U.size() * sizeof(int), cudaMemcpyHostToDevice));

    for (int dim = 0; dim < dims-1; dim++){
        int root = search_causal_order_Block_Worker_V1(all_data, all_data_device,
                                        x_i_device, x_j_device, ri_j_device, rj_i_device,
                                        cov_mat_device, U, M_list_device,
                                        U_device, Uc_device, Vj_device);

        U.erase(std::remove(U.begin(), U.end(), root), U.end());
        copy(U.begin(), U.end(), temp_U);
        HANDLE_ERROR(cudaMemcpy(U_device, temp_U, U.size() * sizeof(int), cudaMemcpyHostToDevice));
        regress_root_base_V1<<<U.size(), block_size, smemSize>>>(all_data_device, root, base, U_device, U.size(), dims, samples);
        causal_order.push_back(root);
    }
    causal_order.push_back(U[0]);
    deallocate_all_data(all_data);
    delete[] temp_U;
    HANDLE_ERROR(cudaFree(all_device_main));
    HANDLE_ERROR(cudaFree(all_data_device));
    HANDLE_ERROR(cudaFree(ri_j_device));
    HANDLE_ERROR(cudaFree(ri_j_device_main));
    HANDLE_ERROR(cudaFree(rj_i_device));
    HANDLE_ERROR(cudaFree(rj_i_device_main));
    HANDLE_ERROR(cudaFree(x_j_device));
    HANDLE_ERROR(cudaFree(x_j_device_main));
    HANDLE_ERROR(cudaFree(x_i_device));
    HANDLE_ERROR(cudaFree(x_i_device_main));
    HANDLE_ERROR(cudaFree(cov_mat_device));
    HANDLE_ERROR(cudaFree(M_list_device));
    HANDLE_ERROR(cudaFree(U_device));
    HANDLE_ERROR(cudaFree(Uc_device));
    HANDLE_ERROR(cudaFree(Vj_device));
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Runtime: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << endl;

    if (verbose){
        cout << "Causal order : ";
        for (auto i = causal_order.begin(); i!= causal_order.end(); i++){
            std::cout << *i << ' ';
        }
        std::cout << endl;
    }
    return causal_order;
}


vector<int> direct_lingam::fit_Thread_Worker_V1(vector<vector<data_type>> X){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    dims = X.size();
    vector<data_type> temp = *X.begin();
    samples = temp.size();
    setbuf(stdout, NULL);
    causal_order.clear();
    data_type* cov_mat_device = allocate_dim_dim_device();
    data_type** all_data = allocate_all_data();

    vector<data_type**> device_temp = allocate_all_data_device();
    data_type** all_data_device = device_temp[0];
    data_type** all_data_device_copy = device_temp[1];
    data_type* all_device_main = (data_type*)device_temp[2];
    

    data_type* M_list_device;
    int* U_device;
    int* Uc_device;
    int* Vj_device;
    HANDLE_ERROR(cudaMalloc((void**)&M_list_device, dims * sizeof(data_type)));
    HANDLE_ERROR(cudaMalloc((void**)&U_device, dims * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&Uc_device, dims * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&Vj_device, dims * sizeof(int)));


    for(int i = 0; i < dims; i++){
        copy(X[i].begin(), X[i].end(), all_data[i]);
    }

    for(int i = 0; i < dims; i++){
        HANDLE_ERROR(cudaMemcpy(all_data_device_copy[i], all_data[i],
            samples * sizeof(data_type), cudaMemcpyHostToDevice));
    }

    vector<int> U(dims);
    std::iota(U.begin(), U.end(), 0);
    int grid_size = U.size();
    int block_size = 512;
    int smemSize = block_size * sizeof(data_type);
    int* temp_U = new int[dims];
    copy(U.begin(), U.end(), temp_U);
    HANDLE_ERROR(cudaMemcpy(U_device, temp_U, U.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    for (int dim = 0; dim < dims-1; dim++){
        int root = search_causal_order_Thread_Worker_V1(all_data, all_data_device,
                                        cov_mat_device, U, M_list_device, U_device,
                                        Uc_device, Vj_device);

        U.erase(std::remove(U.begin(), U.end(), root), U.end());
        copy(U.begin(), U.end(), temp_U);
        HANDLE_ERROR(cudaMemcpy(U_device, temp_U, U.size() * sizeof(int), cudaMemcpyHostToDevice));
        regress_root_base_V1<<<U.size(), block_size, smemSize>>>(all_data_device, root, base, U_device, U.size(), dims, samples);
        causal_order.push_back(root);
    }
    causal_order.push_back(U[0]);
    deallocate_all_data(all_data);
    delete[] temp_U;
    HANDLE_ERROR(cudaFree(all_device_main));
    HANDLE_ERROR(cudaFree(all_data_device));
    HANDLE_ERROR(cudaFree(cov_mat_device));
    HANDLE_ERROR(cudaFree(M_list_device));
    HANDLE_ERROR(cudaFree(U_device));
    HANDLE_ERROR(cudaFree(Uc_device));
    HANDLE_ERROR(cudaFree(Vj_device));
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Runtime: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << endl;

    if (verbose){
        cout << "Causal order : ";
        for (auto i = causal_order.begin(); i!= causal_order.end(); i++){
            std::cout << *i << ' ';
        }
        std::cout << endl;
    }
    return causal_order;
}


vector<int> direct_lingam::fit_Block_Compare_V1(vector<vector<data_type>> X){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    dims = X.size();
    vector<data_type> temp = *X.begin();
    samples = temp.size();
    setbuf(stdout, NULL);
    causal_order.clear();
    data_type* cov_mat_device = allocate_dim_dim_device();
    data_type** all_data = allocate_all_data();

    vector<data_type**> device_temp = allocate_all_data_device();
    data_type** all_data_device = device_temp[0];
    data_type** all_data_device_copy = device_temp[1];
    data_type* all_device_main = (data_type*)device_temp[2];
    
    data_type* M_list_device;
    int* U_device;
    int* Uc_device;
    int* Vj_device;
    data_type* M_all;
    HANDLE_ERROR(cudaMalloc((void**)&M_list_device, dims * sizeof(data_type)));
    HANDLE_ERROR(cudaMalloc((void**)&U_device, dims * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&Uc_device, dims * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&Vj_device, dims * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&M_all, dims * dims * sizeof(data_type)));


    for(int i = 0; i < dims; i++){
        copy(X[i].begin(), X[i].end(), all_data[i]);
    }

    for(int i = 0; i < dims; i++){
        HANDLE_ERROR(cudaMemcpy(all_data_device_copy[i], all_data[i],
            samples * sizeof(data_type), cudaMemcpyHostToDevice));
    }

    vector<int> U(dims);
    std::iota(U.begin(), U.end(), 0);
    int grid_size = U.size();
    int block_size = 512;
    int smemSize = block_size * sizeof(data_type);
    int* temp_U = new int[dims];
    copy(U.begin(), U.end(), temp_U);
    HANDLE_ERROR(cudaMemcpy(U_device, temp_U, U.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    for (int dim = 0; dim < dims-1; dim++){
        int root = search_causal_order_Block_Compare_V1(all_data, all_data_device,
                                        cov_mat_device, U, M_list_device, U_device,
                                        Uc_device, Vj_device, M_all);

        U.erase(std::remove(U.begin(), U.end(), root), U.end());
        copy(U.begin(), U.end(), temp_U);
        HANDLE_ERROR(cudaMemcpy(U_device, temp_U, U.size() * sizeof(int), cudaMemcpyHostToDevice));
        regress_root_base_V1<<<U.size(), block_size, smemSize>>>(all_data_device, root, base, U_device, U.size(), dims, samples);
        causal_order.push_back(root);
    }
    causal_order.push_back(U[0]);
    deallocate_all_data(all_data);
    delete[] temp_U;
    HANDLE_ERROR(cudaFree(all_device_main));
    HANDLE_ERROR(cudaFree(all_data_device));
    HANDLE_ERROR(cudaFree(cov_mat_device));
    HANDLE_ERROR(cudaFree(M_list_device));
    HANDLE_ERROR(cudaFree(U_device));
    HANDLE_ERROR(cudaFree(Uc_device));
    HANDLE_ERROR(cudaFree(Vj_device));
    HANDLE_ERROR(cudaFree(M_all));
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Runtime: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << endl;

    if (verbose){
        cout << "Causal order : ";
        for (auto i = causal_order.begin(); i!= causal_order.end(); i++){
            std::cout << *i << ' ';
        }
        std::cout << endl;
    }
    return causal_order;
}


vector<vector<int>> direct_lingam::search_candidate(vector<int> U){
    vector<vector<int>> result;
    vector<int> empty_vector;
    result.push_back(U);
    result.push_back(empty_vector);
    return result;
}

int direct_lingam::search_causal_order(data_type** &all_data_device_base, data_type** all_data_device,
                                        data_type** &cov_mat, data_type* cov_mat_device,
                                        data_type** &entropy_result, data_type* entropy_result_device,
                                        vector<int> U, data_type* M_list_device,
                                        int* U_device, int* Uc_device, int* Vj_device,
                                        bool* threshold_flag, bool* threshold_flag_block,
                                        bool* entropy_flag_device, volatile int* entropy_Lock, int* checkpoint,
                                        bool* messages, bool* done_dims, data_type* M_all){
    vector<int> Uc;
    vector<int> Vj;
    bool finish = false;
    bool* temp_finish = new bool[1];
    bool* temp_done = new bool[dims*dims];

    vector<vector<int>> candidates = search_candidate(U);
    Uc = candidates[0];
    Vj = candidates[1];
    
    
    if(Uc.size() == 1){
        return Uc[0];
    }
    int* temp = new int[dims];
    if(Uc.size() > 0){
        copy(Uc.begin(), Uc.end(), temp);
        HANDLE_ERROR(cudaMemcpy(Uc_device, temp, Uc.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    if(Vj.size() > 0){
        copy(Vj.begin(), Vj.end(), temp);
        HANDLE_ERROR(cudaMemcpy(Vj_device, temp, Vj.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    copy(U.begin(), U.end(), temp);
    HANDLE_ERROR(cudaMemcpy(U_device, temp, U.size() * sizeof(int), cudaMemcpyHostToDevice));
    make_checkpoint(temp, Uc, U);

    HANDLE_ERROR(cudaMemcpy(checkpoint, temp, Uc.size() * sizeof(int), cudaMemcpyHostToDevice));

    int grid_size = Uc.size();
    int block_size = 512;
    int smemSize = (block_size) * sizeof(data_type) + 2 * sizeof(bool) + sizeof(int);
    
    if(base){
        normalize_all_device_base <<< grid_size, block_size, smemSize>>> (all_data_device_base, all_data_device, U_device, dims, samples);
        calculate_cov_mat_base <<<grid_size, block_size, smemSize>>>(cov_mat_device, all_data_device_base, U_device, dims, samples);
    }

    HANDLE_ERROR(cudaMemset(entropy_result_device, 0, dims * dims * sizeof(data_type)));
    HANDLE_ERROR(cudaMemset(threshold_flag, 0, sizeof(bool)));
    HANDLE_ERROR(cudaMemset(threshold_flag_block, 0, dims * sizeof(bool)));
    HANDLE_ERROR(cudaMemset(M_list_device, 0, dims * sizeof(data_type)));
    HANDLE_ERROR(cudaMemset(entropy_flag_device, 0, dims * dims * sizeof(bool)));
    HANDLE_ERROR(cudaMemset((void *)entropy_Lock, 0, dims * dims * sizeof(volatile int)));
    HANDLE_ERROR(cudaMemset(messages, 0, dims * dims * sizeof(bool)));
    HANDLE_ERROR(cudaMemset(done_dims, 0, dims * dims * sizeof(bool)));
    HANDLE_ERROR(cudaMemset(M_all, 0, dims * dims * sizeof(data_type)));

    vector<data_type> M_list;

    while(true){
        if (base){
            do_all_device_global <<<grid_size, block_size, smemSize>>> (all_data_device_base, cov_mat_device, entropy_result_device,
                            M_list_device, U_device, U.size(), Uc_device, Uc.size(), Vj_device, Vj.size(), dims, samples,
                            threshold, threshold_flag, threshold_flag_block, entropy_flag_device, entropy_Lock, checkpoint,
                            messages, done_dims, M_all);
        }
        else {
            do_all_device_global <<<grid_size, block_size, smemSize>>> (all_data_device, cov_mat_device, entropy_result_device,
                            M_list_device, U_device, U.size(), Uc_device, Uc.size(), Vj_device, Vj.size(), dims, samples,
                            threshold, threshold_flag, threshold_flag_block, entropy_flag_device, entropy_Lock, checkpoint,
                            messages, done_dims, M_all);
        }
        HANDLE_ERROR(cudaDeviceSynchronize());
        HANDLE_ERROR(cudaMemcpy(temp_finish, threshold_flag, sizeof(bool), cudaMemcpyDeviceToHost));
        finish = temp_finish[0];
        if(finish){
            HANDLE_ERROR(cudaMemcpy(temp_done, done_dims, dims * dims * sizeof(bool), cudaMemcpyDeviceToHost));
            int counter = 0;
            for (int i = 0; i < dims * dims ; i++){
                if (temp_done[i] == true){
                    counter += 1;
                }
            }
            num_compare += U.size() * (U.size()-1);
            num_perform += counter;
            break;
        }
        else{
            data_type* M_list_copy = new data_type[dims];
            HANDLE_ERROR(cudaMemcpy(M_list_copy, M_list_device, dims * sizeof(data_type), cudaMemcpyDeviceToHost));
            data_type max = -std::numeric_limits<double>::max();
            for(auto i = U.begin(); i != U.end() ; i++){
                if(M_list_copy[*i] > max){
                    max = M_list_copy[*i];
                }
            }
            max = - max;
            while(max > threshold ){
                threshold *= update_rate;
            }
        }
    }

    data_type* M_list_copy = new data_type[dims];
    HANDLE_ERROR(cudaMemcpy(M_list_copy, M_list_device, dims * sizeof(data_type), cudaMemcpyDeviceToHost));
    data_type max = -std::numeric_limits<double>::max();
    int max_index = -1;
    for(auto i = U.begin(); i != U.end() ; i++){
        if(M_list_copy[*i] > max){
            max_index = *i;
            max = M_list_copy[*i];
        }
    }

    if(max_index == -1){
        cout << "bad threshold" << endl;
    }
    delete[] temp_finish;
    delete[] temp;
    delete[] M_list_copy;
    delete[] temp_done;
    return max_index;
}

int direct_lingam::search_causal_order_Block_Worker_V1(data_type** &all_data, data_type** all_data_device,
                                data_type** &x_i_device, data_type** &x_j_device,                            
                                data_type** &ri_j_device, data_type** &rj_i_device,
                                data_type* cov_mat_device,
                                vector<int> U, data_type* M_list_device,
                                int* U_device, int* Uc_device, int* Vj_device){
    vector<int> Uc;
    vector<int> Vj;

    vector<vector<int>> candidates = search_candidate(U);
    Uc = candidates[0];
    Vj = candidates[1];
    
    if(Uc.size() == 1){
        return Uc[0];
    }
    int* temp = new int[dims];
    if(Uc.size() > 0){
        copy(Uc.begin(), Uc.end(), temp);
        HANDLE_ERROR(cudaMemcpy(Uc_device, temp, Uc.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    if(Vj.size() > 0){
        copy(Vj.begin(), Vj.end(), temp);
        HANDLE_ERROR(cudaMemcpy(Vj_device, temp, Vj.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    copy(U.begin(), U.end(), temp);
    HANDLE_ERROR(cudaMemcpy(U_device, temp, U.size() * sizeof(int), cudaMemcpyHostToDevice));

    int grid_size = Uc.size();
    int block_size = 512;
    vector<data_type> M_list;

    int smemSize = (block_size) * sizeof(data_type) + sizeof(bool);
    
    HANDLE_ERROR(cudaMemset(M_list_device, 0, dims * sizeof(data_type)));
    
    do_all_device_Block_Worker_V1 <<<grid_size, block_size, smemSize>>>  (all_data_device,
                                    x_i_device, x_j_device, ri_j_device, rj_i_device, cov_mat_device,
                                    M_list_device, base, U_device, U.size(), Uc_device,
                                    Uc.size(), Vj_device, Vj.size(), dims, samples);

    data_type* M_list_copy = new data_type[dims];
    HANDLE_ERROR(cudaMemcpy(M_list_copy, M_list_device, dims * sizeof(data_type), cudaMemcpyDeviceToHost));
    data_type max = -std::numeric_limits<double>::max();
    int max_index = -1;
    for(auto i = U.begin(); i != U.end() ; i++){
        if(M_list_copy[*i] > max){
            max_index = *i;
            max = M_list_copy[*i];
        }
    }
    if(max_index == -1){
        cout << "bad threshold" << endl;
    }

    delete[] temp;
    delete[] M_list_copy;
    return max_index;
}


int direct_lingam::search_causal_order_Thread_Worker_V1(data_type** &all_data, data_type** all_data_device,
                                data_type* cov_mat_device,
                                vector<int> U, data_type* M_list_device,
                                int* U_device, int* Uc_device, int* Vj_device){
    vector<int> Uc;
    vector<int> Vj;

    vector<vector<int>> candidates = search_candidate(U);
    Uc = candidates[0];
    Vj = candidates[1];
    
    
    if(Uc.size() == 1){
        return Uc[0];
    }
    int* temp = new int[dims];
    if(Uc.size() > 0){
        copy(Uc.begin(), Uc.end(), temp);
        HANDLE_ERROR(cudaMemcpy(Uc_device, temp, Uc.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    if(Vj.size() > 0){
        copy(Vj.begin(), Vj.end(), temp);
        HANDLE_ERROR(cudaMemcpy(Vj_device, temp, Vj.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    copy(U.begin(), U.end(), temp);
    HANDLE_ERROR(cudaMemcpy(U_device, temp, U.size() * sizeof(int), cudaMemcpyHostToDevice));

    int grid_size = Uc.size();
    int block_size = 512;
    vector<data_type> M_list;
    int smemSize = (block_size) * sizeof(data_type);
    
    HANDLE_ERROR(cudaMemset(M_list_device, 0, dims * sizeof(data_type)));
    
    do_all_device_Thread_Worker_V1 <<<grid_size, block_size, smemSize>>>  (all_data_device,cov_mat_device,
                                    M_list_device, base, U_device, U.size(), Uc_device,
                                    Uc.size(), Vj_device, Vj.size(), dims, samples);

    data_type* M_list_copy = new data_type[dims];
    HANDLE_ERROR(cudaMemcpy(M_list_copy, M_list_device, dims * sizeof(data_type), cudaMemcpyDeviceToHost));
    data_type max = -std::numeric_limits<double>::max();
    int max_index = -1;
    for(auto i = U.begin(); i != U.end() ; i++){
        if(M_list_copy[*i] > max){
            max_index = *i;
            max = M_list_copy[*i];
        }
    }
    if(max_index == -1){
        cout << "bad threshold" << endl;
    }
    delete[] temp;
    delete[] M_list_copy;
    return max_index;
}

int direct_lingam::search_causal_order_Block_Compare_V1(data_type** &all_data, data_type** all_data_device,
                                        data_type* cov_mat_device,
                                        vector<int> U, data_type* M_list_device,
                                        int* U_device, int* Uc_device, int* Vj_device, data_type* M_all){
    vector<int> Uc;
    vector<int> Vj;

    
    vector<vector<int>> candidates = search_candidate(U);
    Uc = candidates[0];
    Vj = candidates[1];
    
    
    if(Uc.size() == 1){
        return Uc[0];
    }
    int* temp = new int[dims];
    if(Uc.size() > 0){
        copy(Uc.begin(), Uc.end(), temp);
        HANDLE_ERROR(cudaMemcpy(Uc_device, temp, Uc.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    if(Vj.size() > 0){
        copy(Vj.begin(), Vj.end(), temp);
        HANDLE_ERROR(cudaMemcpy(Vj_device, temp, Vj.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    copy(U.begin(), U.end(), temp);
    HANDLE_ERROR(cudaMemcpy(U_device, temp, U.size() * sizeof(int), cudaMemcpyHostToDevice));
    make_checkpoint(temp, Uc, U);
    
    dim3 grid_size(Uc.size(), Uc.size(), 1);
    int grid_size_reduce = Uc.size();
    int block_size = 512;
    int smemSize = (block_size) * sizeof(data_type);

    HANDLE_ERROR(cudaMemset(M_list_device, 0, dims * sizeof(data_type)));
    HANDLE_ERROR(cudaMemset(M_all, 0, dims * dims * sizeof(data_type)));

    
    do_all_device_Block_Compare_V1 <<<grid_size, block_size, smemSize>>>  (all_data_device,cov_mat_device,
                                    M_all, base, U_device, U.size(), Uc_device,
                                    Uc.size(), Vj_device, Vj.size(), dims, samples);
    reduce_Block_Comapare<<<grid_size_reduce, block_size, smemSize>>> (M_all, M_list_device,
                                    U_device, U.size(), Uc_device, Uc.size(), dims, samples);
    
    data_type* M_list_copy = new data_type[dims];
    HANDLE_ERROR(cudaMemcpy(M_list_copy, M_list_device, dims * sizeof(data_type), cudaMemcpyDeviceToHost));
    data_type* M_all_copy = new data_type[dims*dims];

    data_type max = -std::numeric_limits<double>::max();
    int max_index = -1;
    for(auto i = U.begin(); i != U.end() ; i++){
        if(M_list_copy[*i] > max){
            max_index = *i;
            max = M_list_copy[*i];
        }
    }

    if(max_index == -1){
        cout << "bad threshold" << endl;
    }
    delete[] temp;
    delete[] M_list_copy;
    return max_index;
}


data_type** direct_lingam::allocate_all_data(){
    data_type** all_data = new data_type*[dims];
    for(int i = 0; i < dims; i++){
        all_data[i] = new data_type [samples];
    }
    return all_data;
}

vector<data_type**> direct_lingam::allocate_all_data_device(){
    vector<data_type**> result;
    data_type** all_data_device;
    data_type* all_device_main;
    HANDLE_ERROR(cudaMalloc((void***)&all_data_device, dims * sizeof(data_type*)));
    HANDLE_ERROR(cudaMalloc((void**)&all_device_main, samples * dims * sizeof(data_type)));
    data_type** first_layer_pointer = new data_type*[dims];
    for(int i = 0; i < dims ; i++){
        first_layer_pointer[i] = all_device_main + i * samples;
    }
    HANDLE_ERROR(cudaMemcpy(all_data_device, first_layer_pointer, dims * sizeof(data_type*), cudaMemcpyHostToDevice));
    result.push_back(all_data_device);
    result.push_back(first_layer_pointer);
    result.push_back((data_type**)all_device_main);
    return result;
}

void direct_lingam::deallocate_all_data(data_type** all_data){
    delete[] all_data;
}

data_type** direct_lingam::allocate_dim_dim(){
    data_type** cov_mat = new data_type*[dims];
    for(int i = 0; i < dims; i++){
        cov_mat[i] = new data_type[dims];
    }
    return cov_mat;
}

data_type* direct_lingam::allocate_dim_dim_device(){
    data_type* all_data_device;
    HANDLE_ERROR(cudaMalloc((void**)&all_data_device, dims * dims * sizeof(data_type)));
    return all_data_device;
}

void direct_lingam::deallocate_dim_dim(data_type** cov_mat){
    for(int i = 0; i < dims; i++){
        delete[] cov_mat[i];
    }
    delete[] cov_mat;
}

void direct_lingam::make_checkpoint(int* result, vector<int> Uc, vector<int> U){
    int j= 0;
    for(int i = 0; i < U.size(); i++){
        if(Uc[j] == U[i]){
            result[j] = i;
            j++;
        }
    }
}

void direct_lingam::set_update_rate(data_type rate){
    update_rate = rate;
}

void direct_lingam::set_base_mode(bool mode){
    base = mode;
}

void direct_lingam::set_verbose_mode(bool mode){
    verbose = mode;
}

void direct_lingam::reset_saved_comparisons(){
    saved_comparisons.clear();
}

float direct_lingam::average_saved_comparisons(){
    return std::accumulate(saved_comparisons.begin(), saved_comparisons.end(), 0.0)/ float(saved_comparisons.size());
}

void direct_lingam::reset_runtimes(){
    runtimes.clear();
}

float direct_lingam::average_runtimes(){
    return std::accumulate(runtimes.begin(), runtimes.end(), 0.0)/ float(runtimes.size());
}