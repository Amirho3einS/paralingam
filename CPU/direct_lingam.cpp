#include "direct_lingam.hpp"

direct_lingam::direct_lingam(/* args */){
}

direct_lingam::~direct_lingam(){
}


vector<int> direct_lingam::fit(vector<vector<data_type>> X){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    dims = X.size();
    vector<data_type> temp = *X.begin();
    samples = temp.size();
    causal_order.clear();
    vector<int> U(dims);
    std::iota(U.begin(), U.end(), 0);

    for (int dim = 0; dim < dims; dim++){
        int root = search_causal_order(X, U);
        for(auto i = U.begin(); i != U.end(); i++){
            if( *i != root){
                if(base){
                    X[*i] = residual_base(X[*i], X[root]);
                }
                else{
                    X[*i] = residual(X[*i], X[root]);
                }
            }
        }
        U.erase(std::remove(U.begin(), U.end(), root), U.end());
        causal_order.push_back(root);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Runtime:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << endl;
    
    if (verbose){
        cout << "causal order : ";
        for (auto i = causal_order.begin(); i!= causal_order.end(); i++){
            std::cout << *i << ' ';
        }
        std::cout << endl;
    }
    return causal_order;
}


vector<int> direct_lingam::fit_opt(vector<vector<data_type>> X){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    dims = X.size();
    vector<data_type> temp = *X.begin();
    samples = temp.size();
    causal_order.clear();
    vector<int> U(dims);
    std::iota(U.begin(), U.end(), 0);
    // the main threshold is 1e-7, it will be update in the function
    data_type threshold = 1e-7/2;
    for (int dim = 0; dim < dims; dim++){
        int root = search_causal_order_opt(X, U, threshold);
        for(auto i = U.begin(); i != U.end(); i++){
            if( *i != root){
                if(base){
                    X[*i] = residual_base(X[*i], X[root]);
                }
                else{
                    X[*i] = residual(X[*i], X[root]);
                }
            }
        }
        U.erase(std::remove(U.begin(), U.end(), root), U.end());
        causal_order.push_back(root);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Runtime:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << endl;
    
    if (verbose){
        cout << "causal order : ";
        for (auto i = causal_order.begin(); i!= causal_order.end(); i++){
            std::cout << *i << ' ';
        }
        std::cout << endl;
    }
    return causal_order;
}


vector<vector<data_type>> direct_lingam::estimate_adjacency_matrix(vector<vector<data_type>> X){
    vector<vector<data_type>> result;
    return result;
}


vector<int> direct_lingam::get_causal_order(){
    return causal_order;
}

vector<data_type> direct_lingam::residual(vector<data_type> xi, vector<data_type> xj){
    vector<data_type> result;
    data_type cov = covariance(xi, xj);
    data_type var = variance(xj);

    if (base){
        data_type temp = pow((1. - pow(cov,2)), 0.5);
        for(int i = 0; i < xi.size(); i++){
            result.push_back((xi[i] - cov * xj[i])/temp);
        }
    }
    else {
        for(int i = 0; i < xi.size(); i++){
            result.push_back(xi[i] - cov/var * xj[i]);
        }
    }
    return result;
}

vector<data_type> direct_lingam::residual_base(vector<data_type> xi, vector<data_type> xj){
    vector<data_type> result;
    data_type cov = covariance(xi, xj);
    data_type var = variance(xj);

    for(int i = 0; i < xi.size(); i++){
        result.push_back(xi[i] - cov * xj[i]);
    }
    return result;
}

data_type direct_lingam::entropy(vector<data_type> u){
    data_type k1 = 79.047;
    data_type k2 = 7.4129;
    data_type gamma = 0.37457;
    data_type result = 0;
    data_type cal_1;
    data_type cal_2;
    data_type _size = data_type(u.size());
    cal_1 = accumulate(u.begin(), u.end(), 0.0, entropy_cal_1)/_size;
    cal_2 = accumulate(u.begin(), u.end(), 0.0, entropy_cal_2)/_size;
    return (1 + log(2 * M_PI)) / 2 - k1 * pow(cal_1 - gamma, 2) - k2 * pow(cal_2, 2);
}

data_type direct_lingam::diff_mutual_info(vector<data_type> xi_std, vector<data_type> xj_std, 
                                            vector<data_type> ri_j, vector<data_type> rj_i){

    data_type std_ri_j = pow(variance(ri_j), 0.5);
    data_type std_rj_i = pow(variance(rj_i), 0.5);
    if (base){
        std_ri_j = 1.0;
        std_rj_i = 1.0;
    }
    int _size = xi_std.size();
    for(int i = 0; i < _size; i++){
        ri_j[i] = ri_j[i] / std_ri_j;
        rj_i[i] = rj_i[i] / std_rj_i;
    }
    return (entropy(xj_std) + entropy(ri_j)) - (entropy(xi_std) + entropy(rj_i));
}

vector<vector<int>> direct_lingam::search_candidate(vector<int> U){
    vector<vector<int>> result;
    vector<int> empty_vector;
    result.push_back(U);
    result.push_back(empty_vector);
    return result;
}

int direct_lingam::search_causal_order(vector<vector<data_type>> &X, vector<int> U){
    vector<int> Uc;
    vector<int> Vj;
    vector<vector<int>> candidates = search_candidate(U);
    Uc = candidates[0];
    Vj = candidates[1];

    if(Uc.size() == 1){
        return Uc[0];
    }

    vector<data_type> M_list;
    vector<data_type> xi_std;
    vector<data_type> xj_std;
    vector<data_type> ri_j;
    vector<data_type> rj_i;

    int temp = 0;

    for(auto i = Uc.begin(); i != Uc.end(); i++){
        data_type M = 0;
        for (auto j = U.begin(); j != U.end(); j++){
            if (*i != *j){
                xi_std = normalize(X[*i]);
                xj_std = normalize(X[*j]);

                if((std::find(Vj.begin(), Vj.end(), *i) != Vj.end()) && 
                        (std::find(Uc.begin(), Uc.end(), *j) != Uc.end())){
                    ri_j = xi_std;
                }
                else{
                    ri_j = residual(xi_std, xj_std);
                }
                if((std::find(Vj.begin(), Vj.end(), *j) != Vj.end()) && 
                        (std::find(Uc.begin(), Uc.end(), *i) != Uc.end())){
                    rj_i = xj_std;
                }
                else{
                    rj_i = residual(xj_std,  xi_std);
                }

                M += pow(std::min(data_type(0), diff_mutual_info(xi_std, xj_std, ri_j, rj_i)), 2);
            }
        }
        M_list.push_back( -1.0 * M);
    }
    return Uc[std::max_element(M_list.begin(), M_list.end()) - M_list.begin()];
}

int direct_lingam::search_causal_order_opt(vector<vector<data_type>> &X, vector<int> U, 
                                        data_type threshold){

    vector<int> Uc;
    vector<int> Vj;
    vector<vector<int>> candidates = search_candidate(U);
    Uc = candidates[0];
    Vj = candidates[1];

    if(Uc.size() == 1){
        return Uc[0];
    }

    vector<data_type> M_list(Uc.size(), 0);
    vector<data_type> xi_std;
    vector<data_type> xj_std;
    vector<data_type> ri_j;
    vector<data_type> rj_i;

    vector<vector<bool>> done_dims(Uc.size(), vector<bool>(U.size(), false));

    vector<int> C(Uc.size(), 0);
    for(int i = 0; i < Uc.size(); i++){
        done_dims[i][i] = true;
        C[i] = i+1;
    }
    C[Uc.size()-1] = 0;

    bool finish = false;

    data_type temp;

    data_type update_rate = 2.0;

    while(!finish){
        threshold *= update_rate;
        for(int i = 0; i < Uc.size(); i++){
            while (!std::all_of(done_dims[i].begin(), done_dims[i].end(), [](bool i){return i;})){
                if (-M_list[i] > threshold || C[i] == i){
                    break;
                }
                // Doing signle comoparison
                int j = C[i];
                if (Uc[i] != U[j]){
                    xi_std = normalize(X[Uc[i]]);
                    xj_std = normalize(X[U[j]]);

                    if((std::find(Vj.begin(), Vj.end(), Uc[i]) != Vj.end()) && 
                            (std::find(Uc.begin(), Uc.end(), U[j]) != Uc.end())){
                        std::cout << " first error " << threshold << std::endl;
                        ri_j = xi_std;
                    }
                    else{
                        ri_j = residual(xi_std, xj_std);
                    }
                    if((std::find(Vj.begin(), Vj.end(), U[j]) != Vj.end()) && 
                            (std::find(Uc.begin(), Uc.end(), Uc[i]) != Uc.end())){
                        std::cout << " second error " << threshold << std::endl;
                        rj_i = xj_std;
                    }
                    else{
                        rj_i = residual(xj_std,  xi_std);
                    }
                    temp = diff_mutual_info(xi_std, xj_std, ri_j, rj_i);
                    M_list[i] -= pow(std::min(data_type(0), temp), 2);
                    done_dims[i][j] = true;
                    // msg
                    M_list[j] -= pow(std::min(data_type(0), -temp), 2);
                    done_dims[j][i] = true;
                }
                do{
                    C[i] = (C[i]+1) % Uc.size();
                } while (done_dims[i][C[i]] && C[i] != i);
                
            }
            // here we check that all of the comparisons are done
            // and a worker's score is lower than threshold
            if (std::all_of(done_dims[i].begin(), done_dims[i].end(), [](bool i){return i;}) 
                && (- M_list[i]) < threshold) {
                finish = true;
            }
        }
        // at this point if at least one of the workers has finished it's task
        // we are going to choose the root
    }
    return Uc[std::max_element(M_list.begin(), M_list.end()) - M_list.begin()];
}

vector<data_type> normalize(vector<data_type> X){
    data_type mean = accumulate(X.begin(), X.end(), 0.0)/ data_type(X.size());
    data_type var = 0;
    for(auto i = X.begin(); i != X.end(); i++){
        var += pow((*i - mean), 2);
    }
    data_type std_cal = sqrt(var/data_type(X.size() - 1));
    for(auto i = X.begin(); i != X.end(); i++){
        *i = (*i - mean) / std_cal;
    }
    return X;
}

void vector_print(vector<data_type> data){    
    for(auto i = data.begin(); i != data.end(); i++){
        std::cout << *i << ' ';
    }
    std::cout << endl;
}

data_type variance(vector<data_type> X){
    data_type mean = accumulate(X.begin(), X.end(), 0.0)/ data_type(X.size());
    data_type var = 0;
    for(auto i = X.begin(); i != X.end(); i++){
        var += pow((*i - mean), 2);
    }
    return var/(data_type(X.size()-1));
}

data_type direct_lingam::covariance(vector<data_type> X, vector<data_type> Y){
    int n = X.size();
    data_type X_mean = accumulate(X.begin(), X.end(), 0.0)/ data_type(n);
    data_type Y_mean = accumulate(Y.begin(), Y.end(), 0.0)/ data_type(n);
    data_type sum = 0;
    for(int i = 0; i < n; i++){
        sum += (X[i] - X_mean) * (Y[i] - Y_mean);
    }
    data_type result = sum/data_type(n-1);
    if(base){
        if(result > 0.99){
            result = 0.99;
        }
        else if(result < -0.99){
            result = -0.99;
        }
    }
    return result;
}


data_type entropy_cal_1(data_type x, data_type y){
    return x + log(cosh(y));
}

data_type entropy_cal_2(data_type x, data_type y){
    return x + y * (exp(-0.5* pow(y, 2)));
}

void direct_lingam::set_base_mode(bool mode){
    base = mode;
}

void direct_lingam::set_verbose_mode(bool mode){
    verbose = mode;
}