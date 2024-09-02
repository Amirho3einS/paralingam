#ifndef DIRECT_LINGAM_HPP
#define DIRECT_LINGAM_HPP

#include <iostream>
#include <vector>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <chrono>

#define data_type double

using namespace std;

class direct_lingam{
private:
    int dims = 0;
    int samples = 0;
    bool base = false;
    bool verbose = true;
    vector<int> causal_order;
public:
    direct_lingam();
    ~direct_lingam();

    vector<int> fit(vector<vector<data_type>> X);
    vector<int> fit_opt(vector<vector<data_type>> X);
    vector<vector<data_type>> estimate_adjacency_matrix(vector<vector<data_type>> X);
    vector<vector<data_type>> get_adjacency_matrix();
    vector<int> get_causal_order();
    vector<data_type> residual(vector<data_type> xi, vector<data_type> xj);
    vector<data_type> residual_base(vector<data_type> xi, vector<data_type> xj);
    data_type entropy(vector<data_type> u);
    data_type diff_mutual_info(vector<data_type> xi_std, vector<data_type> xj_std, 
                                vector<data_type> ri_j, vector<data_type> rj_i);
    vector<vector<int>> search_candidate(vector<int> U);
    int search_causal_order(vector<vector<data_type>> &X, vector<int> U);
    int search_causal_order_opt(vector<vector<data_type>> &X, vector<int> U,
                            data_type threshold);
    void set_base_mode(bool base);
    void set_verbose_mode(bool verbose);
    data_type covariance(vector<data_type> X, vector<data_type> Y);

};


vector<data_type> normalize(vector<data_type> X);
void vector_print(vector<data_type> data);
data_type variance(vector<data_type> X);

data_type entropy_cal_1(data_type x, data_type y);
data_type entropy_cal_2(data_type x, data_type y);


#endif