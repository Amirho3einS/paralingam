#include "main.hpp"

int main(int argc, char** argv){
    direct_lingam model;
    vector<vector<data_type>> B, X;
    vector<int> causal_order;
    string dir;
    vector<string> real_data{"e_coli_core", "pathfinder", "andes", "diabetes", "pigs", "link",
                                "iJR904", "munin", "iAF1260b", "iAF1260", "iY75_1357",
                                "iECDH10B_1368", "iML1515", "iEC1372_W3110"};

    switch (atoi(argv[1])) {
    case 4: {
        model.set_base_mode(true);
        for(auto s = real_data.begin(); s != real_data.end(); s++){
            dir = "../real/" + *s + ".csv";
            cout << *s << endl;
            X = read_csv(dir);
            if(!vector_check(X)){
                cout << "invalid data sizes" << endl;
            }
            causal_order = model.fit(X);
            cout << endl;
        }
        break;
    }
    case 5:{
        vector<int> dims{100, 200, 500, 1000};
        vector<int> samples{1024, 2048, 4096, 8192};
        vector<string> graph_type{"S", "D", "SS"};
        string dir;
        for(auto d = dims.begin(); d != dims.end(); d++){
            for(auto s = samples.begin(); s != samples.end(); s++){
                for(auto t = graph_type.begin(); t != graph_type.end(); t++){
                    for(int c = 0; c < 1; c++){
                        cout << *t << ' ' << *d << ' ' << *s << endl;
                        dir = "../synthetic/B_" + *t + '_' + to_string(*d) + '_' + to_string(*s) + '_' + to_string(c)  + ".csv";
                        B = read_csv(dir);
                        if(!vector_check(B)){
                            cout << "invalid data sizes" << endl;
                        }
                        dir = "../synthetic/X_" + *t + '_' + to_string(*d) + '_' + to_string(*s) + '_' + to_string(c)  + ".csv";
                        X = read_csv(dir);
                        if(!vector_check(X)){
                            cout << "invalid data sizes" << endl;
                        }
                        causal_order = model.fit(X);
                        cout << endl;
                    }
                }
            }
        }
        break;
    }
    case 6: {
        model.set_base_mode(true);
        for(auto s = real_data.begin(); s != real_data.end(); s++){
            dir = "../real/" + *s + ".csv";
            cout << *s << endl;
            X = read_csv(dir);
            if(!vector_check(X)){
                cout << "invalid data sizes" << endl;
            }
            causal_order = model.fit_opt(X);
            cout << endl;
        }
        break;
    }
    case 7:{
        vector<int> dims{100, 200, 500, 1000};
        vector<int> samples{1024, 2048, 4096, 8192};
        vector<string> graph_type{"S", "D", "SS"};
        string dir;
        for(auto d = dims.begin(); d != dims.end(); d++){
            for(auto s = samples.begin(); s != samples.end(); s++){
                for(auto t = graph_type.begin(); t != graph_type.end(); t++){
                    for(int c = 0; c < 1; c++){
                        cout << *t << ' ' << *d << ' ' << *s << endl;
                        dir = "../synthetic/B_" + *t + '_' + to_string(*d) + '_' + to_string(*s) + '_' + to_string(c)  + ".csv";
                        B = read_csv(dir);
                        if(!vector_check(B)){
                            cout << "invalid data sizes" << endl;
                        }
                        dir = "../synthetic/X_" + *t + '_' + to_string(*d) + '_' + to_string(*s) + '_' + to_string(c)  + ".csv";
                        X = read_csv(dir);
                        if(!vector_check(X)){
                            cout << "invalid data sizes" << endl;
                        }
                        causal_order = model.fit_opt(X);
                        cout << endl;
                    }
                }
            }
        }
        break;
    }
    default:{
        cout << "Invalid option" << endl;
        break;
    }
    }
    return 0; 
}

vector<vector<data_type>> read_csv(string dir){
    ifstream file(dir);
    string line;
    vector<vector<data_type>> data;
    vector<data_type> row;
    getline(file, line);
    while(line != ""){
        vector<data_type> ().swap(row);
        while( line.find(',') != string::npos){
            row.push_back(stod(line.substr(0, line.find(','))));
            line.erase(0, line.find(',') + 1);
        }
        row.push_back(stod(line));
        getline(file, line);
        data.push_back(row);
    }
    return data;
}

bool vector_check(vector<vector<data_type>> data){
    vector<data_type> temp = *data.begin();
    int size = temp.size();
    for (auto i = data.begin(); i!= data.end(); ++i){
        temp = *i;
        if (temp.size() != size)
            return false;
    }
    return true;
}
