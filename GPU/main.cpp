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
    case 0: {
        model.set_base_mode(true);
        model.set_update_rate(2.0);
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
    case 1: {
        vector<int> dims{100, 200, 500, 1000};
        vector<int> samples{1024, 2048, 4096, 8192};
        vector<string> graph_type{"S", "D", "SS"};
        model.set_update_rate(2.0);
        for(auto t = graph_type.begin(); t != graph_type.end(); t++){
            for(auto d = dims.begin(); d != dims.end(); d++){
                for(auto s = samples.begin(); s != samples.end(); s++){
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
    case 2:{
        model.set_base_mode(true);
        vector<data_type> c{1.05, 1.5, 2.0, 5.0, 10};
        for (auto i = c.begin(); i != c.end(); i++){
            cout << "----- rate = " << *i << " -----" << endl;
            model.set_update_rate(*i);
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
        }
        break;
    }
    case 3: {
        model.set_base_mode(true);
        for(auto s = real_data.begin(); s != real_data.end(); s++){
            dir = "../real/" + *s + ".csv";
            X = read_csv(dir);
            if(!vector_check(X)){
                cout << "invalid data sizes" << endl;
            }
            for (int i = 0; i < 4; i++){
                switch(i){
                    case 0 : {
                        cout <<*s << " : " << "Block Worker" <<endl;
                        causal_order = model.fit_Block_Worker_V1(X);
                        break;
                    }
                    case 1 : {
                        cout <<*s << " : " << "Thread Worker" <<endl;
                        causal_order = model.fit_Thread_Worker_V1(X);

                        break;
                    }
                    case 2 : {
                        cout <<*s << " : " << "Block Compare" <<endl;
                        causal_order = model.fit_Block_Compare_V1(X);
                        break;
                    }
                    case 3 : {
                        cout <<*s << " : " << "ParaLiNGAM" <<endl;
                        model.set_update_rate(2.0);
                        causal_order = model.fit(X);
                        break;
                    }
                }
            }
            cout << endl;
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
    data_type temp;
    data_type max = 0.0;
    while(line != ""){
        vector<data_type> ().swap(row);
        while( line.find(',') != string::npos){
            temp = stod(line.substr(0, line.find(',')));
            row.push_back(temp);
            line.erase(0, line.find(',') + 1);
            if(temp > max){
                max = temp;
            }
        }
        temp = stod(line);
        row.push_back(temp);
        if(temp > max){
            max = temp;
        }
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
