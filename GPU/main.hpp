#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <random>
#include "direct_lingam.hpp"



using namespace std;

vector<vector<data_type>> read_csv(string dir);
bool vector_check(vector<vector<data_type>> data);
int main(int argc, char** argv);