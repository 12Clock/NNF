#include "NNFTensor.h"
#include <vector>
#include <iostream>
#include <typeinfo>

int main()
{
    nnf::NNF_Array(Int) x = {0, 1, 2, 3, 4};
    nnf::NNF_Array(Int) y = {5, 6, 7, 8, 9};
    x.concat_(y);
    x[6] = 1;
    std::cout<<x[6]<<std::endl;
    for(int i=0;i<x.length();i++) std::cout<<x[i]<<" \n"[i==x.length()-1];
    return 0;
}

