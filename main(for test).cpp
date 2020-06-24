#include <iostream>
#include <src/NNFBase/NNFBaseMacros.h>

using namespace std;

int main()
{
    try{
        // NNF_CHECK(-1>0, "msg");
        throw nnf::utils::NNF_Error({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, "exp1");
        cout<<"hhhh"<<endl;
    }catch(nnf::utils::NNF_Error& e){
        cout<<e.msg()<<endl;
    }

    cout<<"UnKonw Error"<<endl;
    return 0;
}