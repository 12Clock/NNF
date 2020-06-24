#ifndef NNF_EXCEPTION_CPP
#define NNF_EXCEPTION_CPP

#include <src/NNFBase/utils/NNFException.h>
#include <src/NNFBase/details/NNFBaseString.cpp>

namespace nnf{

namespace utils{

void NNF_Error::refresh_what()
{
    std::stringstream oss;
    oss<<msg_;

    if(context_.size() == 1){
        oss<<" ("<<context_[0]<<")";
    }else{
        for(const auto& str: context_){
            oss << "\n  " << str;
        }
    }

    what_ = oss.str();
}

NNF_Error::NNF_Error(SourceLocation source_location, const std::string& msg)
{
    std::ostringstream oss;
    std::ostringstream os;
    os << (oss, source_location) << ":\n" << msg << "\n";
    msg_ = os.str();
    refresh_what();
}

NNF_Error::NNF_Error(const std::string& msg)
: msg_(std::move(msg)) {refresh_what();}

NNF_Error::NNF_Error(const char* file, const uint32_t line, const char* condition, const std::string& msg)
: NNF_Error(
    nnf::details::str(
        "[ File: ", file, " in line ", line, " ] :",
        condition, ".\n", msg, "\n"
    )
) {refresh_what();}

void NNF_Error::add_context(std::string new_msg)
{
    context_.push_back(std::move(new_msg));
    refresh_what();
}

} // namespace utils

} // namespace nnf

#endif