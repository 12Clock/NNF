#ifndef NNF_EXCEPTION_H
#define NNF_EXCEPTION_H

#include <string>
#include <vector>
#include <src/NNFBase/utils/NNFStringUtils.h>

namespace nnf {

namespace utils{

/*
Basic types of all errors in NNF
*/
class NNF_Error: public std::exception
{
    private:
        std::string msg_;                    // Message about the current error
        std::vector<std::string> context_;   // Context error message
        std::string what_;                   // Including the above two kinds of information

        void refresh_what();                 // When add a context error message, refresh what

    public:
        NNF_Error(SourceLocation source_location, const std::string& msg);
        NNF_Error(const std::string& msg);
        NNF_Error(const char* file, const uint32_t line, const char* condition, const std::string& msg);
        void add_context(std::string new_msg);

        const std::string& msg() const{
            return msg_;
        }

        const std::vector<std::string>& context() const{
            return context_;
        }

        const char* what() const noexcept override{
            return what_.c_str();
        }
};

} // namespace utils

} // namespace nnf


#endif