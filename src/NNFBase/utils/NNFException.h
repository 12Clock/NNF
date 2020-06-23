#ifndef NNF_EXCEPTION_H
#define NNF_EXCEPTION_H

#include <string>
#include <vector>

namespace nnf {

namespace utils{

class NNF_Error: public std::exception
{
    private:
        std::string msg_;
        std::vector<std::string> context_;
        std::string what_;

        void refresh_what();

    public:
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