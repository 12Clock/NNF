#ifndef NNF_BASE_STRING_CPP
#define NNF_BASE_STRING_CPP

#include <sstream>
#include <string>

namespace nnf{

namespace details{

    /*
    Recursive implementation of inputting multiple character types into the character stream
    (递归实现将多种字符类型数据输入字符流中)
    */
    inline std::ostream& _str(std::ostream& ss) {
        return ss;
    }

    template <typename T>
    inline std::ostream& _str(std::ostream& ss, const T& t) {
        ss << t;
        return ss;
    }

    template <typename T, typename... Args>
    inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
        return _str(_str(ss, t), args...);
    }

    /*
    Splicing strings to facilitate printing information
    (拼接字符串以方便打印信息)
    */
    template<typename... Args>
    inline std::string str(const Args&... args)
    {
        std::stringstream oss;
        _str(oss, args...);
        return oss.str();
    }

    /*
    Get the file name from the path
    (从路径中得到文件名)
    */
    std::string StripBasename(const std::string& full_path) {
        const char kSeparator = '/';
        size_t pos = full_path.rfind(kSeparator);
        if (pos != std::string::npos) {
            return full_path.substr(pos + 1, std::string::npos);
        } else {
            return full_path;
        }
    }

    /*
    If the x string is empty, return the y string
    (如果字符串x为空，则返回字符串y)
    */
    inline std::string if_empty_then(std::string x, std::string y){
        if(x.empty()) {
            return y;
        }else{
            return x;
        }
    }
} // namespace details

} // namespace nnf

#endif