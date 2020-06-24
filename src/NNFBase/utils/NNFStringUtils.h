#ifndef NNF_STRING_UTILS_H
#define NNF_STRING_UTILS_H

#include <iostream>

namespace nnf{

namespace utils{

typedef unsigned int uint32_t;

// To save local error information, like file, function and line
struct SourceLocation{
    const char* function;
    const char* file;
    uint32_t line;

    SourceLocation(const char* fun, const char* file, uint32_t line)
    : function(fun)
    , file(file)
    , line(line) {}
};

std::ostream& operator<<(std::ostream& os, SourceLocation& log)
{
    os << "( " << log.function << " )[ File: " << log.file << " in line " << log.line << " ] ";
    return os;
}

} // namespace utils

} // namespace nnf

#endif