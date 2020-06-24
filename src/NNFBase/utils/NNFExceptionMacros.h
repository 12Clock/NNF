#ifndef NNF_EXCEPTION_MACROS_H
#define NNF_EXCEPTION_MACROS_H

#include <src/NNFBase/utils/NNFException.cpp>

#define NNF_LIKELY(expr)   (__builtin_expect(static_cast<bool>(expr), 1))
#define NNF_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))

// Throw error message
#define NNF_THROW_ERROR(err_type, msg) throw err_type({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, msg)

// Conditionally throw the error message
#define NNF_CHECH_WITH_MSG(err_type, cond, ...)                                              \
    if(NNF_UNLIKELY(!(cond))){                                                                  \
        NNF_THROW_ERROR(err_type,                                                            \
            nnf::details::if_empty_then(                                                     \
                nnf::details::str(__VA_ARGS__),                                             \
                nnf::details::str("Expected ", #cond, " to be true, but got false.  ")       \
            ));                                                                              \
    }

#define NNF_CHECK_WITH(err_type, cond, ...) NNF_CHECH_WITH_MSG(err_type, cond, __VA_ARGS__)
#define NNF_CHECK(cond, ...) NNF_CHECK_WITH(nnf::utils::NNF_Error, cond, __VA_ARGS__)

// Assert of NNF
#define NNF_INTERNAL_ASSERT(cond, ...)                            \
    if(NNF_UNLIKELY(!(cond))){                                                  \
        NNF_THROW_ERROR(                                          \
            nnf::utils::NNF_Error, nnf::details::str(             \
                #cond, " INTERNAL ASSERT FAILD : ",               \
                nnf::details::str(__VA_ARGS__)                    \
            )                                                     \
        );                                                        \
    }

// Throw the error message
#define NNF_ERROR(...) NNF_THROW_ERROR(nnf::utils::NNF_Error, nnf::details::str(__VA_ARGS__))

#define NNF_UNUSED __attribute__((__unused__))

#endif