#ifndef NNF_ARRAY_H
#define NNF_ARRAY_H

#include <initializer_list>

namespace nnf{

typedef int int32_t;

struct float_array
{
    float *array;
    int32_t length;
};

struct int_array
{
    int32_t *array;
    int32_t length;
};

class NNF_FloatArray
{
    private:
        float_array data;
    public:
        NNF_FloatArray() = default;
        NNF_FloatArray(float * _array, int32_t length);
        NNF_FloatArray(const std::initializer_list<float> & _array);
        template<class T> NNF_FloatArray(const T & _array);
        template<typename T> NNF_FloatArray(T _array, int32_t length);
        ~NNF_FloatArray();
        void concat_(const NNF_FloatArray & _array);
        NNF_FloatArray concat(const NNF_FloatArray & _array);
        float & operator[](int32_t index);
        void operator=(const NNF_FloatArray & _array);
        // template<typename T> void operator=(T _array);
        template<class T> void operator=(const T & _array);
        float * const _get_ptr() const;
        const int32_t length() const;
        const int32_t size() const;
};

class NNF_IntArray
{
    private:
        int_array data;
    public:
        NNF_IntArray() = default;
        NNF_IntArray(int32_t * _array, int32_t length);
        NNF_IntArray(const std::initializer_list<int32_t> & _array);
        template<class T> NNF_IntArray(const T & _array);
        template<typename T> NNF_IntArray(T _array, int32_t length);
        ~NNF_IntArray();
        void concat_(const NNF_IntArray & _array);
        NNF_IntArray concat(const NNF_IntArray & _array);
        int32_t & operator[](int32_t index);
        void operator=(const NNF_IntArray & _array);
        // template<typename T> void operator[]=(T _array);
        template<class T> void operator=(const T & _array);
        int32_t * const _get_ptr() const;
        const int32_t length() const;
        const int32_t size() const;
};

}

#define NNF_Array(type) NNF_##type##Array

#endif