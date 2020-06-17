#include "NNFArray.h"
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

namespace nnf {

    // NNF_FloatArray 
    /*
    Author: LineZ
    Update: 2020/6/12
    */
    NNF_FloatArray::NNF_FloatArray(float * _array, int32_t length)
    {
        assert(_array != NULL);
        this->data.array = _array;
        this->data.length = length;
    }

    NNF_FloatArray::NNF_FloatArray(const std::initializer_list<float> & _array)
    {
        this->data.array = (float*)malloc(sizeof(float) * _array.size());

        if(this->data.array != NULL){
            this->data.length = _array.size();
            float *ptr = this->data.array;
            for(std::initializer_list<float>::iterator i = _array.begin(); i != _array.end(); i++){
                *ptr++ = *i;
            }
        }
    }

    template<class T> NNF_FloatArray::NNF_FloatArray(const T & _array)
    {
        this->data.array = (float*)malloc(sizeof(float) * _array.size());

        if(this->data.array != NULL){
            this->data.length = _array.size();
            float *ptr = this->data.array;
            for(typename T::iterator i = _array.begin(); i != _array.end(); i++){
                *ptr++ = *i;
            }
        }
    }

    template<typename T> NNF_FloatArray::NNF_FloatArray(T _array, int32_t length)
    {
        this->data.array = (float*)malloc(sizeof(float) * length);
        if(this->data.array != NULL){
            this->data.length = length;
            float *ptr = this->data.array;
            for(int32_t i = 0; i < length; i++){
                *ptr++ = _array[i];
            }
        }
    }

    NNF_FloatArray::~NNF_FloatArray()
    {
        free(this->data.array);
    }

    void NNF_FloatArray::concat_(const NNF_FloatArray & _array)
    {
        const int32_t length = this->data.length + _array.length();
        float *new_array= (float*)malloc(sizeof(float) * length);

        if(new_array != NULL){
            float *ptr1 = new_array; 
            float *ptr2 = this->data.array;
            float const *ptr3 = _array._get_ptr();

            for(int32_t i = 0; i < this->data.length; i++){
                *ptr1++ = *ptr2++;
            }
            for(int32_t i = 0; i < _array.length(); i++){
                *ptr1++ = *ptr3++;
            }

            free(this->data.array);
            this->data.array = new_array;
            this->data.length = length;
        }else{
            std::cout<<"malloc failed!"<<std::endl;
            exit(-1);
        }
    }

    NNF_FloatArray NNF_FloatArray::concat(const NNF_FloatArray & _array)
    {
        const int32_t length = this->data.length + _array.length();
        float *new_array= (float*)malloc(sizeof(float) * length);

        if(new_array != NULL){
            float *ptr1 = new_array; 
            float *ptr2 = this->data.array;
            float const *ptr3 = _array._get_ptr();

            for(int32_t i = 0; i < this->data.length; i++){
                *ptr1++ = *ptr2++;
            }
            for(int32_t i = 0; i < _array.length(); i++){
                *ptr1++ = *ptr3++;
            }

            return NNF_FloatArray(new_array, length);
        }else{
            std::cout<<"malloc failed!"<<std::endl;
            exit(-1);
        }
    }

    float & NNF_FloatArray::operator[](int32_t index)
    {
        assert(index < this->data.length);
        return *(this->data.array + index);
    }

    void NNF_FloatArray::operator=(const NNF_FloatArray & _array)
    {
        float *new_array = (float*)malloc(sizeof(float) * this->data.length);

        if(new_array != NULL){
            free(this->data.array);
            this->data.length = _array.length();
            this->data.array = new_array;
            
            float *ptr1 = this->data.array;
            float const *ptr2 = _array._get_ptr();

            for (int32_t i = 0; i < this->data.length; i++){
                *ptr1++ = *ptr2++;
            }
        }
    }

    template<class T> void NNF_FloatArray::operator=(const T & _array)
    {
        float *new_array = (float*)malloc(sizeof(float) * _array.size());

        if(new_array != NULL){
            free(this->data.array);
            this->data.length = _array.size();
            this->data.array = new_array;
            float *ptr = this->data.array;
            for(typename T::iterator i = _array.begin(); i != _array.end(); i++){
                *ptr++ = *i;
            }
        }
    }

    float * const NNF_FloatArray::_get_ptr() const
    {
        return this->data.array;
    }

    const int32_t NNF_FloatArray::length() const
    {
        return this->data.length;
    }

    const int32_t NNF_FloatArray::size() const
    {
        return this->data.length;
    }


    // NNF_IntArray 
    /*
    Author: LineZ
    Update: 2020/6/12
    */
    NNF_IntArray::NNF_IntArray(int32_t * _array, int32_t length)
    {
        assert(_array != NULL);
        this->data.array = _array;
        this->data.length = length;
    }

    NNF_IntArray::NNF_IntArray(const std::initializer_list<int32_t> & _array)
    {
        this->data.array = (int32_t*)malloc(sizeof(int32_t) * _array.size());
        if(this->data.array != NULL){
            this->data.length = _array.size();
            int32_t *ptr = this->data.array;
            for(std::initializer_list<int32_t>::iterator i = _array.begin(); i != _array.end(); i++){
                *ptr++ = *i;
            }
        }
    }

    template<class T> NNF_IntArray::NNF_IntArray(const T & _array)
    {
        this->data.array = (int32_t*)malloc(sizeof(int32_t) * _array.size());
        if (this->data.array != NULL){
            this->data.length = _array.size();

            int32_t *ptr = this->data.array;
            for(typename T::iterator i = _array.begin(); i != _array.end(); i++){
                *ptr++ = *i;
            }
        }
    }

    template<typename T> NNF_IntArray::NNF_IntArray(T _array, int32_t length)
    {
        this->data.array = (int32_t*)malloc(sizeof(int32_t) * length);

        if(this->data.array != NULL){
            this->data.length = length;
            int32_t *ptr = this->data.array;
            for(int i = 0; i < length; i++){
                *ptr++ = _array[i];
            }
        }
    }

    NNF_IntArray::~NNF_IntArray()
    {
        free(this->data.array);
    }

    void NNF_IntArray::concat_(const NNF_IntArray & _array)
    {
        const int32_t length = this->data.length + _array.length();
        int32_t *new_array = (int32_t*)malloc(sizeof(int32_t) * length);

        if(new_array != NULL){
            int32_t *ptr1 = new_array;
            int32_t *ptr2 = this->data.array;
            int32_t const *ptr3 = _array._get_ptr();

            for(int i = 0; i < this->data.length; i++){
                *ptr1++ = *ptr2++;
            }
            for(int i = 0; i < _array.length(); i++){
                *ptr1++ = *ptr3++;
            }

            free(this->data.array);
            this->data.array = new_array;
            this->data.length = length;
        }else{
            std::cout<<"malloc failed!"<<std::endl;
            exit(-1);
        }
    }

    NNF_IntArray NNF_IntArray::concat(const NNF_IntArray & _array)
    {
        const int32_t length = this->data.length + _array.length();
        int32_t *new_array = (int32_t*)malloc(sizeof(int32_t) * length);

        if(new_array != NULL){
            int32_t *ptr1 = new_array;
            int32_t *ptr2 = this->data.array;
            int32_t const *ptr3 = _array._get_ptr();

            for(int i = 0; i < this->data.length; i++){
                *ptr1++ = *ptr2++;
            }
            for(int i = 0; i < _array.length(); i++){
                *ptr1++ = *ptr3++;
            }
            return NNF_IntArray(new_array, length);
        }else{
            std::cout<<"malloc failed!"<<std::endl;
            exit(-1);
        }
    }

    int32_t & NNF_IntArray::operator[](int32_t index)
    {
        assert(index < this->data.length);
        return *(this->data.array+index);
    }

    void NNF_IntArray::operator=(const NNF_IntArray & _array)
    {
        int32_t *new_array = (int32_t*)malloc(sizeof(int32_t) * this->data.length);

        if(new_array != NULL){
            free(this->data.array);
            this->data.length = _array.length();
            this->data.array = new_array;

            int32_t *ptr1 = this->data.array;
            int32_t *ptr2 = _array._get_ptr();

            for(int32_t i = 0; i < this->data.length; i++){
                *ptr1++ = *ptr2++;
            }
        }
    }

    template<class T> void NNF_IntArray::operator=(const T & _array)
    {
        int32_t *new_array = (int32_t*)malloc(sizeof(int32_t) * _array.size());
        
        if(new_array != NULL){
            free(this->data.array);
            this->data.length = _array.size();
            this->data.array = new_array;

            int32_t *ptr1 = this->data.array;
            for(typename T::iterator i = _array.begin(); i != _array.end(); i++){
                *ptr1++ = *i;
            }
        }
    }

    int32_t * const NNF_IntArray::_get_ptr() const
    {
        return this->data.array;
    }

    const int32_t NNF_IntArray::length() const
    {
        return this->data.length;
    }

    const int32_t NNF_IntArray::size() const
    {
        return this->data.length;
    }

}