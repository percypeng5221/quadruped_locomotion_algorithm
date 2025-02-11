/******************************************************************************
* Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the NVIDIA CORPORATION nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
******************************************************************************/

#pragma once

#include <limits>
#include <memory>
#include <ostream>

#include <thrust/device_vector.h>

namespace c2h
{

class custom_type_state_t
{
  std::size_t m_key{};
  std::size_t m_val{};

public:
  __host__ __device__ void set_key(std::size_t key) { m_key = key; }
  __host__ __device__ std::size_t get_key() const { return m_key; }
  __host__ __device__ void set_val(std::size_t val) { m_val = val; }
  __host__ __device__ std::size_t get_val() const { return m_val; }
};

template <template<typename> class... Policies>
class custom_type_t : public custom_type_state_t
                    , public Policies<custom_type_t<Policies...>>...
{

public:
  friend __host__ std::ostream &operator<<(std::ostream &os, 
                                           const custom_type_t &self) 
  { 
    return os << "{ " << self.get_key() << ", " << self.get_val() << " }";
  }

};

template <class CustomType>
class less_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the 
  // Microsoft host compiler does not.
  char workaround_msvc;

public:
  __host__ __device__ bool operator<(const CustomType& other) const
  {
    return static_cast<const CustomType&>(*this).get_key() 
         < other.get_key();
  }
};

template <class CustomType>
class equal_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the 
  // Microsoft host compiler does not.
  char workaround_msvc;

public:
  __host__ __device__ bool operator==(const CustomType& other) const
  {
    const CustomType& self = static_cast<const CustomType&>(*this);
    
    return self.get_key() == other.get_key() &&
           self.get_val() == other.get_val();
  }
};

template <class CustomType>
class subtractable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the 
  // Microsoft host compiler does not.
  char workaround_msvc;

public:
  __host__ __device__ CustomType operator-(const CustomType& other) const
  {
    CustomType result{};

    const CustomType& self = static_cast<const CustomType&>(*this);

    result.set_key(self.get_key() - other.get_key());
    result.set_val(self.get_val() - other.get_val());
    
    return result;
  }
};

template <class CustomType>
class accumulateable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the 
  // Microsoft host compiler does not.
  char workaround_msvc;

public:
  __host__ __device__ CustomType operator+(const CustomType& other) const
  {
    CustomType result{};

    const CustomType& self = static_cast<const CustomType&>(*this);

    result.set_key(self.get_key() + other.get_key());
    result.set_val(self.get_val() + other.get_val());
    
    return result;
  }
};

} // c2h

namespace std {
  template<template<typename> class... Policies> 
  class numeric_limits<c2h::custom_type_t<Policies...>> 
  {
  public:
     static c2h::custom_type_t<Policies...> max() 
     {
       c2h::custom_type_t<Policies...> val;
       val.set_key(std::numeric_limits<std::size_t>::max());
       val.set_val(std::numeric_limits<std::size_t>::max());
       return val;
     }

     static c2h::custom_type_t<Policies...> lowest() 
     {
       c2h::custom_type_t<Policies...> val;
       val.set_key(std::numeric_limits<std::size_t>::lowest());
       val.set_val(std::numeric_limits<std::size_t>::lowest());
       return val;
     }
  };
}

