#ifndef HIP_ASSERT_H
#define HIP_ASSERT_H

#define __HIP_PLATFORM_AMD__
#include <iostream>


#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}


#endif
