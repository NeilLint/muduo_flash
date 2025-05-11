#ifndef CTRANSFORMER_HPP
#define CTRANSFORMER_HPP

#include "gpu_model.hpp"
#include "modelConfig.hpp"
#include "../infer/gpu_runState.hpp"
#include <unistd.h>
#include <string>

class GPU_Transformer:public GPU_Model{
public:
    GPU_Transformer();
    ~GPU_Transformer(); 
};

#endif