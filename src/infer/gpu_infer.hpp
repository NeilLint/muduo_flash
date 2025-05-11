#ifndef GPU_INFER_HPP
#define GPU_INFER_HPP

#include<string>

#include "../model/gpu_model.hpp"
#include "../model/gpu_transformer.hpp"
#include "../backend/gpu_backend.hpp"
#include "../util.hpp"
#include "sampler.hpp"

class GPU_Infer{
    private:
        enum ModelType mt;
        enum BackendType bt;

        GPU_Backend *backend;
        GPU_Model *model;
        CTokenizer *tokenizer;
        CSampler *sampler;

        int maxSeqLen;
        float temperature;   
        float topp;          
        int steps;
        unsigned long long rngSeed;

    public:
        GPU_Infer();
        ~GPU_Infer();
        
        void build(std::string modelPath, std::string tknzrPath, ModelType mt, BackendType bt);
        std::tuple<std::string, int, long> generate(std::string prompt);
};

#endif