#include <string>
#include <cstring>
#include <iostream>
#include <chrono>
#include <tuple>

#include "gpu_infer.hpp"
#include "../model/gpu_transformer.hpp"
#include "../util.hpp"
#include "../backend/gpu_backend.hpp"

GPU_Infer::GPU_Infer() {
    this->mt = MODEL_LLAMA;
    this->bt = GPU;
    this->model = NULL;
    this->backend = NULL;

    this->maxSeqLen = 256;
    this->temperature = 0.0;    // 0.0：贪婪解码
    this->topp = 1.0f;          // 核采样中的top-p值
    this->steps = 256;          // 运行的步骤数
    this->rngSeed = 0;          // 随机数种子
}

GPU_Infer::~GPU_Infer(){
    if(this->model != NULL) {
        delete this->model;
    }

    if(this->backend != NULL) {
        delete this->backend;
    }
}

long timeInMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

void printSafeString(const std::string& piece) {

    if (piece.empty()) {
        return;
    }
    if (piece.size() == 1) {
        unsigned char byteVal = static_cast<unsigned char>(piece[0]);
        if (!(std::isprint(byteVal) || std::isspace(byteVal))) {
            return; 
        }
    }

    std::cout << piece << std::flush;
}

void GPU_Infer::build(std::string modelPath,std::string tknzrPath, ModelType mt, BackendType bt){
     if(mt == MODEL_LLAMA){
        model = new GPU_Transformer();
        model->initializeModel(modelPath);
     } else{
        std::cerr<< "[ERROR:] Unsupported model type\n"<<std::endl;
        exit(1);
    }
    
    
    this->bt = bt;
    backend = new GPU_Backend();
    
    tokenizer = new CTokenizer();
    sampler = new CSampler();
	model->backend = backend;
    tokenizer->initializeTokenizer(tknzrPath,model->config.vocabSize);
    sampler->initializeSampler(model->config.vocabSize, temperature, topp, rngSeed);
}

std::tuple<std::string, int, long> GPU_Infer::generate(std::string prompt) {
    std::string result;
    std::string emptyPrompt = "";

    int numPromptTokens = 0;
    int* promptTokens = new int[prompt.size() + 3]; // BOS, EOS, 和空终止符

    model->encode(tokenizer, prompt, 1, 0, promptTokens, &numPromptTokens);

    if (numPromptTokens < 1) {
        std::cerr<<"[ERROR:] Something is wrong, expected at least 1 prompt token\n"<<std::endl;
        exit(EXIT_FAILURE);
    }

    long start = 0;      
    long end = 0;        
    long elapsed;
    int next;                    
    int token = promptTokens[0]; 
    int pos = 0;               

    while (pos < steps) {

        float* logits = model->forward(token, pos, backend);

        if (pos < numPromptTokens - 1) {

            next = promptTokens[pos + 1];

        } else {
            next = sampler->sample(logits);
        }
        pos++;

        if (next == 1) { 
            break; 
        }

        char* decodedToken = model->decode(tokenizer, token, next);
        result += decodedToken;
        printSafeString(decodedToken);
        token = next;

        if (start == 0) { 
            start = timeInMs(); 
        }
    }
    printf("\n");

    if (pos > 1) {
        end = timeInMs();
        elapsed = end - start;
    }

    delete[] promptTokens;
    return std::make_tuple(result, pos - 1, elapsed);
}
