#include <string>
#include <cstring>
#include <iostream>
#include <chrono>
#include <tuple>
#include <stdexcept>

#include "gpu_infer.hpp"
#include "../model/gpu_transformer.hpp"
#include "../util.hpp"
#include "../backend/gpu_backend.hpp"

GPU_Infer::GPU_Infer()
{
    this->mt = MODEL_LLAMA;
    this->bt = GPU;
    this->model = nullptr;
    this->backend = nullptr;
    this->tokenizer = nullptr;
    this->sampler = nullptr;

    this->maxSeqLen = 256;
    this->temperature = 0.0; // 0.0：贪婪解码
    this->topp = 1.0f;       // 核采样中的top-p值
    this->steps = 256;       // 运行的步骤数
    this->rngSeed = 0;       // 随机数种子
}

GPU_Infer::~GPU_Infer()
{
    if (this->model != nullptr)
    {
        delete this->model;
    }

    if (this->backend != nullptr)
    {
        delete this->backend;
    }

    if (this->tokenizer != nullptr)
    {
        delete this->tokenizer;
    }

    if (this->sampler != nullptr)
    {
        delete this->sampler;
    }
}

long timeInMs()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

void printSafeString(const std::string &piece)
{

    if (piece.empty())
    {
        return;
    }
    if (piece.size() == 1)
    {
        unsigned char byteVal = static_cast<unsigned char>(piece[0]);
        if (!(std::isprint(byteVal) || std::isspace(byteVal)))
        {
            return;
        }
    }

    std::cout << piece << std::flush;
}

void GPU_Infer::build(std::string modelPath, std::string tknzrPath, ModelType mt, BackendType bt)
{
    if (mt == MODEL_LLAMA)
    {
        model = new GPU_Transformer();
        model->initializeModel(modelPath);
    }
    else
    {
        throw std::runtime_error("unsupported model type");
    }

    this->bt = bt;
    backend = new GPU_Backend();

    tokenizer = new CTokenizer();
    sampler = new CSampler();
    model->backend = backend;
    tokenizer->initializeTokenizer(tknzrPath, model->config.vocabSize);
    sampler->initializeSampler(model->config.vocabSize, temperature, topp, rngSeed);

    model->config.attentionPattern = (model->config.numKvHeads == model->config.numHeads)
                                        ? CModelConfig::AttentionPattern::MHA
                                        : CModelConfig::AttentionPattern::GQA;
}

void GPU_Infer::setAttentionKernel(const std::string &kernelName)
{
    if (model == nullptr)
    {
        throw std::runtime_error("setAttentionKernel must be called after build()");
    }
    if (kernelName == "flash")
    {
        model->config.attentionKernel = CModelConfig::AttentionKernel::FLASH;
    }
    else if (kernelName == "classic")
    {
        model->config.attentionKernel = CModelConfig::AttentionKernel::CLASSIC;
    }
    else
    {
        throw std::runtime_error("unsupported attention kernel: " + kernelName);
    }
}

std::tuple<std::string, int, long> GPU_Infer::generate(std::string prompt)
{
    std::string result;
    int numPromptTokens = 0;
    int *promptTokens = new int[prompt.size() + 3]; // BOS, EOS, 和空终止符

    model->encode(tokenizer, prompt, 1, 0, promptTokens, &numPromptTokens);

    if (numPromptTokens < 1)
    {
        throw std::runtime_error("expected at least 1 prompt token");
    }

    long start = 0;
    long end = 0;
    long elapsed;
    int next;
    int token = promptTokens[0];
    int pos = 0;
    while (pos < steps)
    {
        model->forward(token, pos, backend);

        if (pos < numPromptTokens - 1)
        {

            next = promptTokens[pos + 1];
        }
        else
        {
            next = sampler->sample(model->state.h_logits);
        }
        pos++;

        if (next == 1)
        {
            break;
        }

        char *decodedToken = model->decode(tokenizer, token, next);
        result += decodedToken;
        printSafeString(decodedToken);
        token = next;

        if (start == 0)
        {
            start = timeInMs();
        }
    }
    printf("\n");

    if (pos > 1)
    {
        end = timeInMs();
        elapsed = end - start;
    }

    delete[] promptTokens;
    return std::make_tuple(result, pos - 1, elapsed);
}
