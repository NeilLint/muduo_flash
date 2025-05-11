#ifndef SAMPLER_HPP
#define SAMPLER_HPP


struct Candidate {
    float probability;
    int tokenIndex;
};

class CSampler{
private:
    void softmax(float* x, int n);
public:
    int vocabSize;
    Candidate* candidates; // 在 top-p 采样中使用的缓冲区
    float temperature;
    float topP;
    unsigned long long rngState;
    CSampler();
    ~CSampler();
    void initializeSampler(int vocabSize, float temperature, float topP, unsigned long long rngSeed);
    void freeSampler();
    int sample(float* logits);
};

#endif