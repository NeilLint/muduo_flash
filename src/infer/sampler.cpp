#include "sampler.hpp"
#include <algorithm>
#include <iostream>

CSampler::CSampler() {};

CSampler::~CSampler() {
    if (this->candidates != NULL)
        delete this->candidates;
};


int GreedySample(const float* probs, int size) {
    int bestIdx = 0;
    float maxProb = probs[0];
    for (int i = 1; i < size; i++) {
        if (probs[i] > maxProb) {
            bestIdx = i;
            maxProb = probs[i];
        }
    }
    return bestIdx;
}

int MultinomialSample(const float* probs, int size, float randVal) {
    float cumulativeSum = 0.0f;
    for (int i = 0; i < size; i++) {
        cumulativeSum += probs[i];
        if (randVal < cumulativeSum) {
            return i;
        }
    }
    return size - 1;
}

int CompareProbDescending(const void* a, const void* b) {
    const Candidate* p1 = (const Candidate*) a;
    const Candidate* p2 = (const Candidate*) b;
    return (p1->probability > p2->probability) ? -1 : (p1->probability < p2->probability) ? 1 : 0;
}

int TopPSample(float* probabilities, int size, float topP, Candidate* candidates, float randomValue) {
    int count = 0;
    const float threshold = (1.0f - topP) / (size - 1);
   
    for (int i = 0; i < size; i++) {
        if (probabilities[i] >= threshold) {
            candidates[count].tokenIndex = i;
            candidates[count].probability = probabilities[i];
            count++;
        }
    }
    qsort(candidates, count, sizeof(Candidate), CompareProbDescending);

    float cumulativeProb = 0.0f;
    int lastIndex = count - 1; 
    for (int i = 0; i < count; i++) {
        cumulativeProb += candidates[i].probability;
        if (cumulativeProb > topP) {
            lastIndex = i;
            break; 
        }
    }

    float target = randomValue * cumulativeProb;
    float cdf = 0.0f;
    for (int i = 0; i <= lastIndex; i++) {
        cdf += candidates[i].probability;
        if (target < cdf) {
            return candidates[i].tokenIndex;
        }
    }
    return candidates[lastIndex].tokenIndex; 
}

void CSampler::initializeSampler(int vocabSize, float temperature, float ttopP, unsigned long long rngSeed) {
    this->vocabSize = vocabSize;
    this->temperature = temperature;
    this->topP = topP;
    this->rngState = rngSeed;
    this->candidates = new Candidate[this->vocabSize];
}

unsigned int randomu32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float randomf32(unsigned long long *state) { 
    return (randomu32(state) >> 8) / 16777216.0f;
}

void CSampler::softmax(float* x, int n) {
     if (n <= 0 || x == nullptr) return;
 
     // 为了数值稳定性，先减去最大值
     float max_val = x[0];
     for (int i = 1; i < n; ++i) {
         if (x[i] > max_val) max_val = x[i];
     }
 
     // 计算 e^(x[i] - max_val) 和 sum
     float sum = 0.0f;
     for (int i = 0; i < n; ++i) {
         x[i] = std::exp(x[i] - max_val);
         sum += x[i];
     }
 
     // 归一化
     for (int i = 0; i < n; ++i) {
         x[i] /= sum;
     }
 }



int CSampler::sample(float* logits) {
    int next;
    if (this->temperature == 0.0f) {
        next = GreedySample(logits, this->vocabSize);
    } else {
        for (int i = 0; i < this->vocabSize; i++) { 
            logits[i] /= this->temperature; 
        }
        softmax(logits, this->vocabSize);  
        float randomValue = randomf32(&this->rngState);
        if (this->topP <= 0 || this->topP >= 1) {
            next = MultinomialSample(logits, this->vocabSize, randomValue);
        } else {
            next = TopPSample(logits, this->vocabSize, this->topP, this->candidates, randomValue);
        }
    }
    return next;
}