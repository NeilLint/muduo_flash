#include <iostream>
#include <string>
#include <vector>
#include <fstream> 
#include "./infer/gpu_infer.hpp"
#include "util.hpp"
#include <tuple>
std::string modelPath;
std::string tknzrPath;

ModelType mt;
BackendType bt;
std::vector<std::string> prompts; 

void parse(int argc, char* argv[]){

    if (argc <= 1 || argv[1] == nullptr || std::string(argv[1]).empty() ||
        argc <= 2 || argv[2] == nullptr || std::string(argv[2]).empty()) {
        std::cout << "Usage: muduo [model_path] [tokenizer_path] [prompt] [modelType] [backend]" << std::endl;
        exit(1); 
    }
    modelPath = argv[1];
    tknzrPath = argv[2];

    if (argc > 3 && argv[3] != nullptr && std::string(argv[3]) != "") {
        std::string prompt = argv[3];
        std::ifstream infile(prompt);
        if (infile.good()) {
            std::string line;
            while (std::getline(infile, line)) {
                if (!line.empty()) prompts.push_back(line);
            }
            infile.close();
            std::cout << "[MSG:] Loaded prompts from file: " << prompt << std::endl;
        } else {
            prompts.push_back(prompt); 
            std::cout << "[MSG:] Using default prompt: " << prompt << std::endl;
        }
    } else {
        std::cout << "[MSG:] Using default prompt \"once upon a time,\"" << std::endl;  
        prompts.push_back("once upon a time,");
    }

    if (argc > 4 && argv[4] != nullptr && std::string(argv[4]) != "") {
        std::string modelTypeStr = argv[4];
        if (modelTypeStr == "llama") {
            mt = ModelType::MODEL_LLAMA;
        } else {
            std::cout << "Unknown model type: " << modelTypeStr << std::endl;
        }
    }

    if (argc > 5 && argv[5] != nullptr && std::string(argv[5]) != "") {
        std::string backendStr = argv[5];
        if (backendStr == "cpu") {
            bt = BackendType::CPU;
        } else if (backendStr == "gpu") {
            bt = BackendType::GPU;
            std::cout << "[MSG:] Selected GPU backend" << std::endl;
        } else {
            std::cerr << "[ERROR:] Unsupported backend: " << backendStr << std::endl;
            exit(1);
        }
    }
}

void init(){
    modelPath = "";        
    tknzrPath = "";     
    mt = ModelType::MODEL_LLAMA;  
    bt = BackendType::CPU; 
}

std::vector<std::string> loadResponses(const std::string& filename) {
    std::ifstream inFile(filename);
    std::vector<std::string> responses;
    if (!inFile.is_open()) {
        std::cerr << "[ERROR:] Can't open file " << filename << std::endl;
        exit(1);
    }

    std::string line, current;
    while (std::getline(inFile, line)) {
        if (line.empty()) {
            if (!current.empty()) {
                responses.push_back(current);
                current.clear();
            }
        } else {
            if (!current.empty()) current += "\n";
            current += line;
        }
    }
    if (!current.empty()) {
        responses.push_back(current);
    }
    return responses;
}

int main(int argc, char* argv[]){

    init();
    parse(argc, argv);
    bt = BackendType::GPU;
    GPU_Infer infer;
    infer.build(modelPath, tknzrPath, mt, bt);
    int totalTokens = 0;
    long totalTimeMs = 0;
    
    std::vector<std::tuple<std::string, int, long>> results;
    std::vector<std::string> responses = loadResponses("data/responses.txt");

    for (size_t i = 0; i < prompts.size(); ++i) {
        auto [output, tokens, timeMs] = infer.generate(prompts[i]);
        if (output != responses[i]) {
            std::cerr << "[ERROR:] Result mismatch at sample " << i + 1 << "!"<< tokens << " tokens in " << timeMs << " ms, throughput = "
                    << (tokens / (timeMs / 1000.0)) << " tokens/s" << std::endl;
        }
        else {
            std::cout << "Sample " << i+1 << " (Result Validation PASS): " << tokens << " tokens in " << timeMs << " ms, throughput = "
                    << (tokens / (timeMs / 1000.0)) << " tokens/s" << std::endl;
	    std::cout << std::endl;
        }
        results.emplace_back(output, tokens, timeMs);
        totalTokens += tokens;
        totalTimeMs += timeMs;
    }
    
    if (totalTimeMs > 0) {
        double avgThroughput = totalTokens / (totalTimeMs / 1000.0);
        std::cout << "========== Summary ==========\n";
        std::cout << "Total Samples: " << prompts.size() << "\n";
        std::cout << "Total Tokens: " << totalTokens << "\n";
        std::cout << "Total Time: " << totalTimeMs << " ms\n";
        std::cout << "Average Throughput: " << avgThroughput << " tokens/s\n";
    }
}
