#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

class OnnxSession {
public:
    OnnxSession(const std::string& modelPath);

    // Run inference. inputData points to a flat float array matching inputShape.
    // Returns vector of output float arrays (one per output tensor).
    std::vector<std::vector<float>> run(const float* inputData, size_t inputSize);

    // Input/output metadata
    const std::vector<int64_t>& inputShape() const { return inputShape_; }
    const std::string& inputName() const { return inputName_; }
    size_t inputElementCount() const { return inputElements_; }
    const std::vector<std::string>& outputNames() const { return outputNames_; }
    const std::vector<std::vector<int64_t>>& outputShapes() const { return outputShapes_; }

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::SessionOptions sessionOptions_;

    std::string inputName_;
    std::vector<int64_t> inputShape_;
    size_t inputElements_ = 0;

    std::vector<std::string> outputNames_;
    std::vector<std::vector<int64_t>> outputShapes_;
    std::vector<const char*> outputNamePtrs_;
};
