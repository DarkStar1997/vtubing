#include "onnx_session.h"
#include <stdexcept>

OnnxSession::OnnxSession(const std::string& modelPath)
    : env_(ORT_LOGGING_LEVEL_WARNING, "vtuber_live")
{
    sessionOptions_.SetIntraOpNumThreads(1);
    sessionOptions_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);

    // Input metadata
    Ort::TypeInfo inputType = session_.GetInputTypeInfo(0);
    auto inputTensor = inputType.GetTensorTypeAndShapeInfo();
    inputShape_ = inputTensor.GetShape();
    // Replace dynamic dims (negative values) with 1
    for (auto& d : inputShape_) {
        if (d <= 0) d = 1;
    }
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNameAlloc = session_.GetInputNameAllocated(0, allocator);
    inputName_ = inputNameAlloc.get();

    inputElements_ = 1;
    for (auto d : inputShape_) {
        if (d <= 0) d = 1;
        inputElements_ *= static_cast<size_t>(d);
    }

    // Output metadata
    size_t numOutputs = session_.GetOutputCount();
    outputNames_.resize(numOutputs);
    outputShapes_.resize(numOutputs);
    outputNamePtrs_.resize(numOutputs);
    for (size_t i = 0; i < numOutputs; i++) {
        auto nameAlloc = session_.GetOutputNameAllocated(i, allocator);
        outputNames_[i] = nameAlloc.get();
        Ort::TypeInfo outType = session_.GetOutputTypeInfo(i);
        auto outTensor = outType.GetTensorTypeAndShapeInfo();
        outputShapes_[i] = outTensor.GetShape();
        outputNamePtrs_[i] = outputNames_[i].c_str();
    }
}

std::vector<std::vector<float>> OnnxSession::run(const float* inputData, size_t inputSize)
{
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, const_cast<float*>(inputData), inputSize,
        inputShape_.data(), inputShape_.size());

    const char* inputNamePtr = inputName_.c_str();
    auto outputTensors = session_.Run(
        Ort::RunOptions{nullptr},
        &inputNamePtr, &inputTensor, 1,
        outputNamePtrs_.data(), outputNamePtrs_.size());

    std::vector<std::vector<float>> results(outputTensors.size());
    for (size_t i = 0; i < outputTensors.size(); i++) {
        auto& tensor = outputTensors[i];
        auto tensorInfo = tensor.GetTensorTypeAndShapeInfo();
        size_t count = tensorInfo.GetElementCount();
        float* data = tensor.GetTensorMutableData<float>();
        results[i].assign(data, data + count);
    }
    return results;
}
