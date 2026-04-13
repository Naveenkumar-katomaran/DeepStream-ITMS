#include <algorithm>
#include <cmath>
#include <vector>
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseCustomYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

static NvDsInferObjectDetectionInfo convertBox(float cx, float cy, float w, float h, float netW, float netH) {
    NvDsInferObjectDetectionInfo res;
    res.left = std::max(0.0f, (cx - w / 2.0f));
    res.top = std::max(0.0f, (cy - h / 2.0f));
    res.width = std::min(netW - res.left, w);
    res.height = std::min(netH - res.top, h);
    return res;
}

extern "C" bool NvDsInferParseCustomYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
    
    if (outputLayersInfo.empty()) return false;

    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    float* data = (float*)layer.buffer;

    // YOLOv8 output: [ (4 + num_classes) x num_elements ]
    const int num_classes = detectionParams.numClassesConfigured;
    
    // num_elements = (W/8)*(H/8) + (W/16)*(H/16) + (W/32)*(H/32)
    const int num_elements = (networkInfo.width / 8) * (networkInfo.height / 8) + 
                             (networkInfo.width / 16) * (networkInfo.height / 16) + 
                             (networkInfo.width / 32) * (networkInfo.height / 32);

    for (int i = 0; i < num_elements; i++) {
        float max_prob = 0.0f;
        int max_idx = -1;

        for (int c = 0; c < num_classes; c++) {
            float prob = data[(4 + c) * num_elements + i];
            if (prob > max_prob) {
                max_prob = prob;
                max_idx = c;
            }
        }

        if (max_idx != -1 && max_prob > detectionParams.perClassThreshold[max_idx]) {
            float cx = data[0 * num_elements + i];
            float cy = data[1 * num_elements + i];
            float w = data[2 * num_elements + i];
            float h = data[3 * num_elements + i];

            NvDsInferObjectDetectionInfo obj = convertBox(cx, cy, w, h, networkInfo.width, networkInfo.height);
            obj.classId = max_idx;
            obj.detectionConfidence = max_prob;
            objectList.push_back(obj);
        }
    }

    return true;
}
