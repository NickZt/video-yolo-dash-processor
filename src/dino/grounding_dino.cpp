#include "grounding_dino.h"
#include "string_utility.hpp"
#include <algorithm>
#include <cstring>
#include <math.h>

using namespace cv;
using namespace std;
using namespace Ort;

GroundingDINO::GroundingDINO(string modelpath, float box_threshold,
                             string vocab_path, float text_threshold,
                             int num_threads, int img_size)
    : env(ORT_LOGGING_LEVEL_ERROR, "GroundingDINO")
{
    this->size[0] = img_size;
    this->size[1] = img_size;

    sessionOptions.SetIntraOpNumThreads(num_threads);
    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    ort_session =
        std::make_unique<Ort::Session>(env, modelpath.c_str(), sessionOptions);

    this->load_tokenizer(vocab_path);
    this->box_threshold = box_threshold;
    this->text_threshold = text_threshold;
}

bool GroundingDINO::load_tokenizer(std::string vocab_path)
{
    tokenizer.reset(new TokenizerClip);
    return tokenizer->load_tokenize(vocab_path);
}

void GroundingDINO::preprocess(Mat img)
{
    Mat rgbimg;
    cvtColor(img, rgbimg, COLOR_BGR2RGB);
    resize(rgbimg, rgbimg, cv::Size(this->size[0], this->size[1]));
    vector<cv::Mat> rgbChannels(3);
    split(rgbimg, rgbChannels);
    for (int c = 0; c < 3; c++)
    {
        rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1.0 / (255.0 * std[c]),
                                 (0.0 - mean[c]) / std[c]);
    }

    const int image_area = this->size[0] * this->size[1];
    this->input_img.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_img.data(), (float*)rgbChannels[0].data, single_chn_size);
    memcpy(this->input_img.data() + image_area, (float*)rgbChannels[1].data,
           single_chn_size);
    memcpy(this->input_img.data() + image_area * 2, (float*)rgbChannels[2].data,
           single_chn_size);
}

vector<DINOObject> GroundingDINO::detect(Mat srcimg, string text_prompt)
{
    this->preprocess(srcimg);
    const int srch = srcimg.rows, srcw = srcimg.cols;

    std::transform(text_prompt.begin(), text_prompt.end(), text_prompt.begin(),
                   ::tolower);
    string caption = strip(text_prompt);
    if (endswith(caption, ".") == 0)
    {
        caption += " .";
    }

    this->input_ids.resize(1);
    this->attention_mask.resize(1);
    this->token_type_ids.resize(1);
    std::vector<int64_t> ids;
    tokenizer->encode_text(caption, ids);
    int len_ids = ids.size();
    int trunc_len = len_ids <= this->max_text_len ? len_ids : this->max_text_len;
    input_ids[0].resize(trunc_len);
    token_type_ids[0].resize(trunc_len);
    attention_mask[0].resize(trunc_len);
    for (int i = 0; i < trunc_len; i++)
    {
        input_ids[0][i] = ids[i];
        token_type_ids[0][i] = 0;
        attention_mask[0][i] = ids[i] > 0 ? 1 : 0;
    }

    const int num_token = input_ids[0].size();
    vector<int> idxs;
    for (int i = 0; i < num_token; i++)
    {
        for (int j = 0; j < this->specical_tokens.size(); j++)
        {
            if (input_ids[0][i] == this->specical_tokens[j])
            {
                idxs.push_back(i);
            }
        }
    }

    len_ids = idxs.size();
    trunc_len = num_token <= this->max_text_len ? num_token : this->max_text_len;
    text_self_attention_masks.resize(1);
    text_self_attention_masks[0].resize(trunc_len * trunc_len);
    position_ids.resize(1);
    position_ids[0].resize(trunc_len);
    for (int i = 0; i < trunc_len; i++)
    {
        for (int j = 0; j < trunc_len; j++)
        {
            text_self_attention_masks[0][i * trunc_len + j] = (i == j ? 1 : 0);
        }
        position_ids[0][i] = 0;
    }
    int previous_col = 0;
    for (int i = 0; i < len_ids; i++)
    {
        const int col = idxs[i];
        if (col == 0 || col == num_token - 1)
        {
            text_self_attention_masks[0][col * trunc_len + col] = true;
            position_ids[0][col] = 0;
        }
        else
        {
            for (int j = previous_col + 1; j <= col; j++)
            {
                for (int k = previous_col + 1; k <= col; k++)
                {
                    text_self_attention_masks[0][j * trunc_len + k] = true;
                }
                position_ids[0][j] = j - previous_col - 1;
            }
        }
        previous_col = col;
    }

    const int seq_len = input_ids[0].size();
    std::vector<int64_t> input_img_shape = {1, 3, this->size[1], this->size[0]};
    std::vector<int64_t> input_ids_shape = {1, seq_len};
    std::vector<int64_t> pixel_mask_shape = {1, this->size[1], this->size[0]};
    std::vector<int64_t> pixel_mask(this->size[1] * this->size[0], 1);

    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back((Ort::Value::CreateTensor<float>(
        memory_info_handler, input_img.data(), input_img.size(),
        input_img_shape.data(), input_img_shape.size())));
    inputTensors.push_back((Ort::Value::CreateTensor<int64_t>(
        memory_info_handler, input_ids[0].data(), input_ids[0].size(),
        input_ids_shape.data(), input_ids_shape.size())));
    inputTensors.push_back((Ort::Value::CreateTensor<int64_t>(
        memory_info_handler, token_type_ids[0].data(), token_type_ids[0].size(),
        input_ids_shape.data(), input_ids_shape.size())));
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_handler, attention_mask[0].data(), attention_mask[0].size(),
        input_ids_shape.data(), input_ids_shape.size()));
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_handler, pixel_mask.data(), pixel_mask.size(),
        pixel_mask_shape.data(), pixel_mask_shape.size()));

    std::vector<Ort::Value> ort_outputs = ort_session->Run(
        Ort::RunOptions{nullptr}, input_names, inputTensors.data(),
        inputTensors.size(), output_names, 2);

    const float* ptr_logits = ort_outputs[0].GetTensorMutableData<float>();
    std::vector<int64_t> logits_shape =
        ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* ptr_boxes = ort_outputs[1].GetTensorMutableData<float>();
    const int outw = logits_shape[2];

    vector<int> filt_inds;
    vector<float> scores;
    for (int i = 0; i < logits_shape[1]; i++)
    {
        float max_data = 0;
        for (int j = 0; j < outw; j++)
        {
            float x = sigmoid(ptr_logits[i * outw + j]);
            if (max_data < x)
            {
                max_data = x;
            }
        }
        if (max_data > this->box_threshold)
        {
            filt_inds.push_back(i);
            scores.push_back(max_data);
        }
    }

    std::vector<DINOObject> objects;
    for (int i = 0; i < filt_inds.size(); i++)
    {
        const int ind = filt_inds[i];
        const int left_idx = 0, right_idx = 255;
        for (int j = left_idx + 1; j < right_idx; j++)
        {
            float x = sigmoid(ptr_logits[ind * outw + j]);
            if (x > this->text_threshold)
            {
                const int64_t token_id = input_ids[0][j];
                DINOObject obj;
                obj.text = this->tokenizer->tokenizer_idx2token[token_id];
                obj.prob = scores[i];

                int xmin =
                    int((ptr_boxes[ind * 4] - ptr_boxes[ind * 4 + 2] * 0.5) * srcw);
                int ymin =
                    int((ptr_boxes[ind * 4 + 1] - ptr_boxes[ind * 4 + 3] * 0.5) * srch);
                int w = int(ptr_boxes[ind * 4 + 2] * srcw);
                int h = int(ptr_boxes[ind * 4 + 3] * srch);
                obj.box = Rect(xmin, ymin, w, h);
                objects.push_back(obj);

                break;
            }
        }
    }
    return objects;
}
