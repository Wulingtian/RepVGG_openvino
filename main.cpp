#include "classifier.h"

//python /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py --input_model /home/willer/RepVGG_TensorRT_int8_local/RepVGG-A0-simple.onnx --output_dir .
int main(int argc, char const *argv[])
{
    int test_echo = 20;
    Classifier* classifier = new Classifier;
    string xml_path = "../models/RepVGG-A0-simple.xml";
    std::vector<float> img_mean = {0.485, 0.456, 0.406};
    std::vector<float> img_std = { 0.229, 0.224, 0.225 };
    int num_class = 2;
    int input_w = 224;
    int input_h = 224;

    classifier->init(xml_path,input_w,input_h,num_class,img_mean,img_std);
    Mat src = imread("../test_imgs/21.jpg");
    int total=0;
    std::vector<float> output;
    for (int j = 0; j < test_echo; ++j) {
        auto t_start = std::chrono::high_resolution_clock::now();
        output = classifier->process_frame(src);

        auto t_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        total += ms;
        std::cout << "[ " << j << " ] " << ms << " ms." << std::endl;
    }

    int index = 0;
    float max = output.data()[0];
    for (int i = 0; i < num_class; i++) {
        if (max < output.data()[i]) {
            max = output.data()[i];
            index = i; 
        }
    }       
    //std::cout << output.data()[0] << " " << output.data()[1] << std::endl;    
    std::cout << "prob: " << index << std::endl;
    total /= test_echo;
    std::cout << "Average over " << test_echo << " runs is " << total << " ms." << std::endl;
}
