#ifndef DETECTOR_H
#define DETECTOR_H
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
using namespace std;
using namespace cv;
using namespace InferenceEngine;

class Classifier
{
public:

    Classifier();
    ~Classifier();
    //初始化
    bool init(string xml_path,int input_w, int input_h, int num_class, std::vector<float> i_mean, std::vector<float> i_std);
    //释放资源
    bool uninit();
    //处理图像获取结果
    std::vector<float> process_frame(Mat& inframe);

private:
    std::vector<float> prepareImage(cv::Mat &src_img);
    //存储初始化获得的可执行网络
    ExecutableNetwork _network;
    OutputsDataMap _outputinfo;
    string _input_name;
    //参数区
    string _xml_path;//OpenVINO模型xml文件路径
    int INPUT_W;
    int INPUT_H;
    int NUM_CLASS;
    std::vector<float> img_mean;
    std::vector<float> img_std;
};
#endif
