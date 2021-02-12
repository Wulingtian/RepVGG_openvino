#include "classifier.h"

Classifier::Classifier(){}

Classifier::~Classifier(){}

std::vector<float> Classifier::prepareImage(cv::Mat &src_img){
    std::vector<float> result(INPUT_W * INPUT_H * 3);
    float *data = result.data();
    float ratio = float(INPUT_W) / float(src_img.cols) < float(INPUT_H) / float(src_img.rows) ? float(INPUT_W) / float(src_img.cols) : float(INPUT_H) / float(src_img.rows);
    cv::Mat flt_img = cv::Mat::zeros(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
    cv::Mat rsz_img = cv::Mat::zeros(cv::Size(src_img.cols*ratio, src_img.rows*ratio), CV_8UC3);
    cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    flt_img.convertTo(flt_img, CV_32FC3);

    int channelLength = INPUT_W * INPUT_H;
    std::vector<cv::Mat> split_img = {
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data + channelLength * 2),
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data + channelLength),
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data)
    };

    auto pr_start = std::chrono::high_resolution_clock::now();
    cv::split(flt_img, split_img);
    for (int i = 0; i < 3; i++) {
            split_img[i] = (split_img[i]/255 - img_mean[i]) / img_std[i];
    }
    auto pr_end = std::chrono::high_resolution_clock::now();

    auto po_ms = std::chrono::duration<float, std::milli>(pr_end - pr_start).count();
    //std::cout << "********** " << po_ms << " ms." << "********** " << std::endl;
    return result;
}

//初始化
bool Classifier::init(string xml_path,int input_w, int input_h, int num_class, std::vector<float> i_mean,std::vector<float> i_std){
    _xml_path = xml_path;
    INPUT_W = input_w;
    INPUT_H = input_h;
    img_mean = i_mean;
    img_std = i_std;
    NUM_CLASS = num_class;
    Core ie;
    auto cnnNetwork = ie.ReadNetwork(_xml_path); 
    //输入设置
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;
    input->setPrecision(Precision::FP32);
    input->getInputData()->setLayout(Layout::NCHW);
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);
    //输出设置
    _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
    for (auto &output : _outputinfo) {
        output.second->setPrecision(Precision::FP32);
    }
    //获取可执行网络
    //_network =  ie.LoadNetwork(cnnNetwork, "GPU");
    _network =  ie.LoadNetwork(cnnNetwork, "CPU");
    return true;
}

//释放资源
bool Classifier::uninit(){
    return true;
}

//处理图像获取结果
std::vector<float> Classifier::process_frame(Mat& inframe){
    cv::Mat showImage = inframe.clone();
    std::vector<float> pr_img = prepareImage(inframe);
    InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();
    Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();

    memcpy(blob_data, pr_img.data(), 3 * INPUT_H * INPUT_W * sizeof(float));

    //执行预测
    infer_request->Infer();
    //获取各层结果
    std::vector<float> output_blob;
    for (auto &output : _outputinfo) {
        auto output_name = output.first;
        Blob::Ptr blob = infer_request->GetBlob(output_name);
        LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
        for (int i; i< NUM_CLASS; ++i){
        output_blob.push_back(blobMapped.as<float *>()[i]);
        }    
    }
    return output_blob;
}

