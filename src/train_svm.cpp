#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>
#include <opencv2/opencv.hpp>

struct Data{
    float CPU_Usage_Percent;
    float Memory_Usage_Percent;

    float Disk_Usage_Percent;
};

//steps:
// Read CSV file


// Extract numeric features
    const std::string MODEL_OUTPUT_PATH = "svm_model.yml";
    const std::string NORM_STATS_PATH = "norm_stats.yml";

    // Determine CSV path: prefer argv[1], otherwise try common relative locations
    std::string csvPath;
    
    std::vector<std::string> candidates = {
            "system_performance_data.csv",
            "data/system_performance_data.csv",
            "data\\system_performance_data.csv",
            "../data/system_performance_data.csv",
            "../../data/system_performance_data.csv"
    };
    for (const auto &p : candidates) {
        if (std::filesystem::exists(p)) {
            csvPath = p;
            break;
        }
    }
    

    if (csvPath.empty()) {
        std::cerr << "Error: could not find 'system_performance_data.csv'.\nTried common locations and no path was provided.\n";
        std::cerr << "Pass the CSV path as the first argument or place the file in the program working directory or in 'data/'.\n";
        return -1;
    }

    std::ifstream file(csvPath);
    if(!file.is_open()){
        std::cerr << "Error opening file: " << csvPath << std::endl;
        return -1;
    }
// Save the model

// Save normalization parameters


int main(){
    const std:: string MODEL_OUTPUT_PATH = "svm_model.yml";
    const std::string NORM_STATS_PATH = "norm_stats.yml";

    const std::string DATA_FILE_PATH = "C:\\Users\\kheza\\Desktop\\hidden_desktop\\svm-learning\\data\\system_performance_data.csv";
    std::ifstream file(DATA_FILE_PATH);

    if(!file.is_open()){
        std::cerr << "Error opening file: " << DATA_FILE_PATH << std::endl;
        return -1;
    }
    
    // Skip header row if it exists
    std::string header;
    std::getline(file, header);
    std::vector<Data> samples;
    std::string line;

    while(std::getline(file , line )){
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> row;
        // 3. Parse each line using the comma delimiter
        while (std::getline(ss, field, ',')) {
            row.push_back(field);
        }
        if(row.size() !=3){
            continue; // Skip malformed lines
        }
        Data data;
        data.CPU_Usage_Percent = std::stof(row[0]);
        data.Memory_Usage_Percent = std::stof(row[1]);
        data.Disk_Usage_Percent = std::stof(row[2]);
        samples.push_back(data);
            


    }

    file.close();   
    if (samples.empty()) {
        std::cerr << "Error: no samples loaded from CSV." << std::endl;
        return -1;
    }

    std::cout << "Loaded " << samples.size() << " samples." << std::endl;

    //create the matrix to hold the data

    const int nbSamples = static_cast<int>(samples.size());
    const int nbFeatures =3; // CPU, Memory, Disk

    cv::Mat featureMatrix(nbSamples, nbFeatures, CV_32F);

    for(int i=0; i< nbSamples; ++i){
        featureMatrix.at<float>(i,0) = samples[i].CPU_Usage_Percent;
        featureMatrix.at<float>(i,1) = samples[i].Memory_Usage_Percent;
        featureMatrix.at<float>(i,2) = samples[i].Disk_Usage_Percent;
    }

    std::cout << "Feature matrix created with size: " << featureMatrix.size() << std::endl;

    // compute mean and std deviation
    cv::Mat mean, stddev;
    cv::meanStdDev(featureMatrix, mean, stddev);

    // mean and stddev are returned as 1 x nbFeatures (type CV_64F)
    float mean_cpu   = static_cast<float>(mean.at<double>(0, 0));
    float mean_mem   = static_cast<float>(mean.at<double>(0, 1));
    float mean_disk  = static_cast<float>(mean.at<double>(0, 2));

    float std_cpu    = static_cast<float>(stddev.at<double>(0, 0));
    float std_mem    = static_cast<float>(stddev.at<double>(0, 1));
    float std_disk   = static_cast<float>(stddev.at<double>(0, 2));

    // Safety check (avoid division by zero later)
    if (std_cpu == 0)  std_cpu = 1.0f;
    if (std_mem == 0)  std_mem = 1.0f;
    if (std_disk == 0) std_disk = 1.0f;

    //normalize the data
    
    for (int i = 0; i < featureMatrix.rows; ++i) {
    featureMatrix.at<float>(i,0) = (featureMatrix.at<float>(i,0) - mean_cpu) / std_cpu;
    featureMatrix.at<float>(i,1) = (featureMatrix.at<float>(i,1) - mean_mem) / std_mem;
    featureMatrix.at<float>(i,2) = (featureMatrix.at<float>(i,2) - mean_disk) / std_disk;
}
    //save the normlised data
    cv::FileStorage fs(NORM_STATS_PATH, cv::FileStorage::WRITE);

    if (!fs.isOpened()) {
        std::cerr << "Error: Unable to open normalization stats file for writing."
                << std::endl;
        return -1;
    }

    // Save feature order (CRITICAL)
    fs << "feature_order" << "["
    << "CPU_Usage_Percent"
    << "Memory_Usage_Percent"
    << "Disk_Usage_Percent"
    << "]";

    // Save means
    fs << "mean" << "["
    << mean_cpu
    << mean_mem
    << mean_disk
    << "]";

    // Save standard deviations
    fs << "std" << "["
    << std_cpu
    << std_mem
    << std_disk
    << "]";

    fs.release();

    std::cout << "Normalization stats saved to: "
            << NORM_STATS_PATH << std::endl;


    // Train one-class SVM
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::ONE_CLASS);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setNu(0.1); // Set the nu parameter (adjustable)
    svm->setGamma(0.5); // Set the gamma parameter (adjustable)

    svm->train(featureMatrix, cv::ml::ROW_SAMPLE, cv::Mat());
    //train the SVM model
    std::cout << "SVM training completed." << std::endl;
    //save the model
    svm->save(MODEL_OUTPUT_PATH);
    std::cout << "SVM model saved to: " << MODEL_OUTPUT_PATH << std::endl;



    return 0;
}
