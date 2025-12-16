#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

//Goal of monitor.cpp
//Read live system metrics, normalize them, and decide if the laptop is behaving normally or not.

const std::string MODEL_PATH = "svm_model.yml";
const std::string NORM_STATS_PATH = "norm_stats.yml";

struct Stats {
    std::vector<std::string> feature_order;
    cv::Mat mean;
    cv::Mat stdev;
};

bool loadNormalizationStats(const std::string& path, Stats& stats) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        std::cerr << "Error: Cannot open normalization stats file: " << path << std::endl;
        return false;
    }
    
    // Read feature order
    cv::FileNode order_node = fs["feature_order"];
    if (order_node.isSeq()) {
        for (size_t i = 0; i < order_node.size(); ++i) {
            stats.feature_order.push_back(static_cast<std::string>(order_node[i]));
        }
    }
    
    // Read mean values
    cv::FileNode mean_node = fs["mean"];
    if (mean_node.isSeq()) {
        stats.mean = cv::Mat(1, mean_node.size(), CV_32F);
        for (int i = 0; i < mean_node.size(); ++i) {
            stats.mean.at<float>(0, i) = static_cast<float>(mean_node[i]);
        }
    }
    
    // Read std deviation values
    cv::FileNode std_node = fs["std"];
    if (std_node.isSeq()) {
        stats.stdev = cv::Mat(1, std_node.size(), CV_32F);
        for (int i = 0; i < std_node.size(); ++i) {
            stats.stdev.at<float>(0, i) = static_cast<float>(std_node[i]);
        }
    }
    
    fs.release();
    return true;
}

cv::Mat normalizeSample(const cv::Mat& sample, const Stats& stats) {
    cv::Mat normalized = sample.clone();
    
    for (int i = 0; i < sample.cols; ++i) {
        float mean_val = stats.mean.at<float>(0, i);
        float std_val = stats.stdev.at<float>(0, i);
        
        if (std_val == 0) std_val = 1.0f; // Avoid division by zero
        
        normalized.at<float>(0, i) = (sample.at<float>(0, i) - mean_val) / std_val;
    }
    
    return normalized;
}

int main() {
    // Load normalization stats
    Stats stats;
    if (!loadNormalizationStats(NORM_STATS_PATH, stats)) {
        return -1;
    }
    
    std::cout << "Loaded normalization stats. Feature order: ";
    for (const auto& feature : stats.feature_order) {
        std::cout << feature << " ";
    }
    std::cout << std::endl;
    // Basic validation of normalization stats
    if (stats.mean.empty() || stats.stdev.empty()) {
        std::cerr << "Error: normalization stats are empty or malformed." << std::endl;
        return -1;
    }
    if (stats.mean.cols != stats.stdev.cols) {
        std::cerr << "Error: mean and std size mismatch." << std::endl;
        return -1;
    }
    
    // Load SVM model
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(MODEL_PATH);
    if (svm.empty()) {
        std::cerr << "Error: Cannot load SVM model from: " << MODEL_PATH << std::endl;
        return -1;
    }
    
    std::cout << "SVM model loaded successfully!" << std::endl;
    
    // Example: Monitor a single sample
    // In real implementation, you would get these from system monitoring
    int nFeatures = stats.mean.cols;
    cv::Mat sample(1, nFeatures, CV_32F);
    // Populate with example values (or real monitoring values)
    if (nFeatures >= 3) {
        sample.at<float>(0, 0) = 55.0f;  // CPU usage
        sample.at<float>(0, 1) = 65.0f;  // Memory usage
        sample.at<float>(0, 2) = 75.0f;  // Disk usage
        for (int i = 3; i < nFeatures; ++i) sample.at<float>(0, i) = 0.0f;
    } else {
        for (int i = 0; i < nFeatures; ++i) sample.at<float>(0, i) = 0.0f;
    }

    std::cout << "\nMonitoring sample:" << std::endl;
    for (int i = 0; i < sample.cols; ++i) {
        std::cout << stats.feature_order.size() << i ? stats.feature_order[i] : std::to_string(i);
        std::cout << ": " << sample.at<float>(0, i) << "%";
        if (i + 1 < sample.cols) std::cout << ", ";
    }
    std::cout << std::endl;
    
    // Normalize the sample
    cv::Mat normalized = normalizeSample(sample, stats);
    
    // Predict using SVM
    float response = svm->predict(normalized);
    
    // For one-class SVM:
    // response = +1 : normal (inside the learned region)
    // response = -1 : anomaly (outside the learned region)
    
    std::cout << "SVM response: " << response << std::endl;
    if (response > 0) {
        std::cout << "Status: NORMAL" << std::endl;
    } else {
        std::cout << "Status: ANOMALY DETECTED!" << std::endl;
    }
    
    return 0;
}