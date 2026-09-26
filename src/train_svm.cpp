// Trains a one-class SVM on system performance samples collected by
// scripts/collect-data.py, then persists both the model and the normalization
// statistics so that monitor.cpp can reproduce the exact same feature scaling.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/ml.hpp>

namespace {

constexpr const char* kDefaultCsvPath = "data/system_performance_data.csv";
constexpr const char* kDefaultModelPath = "models/svm_model.yml";
constexpr const char* kDefaultStatsPath = "data/norm_stats.yml";

// Column order shared by collect-data.py, this trainer and monitor.cpp.
const std::vector<std::string> kFeatureOrder = {
    "CPU_Usage_Percent",
    "Memory_Usage_Percent",
    "Disk_Usage_Percent",
};

// Parses a float without throwing on malformed input.
bool parseFloat(const std::string& text, float& value) {
    try {
        std::size_t consumed = 0;
        const float parsed = std::stof(text, &consumed);
        if (consumed != text.size()) {
            return false;
        }
        value = parsed;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// Reads the CSV, skipping the header and any row that cannot be parsed.
std::vector<std::vector<float>> readSamples(const std::string& path, int& skipped) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open dataset: " << path << "\n"
                  << "       run 'python scripts/collect-data.py' first, or pass a path.\n";
        return {};
    }

    std::vector<std::vector<float>> samples;
    std::string line;
    std::getline(file, line);  // header
    skipped = 0;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream stream(line);
        std::string field;
        std::vector<std::string> row;
        while (std::getline(stream, field, ',')) {
            row.push_back(field);
        }

        if (row.size() != kFeatureOrder.size() + 1) {
            ++skipped;
            continue;
        }

        std::vector<float> features(kFeatureOrder.size());
        bool valid = true;
        for (std::size_t i = 0; i < features.size(); ++i) {
            if (!parseFloat(row[i + 1], features[i])) {
                valid = false;
                break;
            }
        }

        if (valid) {
            samples.push_back(std::move(features));
        } else {
            ++skipped;
        }
    }

    return samples;
}

void printUsage(const char* executable) {
    std::cout << "Usage: " << executable << " [options]\n\n"
              << "Options:\n"
              << "  --csv <path>     Training CSV (default: " << kDefaultCsvPath << ")\n"
              << "  --model <path>   Output model YAML (default: " << kDefaultModelPath << ")\n"
              << "  --stats <path>   Output normalization stats YAML (default: "
              << kDefaultStatsPath << ")\n"
              << "  --nu <value>     One-class SVM nu, 0 < nu <= 1 (default: 0.1)\n"
              << "  --gamma <value>  RBF kernel gamma, gamma > 0 (default: 0.5)\n"
              << "  -h, --help       Show this help\n";
}

std::string argumentValue(int argc, char* argv[], int& index, const char* flag) {
    if (index + 1 >= argc) {
        std::cerr << "Error: " << flag << " requires a value.\n";
        std::exit(1);
    }
    return argv[++index];
}

double parseDouble(const std::string& flag, const std::string& text) {
    try {
        return std::stod(text);
    } catch (const std::exception&) {
        std::cerr << "Error: " << flag << " expects a number, got '" << text << "'.\n";
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string csvPath = kDefaultCsvPath;
    std::string modelPath = kDefaultModelPath;
    std::string statsPath = kDefaultStatsPath;
    double nu = 0.1;
    double gamma = 0.5;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--csv") {
            csvPath = argumentValue(argc, argv, i, "--csv");
        } else if (arg == "--model") {
            modelPath = argumentValue(argc, argv, i, "--model");
        } else if (arg == "--stats") {
            statsPath = argumentValue(argc, argv, i, "--stats");
        } else if (arg == "--nu") {
            nu = parseDouble("--nu", argumentValue(argc, argv, i, "--nu"));
        } else if (arg == "--gamma") {
            gamma = parseDouble("--gamma", argumentValue(argc, argv, i, "--gamma"));
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Error: unknown argument '" << arg << "'\n\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (nu <= 0.0 || nu > 1.0) {
        std::cerr << "Error: --nu must be in (0, 1].\n";
        return 1;
    }
    if (gamma <= 0.0) {
        std::cerr << "Error: --gamma must be > 0.\n";
        return 1;
    }

    int skippedRows = 0;
    const std::vector<std::vector<float>> samples = readSamples(csvPath, skippedRows);
    if (samples.empty()) {
        return 1;
    }

    const int nbSamples = static_cast<int>(samples.size());
    const int nbFeatures = static_cast<int>(kFeatureOrder.size());

    std::cout << "Dataset      : " << csvPath << "\n"
              << "Samples      : " << nbSamples << "\n"
              << "Skipped rows : " << skippedRows << "\n"
              << "Features     : " << nbFeatures << "\n\n";

    cv::Mat features(nbSamples, nbFeatures, CV_32F);
    for (int i = 0; i < nbSamples; ++i) {
        for (int j = 0; j < nbFeatures; ++j) {
            features.at<float>(i, j) = samples[i][j];
        }
    }

    // cv::meanStdDev() returns ONE value per channel, i.e. a 1x1 CV_64F matrix
    // for a single-channel matrix - not one value per column. So it has to be
    // called once per column to get per-feature statistics.
    std::vector<float> means(nbFeatures);
    std::vector<float> stds(nbFeatures);
    for (int j = 0; j < nbFeatures; ++j) {
        cv::Mat mean;
        cv::Mat stddev;
        cv::meanStdDev(features.col(j), mean, stddev);
        if (mean.empty() || stddev.empty()) {
            std::cerr << "Error: cv::meanStdDev returned nothing for column " << j << ".\n";
            return 1;
        }

        means[j] = static_cast<float>(mean.at<double>(0, 0));
        stds[j] = static_cast<float>(stddev.at<double>(0, 0));

        // Replace zero-variance features so normalization stays finite and the
        // saved statistics remain usable by monitor.cpp.
        if (stds[j] < 1e-6f) {
            std::cout << "Warning: '" << kFeatureOrder[j]
                      << "' has (near) zero variance in this dataset; using std = 1.0.\n";
            stds[j] = 1.0f;
        }
    }

    std::cout << "\nFeature statistics\n" << std::string(52, '-') << "\n"
              << std::left << std::setw(24) << "feature" << std::right << std::setw(12) << "mean"
              << std::setw(12) << "std" << std::setw(12) << "min" << std::setw(12) << "max"
              << "\n";
    for (int j = 0; j < nbFeatures; ++j) {
        double minValue = 0.0;
        double maxValue = 0.0;
        cv::minMaxIdx(features.col(j), &minValue, &maxValue);

        std::cout << std::left << std::setw(24) << kFeatureOrder[j] << std::right << std::setw(12)
                  << means[j] << std::setw(12) << stds[j] << std::setw(12) << minValue
                  << std::setw(12) << maxValue << "\n";
    }
    std::cout << std::string(52, '-') << "\n";

    // Normalize with the same statistics that are written to disk.
    for (int j = 0; j < nbFeatures; ++j) {
        features.col(j) = (features.col(j) - means[j]) / stds[j];
    }

    if (!cv::checkRange(features, true)) {
        std::cerr << "Error: normalized data contains NaN or Inf values.\n";
        return 1;
    }

    // Persist normalization stats; monitor.cpp depends on this file.
    const std::filesystem::path statsFile(statsPath);
    if (statsFile.has_parent_path()) {
        std::filesystem::create_directories(statsFile.parent_path());
    }

    cv::FileStorage statsWriter(statsPath, cv::FileStorage::WRITE);
    if (!statsWriter.isOpened()) {
        std::cerr << "Error: cannot write normalization stats to: " << statsPath << "\n";
        return 1;
    }
    statsWriter << "feature_order" << "[";
    for (const std::string& feature : kFeatureOrder) {
        statsWriter << feature;
    }
    statsWriter << "]";
    statsWriter << "mean" << means;
    statsWriter << "std" << stds;
    statsWriter.release();
    std::cout << "\nNormalization stats saved to: " << statsPath << "\n";

    // One-class SVM learns the boundary of "normal" behaviour; anything outside
    // the learned region is reported as an anomaly.
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::ONE_CLASS);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setNu(nu);
    svm->setGamma(gamma);
    svm->train(features, cv::ml::ROW_SAMPLE, cv::Mat());

    const std::filesystem::path modelFile(modelPath);
    if (modelFile.has_parent_path()) {
        std::filesystem::create_directories(modelFile.parent_path());
    }

    svm->save(modelPath);
    std::cout << "SVM model saved to      : " << modelPath << "\n"
              << "  type                 : ONE_CLASS (RBF kernel)\n"
              << "  nu / gamma           : " << nu << " / " << gamma << "\n"
              << "  support vectors      : " << svm->getSupportVectors().rows << "\n";

    std::cout << "\nNext step: run monitor.exe to score live samples against this model.\n";
    return 0;
}
