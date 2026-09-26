// Loads the trained one-class SVM and its normalization stats, then scores
// system performance samples to decide whether the laptop is behaving normally.
//
// Usage:
//   monitor.exe [cpu memory disk]...     score explicit values (percentages)
//   monitor.exe --csv <path> [interval]  score every row of a CSV
//
// With no arguments the bundled demo sample (55 / 65 / 75) is scored.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/ml.hpp>

namespace {

constexpr const char* kDefaultModelPath = "models/svm_model.yml";
constexpr const char* kDefaultStatsPath = "data/norm_stats.yml";

struct NormalizationStats {
    std::vector<std::string> featureOrder;
    std::vector<float> mean;
    std::vector<float> stdev;
};

bool readSequence(cv::FileNode node, std::vector<float>& out) {
    if (!node.isSeq()) {
        return false;
    }
    out.clear();
    out.reserve(static_cast<std::size_t>(node.size()));
    for (int i = 0; i < node.size(); ++i) {
        out.push_back(static_cast<float>(static_cast<double>(node[i])));
    }
    return true;
}

// Loads feature order, mean and standard deviation used during training.
bool loadNormalizationStats(const std::string& path, NormalizationStats& stats) {
    cv::FileStorage file(path, cv::FileStorage::READ);
    if (!file.isOpened()) {
        std::cerr << "Error: cannot open normalization stats: " << path << "\n"
                  << "       run train_svm.exe first.\n";
        return false;
    }

    cv::FileNode orderNode = file["feature_order"];
    if (!orderNode.isSeq()) {
        std::cerr << "Error: '" << path << "' has no feature_order sequence.\n";
        return false;
    }
    for (int i = 0; i < orderNode.size(); ++i) {
        stats.featureOrder.push_back(static_cast<std::string>(orderNode[i]));
    }

    if (!readSequence(file["mean"], stats.mean) || !readSequence(file["std"], stats.stdev)) {
        std::cerr << "Error: '" << path << "' is missing a valid mean/std sequence.\n";
        return false;
    }

    file.release();

    if (stats.mean.size() != stats.stdev.size()) {
        std::cerr << "Error: mean and std sizes differ (" << stats.mean.size() << " vs "
                  << stats.stdev.size() << ").\n";
        return false;
    }
    if (!stats.featureOrder.empty() && stats.featureOrder.size() != stats.mean.size()) {
        std::cerr << "Error: feature_order has " << stats.featureOrder.size()
                  << " entries but mean/std have " << stats.mean.size() << ".\n";
        return false;
    }
    return true;
}

// Applies the training-time scaling: z = (x - mean) / std.
std::vector<float> normalize(const std::vector<float>& sample, const NormalizationStats& stats) {
    std::vector<float> normalized(sample.size());
    for (std::size_t i = 0; i < sample.size(); ++i) {
        const float stddev = (i < stats.stdev.size() && stats.stdev[i] != 0.0f) ? stats.stdev[i]
                                                                               : 1.0f;
        const float mean = (i < stats.mean.size()) ? stats.mean[i] : 0.0f;
        normalized[i] = (sample[i] - mean) / stddev;
    }
    return normalized;
}

std::string featureName(const NormalizationStats& stats, std::size_t index) {
    return index < stats.featureOrder.size() ? stats.featureOrder[index]
                                             : "feature_" + std::to_string(index);
}

void printFeatureStats(const NormalizationStats& stats) {
    std::cout << "Normalization stats\n" << std::string(52, '-') << "\n"
              << std::left << std::setw(24) << "feature" << std::right << std::setw(12) << "mean"
              << std::setw(12) << "std" << "\n";
    for (std::size_t i = 0; i < stats.mean.size(); ++i) {
        std::cout << std::left << std::setw(24) << featureName(stats, i) << std::right
                  << std::setw(12) << stats.mean[i] << std::setw(12) << stats.stdev[i] << "\n";
    }
    std::cout << std::string(52, '-') << "\n";
}

// Scores one sample; returns the SVM response (1 = inside, 0 = outside).
float score(const cv::ml::SVM& svm,
            const NormalizationStats& stats,
            const std::vector<float>& sample) {
    if (sample.size() != stats.mean.size()) {
        std::cerr << "Error: expected " << stats.mean.size() << " features, got " << sample.size()
                  << ".\n";
        return 0.0f;
    }

    const std::vector<float> normalized = normalize(sample, stats);

    cv::Mat row(1, static_cast<int>(normalized.size()), CV_32F);
    for (std::size_t i = 0; i < normalized.size(); ++i) {
        row.at<float>(0, static_cast<int>(i)) = normalized[i];
    }

    std::cout << "\nSample\n" << std::string(52, '-') << "\n"
              << std::left << std::setw(24) << "feature" << std::right << std::setw(14)
              << "raw" << std::setw(14) << "normalized" << "\n";
    for (std::size_t i = 0; i < normalized.size(); ++i) {
        std::cout << std::left << std::setw(24) << featureName(stats, i) << std::right
                  << std::setw(14) << sample[i] << std::setw(14) << normalized[i] << "\n";
    }
    std::cout << std::string(52, '-') << "\n";

    // cv::ml::SVM::predict() returns 1 for a sample inside the learned region
    // and 0 for one outside it. The old two-argument overload that exposed the
    // raw decision value was removed in OpenCV 4.
    const float response = svm.predict(row);
    std::cout << "SVM response   : " << std::fixed << std::setprecision(3) << response << "\n"
              << "Status         : " << (response > 0 ? "NORMAL" : "ANOMALY DETECTED") << "\n";
    return response;
}

int scoreCsv(const std::string& path,
             const cv::ml::SVM& svm,
             const NormalizationStats& stats,
             int intervalSeconds) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open CSV: " << path << "\n";
        return 1;
    }

    std::string line;
    std::getline(file, line);  // header

    int anomalies = 0;
    int total = 0;
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
        if (row.size() != stats.mean.size() + 1) {
            continue;
        }

        std::vector<float> sample;
        sample.reserve(stats.mean.size());
        bool valid = true;
        for (std::size_t i = 1; i < row.size(); ++i) {
            try {
                sample.push_back(std::stof(row[i]));
            } catch (const std::exception&) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            continue;
        }

        ++total;
        const float response = score(svm, stats, sample);
        if (response <= 0) {
            ++anomalies;
        }

        if (intervalSeconds > 0) {
            std::this_thread::sleep_for(std::chrono::seconds(intervalSeconds));
        }
    }

    std::cout << "\nScored " << total << " samples, " << anomalies << " flagged as anomalies.\n";
    return 0;
}

void printUsage(const char* executable) {
    std::cout << "Usage: " << executable << " [options] [values]\n\n"
              << "Options:\n"
              << "  --csv <path>       Score every row of a CSV file\n"
              << "  --interval <sec>   Pause between CSV rows (default: 0)\n"
              << "  --model <path>     Model YAML (default: " << kDefaultModelPath << ")\n"
              << "  --stats <path>     Normalization stats YAML (default: " << kDefaultStatsPath
              << ")\n"
              << "  -h, --help        Show this help\n\n"
              << "Examples:\n"
              << "  " << executable << " 55 65 75\n"
              << "  " << executable << " --csv data/system_performance_data.csv\n";
}

std::string argumentValue(int argc, char* argv[], int& index, const char* flag) {
    if (index + 1 >= argc) {
        std::cerr << "Error: " << flag << " requires a value.\n";
        std::exit(1);
    }
    return argv[++index];
}

int parseInteger(const std::string& flag, const std::string& text) {
    try {
        return std::stoi(text);
    } catch (const std::exception&) {
        std::cerr << "Error: " << flag << " expects an integer, got '" << text << "'.\n";
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string modelPath = kDefaultModelPath;
    std::string statsPath = kDefaultStatsPath;
    std::string csvPath;
    int intervalSeconds = 0;
    std::vector<float> values;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--csv") {
            csvPath = argumentValue(argc, argv, i, "--csv");
        } else if (arg == "--interval") {
            intervalSeconds = parseInteger("--interval", argumentValue(argc, argv, i, "--interval"));
        } else if (arg == "--model") {
            modelPath = argumentValue(argc, argv, i, "--model");
        } else if (arg == "--stats") {
            statsPath = argumentValue(argc, argv, i, "--stats");
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            try {
                values.push_back(std::stof(arg));
            } catch (const std::exception&) {
                std::cerr << "Error: unknown argument '" << arg << "'\n\n";
                printUsage(argv[0]);
                return 1;
            }
        }
    }

    if (!std::filesystem::exists(modelPath) || !std::filesystem::exists(statsPath)) {
        std::cerr << "Error: model or stats file is missing. Train first: train_svm.exe\n";
        return 1;
    }

    NormalizationStats stats;
    if (!loadNormalizationStats(statsPath, stats)) {
        return 1;
    }

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(modelPath);
    if (svm.empty()) {
        std::cerr << "Error: cannot load SVM model from: " << modelPath << "\n";
        return 1;
    }

    const int expected = static_cast<int>(stats.mean.size());
    if (svm->getVarCount() != expected) {
        std::cerr << "Error: model expects " << svm->getVarCount() << " features but stats have "
                  << expected << ".\n";
        return 1;
    }

    std::cout << "Model : " << modelPath << "\n"
              << "Stats : " << statsPath << "\n";
    printFeatureStats(stats);

    if (!csvPath.empty()) {
        return scoreCsv(csvPath, *svm, stats, intervalSeconds);
    }

    if (values.empty()) {
        std::cout << "\nNo sample supplied; scoring the built-in demo sample "
                     "(55 / 65 / 75).\n"
                  << "Pass real values, e.g. monitor.exe 55 65 75\n";
        values = {55.0f, 65.0f, 75.0f};
    }

    if (static_cast<int>(values.size()) != expected) {
        std::cerr << "Error: expected " << expected << " values, got " << values.size() << ".\n";
        return 1;
    }

    score(*svm, stats, values);
    return 0;
}
