#ifndef PREPROCESSING_ENCODERS_HPP
#define PREPROCESSING_ENCODERS_HPP

#include <vector>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>

namespace preprocessing {

class OneHotEncoder {
private:
    std::vector<std::map<std::string, int>> feature_maps_;
    std::vector<int> n_features_per_category_;
    bool fitted_ = false;
    
public:
    void fit(const std::vector<std::vector<std::string>>& X) {
        if (X.empty()) return;
        
        size_t n_features = X[0].size();
        feature_maps_.resize(n_features);
        n_features_per_category_.resize(n_features);
        
        for (size_t j = 0; j < n_features; ++j) {
            std::set<std::string> unique_categories;
            for (const auto& row : X) {
                unique_categories.insert(row[j]);
            }
            
            int idx = 0;
            for (const auto& cat : unique_categories) {
                feature_maps_[j][cat] = idx++;
            }
            n_features_per_category_[j] = unique_categories.size();
        }
        
        fitted_ = true;
    }
    
    std::vector<std::vector<double>> transform(const std::vector<std::vector<std::string>>& X) const {
        if (!fitted_) throw std::runtime_error("Encoder not fitted!");
        
        size_t n_samples = X.size();
        size_t total_features = 0;
        for (int n : n_features_per_category_) total_features += n;
        
        std::vector<std::vector<double>> X_encoded(n_samples, std::vector<double>(total_features, 0.0));
        
        size_t offset = 0;
        for (size_t j = 0; j < feature_maps_.size(); ++j) {
            for (size_t i = 0; i < n_samples; ++i) {
                auto it = feature_maps_[j].find(X[i][j]);
                if (it != feature_maps_[j].end()) {
                    int idx = it->second;
                    X_encoded[i][offset + idx] = 1.0;
                }
            }
            offset += n_features_per_category_[j];
        }
        
        return X_encoded;
    }
    
    std::vector<std::vector<double>> fit_transform(const std::vector<std::vector<std::string>>& X) {
        fit(X);
        return transform(X);
    }
};

class OrdinalEncoder {
private:
    std::vector<std::map<std::string, int>> feature_maps_;
    bool fitted_ = false;
    
public:
    void fit(const std::vector<std::vector<std::string>>& X) {
        if (X.empty()) return;
        
        size_t n_features = X[0].size();
        feature_maps_.resize(n_features);
        
        for (size_t j = 0; j < n_features; ++j) {
            std::set<std::string> unique_categories;
            for (const auto& row : X) {
                unique_categories.insert(row[j]);
            }
            
            int idx = 0;
            for (const auto& cat : unique_categories) {
                feature_maps_[j][cat] = idx++;
            }
        }
        
        fitted_ = true;
    }
    
    std::vector<std::vector<int>> transform(const std::vector<std::vector<std::string>>& X) const {
        if (!fitted_) throw std::runtime_error("Encoder not fitted!");
        
        std::vector<std::vector<int>> X_encoded(X.size(), std::vector<int>(X[0].size()));
        
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < X[0].size(); ++j) {
                auto it = feature_maps_[j].find(X[i][j]);
                if (it != feature_maps_[j].end()) {
                    X_encoded[i][j] = it->second;
                }
            }
        }
        
        return X_encoded;
    }
    
    std::vector<std::vector<int>> fit_transform(const std::vector<std::vector<std::string>>& X) {
        fit(X);
        return transform(X);
    }
};

class LabelEncoder {
private:
    std::map<std::string, int> class_to_int_;
    std::map<int, std::string> int_to_class_;
    bool fitted_ = false;
    
public:
    void fit(const std::vector<std::string>& y) {
        std::set<std::string> unique_classes(y.begin(), y.end());
        
        int idx = 0;
        for (const auto& cls : unique_classes) {
            class_to_int_[cls] = idx++;
        }
        
        for (const auto& pair : class_to_int_) {
            int_to_class_[pair.second] = pair.first;
        }
        
        fitted_ = true;
    }
    
    std::vector<int> transform(const std::vector<std::string>& y) const {
        if (!fitted_) throw std::runtime_error("Encoder not fitted!");
        
        std::vector<int> y_encoded;
        y_encoded.reserve(y.size());
        for (const auto& label : y) {
            auto it = class_to_int_.find(label);
            if (it != class_to_int_.end()) {
                y_encoded.push_back(it->second);
            }
        }
        return y_encoded;
    }
    
    std::vector<std::string> inverse_transform(const std::vector<int>& y) const {
        if (!fitted_) throw std::runtime_error("Encoder not fitted!");
        
        std::vector<std::string> y_decoded;
        y_decoded.reserve(y.size());
        for (int label : y) {
            auto it = int_to_class_.find(label);
            if (it != int_to_class_.end()) {
                y_decoded.push_back(it->second);
            }
        }
        return y_decoded;
    }
    
    std::vector<int> fit_transform(const std::vector<std::string>& y) {
        fit(y);
        return transform(y);
    }
};

class MultiLabelBinarizer {
private:
    std::vector<std::string> classes_;
    std::map<std::string, int> class_to_idx_;
    bool fitted_ = false;
    
public:
    void fit(const std::vector<std::vector<std::string>>& y) {
        std::set<std::string> unique_classes;
        for (const auto& labels : y) {
            for (const auto& label : labels) {
                unique_classes.insert(label);
            }
        }
        
        classes_.assign(unique_classes.begin(), unique_classes.end());
        for (size_t i = 0; i < classes_.size(); ++i) {
            class_to_idx_[classes_[i]] = i;
        }
        
        fitted_ = true;
    }
    
    std::vector<std::vector<int>> transform(const std::vector<std::vector<std::string>>& y) const {
        if (!fitted_) throw std::runtime_error("Binarizer not fitted!");
        
        std::vector<std::vector<int>> y_binarized(y.size(), std::vector<int>(classes_.size(), 0));
        
        for (size_t i = 0; i < y.size(); ++i) {
            for (const auto& label : y[i]) {
                auto it = class_to_idx_.find(label);
                if (it != class_to_idx_.end()) {
                    y_binarized[i][it->second] = 1;
                }
            }
        }
        
        return y_binarized;
    }
    
    std::vector<std::vector<int>> fit_transform(const std::vector<std::vector<std::string>>& y) {
        fit(y);
        return transform(y);
    }
};

template<typename T = float>
class TargetEncoder {
private:
    struct CategoryStats {
        T sum = 0.0f;
        int count = 0;
    };
    
    std::vector<std::unordered_map<std::string, CategoryStats>> feature_stats_;
    std::vector<T> global_means_;
    T smoothing_ = 10.0f;
    bool fitted_ = false;
    
public:
    TargetEncoder(T smoothing = 10.0f) : smoothing_(smoothing) {}
    
    void fit(const std::vector<std::vector<std::string>>& X, 
             const std::vector<T>& y) {
        if (X.empty() || X[0].empty()) return;
        
        size_t n_features = X[0].size();
        feature_stats_.resize(n_features);
        global_means_.resize(n_features, 0.0f);
        
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < n_features; ++j) {
                auto& stats = feature_stats_[j][X[i][j]];
                stats.sum += y[i];
                stats.count++;
                global_means_[j] += y[i];
            }
        }
        
        T total_samples = static_cast<T>(X.size());
        for (size_t j = 0; j < n_features; ++j) {
            global_means_[j] /= total_samples;
        }
        
        fitted_ = true;
    }
    
    std::vector<std::vector<T>> transform(const std::vector<std::vector<std::string>>& X) const {
        if (!fitted_) throw std::runtime_error("Encoder not fitted!");
        
        std::vector<std::vector<T>> X_encoded;
        X_encoded.reserve(X.size());
        
        for (const auto& row : X) {
            std::vector<T> encoded_row;
            encoded_row.reserve(row.size());
            
            for (size_t j = 0; j < row.size(); ++j) {
                const auto& stats = feature_stats_[j];
                auto it = stats.find(row[j]);
                
                if (it != stats.end()) {
                    T category_mean = it->second.sum / it->second.count;
                    T k = smoothing_ / (smoothing_ + it->second.count);
                    T encoded = k * global_means_[j] + (1 - k) * category_mean;
                    encoded_row.push_back(encoded);
                } else {
                    encoded_row.push_back(global_means_[j]);
                }
            }
            X_encoded.push_back(std::move(encoded_row));
        }
        
        return X_encoded;
    }
};

} // namespace preprocessing

#endif