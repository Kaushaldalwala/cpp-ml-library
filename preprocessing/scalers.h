#ifndef PREPROCESSING_SCALERS_HPP
#define PREPROCESSING_SCALERS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iostream>

namespace preprocessing {

// Memory-efficient StandardScaler (no extra copies)
template<typename T = float>  // Use float for memory efficiency
class StandardScaler {
private:
    std::vector<T> mean_;
    std::vector<T> scale_;
    bool fitted_ = false;
    
public:
    StandardScaler() = default;
    
    void fit(const std::vector<std::vector<T>>& X) {
        if (X.empty() || X[0].empty()) return;
        
        size_t n_samples = X.size();
        size_t n_features = X[0].size();
        
        mean_.assign(n_features, 0.0f);
        scale_.assign(n_features, 0.0f);
        
        // Single pass mean computation
        for (const auto& row : X) {
            for (size_t j = 0; j < n_features; ++j) {
                mean_[j] += row[j];
            }
        }
        
        T inv_n = 1.0f / n_samples;
        for (size_t j = 0; j < n_features; ++j) {
            mean_[j] *= inv_n;
        }
        
        // Single pass variance computation
        std::vector<T> variance(n_features, 0.0f);
        for (const auto& row : X) {
            for (size_t j = 0; j < n_features; ++j) {
                T diff = row[j] - mean_[j];
                variance[j] += diff * diff;
            }
        }
        
        for (size_t j = 0; j < n_features; ++j) {
            variance[j] *= inv_n;
            scale_[j] = std::sqrt(variance[j] + 1e-8f);
        }
        
        fitted_ = true;
    }
    
    // In-place transform to save memory
    void transform_inplace(std::vector<std::vector<T>>& X) const {
        if (!fitted_) throw std::runtime_error("Scaler not fitted!");
        
        for (auto& row : X) {
            for (size_t j = 0; j < row.size(); ++j) {
                row[j] = (row[j] - mean_[j]) / scale_[j];
            }
        }
    }
    
    std::vector<std::vector<T>> transform(const std::vector<std::vector<T>>& X) const {
        if (!fitted_) throw std::runtime_error("Scaler not fitted!");
        
        auto result = X;  // Copy once
        transform_inplace(result);
        return result;
    }
    
    void fit_transform_inplace(std::vector<std::vector<T>>& X) {
        fit(X);
        transform_inplace(X);
    }
    
    const std::vector<T>& get_mean() const { return mean_; }
    const std::vector<T>& get_scale() const { return scale_; }
};

// Memory-efficient MinMaxScaler
template<typename T = float>
class MinMaxScaler {
private:
    std::vector<T> min_;
    std::vector<T> max_;
    T feature_range_min_ = 0.0f;
    T feature_range_max_ = 1.0f;
    bool fitted_ = false;
    
public:
    MinMaxScaler(T range_min = 0.0f, T range_max = 1.0f) 
        : feature_range_min_(range_min), feature_range_max_(range_max) {}
    
    void fit(const std::vector<std::vector<T>>& X) {
        if (X.empty() || X[0].empty()) return;
        
        size_t n_features = X[0].size();
        min_.assign(n_features, std::numeric_limits<T>::max());
        max_.assign(n_features, std::numeric_limits<T>::lowest());
        
        for (const auto& row : X) {
            for (size_t j = 0; j < n_features; ++j) {
                if (row[j] < min_[j]) min_[j] = row[j];
                if (row[j] > max_[j]) max_[j] = row[j];
            }
        }
        
        fitted_ = true;
    }
    
    void transform_inplace(std::vector<std::vector<T>>& X) const {
        if (!fitted_) throw std::runtime_error("Scaler not fitted!");
        
        T range = feature_range_max_ - feature_range_min_;
        for (auto& row : X) {
            for (size_t j = 0; j < row.size(); ++j) {
                if (max_[j] - min_[j] != 0) {
                    row[j] = feature_range_min_ + 
                            (row[j] - min_[j]) * range / (max_[j] - min_[j]);
                } else {
                    row[j] = feature_range_min_;
                }
            }
        }
    }
    
    void fit_transform_inplace(std::vector<std::vector<T>>& X) {
        fit(X);
        transform_inplace(X);
    }
};

// Memory-efficient MaxAbsScaler (perfect for sparse data)
template<typename T = float>
class MaxAbsScaler {
private:
    std::vector<T> max_abs_;
    bool fitted_ = false;
    
public:
    void fit(const std::vector<std::vector<T>>& X) {
        if (X.empty() || X[0].empty()) return;
        
        size_t n_features = X[0].size();
        max_abs_.assign(n_features, 0.0f);
        
        for (const auto& row : X) {
            for (size_t j = 0; j < n_features; ++j) {
                T abs_val = std::abs(row[j]);
                if (abs_val > max_abs_[j]) max_abs_[j] = abs_val;
            }
        }
        
        fitted_ = true;
    }
    
    void transform_inplace(std::vector<std::vector<T>>& X) const {
        if (!fitted_) throw std::runtime_error("Scaler not fitted!");
        
        for (auto& row : X) {
            for (size_t j = 0; j < row.size(); ++j) {
                if (max_abs_[j] != 0) {
                    row[j] = row[j] / max_abs_[j];
                }
            }
        }
    }
    
    void fit_transform_inplace(std::vector<std::vector<T>>& X) {
        fit(X);
        transform_inplace(X);
    }
};

// Memory-efficient RobustScaler
template<typename T = float>
class RobustScaler {
private:
    std::vector<T> median_;
    std::vector<T> iqr_;
    bool fitted_ = false;
    
    T compute_median(std::vector<T>& values) const {
        std::sort(values.begin(), values.end());
        size_t n = values.size();
        if (n % 2 == 0) {
            return (values[n/2 - 1] + values[n/2]) * 0.5f;
        } else {
            return values[n/2];
        }
    }
    
    T compute_iqr(std::vector<T>& values) const {
        std::sort(values.begin(), values.end());
        size_t n = values.size();
        T q1 = values[n/4];
        T q3 = values[3*n/4];
        return q3 - q1;
    }
    
public:
    void fit(const std::vector<std::vector<T>>& X) {
        if (X.empty() || X[0].empty()) return;
        
        size_t n_samples = X.size();
        size_t n_features = X[0].size();
        
        median_.assign(n_features, 0.0f);
        iqr_.assign(n_features, 0.0f);
        
        for (size_t j = 0; j < n_features; ++j) {
            std::vector<T> column;
            column.reserve(n_samples);
            for (const auto& row : X) {
                column.push_back(row[j]);
            }
            median_[j] = compute_median(column);
            iqr_[j] = compute_iqr(column);
            if (iqr_[j] == 0) iqr_[j] = 1.0f;
        }
        
        fitted_ = true;
    }
    
    void transform_inplace(std::vector<std::vector<T>>& X) const {
        if (!fitted_) throw std::runtime_error("Scaler not fitted!");
        
        for (auto& row : X) {
            for (size_t j = 0; j < row.size(); ++j) {
                row[j] = (row[j] - median_[j]) / iqr_[j];
            }
        }
    }
    
    void fit_transform_inplace(std::vector<std::vector<T>>& X) {
        fit(X);
        transform_inplace(X);
    }
};

} // namespace preprocessing

#endif