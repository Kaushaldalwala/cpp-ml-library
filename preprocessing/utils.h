#ifndef PREPROCESSING_UTILS_HPP
#define PREPROCESSING_UTILS_HPP

#include <vector>
#include <cmath>
#include <random>
#include "scalers.h"
#include "discretizers.h"
#include "normalizers.h"

namespace preprocessing {

template<typename T = double>
class KernelCenterer {
private:
    std::vector<T> K_mean_;
    T total_mean_;
    bool fitted_ = false;
    
public:
    void fit(const std::vector<std::vector<T>>& K) {
        size_t n = K.size();
        K_mean_.assign(n, 0.0);
        total_mean_ = 0.0;
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                K_mean_[i] += K[i][j];
            }
            K_mean_[i] /= n;
            total_mean_ += K_mean_[i];
        }
        total_mean_ /= n;
        
        fitted_ = true;
    }
    
    std::vector<std::vector<T>> transform(const std::vector<std::vector<T>>& K) const {
        if (!fitted_) throw std::runtime_error("Centerer not fitted!");
        
        size_t n = K.size();
        std::vector<std::vector<T>> K_centered = K;
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                K_centered[i][j] = K[i][j] - K_mean_[i] - K_mean_[j] + total_mean_;
            }
        }
        
        return K_centered;
    }
    
    std::vector<std::vector<T>> fit_transform(const std::vector<std::vector<T>>& K) {
        fit(K);
        return transform(K);
    }
};

template<typename T = double>
std::vector<std::vector<T>> add_dummy_feature(const std::vector<std::vector<T>>& X, T value = 1.0) {
    if (X.empty()) return {};
    
    std::vector<std::vector<T>> X_with_dummy(X.size(), std::vector<T>(X[0].size() + 1, value));
    
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[0].size(); ++j) {
            X_with_dummy[i][j + 1] = X[i][j];
        }
    }
    
    return X_with_dummy;
}

// Functional API equivalents - Note the template parameters!
template<typename T = double>
std::vector<std::vector<T>> scale(const std::vector<std::vector<T>>& X) {
    StandardScaler<T> scaler;
    return scaler.fit_transform(X);
}

template<typename T = double>
std::vector<std::vector<T>> minmax_scale(const std::vector<std::vector<T>>& X, 
                                         T range_min = 0.0, T range_max = 1.0) {
    MinMaxScaler<T> scaler(range_min, range_max);
    return scaler.fit_transform(X);
}

template<typename T = double>
std::vector<std::vector<T>> maxabs_scale(const std::vector<std::vector<T>>& X) {
    MaxAbsScaler<T> scaler;
    return scaler.fit_transform(X);
}

template<typename T = double>
std::vector<std::vector<T>> robust_scale(const std::vector<std::vector<T>>& X) {
    RobustScaler<T> scaler;
    return scaler.fit_transform(X);
}

template<typename T = double>
std::vector<std::vector<T>> normalize(const std::vector<std::vector<T>>& X, 
                                       typename Normalizer<T>::Norm norm = Normalizer<T>::L2) {
    Normalizer<T> normalizer(norm);
    return normalizer.transform(X);
}

template<typename T = double>
std::vector<std::vector<T>> quantile_transform(const std::vector<std::vector<T>>& X, 
                                                int n_quantiles = 1000) {
    QuantileTransformer<T> transformer(n_quantiles);
    return transformer.fit_transform(X);
}

template<typename T = double>
std::vector<std::vector<int>> binarize(const std::vector<std::vector<T>>& X, T threshold = 0.0) {
    Binarizer<T> binarizer(threshold);
    return binarizer.transform(X);
}

// Convenience function for label binarization (functional API)
template<typename T = double>
std::vector<std::vector<T>> label_binarize(const std::vector<int>& y, 
                                           const std::vector<int>& classes) {
    std::vector<std::vector<T>> y_bin(y.size(), std::vector<T>(classes.size(), 0.0));
    
    for (size_t i = 0; i < y.size(); ++i) {
        for (size_t j = 0; j < classes.size(); ++j) {
            if (y[i] == classes[j]) {
                y_bin[i][j] = 1.0;
                break;
            }
        }
    }
    
    return y_bin;
}

} // namespace preprocessing

#endif