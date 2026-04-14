#ifndef PREPROCESSING_POWER_TRANSFORMS_HPP
#define PREPROCESSING_POWER_TRANSFORMS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace preprocessing {

// Memory-efficient QuantileTransformer
template<typename T = float>
class QuantileTransformer {
private:
    int n_quantiles_;
    bool output_distribution_;
    std::vector<std::vector<T>> quantiles_;
    bool fitted_ = false;
    
    std::vector<T> compute_quantiles(const std::vector<T>& values, int n_quantiles) const {
        std::vector<T> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        
        std::vector<T> quantiles;
        quantiles.reserve(n_quantiles + 1);
        
        for (int i = 0; i <= n_quantiles; ++i) {
            size_t idx = static_cast<size_t>(i) * (sorted.size() - 1) / n_quantiles;
            quantiles.push_back(sorted[idx]);
        }
        
        return quantiles;
    }
    
public:
    QuantileTransformer(int n_quantiles = 1000, bool output_distribution = false)
        : n_quantiles_(n_quantiles), output_distribution_(output_distribution) {}
    
    void fit(const std::vector<std::vector<T>>& X) {
        if (X.empty() || X[0].empty()) return;
        
        size_t n_features = X[0].size();
        size_t n_samples = X.size();
        
        quantiles_.resize(n_features);
        
        for (size_t j = 0; j < n_features; ++j) {
            std::vector<T> column;
            column.reserve(n_samples);
            for (const auto& row : X) {
                column.push_back(row[j]);
            }
            quantiles_[j] = compute_quantiles(column, n_quantiles_);
        }
        
        fitted_ = true;
    }
    
    void transform_inplace(std::vector<std::vector<T>>& X) const {
        if (!fitted_) throw std::runtime_error("Transformer not fitted!");
        
        for (auto& row : X) {
            for (size_t j = 0; j < row.size(); ++j) {
                auto& quant = quantiles_[j];
                
                // Binary search for the correct quantile
                int idx = std::upper_bound(quant.begin(), quant.end(), row[j]) - quant.begin();
                idx = std::max(0, idx - 1);
                idx = std::min(idx, static_cast<int>(quant.size()) - 2);
                
                // Linear interpolation
                T t = (row[j] - quant[idx]) / (quant[idx + 1] - quant[idx]);
                T quantile_val = (idx + t) / (quant.size() - 1);
                
                if (output_distribution_) {
                    // Approximate inverse CDF for Gaussian
                    quantile_val = std::sqrt(2.0f) * std::erf(2 * quantile_val - 1);
                }
                
                row[j] = quantile_val;
            }
        }
    }
    
    void fit_transform_inplace(std::vector<std::vector<T>>& X) {
        fit(X);
        transform_inplace(X);
    }
};

// Memory-efficient PowerTransformer (Yeo-Johnson)
template<typename T = float>
class PowerTransformer {
private:
    std::vector<T> lambdas_;
    bool standardize_;
    bool fitted_ = false;
    
    T yeo_johnson_transform(T x, T lambda) const {
        if (x >= 0) {
            if (std::abs(lambda) < 1e-6f) {
                return std::log(x + 1);
            } else {
                return (std::pow(x + 1, lambda) - 1) / lambda;
            }
        } else {
            if (std::abs(lambda - 2) < 1e-6f) {
                return -std::log(-x + 1);
            } else {
                return -(std::pow(-x + 1, 2 - lambda) - 1) / (2 - lambda);
            }
        }
    }
    
    // Simplified lambda estimation (using skewness minimization)
    T estimate_lambda(const std::vector<T>& x) {
        T best_lambda = 1.0f;
        T best_skewness = std::numeric_limits<T>::max();
        
        // Test lambdas from -2 to 2
        for (T lambda = -2.0f; lambda <= 2.0f; lambda += 0.5f) {
            std::vector<T> transformed;
            transformed.reserve(x.size());
            
            for (T val : x) {
                transformed.push_back(yeo_johnson_transform(val, lambda));
            }
            
            // Compute skewness
            T mean = std::accumulate(transformed.begin(), transformed.end(), 0.0f) / transformed.size();
            T variance = 0.0f;
            T skewness = 0.0f;
            
            for (T val : transformed) {
                T diff = val - mean;
                variance += diff * diff;
            }
            variance /= transformed.size();
            T stddev = std::sqrt(variance);
            
            for (T val : transformed) {
                T diff = val - mean;
                skewness += diff * diff * diff;
            }
            skewness /= (transformed.size() * stddev * stddev * stddev);
            
            if (std::abs(skewness) < std::abs(best_skewness)) {
                best_skewness = skewness;
                best_lambda = lambda;
            }
        }
        
        return best_lambda;
    }
    
public:
    PowerTransformer(bool standardize = true) : standardize_(standardize) {}
    
    void fit(const std::vector<std::vector<T>>& X) {
        if (X.empty() || X[0].empty()) return;
        
        size_t n_features = X[0].size();
        size_t n_samples = X.size();
        lambdas_.resize(n_features);
        
        for (size_t j = 0; j < n_features; ++j) {
            std::vector<T> column;
            column.reserve(n_samples);
            for (const auto& row : X) {
                column.push_back(row[j]);
            }
            lambdas_[j] = estimate_lambda(column);
        }
        
        fitted_ = true;
    }
    
    void transform_inplace(std::vector<std::vector<T>>& X) const {
        if (!fitted_) throw std::runtime_error("Transformer not fitted!");
        
        for (auto& row : X) {
            for (size_t j = 0; j < row.size(); ++j) {
                row[j] = yeo_johnson_transform(row[j], lambdas_[j]);
            }
        }
        
        if (standardize_) {
            StandardScaler<T> scaler;
            scaler.fit_transform_inplace(X);
        }
    }
    
    void fit_transform_inplace(std::vector<std::vector<T>>& X) {
        fit(X);
        transform_inplace(X);
    }
};

} // namespace preprocessing

#endif