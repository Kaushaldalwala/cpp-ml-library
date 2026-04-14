#ifndef PAIRWISE_METRICS_HPP
#define PAIRWISE_METRICS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <iostream>
#include <numeric>
#include <tuple>
#include <queue>

template<typename T = double>
class PairwiseMetrics {
private:
    static const T EPSILON;
    static const T INFINITY_VAL;

public:
    // ==================== EUCLIDEAN DISTANCES ====================
    
    // 1. Euclidean Distance between two vectors
    static T euclidean_distance(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) return -1;
        T sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            T diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    // 2. Euclidean Distance Matrix (all pairwise distances)
    static std::vector<std::vector<T>> euclidean_distances(
        const std::vector<std::vector<T>>& X,
        const std::vector<std::vector<T>>& Y = {}) {
        
        const auto& Y_ref = Y.empty() ? X : Y;
        size_t n_samples_X = X.size();
        size_t n_samples_Y = Y_ref.size();
        
        std::vector<std::vector<T>> distances(n_samples_X, std::vector<T>(n_samples_Y, 0.0));
        
        for (size_t i = 0; i < n_samples_X; ++i) {
            for (size_t j = 0; j < n_samples_Y; ++j) {
                distances[i][j] = euclidean_distance(X[i], Y_ref[j]);
            }
        }
        
        return distances;
    }
    
    // 3. Nan Euclidean Distance (handles missing values)
    static T nan_euclidean_distance(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) return -1;
        
        T sum = 0.0;
        size_t valid_dims = 0;
        
        for (size_t i = 0; i < a.size(); ++i) {
            // Check for NaN (using self-comparison trick)
            if (a[i] != a[i] || b[i] != b[i]) continue;
            
            T diff = a[i] - b[i];
            sum += diff * diff;
            valid_dims++;
        }
        
        if (valid_dims == 0) return INFINITY_VAL;
        
        // Scale by ratio of dimensions
        T scaling = static_cast<T>(a.size()) / valid_dims;
        return std::sqrt(sum * scaling);
    }
    
    // 4. Nan Euclidean Distance Matrix
    static std::vector<std::vector<T>> nan_euclidean_distances(
        const std::vector<std::vector<T>>& X,
        const std::vector<std::vector<T>>& Y = {}) {
        
        const auto& Y_ref = Y.empty() ? X : Y;
        size_t n_samples_X = X.size();
        size_t n_samples_Y = Y_ref.size();
        
        std::vector<std::vector<T>> distances(n_samples_X, std::vector<T>(n_samples_Y, 0.0));
        
        for (size_t i = 0; i < n_samples_X; ++i) {
            for (size_t j = 0; j < n_samples_Y; ++j) {
                distances[i][j] = nan_euclidean_distance(X[i], Y_ref[j]);
            }
        }
        
        return distances;
    }
    
    // ==================== PAIRWISE DISTANCES ====================
    
    // 5. Generic Pairwise Distance with custom metric
    using DistanceFunc = std::function<T(const std::vector<T>&, const std::vector<T>&)>;
    
    static std::vector<std::vector<T>> pairwise_distances(
        const std::vector<std::vector<T>>& X,
        const std::vector<std::vector<T>>& Y,
        DistanceFunc metric) {
        
        size_t n_samples_X = X.size();
        size_t n_samples_Y = Y.size();
        
        std::vector<std::vector<T>> distances(n_samples_X, std::vector<T>(n_samples_Y, 0.0));
        
        for (size_t i = 0; i < n_samples_X; ++i) {
            for (size_t j = 0; j < n_samples_Y; ++j) {
                distances[i][j] = metric(X[i], Y[j]);
            }
        }
        
        return distances;
    }
    
    // Overload for single array (self-distances)
    static std::vector<std::vector<T>> pairwise_distances(
        const std::vector<std::vector<T>>& X,
        DistanceFunc metric) {
        return pairwise_distances(X, X, metric);
    }
    
    // 6. Manhattan (L1) Distance
    static T manhattan_distance(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) return -1;
        T sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += std::abs(a[i] - b[i]);
        }
        return sum;
    }
    
    // 7. Chebyshev (L∞) Distance
    static T chebyshev_distance(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) return -1;
        T max_diff = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        }
        return max_diff;
    }
    
    // 8. Minkowski Distance
    static T minkowski_distance(const std::vector<T>& a, const std::vector<T>& b, T p = 3.0) {
        if (a.size() != b.size()) return -1;
        T sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += std::pow(std::abs(a[i] - b[i]), p);
        }
        return std::pow(sum, 1.0 / p);
    }
    
    // 9. Cosine Distance
    static T cosine_distance(const std::vector<T>& a, const std::vector<T>& b) {
        return 1.0 - cosine_similarity(a, b);
    }
    
    static T cosine_similarity(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) return -1;
        T dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a == 0 || norm_b == 0) return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
    
    // 10. Correlation Distance
    static T correlation_distance(const std::vector<T>& a, const std::vector<T>& b) {
        return 1.0 - correlation_similarity(a, b);
    }
    
    static T correlation_similarity(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) return -1;
        
        T mean_a = std::accumulate(a.begin(), a.end(), 0.0) / a.size();
        T mean_b = std::accumulate(b.begin(), b.end(), 0.0) / b.size();
        
        T dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            T diff_a = a[i] - mean_a;
            T diff_b = b[i] - mean_b;
            dot += diff_a * diff_b;
            norm_a += diff_a * diff_a;
            norm_b += diff_b * diff_b;
        }
        
        if (norm_a == 0 || norm_b == 0) return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
    
    // ==================== PAIRWISE DISTANCES ARGMIN ====================
    
    // 11. Find index of minimum distance for each point in X to points in Y
    static std::vector<int> pairwise_distances_argmin(
        const std::vector<std::vector<T>>& X,
        const std::vector<std::vector<T>>& Y,
        DistanceFunc metric) {
        
        std::vector<int> indices(X.size());
        
        for (size_t i = 0; i < X.size(); ++i) {
            T min_dist = INFINITY_VAL;
            int min_idx = -1;
            
            for (size_t j = 0; j < Y.size(); ++j) {
                T dist = metric(X[i], Y[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = static_cast<int>(j);
                }
            }
            indices[i] = min_idx;
        }
        
        return indices;
    }
    
    // 12. Argmin using Euclidean distance
    static std::vector<int> pairwise_distances_argmin_euclidean(
        const std::vector<std::vector<T>>& X,
        const std::vector<std::vector<T>>& Y) {
        return pairwise_distances_argmin(X, Y, euclidean_distance);
    }
    
    // ==================== PAIRWISE DISTANCES CHUNKED ====================
    
    // 13. Chunked pairwise distances for memory efficiency
    static std::vector<std::vector<std::vector<T>>> pairwise_distances_chunked(
        const std::vector<std::vector<T>>& X,
        const std::vector<std::vector<T>>& Y,
        size_t chunk_size = 100,
        DistanceFunc metric = euclidean_distance) {
        
        std::vector<std::vector<std::vector<T>>> chunks;
        size_t n_samples_X = X.size();
        
        for (size_t i = 0; i < n_samples_X; i += chunk_size) {
            size_t end_i = std::min(i + chunk_size, n_samples_X);
            std::vector<std::vector<T>> chunk;
            
            for (size_t j = i; j < end_i; ++j) {
                std::vector<T> row(Y.size());
                for (size_t k = 0; k < Y.size(); ++k) {
                    row[k] = metric(X[j], Y[k]);
                }
                chunk.push_back(row);
            }
            chunks.push_back(chunk);
        }
        
        return chunks;
    }
    
    // 14. Process chunked distances with callback
    static void pairwise_distances_chunked_process(
        const std::vector<std::vector<T>>& X,
        const std::vector<std::vector<T>>& Y,
        size_t chunk_size,
        std::function<void(const std::vector<std::vector<T>>&, size_t, size_t)> callback,
        DistanceFunc metric = euclidean_distance) {
        
        size_t n_samples_X = X.size();
        
        for (size_t i = 0; i < n_samples_X; i += chunk_size) {
            size_t end_i = std::min(i + chunk_size, n_samples_X);
            std::vector<std::vector<T>> chunk;
            
            for (size_t j = i; j < end_i; ++j) {
                std::vector<T> row(Y.size());
                for (size_t k = 0; k < Y.size(); ++k) {
                    row[k] = metric(X[j], Y[k]);
                }
                chunk.push_back(row);
            }
            
            callback(chunk, i, end_i);
        }
    }
    
    // ==================== PAIRWISE KERNELS ====================
    
    // 15. Linear Kernel
    static T linear_kernel(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) return -1;
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    }
    
    // 16. Polynomial Kernel
    static T polynomial_kernel(const std::vector<T>& a, const std::vector<T>& b, 
                               T degree = 3.0, T coef0 = 1.0) {
        T dot = linear_kernel(a, b);
        return std::pow(dot + coef0, degree);
    }
    
    // 17. RBF (Gaussian) Kernel
    static T rbf_kernel(const std::vector<T>& a, const std::vector<T>& b, T gamma = 0.1) {
        T dist_sq = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            T diff = a[i] - b[i];
            dist_sq += diff * diff;
        }
        return std::exp(-gamma * dist_sq);
    }
    
    // 18. Sigmoid Kernel
    static T sigmoid_kernel(const std::vector<T>& a, const std::vector<T>& b, 
                            T gamma = 0.1, T coef0 = 0.0) {
        T dot = linear_kernel(a, b);
        return std::tanh(gamma * dot + coef0);
    }
    
    // 19. Laplacian Kernel
    static T laplacian_kernel(const std::vector<T>& a, const std::vector<T>& b, T gamma = 0.1) {
        T dist_l1 = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            dist_l1 += std::abs(a[i] - b[i]);
        }
        return std::exp(-gamma * dist_l1);
    }
    
    // 20. Chi-squared Kernel
    static T chi2_kernel(const std::vector<T>& a, const std::vector<T>& b, T gamma = 1.0) {
        T sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            T denominator = a[i] + b[i];
            if (denominator > 0) {
                sum += (a[i] - b[i]) * (a[i] - b[i]) / denominator;
            }
        }
        return std::exp(-gamma * sum);
    }
    
    // 21. Pairwise Kernel Matrix
    static std::vector<std::vector<T>> pairwise_kernels(
        const std::vector<std::vector<T>>& X,
        const std::vector<std::vector<T>>& Y,
        std::function<T(const std::vector<T>&, const std::vector<T>&)> kernel) {
        
        size_t n_samples_X = X.size();
        size_t n_samples_Y = Y.size();
        
        std::vector<std::vector<T>> kernel_matrix(n_samples_X, std::vector<T>(n_samples_Y));
        
        for (size_t i = 0; i < n_samples_X; ++i) {
            for (size_t j = 0; j < n_samples_Y; ++j) {
                kernel_matrix[i][j] = kernel(X[i], Y[j]);
            }
        }
        
        return kernel_matrix;
    }
};

// ==================== DISTANCE METRIC CLASS ====================

template<typename T = double>
class DistanceMetric {
private:
    std::string metric_type;
    std::vector<T> params;
    
public:
    // Constructor
    DistanceMetric(const std::string& metric = "euclidean", 
                   const std::vector<T>& params = {}) 
        : metric_type(metric), params(params) {}
    
    // Compute distance between two vectors
    T distance(const std::vector<T>& a, const std::vector<T>& b) const {
        if (metric_type == "euclidean") {
            return PairwiseMetrics<T>::euclidean_distance(a, b);
        }
        else if (metric_type == "manhattan") {
            return PairwiseMetrics<T>::manhattan_distance(a, b);
        }
        else if (metric_type == "chebyshev") {
            return PairwiseMetrics<T>::chebyshev_distance(a, b);
        }
        else if (metric_type == "cosine") {
            return PairwiseMetrics<T>::cosine_distance(a, b);
        }
        else if (metric_type == "correlation") {
            return PairwiseMetrics<T>::correlation_distance(a, b);
        }
        else if (metric_type == "minkowski") {
            T p = (params.size() > 0) ? params[0] : 3.0;
            return PairwiseMetrics<T>::minkowski_distance(a, b, p);
        }
        else if (metric_type == "nan_euclidean") {
            return PairwiseMetrics<T>::nan_euclidean_distance(a, b);
        }
        else {
            std::cerr << "Unknown metric type: " << metric_type << std::endl;
            return -1;
        }
    }
    
    // Compute pairwise distance matrix
    std::vector<std::vector<T>> pairwise_distances(
        const std::vector<std::vector<T>>& X,
        const std::vector<std::vector<T>>& Y = {}) {
        
        const auto& Y_ref = Y.empty() ? X : Y;
        std::vector<std::vector<T>> distances(X.size(), std::vector<T>(Y_ref.size()));
        
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < Y_ref.size(); ++j) {
                distances[i][j] = distance(X[i], Y_ref[j]);
            }
        }
        
        return distances;
    }
    
    // Get metric name
    std::string get_metric() const { return metric_type; }
    
    // Set metric
    void set_metric(const std::string& metric, const std::vector<T>& new_params = {}) {
        metric_type = metric;
        params = new_params;
    }
};

// Static member initialization
template<typename T>
const T PairwiseMetrics<T>::EPSILON = 1e-10;

template<typename T>
const T PairwiseMetrics<T>::INFINITY_VAL = std::numeric_limits<T>::infinity();

#endif // PAIRWISE_METRICS_HPP