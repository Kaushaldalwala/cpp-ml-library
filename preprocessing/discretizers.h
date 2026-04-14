#ifndef PREPROCESSING_DISCRETIZERS_HPP
#define PREPROCESSING_DISCRETIZERS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace preprocessing {

template<typename T = float>
class KBinsDiscretizer {
public:
    enum Strategy { UNIFORM, QUANTILE };
    
private:
    int n_bins_;
    Strategy strategy_;
    std::vector<std::vector<T>> bin_edges_;
    bool fitted_ = false;
    
    std::vector<T> compute_uniform_bins(const std::vector<T>& values, int n_bins) const {
        T min_val = *std::min_element(values.begin(), values.end());
        T max_val = *std::max_element(values.begin(), values.end());
        
        std::vector<T> bins;
        for (int i = 0; i <= n_bins; ++i) {
            bins.push_back(min_val + i * (max_val - min_val) / n_bins);
        }
        return bins;
    }
    
    std::vector<T> compute_quantile_bins(const std::vector<T>& values, int n_bins) const {
        std::vector<T> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        
        std::vector<T> bins;
        for (int i = 0; i <= n_bins; ++i) {
            size_t idx = i * (sorted.size() - 1) / n_bins;
            bins.push_back(sorted[idx]);
        }
        return bins;
    }
    
public:
    KBinsDiscretizer(int n_bins = 5, Strategy strategy = UNIFORM)
        : n_bins_(n_bins), strategy_(strategy) {}
    
    void fit(const std::vector<std::vector<T>>& X) {
        if (X.empty() || X[0].empty()) return;
        
        size_t n_features = X[0].size();
        bin_edges_.resize(n_features);
        
        for (size_t j = 0; j < n_features; ++j) {
            std::vector<T> column;
            column.reserve(X.size());
            for (const auto& row : X) {
                column.push_back(row[j]);
            }
            
            switch (strategy_) {
                case UNIFORM:
                    bin_edges_[j] = compute_uniform_bins(column, n_bins_);
                    break;
                case QUANTILE:
                    bin_edges_[j] = compute_quantile_bins(column, n_bins_);
                    break;
            }
        }
        
        fitted_ = true;
    }
    
    std::vector<std::vector<int>> transform(const std::vector<std::vector<T>>& X) const {
        if (!fitted_) throw std::runtime_error("Discretizer not fitted!");
        
        std::vector<std::vector<int>> X_binned(X.size(), std::vector<int>(X[0].size()));
        
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < X[0].size(); ++j) {
                int bin = 0;
                for (size_t k = 1; k < bin_edges_[j].size(); ++k) {
                    if (X[i][j] >= bin_edges_[j][k]) {
                        bin = static_cast<int>(k);
                    }
                }
                X_binned[i][j] = bin;
            }
        }
        
        return X_binned;
    }
    
    std::vector<std::vector<int>> fit_transform(const std::vector<std::vector<T>>& X) {
        fit(X);
        return transform(X);
    }
};

} // namespace preprocessing

#endif