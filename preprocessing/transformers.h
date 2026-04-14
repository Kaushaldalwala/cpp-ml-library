#ifndef PREPROCESSING_TRANSFORMERS_HPP
#define PREPROCESSING_TRANSFORMERS_HPP

#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <numeric>

namespace preprocessing {

// Memory-efficient PolynomialFeatures (uses combinatorial generation)
template<typename T = float>
class PolynomialFeatures {
private:
    int degree_;
    bool include_bias_;
    bool interaction_only_;
    std::vector<std::vector<int>> combinations_;
    
    void generate_combinations(int n_features, int current_degree, int start, 
                               std::vector<int>& current, 
                               std::vector<std::vector<int>>& result) {
        if (current_degree == 0) {
            if (!current.empty()) {
                result.push_back(current);
            }
            return;
        }
        
        for (int i = start; i < n_features; ++i) {
            current.push_back(i);
            generate_combinations(n_features, current_degree - 1, 
                                 interaction_only_ ? i + 1 : i, current, result);
            current.pop_back();
        }
    }
    
public:
    PolynomialFeatures(int degree = 2, bool include_bias = true, bool interaction_only = false)
        : degree_(degree), include_bias_(include_bias), interaction_only_(interaction_only) {}
    
    void fit(const std::vector<std::vector<T>>& X) {
        if (X.empty() || X[0].empty()) return;
        
        int n_features = X[0].size();
        combinations_.clear();
        
        if (include_bias_) {
            combinations_.push_back({});
        }
        
        for (int d = 1; d <= degree_; ++d) {
            std::vector<int> current;
            generate_combinations(n_features, d, 0, current, combinations_);
        }
    }
    
    // Memory-efficient: reserve and compute directly
    std::vector<std::vector<T>> transform(const std::vector<std::vector<T>>& X) const {
        std::vector<std::vector<T>> X_poly;
        X_poly.reserve(X.size());
        
        for (const auto& row : X) {
            std::vector<T> new_row;
            new_row.reserve(combinations_.size());
            
            for (const auto& combo : combinations_) {
                T value = 1.0f;
                for (int idx : combo) {
                    value *= row[idx];
                }
                new_row.push_back(value);
            }
            X_poly.push_back(std::move(new_row));
        }
        
        return X_poly;
    }
    
    void transform_inplace(std::vector<std::vector<T>>& X) const {
        X = transform(X);
    }
    
    void fit_transform_inplace(std::vector<std::vector<T>>& X) {
        fit(X);
        transform_inplace(X);
    }
};

// Memory-efficient SplineTransformer (B-spline basis)
template<typename T = float>
class SplineTransformer {
private:
    int n_knots_;
    int degree_;
    std::vector<T> knots_;
    
    T bspline_basis(T x, int i, int d, const std::vector<T>& knots) const {
        if (d == 0) {
            return (knots[i] <= x && x < knots[i + 1]) ? 1.0f : 0.0f;
        }
        
        T left_num = (x - knots[i]);
        T left_den = (knots[i + d] - knots[i]);
        T left = (left_den != 0) ? left_num / left_den * 
                 bspline_basis(x, i, d - 1, knots) : 0.0f;
        
        T right_num = (knots[i + d + 1] - x);
        T right_den = (knots[i + d + 1] - knots[i + 1]);
        T right = (right_den != 0) ? right_num / right_den * 
                  bspline_basis(x, i + 1, d - 1, knots) : 0.0f;
        
        return left + right;
    }
    
public:
    SplineTransformer(int n_knots = 5, int degree = 3) 
        : n_knots_(n_knots), degree_(degree) {}
    
    void fit(const std::vector<std::vector<T>>& X) {
        // Find global min and max for knots
        T global_min = std::numeric_limits<T>::max();
        T global_max = std::numeric_limits<T>::lowest();
        
        for (const auto& row : X) {
            for (T val : row) {
                if (val < global_min) global_min = val;
                if (val > global_max) global_max = val;
            }
        }
        
        // Generate equally spaced knots
        knots_.clear();
        knots_.push_back(global_min);
        
        for (int i = 1; i <= n_knots_; ++i) {
            T knot = global_min + i * (global_max - global_min) / n_knots_;
            knots_.push_back(knot);
        }
        
        knots_.push_back(global_max);
        
        // Add repeated knots for boundary
        for (int i = 0; i < degree_; ++i) {
            knots_.insert(knots_.begin(), knots_.front());
            knots_.push_back(knots_.back());
        }
    }
    
    std::vector<std::vector<T>> transform(const std::vector<std::vector<T>>& X) const {
        int n_basis = knots_.size() - degree_ - 1;
        std::vector<std::vector<T>> X_spline;
        X_spline.reserve(X.size());
        
        for (const auto& row : X) {
            std::vector<T> basis_row;
            basis_row.reserve(n_basis * row.size());
            
            for (T val : row) {
                for (int k = 0; k < n_basis; ++k) {
                    basis_row.push_back(bspline_basis(val, k, degree_, knots_));
                }
            }
            X_spline.push_back(std::move(basis_row));
        }
        
        return X_spline;
    }
    
    void fit_transform_inplace(std::vector<std::vector<T>>& X) {
        fit(X);
        X = transform(X);
    }
};

// Memory-efficient FunctionTransformer
template<typename T = float>
class FunctionTransformer {
private:
    std::function<T(T)> func_;
    std::function<T(T)> inverse_func_;
    
public:
    FunctionTransformer(std::function<T(T)> func, 
                       std::function<T(T)> inverse_func = nullptr)
        : func_(func), inverse_func_(inverse_func) {}
    
    void transform_inplace(std::vector<std::vector<T>>& X) const {
        for (auto& row : X) {
            for (T& val : row) {
                val = func_(val);
            }
        }
    }
    
    std::vector<std::vector<T>> transform(const std::vector<std::vector<T>>& X) const {
        auto result = X;
        transform_inplace(result);
        return result;
    }
    
    void inverse_transform_inplace(std::vector<std::vector<T>>& X) const {
        if (!inverse_func_) return;
        for (auto& row : X) {
            for (T& val : row) {
                val = inverse_func_(val);
            }
        }
    }
};

} // namespace preprocessing

#endif