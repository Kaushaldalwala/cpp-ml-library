#ifndef PREPROCESSING_BINARIZERS_HPP
#define PREPROCESSING_BINARIZERS_HPP

#include <vector>
#include <algorithm>

namespace preprocessing {

// Memory-efficient Binarizer
template<typename T = float>
class Binarizer {
private:
    T threshold_;
    
public:
    Binarizer(T threshold = 0.0f) : threshold_(threshold) {}
    
    void transform_inplace(std::vector<std::vector<T>>& X) const {
        for (auto& row : X) {
            for (T& val : row) {
                val = (val > threshold_) ? static_cast<T>(1) : static_cast<T>(0);
            }
        }
    }
    
    std::vector<std::vector<int>> transform(const std::vector<std::vector<T>>& X) const {
        std::vector<std::vector<int>> X_binarized;
        X_binarized.reserve(X.size());
        
        for (const auto& row : X) {
            std::vector<int> bin_row;
            bin_row.reserve(row.size());
            for (T val : row) {
                bin_row.push_back((val > threshold_) ? 1 : 0);
            }
            X_binarized.push_back(std::move(bin_row));
        }
        
        return X_binarized;
    }
    
    void set_threshold(T threshold) { threshold_ = threshold; }
};

} // namespace preprocessing

#endif