#ifndef PREPROCESSING_NORMALIZERS_HPP
#define PREPROCESSING_NORMALIZERS_HPP

#include <vector>
#include <cmath>
#include <algorithm>

namespace preprocessing {

template<typename T = float>
class Normalizer {
public:
    enum Norm { L1, L2, MAX };
    
private:
    Norm norm_;
    
    T compute_norm(const std::vector<T>& x) const {
        T norm_val = 0.0f;
        switch (norm_) {
            case L1:
                for (T val : x) norm_val += std::abs(val);
                break;
            case L2:
                for (T val : x) norm_val += val * val;
                norm_val = std::sqrt(norm_val);
                break;
            case MAX:
                for (T val : x) norm_val = std::max(norm_val, std::abs(val));
                break;
        }
        return norm_val;
    }
    
public:
    Normalizer(Norm norm = L2) : norm_(norm) {}
    
    void transform_inplace(std::vector<std::vector<T>>& X) const {
        for (auto& row : X) {
            T norm_val = compute_norm(row);
            if (norm_val > 0) {
                for (T& val : row) {
                    val /= norm_val;
                }
            }
        }
    }
    
    std::vector<std::vector<T>> transform(const std::vector<std::vector<T>>& X) const {
        auto result = X;
        transform_inplace(result);
        return result;
    }
};

} // namespace preprocessing

#endif