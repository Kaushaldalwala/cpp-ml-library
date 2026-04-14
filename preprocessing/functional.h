#ifndef PREPROCESSING_FUNCTIONAL_HPP
#define PREPROCESSING_FUNCTIONAL_HPP

#include "scalers.h"
#include "power_transformers.h"
#include "binarizers.h"
#include "normalizers.h" // Add this include

namespace preprocessing
{

    // Functional API - Memory efficient versions
    template <typename T = float>
    inline std::vector<std::vector<T>> scale(const std::vector<std::vector<T>> &X)
    {
        StandardScaler<T> scaler;
        auto result = X;
        scaler.fit_transform_inplace(result);
        return result;
    }

    template <typename T = float>
    inline std::vector<std::vector<T>> minmax_scale(const std::vector<std::vector<T>> &X,
                                                    T range_min = 0.0f, T range_max = 1.0f)
    {
        MinMaxScaler<T> scaler(range_min, range_max);
        auto result = X;
        scaler.fit_transform_inplace(result);
        return result;
    }

    template <typename T = float>
    inline std::vector<std::vector<T>> maxabs_scale(const std::vector<std::vector<T>> &X)
    {
        MaxAbsScaler<T> scaler;
        auto result = X;
        scaler.fit_transform_inplace(result);
        return result;
    }

    template <typename T = float>
    inline std::vector<std::vector<T>> robust_scale(const std::vector<std::vector<T>> &X)
    {
        RobustScaler<T> scaler;
        auto result = X;
        scaler.fit_transform_inplace(result);
        return result;
    }

    template <typename T = float>
    inline std::vector<std::vector<T>> normalize(const std::vector<std::vector<T>> &X,
                                                 int norm_type = 2)
    { // 1=L1, 2=L2, 3=Max
        typename Normalizer<T>::Norm norm;
        switch (norm_type)
        {
        case 1:
            norm = Normalizer<T>::L1;
            break;
        case 3:
            norm = Normalizer<T>::MAX;
            break;
        default:
            norm = Normalizer<T>::L2;
        }
        Normalizer<T> normalizer(norm);
        auto result = X;
        normalizer.transform_inplace(result);
        return result;
    }

    template <typename T = float>
    inline std::vector<std::vector<T>> quantile_transform(const std::vector<std::vector<T>> &X,
                                                        int n_quantiles = 1000)
    {
        QuantileTransformer<T> qt(n_quantiles);
        auto result = X;
        qt.fit_transform_inplace(result);
        return result;
    }

    template <typename T = float>
    inline std::vector<std::vector<T>> power_transform(const std::vector<std::vector<T>> &X,
                                                    bool standardize = true)
    {
        PowerTransformer<T> pt(standardize);
        auto result = X;
        pt.fit_transform_inplace(result);
        return result;
    }

    template <typename T = float>
    inline std::vector<std::vector<int>> binarize(const std::vector<std::vector<T>> &X,
                                                T threshold = 0.0f)
    {
        Binarizer<T> binarizer(threshold);
        return binarizer.transform(X);
    }

    template <typename T = float>
    inline std::vector<std::vector<T>> label_binarize(const std::vector<int> &y,
                                                    const std::vector<int> &classes)
    {
        std::vector<std::vector<T>> y_bin(y.size(), std::vector<T>(classes.size(), 0.0f));

        for (size_t i = 0; i < y.size(); ++i)
        {
            for (size_t j = 0; j < classes.size(); ++j)
            {
                if (y[i] == classes[j])
                {
                    y_bin[i][j] = 1.0f;
                    break;
                }
            }
        }

        return y_bin;
    }

    template <typename T = float>
    inline std::vector<std::vector<T>> add_dummy_feature(const std::vector<std::vector<T>> &X,
                                                        T value = 1.0f)
    {
        if (X.empty())
            return {};

        std::vector<std::vector<T>> X_dummy;
        X_dummy.reserve(X.size());

        for (const auto &row : X)
        {
            std::vector<T> new_row;
            new_row.reserve(row.size() + 1);
            new_row.push_back(value);
            new_row.insert(new_row.end(), row.begin(), row.end());
            X_dummy.push_back(std::move(new_row));
        }

        return X_dummy;
    }

} // namespace preprocessing

#endif