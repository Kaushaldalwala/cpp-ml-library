#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

// Include all modules in correct order
#include "scalers.h"
#include "transformers.h"
#include "power_transformers.h"
#include "normalizers.h"
#include "discretizers.h"
#include "binarizers.h"
#include "encoders.h"
#include "functional.h"

namespace preprocessing {

// Demo function
template<typename T = float>
class PreprocessingDemo {
public:
    static void run() {
        std::cout << "\n=== PREPROCESSING MODULE DEMO ===\n" << std::endl;
        
        std::vector<std::vector<float>> X = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f}
        };
        
        StandardScaler<float> scaler;
        auto X_scaled = X;
        scaler.fit_transform_inplace(X_scaled);
        std::cout << "StandardScaler: [" << X_scaled[0][0] << ", " 
                  << X_scaled[0][1] << ", " << X_scaled[0][2] << "]\n";
    }
};

} // namespace preprocessing

#endif