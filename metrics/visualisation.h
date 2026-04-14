#ifndef DISPLAY_METRICS_H
#define DISPLAY_METRICS_H

#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace metrics {

class Display {

public:

    // ================= CONFUSION MATRIX DISPLAY =================
    static void ConfusionMatrixDisplay(const std::vector<int>& y_true,
                                       const std::vector<int>& y_pred,
                                       const std::string& filename = "confusion_matrix.csv") {

        int tp=0, tn=0, fp=0, fn=0;

        for(int i=0;i<y_true.size();i++){
            if(y_true[i]==1 && y_pred[i]==1) tp++;
            else if(y_true[i]==0 && y_pred[i]==0) tn++;
            else if(y_true[i]==0 && y_pred[i]==1) fp++;
            else fn++;
        }

        std::ofstream file(filename);
        file << " ,Pred_0,Pred_1\n";
        file << "Actual_0," << tn << "," << fp << "\n";
        file << "Actual_1," << fn << "," << tp << "\n";
        file.close();

        std::cout << "Confusion matrix saved to " << filename << "\n";
    }

    // ================= ROC CURVE DISPLAY =================
    static void RocCurveDisplay(const std::vector<int>& y_true,
                                const std::vector<double>& y_score,
                                const std::string& filename = "roc_curve.csv") {

        std::vector<std::pair<double,int>> data;

        for(int i=0;i<y_true.size();i++)
            data.push_back({y_score[i], y_true[i]});

        std::sort(data.begin(), data.end(), std::greater<>());

        int tp=0, fp=0;
        int P = std::count(y_true.begin(), y_true.end(), 1);
        int N = y_true.size() - P;

        std::ofstream file(filename);
        file << "FPR,TPR\n";

        for(auto &d : data){
            if(d.second==1) tp++;
            else fp++;

            double tpr = (double)tp / P;
            double fpr = (double)fp / N;

            file << fpr << "," << tpr << "\n";
        }

        file.close();
        std::cout << "ROC curve saved to " << filename << "\n";
    }

    // ================= PRECISION-RECALL DISPLAY =================
    static void PrecisionRecallDisplay(const std::vector<int>& y_true,
                                       const std::vector<double>& y_score,
                                       const std::string& filename = "pr_curve.csv") {

        std::vector<std::pair<double,int>> data;

        for(int i=0;i<y_true.size();i++)
            data.push_back({y_score[i], y_true[i]});

        std::sort(data.begin(), data.end(), std::greater<>());

        int tp=0, fp=0;
        int P = std::count(y_true.begin(), y_true.end(), 1);

        std::ofstream file(filename);
        file << "Recall,Precision\n";

        for(auto &d : data){
            if(d.second==1) tp++;
            else fp++;

            double precision = tp / (double)(tp+fp);
            double recall = tp / (double)P;

            file << recall << "," << precision << "\n";
        }

        file.close();
        std::cout << "PR curve saved to " << filename << "\n";
    }

    // ================= DET CURVE DISPLAY =================
    static void DetCurveDisplay(const std::vector<int>& y_true,
                                const std::vector<double>& y_score,
                                const std::string& filename = "det_curve.csv") {

        std::vector<std::pair<double,int>> data;

        for(int i=0;i<y_true.size();i++)
            data.push_back({y_score[i], y_true[i]});

        std::sort(data.begin(), data.end(), std::greater<>());

        int tp=0, fp=0;
        int P = std::count(y_true.begin(), y_true.end(), 1);
        int N = y_true.size() - P;

        std::ofstream file(filename);
        file << "FPR,FNR\n";

        for(auto &d : data){
            if(d.second==1) tp++;
            else fp++;

            double fnr = 1.0 - (double)tp / P;
            double fpr = (double)fp / N;

            file << fpr << "," << fnr << "\n";
        }

        file.close();
        std::cout << "DET curve saved to " << filename << "\n";
    }

    // ================= PREDICTION ERROR DISPLAY =================
    static void PredictionErrorDisplay(const std::vector<double>& y_true,
                                       const std::vector<double>& y_pred,
                                       const std::string& filename = "prediction_error.csv") {

        std::ofstream file(filename);
        file << "True,Predicted,Error\n";

        for(int i=0;i<y_true.size();i++){
            double error = y_true[i] - y_pred[i];
            file << y_true[i] << "," << y_pred[i] << "," << error << "\n";
        }

        file.close();
        std::cout << "Prediction error data saved to " << filename << "\n";
    }

};

}


#endif