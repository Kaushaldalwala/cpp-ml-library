// header file of the clustering mertrics

#ifndef CLUSTERING_METRICS_HPP
#define CLUSTERING_METRICS_HPP

#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <unordered_map>

class ClusteringMetrics {
private:
    // Helper: Build contingency matrix
    static std::vector<std::vector<int>> build_contingency_matrix(
        const std::vector<int>& labels_true,
        const std::vector<int>& labels_pred) {
        
        std::set<int> true_classes(labels_true.begin(), labels_true.end());
        std::set<int> pred_classes(labels_pred.begin(), labels_pred.end());
        
        std::map<int, int> true_to_idx, pred_to_idx;
        int idx = 0;
        for (int c : true_classes) true_to_idx[c] = idx++;
        idx = 0;
        for (int c : pred_classes) pred_to_idx[c] = idx++;
        
        std::vector<std::vector<int>> contingency(true_classes.size(), 
                                                   std::vector<int>(pred_classes.size(), 0));
        
        for (size_t i = 0; i < labels_true.size(); ++i) {
            contingency[true_to_idx[labels_true[i]]][pred_to_idx[labels_pred[i]]]++;
        }
        
        return contingency;
    }
    
    // Helper: Calculate entropy
    static double entropy(const std::vector<int>& counts, int total) {
        double ent = 0.0;
        for (int count : counts) {
            if (count > 0) {
                double p = static_cast<double>(count) / total;
                ent -= p * std::log(p);
            }
        }
        return ent;
    }
    
    // Helper: Calculate mutual information from contingency matrix
    static double mutual_information(const std::vector<std::vector<int>>& contingency) {
        int total = 0;
        std::vector<int> row_sums, col_sums;
        
        // Calculate totals
        for (const auto& row : contingency) {
            int row_sum = std::accumulate(row.begin(), row.end(), 0);
            row_sums.push_back(row_sum);
            total += row_sum;
        }
        
        for (size_t j = 0; j < contingency[0].size(); ++j) {
            int col_sum = 0;
            for (size_t i = 0; i < contingency.size(); ++i) {
                col_sum += contingency[i][j];
            }
            col_sums.push_back(col_sum);
        }
        
        // Calculate mutual information
        double mi = 0.0;
        for (size_t i = 0; i < contingency.size(); ++i) {
            for (size_t j = 0; j < contingency[0].size(); ++j) {
                int n_ij = contingency[i][j];
                if (n_ij > 0) {
                    double expected = static_cast<double>(row_sums[i]) * col_sums[j] / total;
                    mi += n_ij * std::log(n_ij / expected);
                }
            }
        }
        
        return mi / total;
    }
    
    // Helper: Calculate entropy of true labels
    static double entropy_true(const std::vector<int>& labels_true) {
        std::map<int, int> counts;
        for (int label : labels_true) counts[label]++;
        
        std::vector<int> counts_vec;
        for (auto& p : counts) counts_vec.push_back(p.second);
        
        return entropy(counts_vec, labels_true.size());
    }
    
    // Helper: Calculate entropy of predicted labels
    static double entropy_pred(const std::vector<int>& labels_pred) {
        std::map<int, int> counts;
        for (int label : labels_pred) counts[label]++;
        
        std::vector<int> counts_vec;
        for (auto& p : counts) counts_vec.push_back(p.second);
        
        return entropy(counts_vec, labels_pred.size());
    }
    
    // Helper: Calculate contingency sums
    static void get_contingency_sums(const std::vector<std::vector<int>>& contingency,
                                    std::vector<int>& row_sums,
                                    std::vector<int>& col_sums,
                                    int& total) {
        total = 0;
        row_sums.clear();
        col_sums.clear();
        
        for (const auto& row : contingency) {
            int row_sum = std::accumulate(row.begin(), row.end(), 0);
            row_sums.push_back(row_sum);
            total += row_sum;
        }
        
        if (contingency.empty()) return;
        
        for (size_t j = 0; j < contingency[0].size(); ++j) {
            int col_sum = 0;
            for (size_t i = 0; i < contingency.size(); ++i) {
                col_sum += contingency[i][j];
            }
            col_sums.push_back(col_sum);
        }
    }

public:
    // 1. Rand Score
    static double rand_score(const std::vector<int>& labels_true,
                            const std::vector<int>& labels_pred) {
        if (labels_true.size() != labels_pred.size()) return -1;
        
        long long a = 0, b = 0, c = 0, d = 0;
        size_t n = labels_true.size();
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                bool same_true = (labels_true[i] == labels_true[j]);
                bool same_pred = (labels_pred[i] == labels_pred[j]);
                
                if (same_true && same_pred) a++;
                else if (!same_true && !same_pred) d++;
                else if (same_true && !same_pred) b++;
                else if (!same_true && same_pred) c++;
            }
        }
        
        return static_cast<double>(a + d) / (a + b + c + d);
    }
    
    // 2. Adjusted Rand Score
    static double adjusted_rand_score(const std::vector<int>& labels_true,
                                     const std::vector<int>& labels_pred) {
        if (labels_true.size() != labels_pred.size()) return -1;
        
        auto contingency = build_contingency_matrix(labels_true, labels_pred);
        std::vector<int> row_sums, col_sums;
        int total;
        get_contingency_sums(contingency, row_sums, col_sums, total);
        
        long long sum_comb = 0;
        for (const auto& row : contingency) {
            for (int val : row) {
                if (val >= 2) {
                    sum_comb += static_cast<long long>(val) * (val - 1) / 2;
                }
            }
        }
        
        long long sum_row_comb = 0;
        for (int row_sum : row_sums) {
            if (row_sum >= 2) {
                sum_row_comb += static_cast<long long>(row_sum) * (row_sum - 1) / 2;
            }
        }
        
        long long sum_col_comb = 0;
        for (int col_sum : col_sums) {
            if (col_sum >= 2) {
                sum_col_comb += static_cast<long long>(col_sum) * (col_sum - 1) / 2;
            }
        }
        
        long long total_comb = static_cast<long long>(total) * (total - 1) / 2;
        
        double expected_index = static_cast<double>(sum_row_comb) * sum_col_comb / total_comb;
        double max_index = (sum_row_comb + sum_col_comb) / 2.0;
        
        if (max_index == expected_index) return 1.0;
        
        return (sum_comb - expected_index) / (max_index - expected_index);
    }
    
    // 3. Mutual Information Score
    static double mutual_info_score(const std::vector<int>& labels_true,
                                   const std::vector<int>& labels_pred) {
        if (labels_true.size() != labels_pred.size()) return -1;
        
        auto contingency = build_contingency_matrix(labels_true, labels_pred);
        return mutual_information(contingency);
    }
    
    // 4. Normalized Mutual Information Score
    static double normalized_mutual_info_score(const std::vector<int>& labels_true,
                                               const std::vector<int>& labels_pred,
                                               bool average_method = true) {
        if (labels_true.size() != labels_pred.size()) return -1;
        
        double mi = mutual_info_score(labels_true, labels_pred);
        double h_true = entropy_true(labels_true);
        double h_pred = entropy_pred(labels_pred);
        
        if (h_true == 0 || h_pred == 0) return 1.0;
        
        if (average_method) {
            // Geometric mean
            return mi / std::sqrt(h_true * h_pred);
        } else {
            // Arithmetic mean
            return 2.0 * mi / (h_true + h_pred);
        }
    }
    
    // 5. Adjusted Mutual Information Score
    static double adjusted_mutual_info_score(const std::vector<int>& labels_true,
                                            const std::vector<int>& labels_pred) {
        if (labels_true.size() != labels_pred.size()) return -1;
        
        double mi = mutual_info_score(labels_true, labels_pred);
        
        auto contingency = build_contingency_matrix(labels_true, labels_pred);
        std::vector<int> row_sums, col_sums;
        int total;
        get_contingency_sums(contingency, row_sums, col_sums, total);
        
        // Calculate expected mutual information
        double expected_mi = 0.0;
        for (int n_i : row_sums) {
            for (int n_j : col_sums) {
                double sum = 0.0;
                int min_val = std::max(1, n_i + n_j - total);
                int max_val = std::min(n_i, n_j);
                
                for (int n_ij = min_val; n_ij <= max_val; ++n_ij) {
                    double term1 = std::lgamma(n_i + 1) - std::lgamma(n_i - n_ij + 1) - std::lgamma(n_ij + 1);
                    double term2 = std::lgamma(n_j + 1) - std::lgamma(n_j - n_ij + 1);
                    double term3 = std::lgamma(total - n_i - n_j + n_ij + 1);
                    double term4 = std::lgamma(total + 1);
                    double term5 = std::lgamma(total - n_i + 1) + std::lgamma(total - n_j + 1);
                    
                    double log_prob = term1 + term2 + term3 + term4 - term5;
                    double prob = std::exp(log_prob);
                    
                    if (n_ij > 0) {
                        double log_mi = std::log(static_cast<double>(n_ij) * total / (n_i * n_j));
                        expected_mi += prob * n_ij * log_mi;
                    }
                }
            }
        }
        
        expected_mi /= total;
        
        double h_true = entropy_true(labels_true);
        double h_pred = entropy_pred(labels_pred);
        double max_mi = std::max(h_true, h_pred);
        
        if (max_mi == expected_mi) return 1.0;
        
        return (mi - expected_mi) / (max_mi - expected_mi);
    }
    
    // 6. Homogeneity Score
    static double homogeneity_score(const std::vector<int>& labels_true,
                                   const std::vector<int>& labels_pred) {
        if (labels_true.size() != labels_pred.size()) return -1;
        
        double h_true_given_pred = 0.0;
        auto contingency = build_contingency_matrix(labels_true, labels_pred);
        std::vector<int> row_sums, col_sums;
        int total;
        get_contingency_sums(contingency, row_sums, col_sums, total);
        
        for (size_t j = 0; j < contingency[0].size(); ++j) {
            std::vector<int> col_values;
            for (size_t i = 0; i < contingency.size(); ++i) {
                col_values.push_back(contingency[i][j]);
            }
            double h = entropy(col_values, col_sums[j]);
            h_true_given_pred += static_cast<double>(col_sums[j]) / total * h;
        }
        
        double h_true = entropy_true(labels_true);
        
        if (h_true == 0) return 1.0;
        return 1.0 - (h_true_given_pred / h_true);
    }
    
    // 7. Completeness Score
    static double completeness_score(const std::vector<int>& labels_true,
                                    const std::vector<int>& labels_pred) {
        if (labels_true.size() != labels_pred.size()) return -1;
        
        double h_pred_given_true = 0.0;
        auto contingency = build_contingency_matrix(labels_true, labels_pred);
        std::vector<int> row_sums, col_sums;
        int total;
        get_contingency_sums(contingency, row_sums, col_sums, total);
        
        for (size_t i = 0; i < contingency.size(); ++i) {
            std::vector<int> row_values = contingency[i];
            double h = entropy(row_values, row_sums[i]);
            h_pred_given_true += static_cast<double>(row_sums[i]) / total * h;
        }
        
        double h_pred = entropy_pred(labels_pred);
        
        if (h_pred == 0) return 1.0;
        return 1.0 - (h_pred_given_true / h_pred);
    }
    
    // 8. V-Measure Score (harmonic mean of homogeneity and completeness)
    static double v_measure_score(const std::vector<int>& labels_true,
                                 const std::vector<int>& labels_pred,
                                 double beta = 1.0) {
        double h = homogeneity_score(labels_true, labels_pred);
        double c = completeness_score(labels_true, labels_pred);
        
        if (h == 0 || c == 0) return 0.0;
        
        double beta_sq = beta * beta;
        return (1 + beta_sq) * h * c / (beta_sq * h + c);
    }
    
    // 9. Fowlkes-Mallows Score
    static double fowlkes_mallows_score(const std::vector<int>& labels_true,
                                       const std::vector<int>& labels_pred) {
        if (labels_true.size() != labels_pred.size()) return -1;
        
        long long tp = 0, fp = 0, fn = 0;
        size_t n = labels_true.size();
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                bool same_true = (labels_true[i] == labels_true[j]);
                bool same_pred = (labels_pred[i] == labels_pred[j]);
                
                if (same_true && same_pred) tp++;
                else if (!same_true && same_pred) fp++;
                else if (same_true && !same_pred) fn++;
            }
        }
        
        if (tp + fp == 0 || tp + fn == 0) return 0.0;
        return std::sqrt(static_cast<double>(tp) / (tp + fp) * 
                        static_cast<double>(tp) / (tp + fn));
    }
};

#endif // CLUSTERING_METRICS_HPP