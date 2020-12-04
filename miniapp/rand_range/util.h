#include <Eigen/Core>
#ifdef USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <tuple>

using namespace std;
using namespace Eigen;

template<typename T>
Eigen::MatrixXd get_gaussian(const int rows, const int cols, T* gen) {
    std::normal_distribution<double> norm_dist(0.0, 1.0);
    Eigen::MatrixXd W(rows, cols);
    for(int j = 0; j < cols; j++){
        for(int i = 0; i < rows; i++){
            W(i,j) = norm_dist(*gen);
        }
    }
    Eigen::VectorXd col_norms = W.colwise().norm();
    assert(col_norms.size() == cols);
    assert(col_norms.minCoeff() >= 0);
    if (col_norms.minCoeff() > 0) {
        return W.normalized();
    }
    return W;
}


// template<> MatrixXd get_gaussian(const int rows, const int cols, std::default_random_engine* gen);
// template<> MatrixXd get_gaussian(const int rows, const int cols, std::minstd_rand* gen);
