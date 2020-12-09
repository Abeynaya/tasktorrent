#include <Eigen/Core>
#include <Eigen/Dense>
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
// #include <gtest/gtest.h>
#include <mpi.h>

#include "tasktorrent/tasktorrent.hpp"
#include "util.h"

using namespace std;
using namespace Eigen;
using namespace ttor;

typedef array<int, 2> int2;
typedef array<int, 3> int3;

int VERB = 0;
bool LOG = false;
int n_threads_ = 2;
int n_ = 2;
int M_ = 5;
int N_ = 4;
int p_ = 1;
int q_ = 1;

struct denseQR {

    /** Usual stuff */
    const int rank;
    const int nranks;
    const int n_threads;
    Communicator* comm;
    Threadpool* tp;

    /** Task flows needed **/
    Taskflow<int> dgeqrt_tf;  // A[k,k] = QR
    Taskflow<int2> dtsqrt_tf;
    Taskflow<int2> dlarfb_tf;
    Taskflow<int3> dssrfb_tf;
    Taskflow<int2> gather_tf;

    /** Taskflows obtained from parent function **/
    Taskflow<int>* notify_tf;

    /** Active messages between tasks **/
    ActiveMsg<view<double>,view<double>,view<int>,int>* am_dgeqrt_2_dlarfb;
    ActiveMsg<view<double>,int>* am_dgeqrt_2_dtsqrt;
    ActiveMsg<view<double>,int, int>* am_dtsqrt_2_dtsqrt;
    ActiveMsg<view<double>,view<double>,int,view<int>,int>* am_dtsqrt_2_dssrfb;
    ActiveMsg<view<double>,int, int, int>* am_dlarfb_2_dssrfb;
    ActiveMsg<view<double>,int, int, int>* am_dssrfb_2_dssrfb;
    ActiveMsg<view<double>,view<double>,view<double>,int,int>* am_gather_0;
    ActiveMsg<view<double>,view<double>,int,int>* am_gather_1;
    ActiveMsg<view<double>,int,int>* am_gather_2;

    /** Matrix A (input and output) **/
    MatrixXd* A;
    MatrixXd* Tmat;
    int p; // No. of ranks in x dim
    int q; // No. of ranks in y dim
    int n; // block size
    int M; // no. of blocks in y dim
    int N; // no. of blocks in x dim

    /** Workspace **/
    map<int2, MatrixXd> Mat;
    map<int2, MatrixXd> T;

    /* Constructor */
    denseQR(Communicator* comm_, Threadpool* tp_, Taskflow<int>* notify_tf_): rank(comm_->comm_rank()),
    nranks(comm_->comm_size()), n_threads(tp_->size()), comm(comm_), tp(tp_), notify_tf(notify_tf_),
    dgeqrt_tf(tp), dtsqrt_tf(tp), dlarfb_tf(tp), dssrfb_tf(tp), gather_tf(tp) {

        // Mat and T size ??? 

        // From dgeqrt
        am_dgeqrt_2_dlarfb = comm->make_active_msg(
            [&](view<double> &V, view<double> &tau, view<int>& js, int& k){
                Mat.at({k,k}) = Map<MatrixXd>(V.data(), n, n); // Check strides
                T.at({k,k})   = Map<MatrixXd>(tau.data(), n, n);
                for (auto& j: js){
                    dlarfb_tf.fulfill_promise({k,j});
                }
            }
        );

        am_dgeqrt_2_dtsqrt = comm->make_active_msg(
            [&](view<double> &R, int& k){
                Mat.at({k,k}) = Map<MatrixXd>(R.data(), n, n); // Check strides
                dtsqrt_tf.fulfill_promise({k+1,k});
            }
        );

        // From dsqrt
        am_dtsqrt_2_dtsqrt = comm->make_active_msg(
            [&](view<double> &R, int& i, int& k){
                Mat.at({k,k}) = Map<MatrixXd>(R.data(), n, n); // Check strides
                dtsqrt_tf.fulfill_promise({i,k});
            }
        );

        am_dtsqrt_2_dssrfb = comm->make_active_msg(
            [&](view<double> &V_ik, view<double> &T_ik, int& i, view<int>& js, int& k){
                Mat.at({i,k}) = Map<MatrixXd>(V_ik.data(), n, n); 
                T.at({i,k}) = Map<MatrixXd>(T_ik.data(), n, n); 

                for(auto& j: js){
                    dssrfb_tf.fulfill_promise({i,j,k}); 
                }
            }
        );

        // From dlarfb
        am_dlarfb_2_dssrfb = comm->make_active_msg(
            [&](view<double> &R, int& i, int&j,  int& k){
                Mat.at({k,j}) = Map<MatrixXd>(R.data(), n, n); // Check strides
                dssrfb_tf.fulfill_promise({i,j,k});
            }
        );

        // From dssrfb
        am_dssrfb_2_dssrfb = comm->make_active_msg(
            [&](view<double> &R, int& i, int&j,  int& k){
                Mat.at({k,j}) = Map<MatrixXd>(R.data(), n, n); // Check strides
                dssrfb_tf.fulfill_promise({i,j,k});
            }
        );

        // Define taskflows
        dgeqrt_tf.set_mapping([&] (int k){
                return (k % n_threads);
            })
            .set_indegree([](int){
                return 1;
            })
            .set_task([&] (int k) {
                int info = LAPACKE_dgeqrt(LAPACK_COL_MAJOR, n, n, n, Mat.at({k,k}).data(), n, T.at({k,k}).data(), n);
                assert(info == 0);
            })
            .set_fulfill([&](int k) {
                // Dependencies -- dlarfb 

                map<int, vector<int>> to_fulfill;

                for(int j = k+1; j<N; j++) { // A[k][j] blocks
                    int r = (k%p)+ (j%q)*p;
                    cout << r << endl;
                    if(to_fulfill.count(r) == 0) {
                        to_fulfill[r] = {j};
                    } else {
                        to_fulfill[r].push_back(j);
                    }
                }


                // First dtsqrt
                if (k+1 < M){
                   int r = ((k+1)%p)+ (k%q)*p;
                   if (r == rank){
                       dtsqrt_tf.fulfill_promise({k+1, k});
                   }
                   else {
                       auto R_kk = view<double>(Mat.at({k,k}).data(), n * n );
                       am_dgeqrt_2_dtsqrt->send(r, R_kk, k);

                   } 
                }
                
                // gather on node 0
                gather_tf.fulfill_promise({k,k});


                // Send data and trigger tasks -- dlarfb
                for (auto& p: to_fulfill){
                    int r = p.first; // rank
                    if (r == rank){
                        for(auto& j: p.second){
                            dlarfb_tf.fulfill_promise({k, j}); 
                        }
                    }
                    else {
                        auto V_kk = view<double>(Mat.at({k,k}).data(), n * n );
                        auto T_kk = view<double>(T.at({k,k}).data(), n*n);
                        auto jsv = view<int>(p.second.data(), p.second.size());
                        am_dgeqrt_2_dlarfb->send(r, V_kk, T_kk, jsv, k);
                    }
                }

            })
            .set_name([](int j) {
                return "geqrt_" + to_string(j);
            })
            .set_priority([&](int) {
                return 4;
            });

        dtsqrt_tf.set_mapping([&] (int2 ik){
                return ((ik[0] + ik[1] * N) % n_threads);
            })
            .set_indegree([](int2 ik){
                return (ik[1] == 0 ? 0 : 1) + 1;
            })
            .set_task([&] (int2 ik) {
                int i = ik[0];
                int k = ik[1];
                int info = LAPACKE_dtpqrt(LAPACK_COL_MAJOR, n, n, 0, n, Mat.at({k,k}).data(), n, Mat.at({i,k}).data(), n, T.at({i,k}).data(), n);
                assert(info == 0);
            })
            .set_fulfill([&](int2 ik) {
                int i = ik[0];
                int k = ik[1];
                // Dependencies 
                map<int, vector<int>> to_fulfill;
                
                // Next dtsqrt
                if (i+1 < M){
                    // int r = block2rank({i+1,k});
                    int r = ((i+1)%p)+ (k%q)*p;
                    if (r == rank){
                        dtsqrt_tf.fulfill_promise({i+1, k});
                    }
                    else {
                        auto R_kk = view<double>(Mat.at({k,k}).data(), n * n );
                        int inext = i+1;
                        am_dtsqrt_2_dtsqrt->send(r, R_kk, inext, k) ; // send A{k,k}
                    }
                }

                // gather on node 0
                gather_tf.fulfill_promise({i,k}); 
                

                // dssrfb
                for(int j = k+1; j<N; j++) { 
                    // int r = block2rank({i,j}); // ssrfb
                    int r = (i%p)+ (j%q)*p;
                    if(to_fulfill.count(r) == 0) {
                        to_fulfill[r] = {j};
                    } else {
                        to_fulfill[r].push_back(j);
                    }
                }

                // Send data and trigger tasks
                for (auto& p: to_fulfill){
                    int r = p.first; // rank
                    if (r == rank){
                        for(auto& j: p.second){
                            dssrfb_tf.fulfill_promise({i,j,k}); 
                        }
                    }
                    else {
                        auto V_ik = view<double>(Mat.at({i,k}).data(), n * n );
                        auto T_ik = view<double>(T.at({i,k}).data(), n*n);
                        auto jsv = view<int>(p.second.data(), p.second.size());

                        am_dtsqrt_2_dssrfb->send(r, V_ik, T_ik, i, jsv, k);
                    }
                }

            })
            .set_name([](int2 ik) {
                return "tsqrt_" + to_string(ik[0]) + "_" +to_string(ik[1]);
            })
            .set_priority([&](int2) {
                return 3;
            });

        // larfb
        dlarfb_tf.set_mapping([&] (int2 kj){
                return ((kj[0] + kj[1]*N) % n_threads);
            })
            .set_indegree([](int2 kj){
                return (kj[0] == 0 ? 0 : 1) + 1;
            })
            .set_task([&] (int2 kj) {

                int k=kj[0]; // who is sending
                int j=kj[1]; // me

                int info = LAPACKE_dlarfb(LAPACK_COL_MAJOR, 'L', 'T', 'F', 'C', n, n, n, Mat.at({k,k}).data(), n, T.at({k,k}).data(), n, Mat.at({k,j}).data(), n);
                assert(info == 0);

            })
            .set_fulfill([&](int2 kj) {
                int k = kj[0];
                int j = kj[1];
                
                if (k+1 < M){
                    // int r = block2rank({k+1, j});
                    int r = ((k+1)%p)+ (j%q)*p;
                    if (r == rank){
                        dssrfb_tf.fulfill_promise({k+1, j, k});
                    }
                    else {
                        auto R_kj = view<double>(Mat.at({k,j}).data(), n * n ); 
                        int knext = k+1;
                        am_dlarfb_2_dssrfb->send(r, R_kj, knext, j, k); //send R_kj
                    }
                }
                
                
            })
            .set_name([](int2 kj) {
                return "dlarfb_" + to_string(kj[0]) + "_" +to_string(kj[1]);
            })
            .set_priority([&](int2) {
                return 2;
            });

        dssrfb_tf.set_mapping([&] (int3 ijk){
                return ((ijk[0] + ijk[1] * N + ijk[2] * N * N) % n_threads);
            })
            .set_indegree([](int3 ijk){
                int i = ijk[0];
                int j = ijk[1];
                int k = ijk[2];

                return (k==0 ? 2 : 3);

            })
            .set_task([&] (int3 ijk) {
                int i = ijk[0];
                int j = ijk[1];
                int k = ijk[2];
                
                int info = LAPACKE_dtpmqrt(LAPACK_COL_MAJOR, 'L', 'T', n, n, n, 0, n, 
                                           Mat.at({i,k}).data(), n,
                                           T.at({i,k}).data(), n,
                                           Mat.at({k,j}).data(), n,
                                           Mat.at({i,j}).data(), n);
                assert(info == 0);

            })
            .set_fulfill([&](int3 ijk) {
                int i = ijk[0];
                int j = ijk[1];
                int k = ijk[2];

                if (i+1 < M){
                    // int r = block2rank({i+1,j});
                    int r = ((i+1)%p)+ (j%q)*p;
                    if (r == rank){
                        dssrfb_tf.fulfill_promise({i+1,j,k});
                    }
                    else {
                        auto R_kj = view<double>(Mat.at({k,j}).data(), n * n ); 
                        int inext = i+1;
                        am_dssrfb_2_dssrfb->send(r, R_kj, inext, j, k);  
                    }
                }
                

                if (i == k+1) {
                    if (j == k+1){
                        // same rank
                        dgeqrt_tf.fulfill_promise(i); // i = k+1
                    }
                    else { // j > k+1
                        assert(j > k+1);
                        dlarfb_tf.fulfill_promise({i, j}); // i = k+1
                    }
                }
                else { // i > k+1
                    assert(i > k+1);
                    if (j == k+1) {
                        dtsqrt_tf.fulfill_promise({i, j}); // j = k+1
                    }
                    else {
                        assert(j > k+1);
                        dssrfb_tf.fulfill_promise({i, j, k+1}); 
                    }
                }
                
            })
            .set_name([](int3 ijk) {
                return "dssrfb_" + to_string(ijk[0]) + "_" +to_string(ijk[1]) + "_" + to_string(ijk[2]);
            })
            .set_priority([&](int3) {
                return 1;
            });  

        am_gather_0 = comm->make_active_msg(
        [&](view<double> &R_j, view<double> &V_ij, view<double> &T_ij,  int& i, int& j) {
            A->block(0, j*n, (j+1)*n, n) = Map<MatrixXd>(R_j.data(), (j+1)*n, n) ;
            // MatrixXd Rtemp = Map<MatrixXd>(R_j.data(), (j+1)*n, n) ;
            // for (int k=0; k<j+1; ++k){
            //     Mat.at({k,j}) = Rtemp.block(k*n, 0, n, n);
            // }
            A->block(i*n, j*n, n, n) = Map<MatrixXd>(V_ij.data(), n, n); // Last block
            Tmat->block(i*n, j*n, n, n) = Map<MatrixXd>(T_ij.data(), n, n);
            // Mat.at({i,j}) = Map<MatrixXd>(V_ij.data(), n, n); // Last block
            // T.at({i,j}) = Map<MatrixXd>(T_ij.data(), n, n);
            notify_tf->fulfill_promise(0);
        });

        am_gather_1 = comm->make_active_msg(
        [&](view<double> &V_ij, view<double> &T_ij,  int& i, int& j) {
            A->block(i*n, j*n, n, n) = Map<MatrixXd>(V_ij.data(), n, n); // Last block
            Tmat->block(i*n, j*n, n, n) = Map<MatrixXd>(T_ij.data(), n, n);

            // Mat.at({i,j}) = Map<MatrixXd>(V_ij.data(), n, n); // Last block
            // T.at({i,j}) = Map<MatrixXd>(T_ij.data(), n, n);
            notify_tf->fulfill_promise(0);


        });

        am_gather_2 = comm->make_active_msg(
        [&](view<double> &T_ij,  int& i, int& j) {
            Tmat->block(i*n, j*n, n, n) = Map<MatrixXd>(T_ij.data(), n, n);
            // T.at({i,j}) = Map<MatrixXd>(T_ij.data(), n, n);
            notify_tf->fulfill_promise(0);


        });

        // gather
        gather_tf.set_mapping([&](int2 ij) {
                return ( (ij[0] + ij[1]*N) % n_threads );
            })
            .set_indegree([](int2) {
                return 1;
            })
            .set_task([&](int2 ij) {
                int i = ij[0];
                int j = ij[1];

                if(rank != 0) {
                    if (i == M-1) { // Last row contains all the updated R[:, j]
                        MatrixXd Atemp = MatrixXd::Zero( (j+1)*n, n);
                        for (int k=0; k< j+1; ++k){ 
                            Atemp.block(k*n, 0, n, n) = Mat.at({k,j}); // R[:,j] of A = QR
                        }
                        auto R_j = view<double>(Atemp.data(), (j+1)*n*n);

                        auto V_ij = view<double>(Mat.at({i,j}).data(), n*n);
                        auto T_ij = view<double>(T.at({i,j}).data(), n*n);
                        am_gather_0->send(0, R_j, V_ij, T_ij, i, j);
                    }
                    else if (i > j){ // Just send the V_ij, T_ij
                        auto V_ij = view<double>(Mat.at({i,j}).data(), n*n);
                        auto T_ij = view<double>(T.at({i,j}).data(), n*n);
                        am_gather_1->send(0, V_ij, T_ij, i, j);
                    }
                    else {
                        assert(i == j);
                        auto T_ij = view<double>(T.at({i,j}).data(), n*n);
                        am_gather_2->send(0, T_ij, i, j);
                    }
                } 
                else {
                    if (i == M-1){
                        for (int k=0; k< j+1; ++k){ 
                            A->block(k*n, j*n, n, n) = Mat.at({k,j}); // R[:,j] of A = QR
                        }
                        A->block(i*n, j*n, n, n) = Mat.at({i, j}); // Last block
                        Tmat->block(i*n, j*n, n, n) = T.at({i,j});
                        notify_tf->fulfill_promise(0);
                    }
                    else if (i > j){
                        A->block(i*n, j*n, n, n) = Mat.at({i,j}); // just that block; contains V
                        Tmat->block(i*n, j*n, n, n) = T.at({i,j});
                        notify_tf->fulfill_promise(0);

                    }
                    else {
                        assert(i == j);
                        // Send T
                        Tmat->block(i*n, j*n, n, n) = T.at({i,j});
                        notify_tf->fulfill_promise(0);

                    }
                }
            })
            .set_name([](int2 ij) {
                return "gather_"+ to_string(ij[0]) + "_" +to_string(ij[1]);
            });

    }

    /** Function **/
    void run(MatrixXd* Y_, MatrixXd* Tmat_, int block_size, int p1, int q1){
        this->A = Y_; // A to be performed QR on 
        this->Tmat = Tmat_;  

        p = p1;
        q = q1;

        M = ceil(Y_->rows()/block_size);
        N = ceil(Y_->cols()/block_size);

        n = block_size;

        // Mapper
        auto block2rank = [&](int2 ij){
            int i = ij[0];
            int j = ij[1];
            int ii = i % p;
            int jj = j % q;
            int r = ii + jj * p;
            assert(r <= nranks);
            return r;
        };

        for(int i = 0; i < M; i++) {
            for (int j=0; j < N; j++){
                if(block2rank({i,j}) == rank) {
                    Mat[{i,j}] = Y_->block(i*n, j*n, n, n);
                    if (i>=j) T[{i,j}] = MatrixXd::Zero(n,n);

                } else {
                    Mat[{i,j}] = MatrixXd::Zero(0,0);
                    if (i>=j) T[{i,j}] = MatrixXd::Zero(0,0);
                }
            }
        }

        if (rank == 0){
            dgeqrt_tf.fulfill_promise(0);
        }

    }

};


// Every rank runs this code
int rand_range(int n_threads, int n, int M, int N, int p, int q)
{
    // MPI info
    const int rank = comm_rank();
    const int n_ranks = comm_size();
    if(VERB) printf("[%d] Hello from %s\n", comm_rank(), processor_name().c_str());

    

    assert(p * q == n_ranks);
    assert(p >= 1);
    assert(q >= 1);

    VectorXd x ;
    VectorXd b ;
    VectorXd bref ;
    int samp=10; // Oversampling parameter
    
    MatrixXd A;
    MatrixXd Y;
    MatrixXd Tmat = MatrixXd::Zero(M*n, N*n); 

    {
        std::default_random_engine default_gen = default_random_engine(2020);
        auto gen_gaussian = [&](int i, int j){ return get_gaussian(i, j, &default_gen); };


        // Generate a Gaussian random matrix  
        MatrixXd G = gen_gaussian(M*n, N*n); 


        // Generate a rank deficient matrix
        MatrixXd X = gen_gaussian(M*n, M*n);
        VectorXd d = VectorXd::Zero(M*n);
        d.head(N*n-samp) = 0.1*VectorXd::LinSpaced(N*n-samp,1, N*n-samp);

        DiagonalMatrix<double, Eigen::Dynamic> D(M*n);
        D = d.asDiagonal();
        A = X*D*X.inverse();

        // We need to find QR of A now
        if(rank == 0) {
            x = VectorXd::Random(n * M);
            b = A*x;
            bref = b;
        }
        
        // Multiply Y with G
        Y = A*G; 
        // Tmat = Y;
        cout << Y << endl;
    }

    // Factorize
    {
        // Initialize the communicator structure
        Communicator comm(MPI_COMM_WORLD, VERB);

        // Threadpool
        Threadpool tp(n_threads, &comm, VERB, "[" + to_string(rank) + "]_");
        Taskflow<int> notify_tf(&tp, VERB);

        notify_tf.set_task([&] (int k){
            printf("Randomized range finder is done and gathered on node 0\n");
        })
        .set_indegree([&] (int k){
            return (M* (M +1)/2 - (M-N)*(M-N+1)/2);
        })
        .set_mapping([&] (int k){
            return (k % n_threads);
        })
        .set_name([&](int k) {
            return "notify_tf_" + to_string(k);
        });

        denseQR qr(&comm, &tp, &notify_tf); // Every rank does this. 


        timer t0 = wctime();
        qr.run(&Y, &Tmat, n, p, q);
        tp.join();
        timer t1 = wctime();
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0)
        {
            cout << "Time : " << elapsed(t0, t1) << endl;
        }

    }

    {
        if(rank == 0) {

            cout << "Entering solve..." << endl;
            // MatrixXd I = MatrixXd::Identity(N*n, N*n);
            
            // Test 1   
            // Q^T b
            for (int j = 0; j < N; ++j){
                MatrixXd Ajj = Y.block(j*n, j*n, n, n);
                MatrixXd Tjj = Tmat.block(j*n, j*n, n, n);

                int info = LAPACKE_dlarfb(LAPACK_COL_MAJOR, 'L', 'T', 'F', 'C', n, 1, n, Ajj.data(), n, 
                                          Tjj.data(), n, b.segment(j*n, n).data(), n);
                assert(info == 0);


                for (int i=j+1; i < M; ++i){
                    MatrixXd Aij = Y.block(i*n, j*n, n, n);
                    MatrixXd Tij = Tmat.block(i*n, j*n, n, n);

                    info = LAPACKE_dtpmqrt(LAPACK_COL_MAJOR, 'L', 'T', n, 1, n, 0, n, 
                                               Aij.data(), n,
                                               Tij.data(), n,
                                               b.segment(j*n, n).data(), n,
                                               b.segment(i*n, n).data(), n);
                    assert(info ==0);
                    

                }
            }
            
            // Q Q^T b
            for (int j = N-1; j > -1; --j){

                for (int i=M-1; i > j; --i){
                    MatrixXd Aij = Y.block(i*n, j*n, n, n);
                    MatrixXd Tij = Tmat.block(i*n, j*n, n, n);
                    int info = LAPACKE_dtpmqrt(LAPACK_COL_MAJOR, 'L', 'N', n, 1, n, 0, n, 
                                               Aij.data(), n,
                                               Tij.data(), n,
                                               b.segment(j*n, n).data(), n,
                                               b.segment(i*n, n).data(), n);
                    assert(info ==0);
                }

                MatrixXd Ajj = Y.block(j*n, j*n, n, n);
                MatrixXd Tjj = Tmat.block(j*n, j*n, n, n);
                int info = LAPACKE_dlarfb(LAPACK_COL_MAJOR, 'L', 'N', 'F', 'C', n, 1, n, Ajj.data(), n, 
                                          Tjj.data(), n, b.segment(j*n, n).data(), n);
                assert(info == 0);

            }
            
            double error = (b - bref).norm() / bref.norm();
            cout << "Error solve: " << error << endl;
            // assert(error<=1e-8);
        }
    }
    return 0;
}


int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;

    MPI_Init_thread(NULL, NULL, req, &prov);

    assert(prov == req);

    if (argc >= 2)
    {
        n_threads_ = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        n_ = atoi(argv[2]);
    }

    if (argc >= 4)
    {
        M_ = atoi(argv[3]);
    }

    if (argc >= 5)
    {
        N_ = atoi(argv[4]);
    }

    if (argc >= 6)
    {
        p_ = atoi(argv[5]);
    }

    if (argc >= 7)
    {
        q_ = atoi(argv[6]);
    }

    if (argc >= 8)
    {
        VERB = atoi(argv[7]);
    }

    if (argc >= 9)
    {
        LOG = atoi(argv[8]);
    }

    const int return_flag = rand_range(n_threads_, n_, M_, N_, p_, q_);

    MPI_Finalize();

    return return_flag;
}