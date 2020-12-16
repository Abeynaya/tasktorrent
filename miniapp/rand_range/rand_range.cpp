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

#include <mpi.h>

#include "tasktorrent/tasktorrent.hpp"
#include "util.h"

using namespace std;
using namespace Eigen;
using namespace ttor;

typedef array<int, 2> int2;
typedef array<int, 3> int3;
typedef array<int, 4> int4;


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
    Taskflow<int> scatter_tf; // from origin to helper ranks
    Taskflow<int> start_qr_tf; // All helper ranks are ready 

    Taskflow<int2> dgeqrt_tf;  // A[k,k] = QR
    Taskflow<int3> dtsqrt_tf;

    Taskflow<int3> dlarfb_tf;
    Taskflow<int4> dssrfb_tf;

    Taskflow<int3> gather_tf;
    Taskflow<int> computeQ_tf; // from helper ranks to origin

    /** Taskflows obtained from parent function **/
    Taskflow<int>* notify_tf;

    /** Active messages between tasks **/
    ActiveMsg<>* am_start_qr;
    ActiveMsg<view<double>,view<int>,view<int>, 
              int, int , int, int , int , int , int , int>* am_scatter;

    ActiveMsg<view<double>,view<double>,view<int>,int, int>* am_dgeqrt_2_dlarfb;
    ActiveMsg<view<double>,int, int>* am_dgeqrt_2_dtsqrt;
    ActiveMsg<view<double>,int, int, int>* am_dtsqrt_2_dtsqrt;
    ActiveMsg<view<double>,view<double>,int,view<int>,int, int>* am_dtsqrt_2_dssrfb;
    ActiveMsg<view<double>,int, int, int, int>* am_dlarfb_2_dssrfb;
    ActiveMsg<view<double>,int, int, int, int>* am_dssrfb_2_dssrfb;
    ActiveMsg<view<double>,view<double>,view<double>,int,int>* am_gather_0;
    ActiveMsg<view<double>,view<double>,int,int>* am_gather_1;
    ActiveMsg<view<double>,int,int>* am_gather_2;

    /** Matrix A (input and output) **/
    MatrixXd* A;
    MatrixXd* Tmat;
    MatrixXd* Q;

    /** Workspace **/ //workspace[origin] = map<int2, MatrixXd> 
    vector<map<int2, MatrixXd>> Mat;
    vector<map<int2, MatrixXd>> T;
    vector<int> p_; // No. of ranks in x dim
    vector<int> q_; // No. of ranks in y dim
    vector<int> n_; // block size
    vector<int> M_; // no. of blocks in y dim
    vector<int> N_; // no. of blocks in x dim
    vector<int> notify_index_; // Which notify_tf parent task to notify about completion
    // vector<int> origin; // Which rank to gather output on

    /* Constructor */
    denseQR(Communicator* comm_, Threadpool* tp_, Taskflow<int>* notify_tf_): rank(comm_->comm_rank()),
    nranks(comm_->comm_size()), n_threads(tp_->size()), comm(comm_), tp(tp_), notify_tf(notify_tf_),
    scatter_tf(tp), start_qr_tf(tp), dgeqrt_tf(tp), dtsqrt_tf(tp), dlarfb_tf(tp), dssrfb_tf(tp), gather_tf(tp), computeQ_tf(tp) {

        Mat.resize(nranks);
        T.resize(nranks);
        p_.resize(nranks);
        q_.resize(nranks);
        n_.resize(nranks);
        M_.resize(nranks);
        N_.resize(nranks);
        notify_index_.resize(nranks);
        // origin.reserve(nranks);


        // Message from helper ranks saying they are ready to begin computation
        am_start_qr = comm->make_active_msg([&] () {
            // Will be received in the origin
            start_qr_tf.fulfill_promise(rank);
        });


        am_scatter = comm->make_active_msg([&](view<double> &V, view<int> &ind_i_, view<int>& ind_j_, int& nblocks,
                int& p, int& q, int& n, int& M, int& N, int& notify_index, int& origin_rank_){

                // In the receiving local rank
                int origin = origin_rank_;

                p_[origin] = p;
                q_[origin] = q;
                n_[origin] = n;
                M_[origin] = M;
                N_[origin] = N;
                notify_index_[origin] = notify_index;

                // initialize workspace
                for (int i=0; i < M; ++i){
                    for (int j=0; j < N; ++j){
                        Mat[origin][{i,j}] = MatrixXd::Zero(0,0);
                        if (i>=j) T[origin][{i,j}] = MatrixXd::Zero(0,0);
                    }
                }

                // Copy submatrix blocks received from origin
                MatrixXd Vtemp = Map<MatrixXd>(V.data(), nblocks*n, n);
                VectorXi ind_i = Map<VectorXi>(ind_i_.data(), nblocks);
                VectorXi ind_j = Map<VectorXi>(ind_j_.data(), nblocks);
                
                for (int counter=0; counter < nblocks; ++counter){
                    auto i = ind_i[counter];
                    auto j = ind_j[counter];
                    Mat[origin][{i,j}] = Vtemp.block(counter*n, 0, n, n);
                    if (i>=j) T[origin][{i,j}] = MatrixXd::Zero(n,n);
                }

                // Fullfill promise on start_qr_tf
                start_qr_tf.fulfill_promise(origin);
            }
        );

        // From origin to helper ranks
        scatter_tf.set_mapping([&] (int origin){ 
            return (origin % n_threads);
        })
        .set_indegree([](int){
            return 1;
        })
        .set_task([&] (int origin) {
            assert(rank == origin); 
            int M = M_[origin];
            int N = N_[origin];
            int n = n_[origin];
            int p = p_[origin];
            int q = q_[origin];

            // Mapper
            auto block2rank = [&](int2 ij){
                int i = ij[0];
                int j = ij[1];
                int ii = i % p;
                int jj = j % q;
                int r = (ii + jj * p + origin) % nranks;
                assert(r <= nranks);
                return r;
            };

            vector<map<int2, MatrixXd>> rank2matrix(nranks);


            for(int i = 0; i < M; i++) {
                for (int j=0; j < N; j++){
                    if (block2rank({i,j}) != origin){
                        rank2matrix[block2rank({i,j})][{i,j}] = A->block(i*n, j*n, n, n);
                        // In local workspace
                        Mat[origin][{i,j}] = MatrixXd::Zero(0,0);
                        if (i>=j) T[origin][{i,j}] = MatrixXd::Zero(0,0);
                    }
                    else {
                        Mat[origin][{i,j}] = A->block(i*n, j*n, n, n);
                        if (i>=j) T[origin][{i,j}] = MatrixXd::Zero(n,n);

                    }
                }
            }
            start_qr_tf.fulfill_promise(origin); // local 

            // Send blocks to other ranks 
            for (int i=0; i < nranks; ++i){
                if (i != origin){
                    int msize = rank2matrix[i].size();
                    MatrixXd submatrix = MatrixXd::Zero(msize*n, n); // Each block is of size n by n
                    vector<int> ind_i(msize);
                    vector<int> ind_j(msize);
                    int counter =0;
                    for (auto m: rank2matrix[i]){ // go through the map
                        ind_i[counter] = m.first[0];
                        ind_j[counter] = m.first[1];
                        submatrix.block(counter*n, 0, n, n) = m.second;
                        counter++;
                    }
                    auto submatrix_view = view<double>(submatrix.data(), msize*n*n);
                    auto ind_i_view = view<int>(ind_i.data(), msize);
                    auto ind_j_view = view<int>(ind_j.data(), msize);

                    am_scatter->send(i, submatrix_view, ind_i_view, ind_j_view,  msize, 
                                    p, q, n, M, N,
                                    notify_index_[origin], origin);
                }
            }

        })
        .set_name([](int origin) {
            return "scatter_" + to_string(origin);
        })
        .set_priority([&](int) {
            return 6;
        });

        start_qr_tf.set_mapping([&] (int origin){
            return origin % n_threads;
        })
        .
        set_indegree([&] (int origin) {
            if (rank == origin) return nranks;
            else return 1;
        })
        .set_task([&] (int origin) {
            cout << " Rank " << rank << " ready to begin QR" << endl;
            if (rank == origin){
                dgeqrt_tf.fulfill_promise({origin,0});
            }
            else {
                // Send a message to origin rank
                am_start_qr->send(origin);
            }
        })
        .set_name([](int k) {
            return "start_qr_" + to_string(k);
        })
        .set_priority([&](int) {
            return 6;
        });


        // From dgeqrt
        am_dgeqrt_2_dlarfb = comm->make_active_msg(
            [&](view<double> &V, view<double> &tau, view<int>& js, int& k, int& origin){
                int n = n_[origin];
                Mat[origin].at({k,k}) = Map<MatrixXd>(V.data(), n, n); // Check strides
                T[origin].at({k,k})   = Map<MatrixXd>(tau.data(), n, n);
                for (auto& j: js){
                        cout << "j" << endl;

                    dlarfb_tf.fulfill_promise({origin,k,j});
                }
            }
        );

        am_dgeqrt_2_dtsqrt = comm->make_active_msg(
            [&](view<double> &R, int& k, int& origin){
                int n = n_[origin];
                Mat[origin].at({k,k}) = Map<MatrixXd>(R.data(), n, n); // Check strides
                dtsqrt_tf.fulfill_promise({origin, k+1,k});
            }
        );

        // From dsqrt
        am_dtsqrt_2_dtsqrt = comm->make_active_msg(
            [&](view<double> &R, int& i, int& k, int& origin){
                int n = n_[origin];
                Mat[origin].at({k,k}) = Map<MatrixXd>(R.data(), n, n); // Check strides
                dtsqrt_tf.fulfill_promise({origin,i,k});
            }
        );

        am_dtsqrt_2_dssrfb = comm->make_active_msg(
            [&](view<double> &V_ik, view<double> &T_ik, int& i, view<int>& js, int& k, int& origin){
                int n = n_[origin];
                Mat[origin].at({i,k}) = Map<MatrixXd>(V_ik.data(), n, n); 
                T[origin].at({i,k}) = Map<MatrixXd>(T_ik.data(), n, n); 

                for(auto& j: js){
                    dssrfb_tf.fulfill_promise({origin,i,j,k}); 
                }
            }
        );

        // From dlarfb
        am_dlarfb_2_dssrfb = comm->make_active_msg(
            [&](view<double> &R, int& i, int&j,  int& k, int& origin){
                int n = n_[origin];
                Mat[origin].at({k,j}) = Map<MatrixXd>(R.data(), n, n); // Check strides
                dssrfb_tf.fulfill_promise({origin, i,j,k});
            }
        );

        // From dssrfb
        am_dssrfb_2_dssrfb = comm->make_active_msg(
            [&](view<double> &R, int& i, int&j,  int& k, int& origin){
                int n = n_[origin];
                Mat[origin].at({k,j}) = Map<MatrixXd>(R.data(), n, n); // Check strides
                dssrfb_tf.fulfill_promise({origin, i,j,k});
            }
        );

        // Define taskflows
        dgeqrt_tf.set_mapping([&] (int2 ok){
                int k = ok[1];
                return (k % n_threads);
            })
            .set_indegree([](int2){
                return 1;
            })
            .set_task([&] (int2 ok) {
                int origin = ok[0];
                int k = ok[1];
                int n = n_[origin];
                int info = LAPACKE_dgeqrt(LAPACK_COL_MAJOR, n, n, n, Mat[origin].at({k,k}).data(), n, T[origin].at({k,k}).data(), n);
                assert(info == 0);
            })
            .set_fulfill([&](int2 ok) {
                // Dependencies -- dlarfb 
                int origin = ok[0];
                int k = ok[1];
                int M = M_[origin];
                int N = N_[origin];
                int n = n_[origin];
                int p = p_[origin];
                int q = q_[origin];

                map<int, vector<int>> to_fulfill;

                for(int j = k+1; j<N; j++) { // A[k][j] blocks
                    int r = ((k%p)+ (j%q)*p + origin) % nranks;
                    if(to_fulfill.count(r) == 0) {
                        to_fulfill[r] = {j};
                    } else {
                        to_fulfill[r].push_back(j);
                    }
                }


                // First dtsqrt
                if (k+1 < M){
                   int r = ((k+1)%p + (k%q)*p + origin) % nranks;;
                   if (r == rank){
                       dtsqrt_tf.fulfill_promise({origin, k+1, k});
                   }
                   else {
                       auto R_kk = view<double>(Mat[origin].at({k,k}).data(), n * n );
                       am_dgeqrt_2_dtsqrt->send(r, R_kk, k, origin);

                   } 
                }
                
                // gather on node origin
                gather_tf.fulfill_promise({origin, k,k});


                // Send data and trigger tasks -- dlarfb
                for (auto& p: to_fulfill){
                    int r = p.first; // rank
                    if (r == rank){
                        for(auto& j: p.second){
                            dlarfb_tf.fulfill_promise({origin, k, j}); 
                        }
                    }
                    else {
                        auto V_kk = view<double>(Mat[origin].at({k,k}).data(), n * n );
                        auto T_kk = view<double>(T[origin].at({k,k}).data(), n*n);
                        auto jsv = view<int>(p.second.data(), p.second.size());
                        am_dgeqrt_2_dlarfb->send(r, V_kk, T_kk, jsv, k, origin);
                    }
                }

            })
            .set_name([](int2 ij) {
                return "geqrt_" + to_string(ij[0]) + to_string(ij[1]);
            })
            .set_priority([&](int2) {
                return 4;
            });

        dtsqrt_tf.set_mapping([&] (int3 oik){
                int origin = oik[0];
                int N = N_[origin];
                return ((oik[1] + oik[2] * N) % n_threads);
            })
            .set_indegree([](int3 oik){
                return (oik[2] == 0 ? 0 : 1) + 1;
            })
            .set_task([&] (int3 oik) {
                int origin = oik[0];
                int i = oik[1];
                int k = oik[2];
                int n = n_[origin];

                int info = LAPACKE_dtpqrt(LAPACK_COL_MAJOR, n, n, 0, n, Mat[origin].at({k,k}).data(), n, 
                                          Mat[origin].at({i,k}).data(), n, T[origin].at({i,k}).data(), n);
                assert(info == 0);
            })
            .set_fulfill([&](int3 oik) {
                int origin = oik[0];
                int i = oik[1];
                int k = oik[2];

                int M = M_[origin];
                int N = N_[origin];
                int n = n_[origin];
                int p = p_[origin];
                int q = q_[origin];

                // Dependencies 
                map<int, vector<int>> to_fulfill;
                
                // Next dtsqrt
                if (i+1 < M){
                    // int r = block2rank({i+1,k});
                    int r = ((i+1)%p+ (k%q)*p + origin) % nranks;
                    if (r == rank){
                        dtsqrt_tf.fulfill_promise({origin, i+1, k});
                    }
                    else {
                        auto R_kk = view<double>(Mat[origin].at({k,k}).data(), n * n );
                        int inext = i+1;
                        am_dtsqrt_2_dtsqrt->send(r, R_kk, inext, k, origin) ; // send A{k,k}
                    }
                }

                // gather on node 0
                gather_tf.fulfill_promise({origin, i,k}); 
                

                // dssrfb
                for(int j = k+1; j<N; j++) { 
                    // int r = block2rank({i,j}); // ssrfb
                    int r = ((i%p)+ (j%q)*p + origin) % nranks;
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
                            dssrfb_tf.fulfill_promise({origin, i,j,k}); 
                        }
                    }
                    else {
                        auto V_ik = view<double>(Mat[origin].at({i,k}).data(), n * n );
                        auto T_ik = view<double>(T[origin].at({i,k}).data(), n*n);
                        auto jsv = view<int>(p.second.data(), p.second.size());

                        am_dtsqrt_2_dssrfb->send(r, V_ik, T_ik, i, jsv, k, origin);
                    }
                }

            })
            .set_name([](int3 oik) {
                return "tsqrt_" + to_string(oik[0]) + "_" +to_string(oik[1]) + "_" +to_string(oik[2]);
            })
            .set_priority([&](int3) {
                return 3;
            });

        // larfb
        dlarfb_tf.set_mapping([&] (int3 okj){
                int origin = okj[0];
                int N = N_[origin];
                return ((okj[1] + okj[2]*N) % n_threads);
            })
            .set_indegree([](int3 okj){
                return (okj[1] == 0 ? 0 : 1) + 1;
            })
            .set_task([&] (int3 okj) {
                int origin = okj[0];
                int k=okj[1]; // who is sending
                int j=okj[2]; // me
                int n=n_[origin];

                int info = LAPACKE_dlarfb(LAPACK_COL_MAJOR, 'L', 'T', 'F', 'C', n, n, n,
                                           Mat[origin].at({k,k}).data(), n, T[origin].at({k,k}).data(), 
                                        n, Mat[origin].at({k,j}).data(), n);
                assert(info == 0);

            })
            .set_fulfill([&](int3 okj) {
                int origin = okj[0];
                int k = okj[1];
                int j = okj[2];

                int M = M_[origin];
                int N = N_[origin];
                int n = n_[origin];
                int p = p_[origin];
                int q = q_[origin];
                
                if (k+1 < M){
                    // int r = block2rank({k+1, j});
                    int r = ((k+1)%p+ (j%q)*p + origin) % nranks;
                    if (r == rank){
                        dssrfb_tf.fulfill_promise({origin, k+1, j, k});
                    }
                    else {
                        auto R_kj = view<double>(Mat[origin].at({k,j}).data(), n * n ); 
                        int knext = k+1;
                        am_dlarfb_2_dssrfb->send(r, R_kj, knext, j, k, origin); //send R_kj
                    }
                }
                
                
            })
            .set_name([](int3 okj) {
                return "dlarfb_" + to_string(okj[0]) + "_" +to_string(okj[1]) + "_" +to_string(okj[2]);
            })
            .set_priority([&](int3) {
                return 2;
            });

        dssrfb_tf.set_mapping([&] (int4 oijk){
                int origin = oijk[0];
                int N = N_[origin];
                return ((oijk[1] + oijk[2] * N + oijk[3] * N * N) % n_threads);
            })
            .set_indegree([](int4 oijk){
                int k = oijk[3];
                return (k==0 ? 2 : 3);
            })
            .set_task([&] (int4 oijk) {
                int origin = oijk[0];
                int i = oijk[1];
                int j = oijk[2];
                int k = oijk[3];
                int n = n_[origin];

                
                int info = LAPACKE_dtpmqrt(LAPACK_COL_MAJOR, 'L', 'T', n, n, n, 0, n, 
                                           Mat[origin].at({i,k}).data(), n,
                                           T[origin].at({i,k}).data(), n,
                                           Mat[origin].at({k,j}).data(), n,
                                           Mat[origin].at({i,j}).data(), n);
                assert(info == 0);

            })
            .set_fulfill([&](int4 oijk) {
                int origin = oijk[0];
                int i = oijk[1];
                int j = oijk[2];
                int k = oijk[3];


                int M = M_[origin];
                int N = N_[origin];
                int n = n_[origin];
                int p = p_[origin];
                int q = q_[origin];

                if (i+1 < M){
                    // int r = block2rank({i+1,j});
                    int r = ((i+1)%p+ (j%q)*p + origin) % nranks;
                    if (r == rank){
                        dssrfb_tf.fulfill_promise({origin, i+1,j,k});
                    }
                    else {
                        auto R_kj = view<double>(Mat[origin].at({k,j}).data(), n * n ); 
                        int inext = i+1;
                        am_dssrfb_2_dssrfb->send(r, R_kj, inext, j, k, origin);  
                    }
                }
                

                if (i == k+1) {
                    if (j == k+1){
                        // same rank
                        dgeqrt_tf.fulfill_promise({origin, i}); // i = k+1
                    }
                    else { // j > k+1
                        assert(j > k+1);
                        dlarfb_tf.fulfill_promise({origin, i, j}); // i = k+1
                    }
                }
                else { // i > k+1
                    assert(i > k+1);
                    if (j == k+1) {
                        dtsqrt_tf.fulfill_promise({origin, i, j}); // j = k+1
                    }
                    else {
                        assert(j > k+1);
                        dssrfb_tf.fulfill_promise({origin, i, j, k+1}); 
                    }
                }
                
            })
            .set_name([](int4 oijk) {
                return "dssrfb_" + to_string(oijk[0]) + "_" +to_string(oijk[1]) + "_" + to_string(oijk[2]) + "_" + to_string(oijk[3]);
            })
            .set_priority([&](int4) {
                return 1;
            });  

        am_gather_0 = comm->make_active_msg(
        [&](view<double> &R_j, view<double> &V_ij, view<double> &T_ij,  int& i, int& j) {
            int n = n_[rank];

            A->block(0, j*n, (j+1)*n, n) = Map<MatrixXd>(R_j.data(), (j+1)*n, n) ;
            A->block(i*n, j*n, n, n) = Map<MatrixXd>(V_ij.data(), n, n); // Last block
            Tmat->block(i*n, j*n, n, n) = Map<MatrixXd>(T_ij.data(), n, n);

            computeQ_tf.fulfill_promise(rank);
        });

        am_gather_1 = comm->make_active_msg(
        [&](view<double> &V_ij, view<double> &T_ij,  int& i, int& j) {
            int n = n_[rank];

            A->block(i*n, j*n, n, n) = Map<MatrixXd>(V_ij.data(), n, n); // Last block
            Tmat->block(i*n, j*n, n, n) = Map<MatrixXd>(T_ij.data(), n, n);

            computeQ_tf.fulfill_promise(rank);


        });

        am_gather_2 = comm->make_active_msg(
        [&](view<double> &T_ij,  int& i, int& j) {
            int n = n_[rank];

            Tmat->block(i*n, j*n, n, n) = Map<MatrixXd>(T_ij.data(), n, n);
            computeQ_tf.fulfill_promise(rank);


        });

        // gather
        gather_tf.set_mapping([&](int3 oij) {
                int origin = oij[0];
                int N = N_[origin];
                return ( (oij[1] + oij[2]*N) % n_threads );
            })
            .set_indegree([](int3) {
                return 1;
            })
            .set_task([&](int3 oij) {
                int origin = oij[0];
                int i = oij[1];
                int j = oij[2];
                int n = n_[origin];
                int M = M_[origin];

                if(rank != origin) {
                    if (i == M-1) { // Last row contains all the updated R[:, j]
                        MatrixXd Atemp = MatrixXd::Zero( (j+1)*n, n);
                        for (int k=0; k< j+1; ++k){ 
                            Atemp.block(k*n, 0, n, n) = Mat[origin].at({k,j}); // R[:,j] of A = QR
                        }
                        auto R_j = view<double>(Atemp.data(), (j+1)*n*n);

                        auto V_ij = view<double>(Mat[origin].at({i,j}).data(), n*n);
                        auto T_ij = view<double>(T[origin].at({i,j}).data(), n*n);
                        am_gather_0->send(origin, R_j, V_ij, T_ij, i, j);
                    }
                    else if (i > j){ // Just send the V_ij, T_ij
                        auto V_ij = view<double>(Mat[origin].at({i,j}).data(), n*n);
                        auto T_ij = view<double>(T[origin].at({i,j}).data(), n*n);
                        am_gather_1->send(origin, V_ij, T_ij, i, j);
                    }
                    else {
                        assert(i == j);
                        auto T_ij = view<double>(T[origin].at({i,j}).data(), n*n);
                        am_gather_2->send(origin, T_ij, i, j);
                    }
                } 
                else {
                    if (i == M-1){
                        for (int k=0; k< j+1; ++k){ 
                            A->block(k*n, j*n, n, n) = Mat[origin].at({k,j}); // R[:,j] of A = QR
                        }
                        A->block(i*n, j*n, n, n) = Mat[origin].at({i, j}); // Last block
                        Tmat->block(i*n, j*n, n, n) = T[origin].at({i,j});
                        computeQ_tf.fulfill_promise(origin);
                    }
                    else if (i > j){
                        A->block(i*n, j*n, n, n) = Mat[origin].at({i,j}); // just that block; contains V
                        Tmat->block(i*n, j*n, n, n) = T[origin].at({i,j});
                        computeQ_tf.fulfill_promise(origin);

                    }
                    else {
                        assert(i == j);
                        // Send T
                        Tmat->block(i*n, j*n, n, n) = T[origin].at({i,j});
                        computeQ_tf.fulfill_promise(origin);

                    }
                }
            })
            .set_name([](int3 oij) {
                return "gather_"+ to_string(oij[0]) + "_" +to_string(oij[1]) + "_" +to_string(oij[2]);
            });


            computeQ_tf.set_task([&] (int k){
                cout << "Computing_Q" << endl;
                int n = n_[rank]; // happens only on origin
                int M = M_[rank];
                int N = N_[rank];

                for (int j = N-1; j > -1; --j){

                    for (int k=0; k<N; ++k){
                        for (int i=M-1; i > j; --i){
                            MatrixXd Aij = A->block(i*n, j*n, n, n);
                            MatrixXd Tij = Tmat->block(i*n, j*n, n, n);
                            // assert(Q->cols() == N*n);
                            MatrixXd Qi = Q->block(i*n, k*n, n, n);
                            MatrixXd Qj = Q->block(j*n, k*n, n, n);

                            int info = LAPACKE_dtpmqrt(LAPACK_COL_MAJOR, 'L', 'N', n, n, n, 0, n, 
                                                       Aij.data(), n,
                                                       Tij.data(), n,
                                                       Qj.data(), n,
                                                       Qi.data(), n);
                            Q->block(i*n, k*n, n, n) = Qi;
                            Q->block(j*n, k*n, n, n) = Qj;

                            assert(info ==0);
                        }

                    }
                    
                    MatrixXd Ajj = A->block(j*n, j*n, n, n);
                    MatrixXd Tjj = Tmat->block(j*n, j*n, n, n);
                    MatrixXd Qjj = Q->block(j*n, 0, n, N*n);

                    int info = LAPACKE_dlarfb(LAPACK_COL_MAJOR, 'L', 'N', 'F', 'C', n, N*n, n, Ajj.data(), n, 
                                              Tjj.data(), n, Qjj.data(), n);

                    Q->block(j*n, 0, n, N*n) = Qjj;

                    assert(info == 0);

                }
            })
            .set_indegree([&] (int k){
                int M = M_[rank];
                int N = N_[rank];
                return (M* (M +1)/2 - (M-N)*(M-N+1)/2);
            })
            .set_mapping([&] (int k){
                return (k % n_threads);
            })
            .set_name([&](int k) {
                return "computeQ_tf_" + to_string(k);
            })
            .set_fulfill([&](int k){
                notify_tf->fulfill_promise(notify_index_[rank]);
            });           
    }

    /** Function **/
    void run(MatrixXd* Y_, MatrixXd* Tmat_, MatrixXd* Q_, int block_size, int k){
        this->A = Y_; // A to be performed QR on  -- only origin will have a copy of this
        this->Tmat = Tmat_;  //-- only origin will have a copy of this
        this->Q = Q_; // thin Q -- only origin will have a copy of this

        int origin = comm_rank();

        p_[origin] = pow(2,ceil(log(nranks)/log(2)));
        q_[origin] = floor(nranks/p_[origin]);

        notify_index_[origin] = k;

        M_[origin] = ceil(Y_->rows()/block_size);
        N_[origin] = ceil(Y_->cols()/block_size);

        n_[origin] = block_size;

        scatter_tf.fulfill_promise(origin);
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
    int samp=0; // Oversampling parameter
    
    MatrixXd A;
    MatrixXd* Y = new MatrixXd(M*n, N*n);
    MatrixXd* Q = new MatrixXd(M*n, N*n); 
    Q->setZero();
    Q->block(0,0,N*n,N*n) = MatrixXd::Identity(N*n, N*n);
    MatrixXd* Tmat = new MatrixXd(M*n, N*n); 
    Tmat->setZero();

    int origin = 0;
    // Factorize
    // {
        // Initialize the communicator structure
        Communicator comm(MPI_COMM_WORLD, VERB);

        // Threadpool
        Threadpool tp(n_threads, &comm, VERB, "[" + to_string(rank) + "]_");
        Taskflow<int> sparsify_tf(&tp, VERB);
        Taskflow<int> notify_tf(&tp, VERB);
        denseQR qr(&comm, &tp, &notify_tf); // Every rank does this. 

        notify_tf.set_task([&] (int k){
            printf("Randomized range finder is done and gathered on rank %d\n",rank);
        })
        .set_indegree([&] (int k){
            return 1;
        })
        .set_mapping([&] (int k){
            return (k % n_threads);
        })
        .set_name([&](int k) {
            return "notify_tf_" + to_string(k);
        });

        
        sparsify_tf.set_mapping([&] (int k){
            return (k % n_threads);
        })
        .set_indegree([&] (int k){
            return 1;
        })
        .set_task([&] (int k){
            printf("Modelling the sparsify task; call the RRQR function\n");
            std::default_random_engine default_gen = default_random_engine(2020+20*k);
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

            // We need to find QR of Y now
            x = VectorXd::Random(n * M);
            b = A*x;
            bref = b;
            
            // Multiply Y with G
            *Y = A*G; 
            // cout << *Y << endl << endl;

            qr.run(Y, Tmat, Q, n, k);
        });

        timer t0 = wctime();
        // if (rank == origin){
            sparsify_tf.fulfill_promise(rank);
        // }
        tp.join();
        timer t1 = wctime();
        MPI_Barrier(MPI_COMM_WORLD);
        // if(rank == origin)
        // {
            cout << "rank: " << rank << " Time : " << elapsed(t0, t1) << endl;
        // }


        if(VERB) {
            // cout << *Y << endl << endl;
            // cout << *Q << endl << endl;

            double error = (A - (*Q)*(Q->transpose())*A).norm();
            cout << "rank: " << rank << " Error solve: " << error << endl;
        }

       delete Q;
       delete Y;
       delete Tmat;

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

    // cout << "returned to main code" << endl;

    MPI_Finalize();

    return return_flag;
}