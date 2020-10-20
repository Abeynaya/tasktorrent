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
// #include <gtest/gtest.h>
#include <mpi.h>

#include "tasktorrent/tasktorrent.hpp"

using namespace std;
using namespace Eigen;
using namespace ttor;

typedef array<int, 2> int2;
typedef array<int, 3> int3;

int VERB = 0;
bool LOG = false;
int n_threads_ = 2;
int n_ = 2;
int N_ = 4;
int p_ = 1;
int q_ = 1;

int denseQR(int n_threads, int n, int N, int p, int q)
{
    // MPI info
    const int rank = comm_rank();
    const int n_ranks = comm_size();
    if(VERB) printf("[%d] Hello from %s\n", comm_rank(), processor_name().c_str());

    assert(p * q == n_ranks);
    assert(p >= 1);
    assert(q >= 1);

    // Form the matrix : let every node have a copy of A for now
    auto gen = [&](int i, int j) { 
        if(i == j) {
            return static_cast<double>(N*n+2);
        } else {
            int k = (i+j)%3;
            if(k == 0) {
                return 0.0;
            } else if (k == 1) {
                return 0.5;
            } else {
                return 1.0;
            }
        }
    };


    // Mapper
    auto block2rank = [&](int2 ij){
        int i = ij[0];
        int j = ij[1];
        int ii = i % p;
        int jj = j % q;
        int r = ii + jj * p;
        assert(r <= n_ranks);
        return r;
    };

    VectorXd x;
    VectorXd b;
    VectorXd bref;

    // Block the matrix for every node
    // Store it in a map
    // Later on we can probably optimize this, to avoid storing the whole thing
    // Note: concurrent access to the map should _NOT_ modify it, otherwise we need a lock
    map<int2, MatrixXd> Mat;
    map<int2, MatrixXd> T;

    {
        // MatrixXd A = MatrixXd::NullaryExpr(N * n, N * n, gen);
       
        if(rank == 0) {
            x = VectorXd::Random(n * N);
            b = x;
            bref = b;
        }


        for(int i = 0; i < N; i++) {
            for (int j=0; j < N; j++){
                if(block2rank({i,j}) == rank) {
                    Mat[{i,j}] = MatrixXd::NullaryExpr(n, n, gen);
                    if (i>=j) T[{i,j}] = MatrixXd::Zero(n,n);

                } else {
                    Mat[{i,j}] = MatrixXd::Zero(0,0);
                    if (i>=j) T[{i,j}] = MatrixXd::Zero(0,0);
            
                }
            }
        }
    }

    // Factorize
    {
        // Initialize the communicator structure
        Communicator comm(MPI_COMM_WORLD, VERB);

        // Threadpool
        Threadpool tp(n_threads, &comm, VERB, "[" + to_string(rank) + "]_");
        Taskflow<int> dgeqrt_tf(&tp, VERB);  // A[k,k] = QR
        Taskflow<int2> dtsqrt_tf(&tp, VERB);
        Taskflow<int2> dlarfb_tf(&tp, VERB);
        Taskflow<int3> dssrfb_tf(&tp, VERB);

        // Log
        DepsLogger dlog(1000000);
        Logger log(1000000);
        if(LOG) {
            tp.set_logger(&log);
            comm.set_logger(&log);
        }
        
        // Active messages
        auto am_dgeqrt_2_dlarfb_Mat = comm.make_large_active_msg(
            [&](int& k, view<int>& js){
                for (auto& j: js){
                    dlarfb_tf.fulfill_promise({k,j});
                }
                return ;
            },
            [&](int& k, view<int>& ){
                Mat.at({k,k}).resize(n,n);
                return Mat.at({k,k}).data();
            },
            [&](int& , view<int>& ){
                return ;
            }
        );

        auto am_dgeqrt_2_dlarfb_T = comm.make_large_active_msg(
            [&](int& k, view<int>& js){
                for (auto& j: js){
                    dlarfb_tf.fulfill_promise({k,j});
                }
                return ;
            },
            [&](int& k, view<int>& ){
                T.at({k,k}).resize(n,n);
                return T.at({k,k}).data();
            },
            [&](int& , view<int>& ){
                return ;
            }
        );

        auto am_dgeqrt_2_dtsqrt = comm.make_large_active_msg(
            [&](int& k){
                dtsqrt_tf.fulfill_promise({k+1,k});
            },
            [&](int& k){
                Mat.at({k,k}).resize(n,n);
                return Mat.at({k,k}).data();
            },
            [&](int& ){
                return ;
            }
        );

        // From dsqrt
        auto am_dtsqrt_2_dtsqrt = comm.make_large_active_msg(
            [&](int& i, int& k){
                dtsqrt_tf.fulfill_promise({i,k});
            },
            [&](int& i, int& k){
                Mat.at({k,k}).resize(n,n);
                return Mat.at({k,k}).data();
            },
            [&](int& , int& ){
                return ;
            }
        );

        auto am_dtsqrt_2_dssrfb_Mat = comm.make_large_active_msg(
            [&](int& i, int& k, view<int>& js){
                for(auto& j: js){
                    dssrfb_tf.fulfill_promise({i,j,k}); 
                }
            },
            [&](int& i, int& k, view<int>&){
                Mat.at({i,k}).resize(n,n);
                return Mat.at({i,k}).data();
            },
            [&](int& , int& , view<int>&){
                return ;
            }
        );

        auto am_dtsqrt_2_dssrfb_T = comm.make_large_active_msg(
            [&](int& i, int& k, view<int>& js){
                for(auto& j: js){
                    dssrfb_tf.fulfill_promise({i,j,k}); 
                }
            },
            [&](int& i, int& k, view<int>&){
                T.at({i,k}).resize(n,n);
                return T.at({i,k}).data();
            },
            [&](int& , int& , view<int>&){
                return ;
            }
        );

        // From dlarfb
        auto am_dlarfb_2_dssrfb = comm.make_large_active_msg(
            [&](int& i, int&j,  int& k){
                dssrfb_tf.fulfill_promise({i,j,k});
            },
            [&](int& i, int&j,  int& k){
                Mat.at({k,j}).resize(n,n);
                return Mat.at({k,j}).data();
            },
            [&](int& , int& , int&){
                return ;
            }
        );

        // From dssrfb
        auto am_dssrfb_2_dssrfb = comm.make_active_msg(
            [&](view<double> &R, int& i, int&j,  int& k){
                Mat.at({k,j}) = Map<MatrixXd>(R.data(), n, n); // Check strides
                dssrfb_tf.fulfill_promise({i,j,k});
            }
        );

        // auto am_dssrfb_2_dssrfb = comm.make_large_active_msg(
        //     [&](int& i, int&j,  int& k){
        //         dssrfb_tf.fulfill_promise({i,j,k});
        //     },
        //     [&](int& , int&j,  int& k){
        //         Mat.at({k,j}).resize(n,n);
        //         return Mat.at({k,j}).data();
        //     },
        //     [&](int& , int& , int&){
        //         return ;
        //     }
        // );


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
                    int r = block2rank({k,j}); // dlarfb
                    if(to_fulfill.count(r) == 0) {
                        to_fulfill[r] = {j};
                    } else {
                        to_fulfill[r].push_back(j);
                    }
                }

                // First dtsqrt
                if (k+1 < N){
                   int r = block2rank({k+1,k}); // Only the next tsqrt
                   if (r == rank){
                       dtsqrt_tf.fulfill_promise({k+1, k});
                   }
                   else {
                       auto R_kk = view<double>(Mat.at({k,k}).data(), n * n );
                       am_dgeqrt_2_dtsqrt->send_large(r, R_kk, k);
                   } 
                }
                
                

                // Send data and trigger tasks -- dlarfb
                for (auto& p: to_fulfill){
                    int r = p.first; // rank
                    if (r == rank){
                        for(auto& j: p.second){
                            dlarfb_tf.fulfill_promise({k, j}); 
                            dlarfb_tf.fulfill_promise({k, j}); /// twice tto account for Mat, T 

                        }
                    }
                    else {
                        auto V_kk = view<double>(Mat.at({k,k}).data(), n * n );
                        auto T_kk = view<double>(T.at({k,k}).data(), n*n);
                        auto jsv = view<int>(p.second.data(), p.second.size());

                        am_dgeqrt_2_dlarfb_Mat->send_large(r, V_kk, k, jsv);
                        am_dgeqrt_2_dlarfb_T->send_large(r, T_kk, k, jsv);
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
                if (i+1 < N){
                    int r = block2rank({i+1,k});
                    if (r == rank){
                        dtsqrt_tf.fulfill_promise({i+1, k});
                    }
                    else {
                        auto R_kk = view<double>(Mat.at({k,k}).data(), n * n );
                        int inext = i+1;
                        am_dtsqrt_2_dtsqrt->send_large(r, R_kk, inext, k) ; // send A{k,k}
                    }
                }
                

                // dssrfb
                for(int j = k+1; j<N; j++) { 
                    int r = block2rank({i,j}); // ssrfb
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
                            dssrfb_tf.fulfill_promise({i,j,k}); // twice

                        }
                    }
                    else {
                        auto V_ik = view<double>(Mat.at({i,k}).data(), n * n );
                        auto T_ik = view<double>(T.at({i,k}).data(), n*n);
                        auto jsv = view<int>(p.second.data(), p.second.size());

                        // am_dtsqrt_2_dssrfb->send(r, V_ik, T_ik, i, jsv, k);
                        am_dtsqrt_2_dssrfb_Mat->send_large(r, V_ik,  i, k, jsv);
                        am_dtsqrt_2_dssrfb_T->send_large(r, T_ik, i, k, jsv);

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
                return (kj[0] == 0 ? 0 : 1) + 2;
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
                
                if (k+1 < N){
                    int r = block2rank({k+1, j});
                    if (r == rank){
                        dssrfb_tf.fulfill_promise({k+1, j, k});
                    }
                    else {
                        auto R_kj = view<double>(Mat.at({k,j}).data(), n * n ); 
                        int knext = k+1;
                        am_dlarfb_2_dssrfb->send_large(r, R_kj, knext, j, k); //send R_kj
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

                return (k==0 ? 2 : 3)+1;

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

                if (i+1 < N){
                    int r = block2rank({i+1,j});
                    if (r == rank){
                        dssrfb_tf.fulfill_promise({i+1,j,k});
                    }
                    else {
                        auto R_kj = view<double>(Mat.at({k,j}).data(), n * n ); 
                        int inext = i+1;
                        am_dlarfb_2_dssrfb->send_large(r, R_kj, inext, j, k);  
                        // am_dssrfb_2_dssrfb->send(r, R_kj, inext, j, k);  

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

            if(rank == 0) printf("Starting QR on a dense matrix\n");
            MPI_Barrier(MPI_COMM_WORLD);
            timer t0 = wctime();
            if (rank == 0){
                dgeqrt_tf.fulfill_promise(0);
            }
            tp.join();
            timer t1 = wctime();
            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == 0)
            {
                cout << "Time : " << elapsed(t0, t1) << endl;
            }

            if(LOG) {
                std::ofstream logfile;
                string filename = "2D_denseQR_"+ to_string(n_ranks)+".log."+to_string(rank);
                logfile.open(filename);
                logfile << log;
                logfile.close();
            }
    }

    // Gather everything on rank 0 and test for accuracy
    {
        Communicator comm(MPI_COMM_WORLD, VERB);
        Threadpool tp(n_threads, &comm, VERB);
        Taskflow<int2> gather_tf(&tp, VERB);

        // 3 active messages with different data
        auto am_gather_0 = comm.make_active_msg(
        [&](view<double> &R_j, view<double> &V_ij, view<double> &T_ij,  int& i, int& j) {
            // A.block(0, j*n, (j+1)*n, n) = Map<MatrixXd>(R_j.data(), (j+1)*n, n) ;
            MatrixXd Rtemp = Map<MatrixXd>(R_j.data(), (j+1)*n, n) ;
            for (int k=0; k<j+1; ++k){
                Mat.at({k,j}) = Rtemp.block(k*n, 0, n, n);
            }
            // A.block(i*n, j*n, n, n) = Map<MatrixXd>(V_ij.data(), n, n); // Last block
            // Tmat.block(i*n, j*n, n, n) = Map<MatrixXd>(T_ij.data(), n, n);
            Mat.at({i,j}) = Map<MatrixXd>(V_ij.data(), n, n); // Last block
            T.at({i,j}) = Map<MatrixXd>(T_ij.data(), n, n);
        });

        auto am_gather_1 = comm.make_active_msg(
        [&](view<double> &V_ij, view<double> &T_ij,  int& i, int& j) {
            // A.block(i*n, j*n, n, n) = Map<MatrixXd>(V_ij.data(), n, n); // Last block
            // Tmat.block(i*n, j*n, n, n) = Map<MatrixXd>(T_ij.data(), n, n);

            Mat.at({i,j}) = Map<MatrixXd>(V_ij.data(), n, n); // Last block
            T.at({i,j}) = Map<MatrixXd>(T_ij.data(), n, n);

        });

        auto am_gather_2 = comm.make_active_msg(
        [&](view<double> &T_ij,  int& i, int& j) {
            // Tmat.block(i*n, j*n, n, n) = Map<MatrixXd>(T_ij.data(), n, n);
            T.at({i,j}) = Map<MatrixXd>(T_ij.data(), n, n);

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
                    if (i == N-1) { // Last row contains all the updated R[:, j]
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
            })
            .set_name([](int2 ij) {
                return "gather_"+ to_string(ij[0]) + "_" +to_string(ij[1]);
            });

        // for(int i = 0; i < N; i++) {
        //     for (int j=0; j <=i ; ++j){
        //         if(block2rank({i,j}) == rank) {
        //             gather_tf.fulfill_promise({i,j});
        //         }
        //     }
        // }


        tp.join();
        MPI_Barrier(MPI_COMM_WORLD);


        // if(rank == 0) {
        //     // Test 1   
        //     // Q^T b
        //     for (int j = 0; j < N; ++j){
        //         // MatrixXd Ajj = A.block(j*n, j*n, n, n);
        //         // MatrixXd Tjj = Tmat.block(j*n, j*n, n, n);
        //         // MatrixXd Tjj = T.at({j,j});

        //         int info = LAPACKE_dlarfb(LAPACK_COL_MAJOR, 'L', 'T', 'F', 'C', n, 1, n, Mat.at({j,j}).data(), n, 
        //                                   T.at({j,j}).data(), n, b.segment(j*n, n).data(), n);
        //         assert(info == 0);

        //         for (int i=j+1; i < N; ++i){
        //             info = LAPACKE_dtpmqrt(LAPACK_COL_MAJOR, 'L', 'T', n, 1, n, 0, n, 
        //                                        Mat.at({i,j}).data(), n,
        //                                        T.at({i,j}).data(), n,
        //                                        b.segment(j*n, n).data(), n,
        //                                        b.segment(i*n, n).data(), n);
        //             assert(info ==0);

        //         }
        //     }

        //     // R^{-1}b
        //     for (int i= N-1; i>-1; --i){
        //         for (int j=N-1; j>i; --j){
        //             b.segment(i*n, n) -= Mat.at({i,j})*b.segment(j*n,n);
        //         }
        //         auto R = Mat.at({i,i}).triangularView<Upper>();
        //         VectorXd y = b.segment(i*n, n);
        //         R.solveInPlace(y);
        //         b.segment(i*n, n) = y;
        //     }
            
        //     double error = (b - x).norm() / x.norm();
        //     cout << "Error solve: " << error << endl;
        //     assert(error<=1e-8);
        // }
    }
    return 0;
}


int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;

    MPI_Init_thread(NULL, NULL, req, &prov);

    assert(prov == req);

    // ::testing::InitGoogleTest(&argc, argv);

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
        N_ = atoi(argv[3]);
    }

    if (argc >= 5)
    {
        p_ = atoi(argv[4]);
    }

    if (argc >= 6)
    {
        q_ = atoi(argv[5]);
    }

    if (argc >= 7)
    {
        VERB = atoi(argv[6]);
    }

    if (argc >= 8)
    {
        LOG = atoi(argv[7]);
    }
    // const int return_flag = RUN_ALL_TESTS();
    const int return_flag = denseQR(n_threads_, n_, N_, p_, q_);

    MPI_Finalize();

    return return_flag;
}
