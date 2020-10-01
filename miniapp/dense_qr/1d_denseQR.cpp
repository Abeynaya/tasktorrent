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
    MatrixXd A = MatrixXd::NullaryExpr(N * n, N * n, gen);

    MatrixXd Aref = A;

    VectorXd beta = VectorXd::Zero(N*n);

    // Mapper
    auto block2rank = [&](int i){
        int ii = i % p;
        int r = ii;
        assert(r <= n_ranks);
        return r;
    };

    // Block the matrix for every node
    // Store it in a map
    // Later on we can probably optimize this, to avoid storing the whole thing
    // Note: concurrent access to the map should _NOT_ modify it, otherwise we need a lock
    map<int, MatrixXd> Mat;
    map<int, VectorXd> tau;

    for(int i = 0; i < N; i++) {
        if(block2rank(i) == rank) {
            Mat[i] = A.block(0, i * n, N*n, n);
            tau[i] = VectorXd::Zero(n);
        } else {
            Mat[i] = MatrixXd::Zero(N*n, n);
            tau[i] = VectorXd::Zero(n);
        }
    }

    // Factorize
    {
        // Initialize the communicator structure
        Communicator comm(MPI_COMM_WORLD, VERB);

        // Threadpool
        Threadpool tp(n_threads, &comm, VERB, "[" + to_string(rank) + "]_");
        Taskflow<int> geqrf_tf(&tp, VERB); // Householder: A = Q^TR
        Taskflow<int2> ormqr_tf(&tp, VERB); // Perform Q^T A = B

        // Log
        DepsLogger dlog(1000000);
        Logger log(1000000);
        if(LOG) {
            tp.set_logger(&log);
            comm.set_logger(&log);
        }
        
        // Active messages
        auto am_ormqr = comm.make_active_msg(
            [&](view<double> &V, view<double> &t, view<int>& is, int& j){
                Mat.at(j).block(j*n, 0, (N-j)*n, n) = Map<MatrixXd>(V.data(), (N*n-j*n), n); // Check strides
                tau.at(j) = Map<VectorXd>(t.data(), n);
                for (auto& i: is){
                    ormqr_tf.fulfill_promise({j,i});
                }
            }
        );

        geqrf_tf.set_mapping([&] (int j){
                return (j % n_threads);
            })
            .set_indegree([](int){
                return 1;
            })
            .set_task([&] (int j) {
                MatrixXd Apiv = Mat.at(j).block(j*n, 0, N*n-j*n, n);
                int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, (N-j)*n, n, Apiv.data(), (N-j)*n, tau.at(j).data());
                Mat.at(j).block(j*n, 0, N*n-j*n, n) = Apiv;
                assert(info == 0);
            })
            .set_fulfill([&](int j) {
                // Dependencies
                map<int,vector<int>> to_fulfill; 
                for(int i = j+1; i<N; i++) {
                    int r = block2rank(i);
                    if(to_fulfill.count(r) == 0) {
                        to_fulfill[r] = {i};
                    } else {
                        to_fulfill[r].push_back(i);
                    }
                }
                // Send data and trigger tasks
                for (auto& p: to_fulfill){
                    int r = p.first; // rank
                    if (r == rank){
                        for(auto& i: p.second){
                            ormqr_tf.fulfill_promise({j, i}); 
                        }
                    }
                    else {
                        MatrixXd Apiv = Mat.at(j).block(j*n, 0, N*n-j*n, n);

                        auto Q_j = view<double>(Apiv.data(), (N*n-j*n)*n);
                        auto tau_j = view<double>(tau.at(j).data(), n);
                        auto isv = view<int>(p.second.data(), p.second.size());
                        am_ormqr->send(r, Q_j, tau_j, isv, j);
                    }
                }
            })
            .set_name([](int j) {
                return "geqrf_" + to_string(j);
            });

        // ormqr
        ormqr_tf.set_mapping([&] (int2 ij){
                return (ij[1] % n_threads);
            })
            .set_indegree([](int2 ij){
                int i = ij[0];
                if (i==0)
                    return 1;
                else 
                    return 2; // previous ormqr and geqrf
            })
            .set_task([&] (int2 ij) {

                int i=ij[0]; // who is sending
                int j=ij[1]; // me

                MatrixXd Q = Mat.at(i).block(i*n, 0, (N-i)*n, n);
                MatrixXd Aj = Mat.at(j).block(i*n, 0, (N-i)*n, n); // Still i


                int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', (N-i)*n, n, n, Q.data(), (N-i)*n, tau.at(i).data(), Aj.data(), (N-i)*n);

                assert(info == 0);
                Mat.at(j).block(i*n, 0, (N-i)*n, n) = Aj;

            })
            .set_fulfill([&](int2 ij) {
                int i = ij[0];
                int j = ij[1];
                // ormqr (i,j) fullfills geqrf (j) if j=i+1
                if (j==i+1){
                    // The block is in the same rank
                    geqrf_tf.fulfill_promise(j); 
                }
                else {
                        ormqr_tf.fulfill_promise({i+1,j});
                }
            })
            .set_name([](int2 ij) {
                return "ormqr_" + to_string(ij[1]);
            });

            if(rank == 0) printf("Starting QR on a dense matrix\n");
            MPI_Barrier(MPI_COMM_WORLD);
            timer t0 = wctime();
            if (rank == 0){
                geqrf_tf.fulfill_promise(0);
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
                string filename = "1D_denseQR_"+ to_string(n_ranks)+".log."+to_string(rank);
                logfile.open(filename);
                logfile << log;
                logfile.close();
            }
    }

    // Gather everything on rank 0 and test for accuracy
    {
        Communicator comm(MPI_COMM_WORLD, VERB);
        Threadpool tp(n_threads, &comm, VERB);
        Taskflow<int> gather_tf(&tp, VERB);
        auto am_gather = comm.make_active_msg(
        [&](view<double> &QR_j, view<double> &tau_j,  int& j) {
        	A.block(0, j * n, N*n, n) = Map<MatrixXd>(QR_j.data(), N*n, n);
            beta.segment(j*n, n) = Map<VectorXd>(tau_j.data(), n);
        });
        // gather
        gather_tf.set_mapping([&](int j) {
                return ( j % n_threads );
            })
            .set_indegree([](int) {
                return 1;
            })
            .set_task([&](int j) {
                
                if(rank != 0) {
                    auto QR_j = view<double>(Mat.at(j).data(), N*n*n);
                    auto tau_j = view<double>(tau.at(j).data(), n);
                    am_gather->send(0, QR_j, tau_j, j);
                } else {
                    A.block(0, j * n, N*n, n) = Mat.at(j);
                    beta.segment(j*n, n) = tau.at(j);

                }
            })
            .set_name([](int j) {
                return "gather_"+ to_string(j);
            });

        for(int i = 0; i < N; i++) {
            if(block2rank(i) == rank) {
                gather_tf.fulfill_promise(i);
            }
        }
        tp.join();
        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0) {
            // Test 1         
            {
                auto R = A.triangularView<Upper>();
                VectorXd x = VectorXd::Random(n * N);
                VectorXd b = Aref*x;
                VectorXd bref = b;
                int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', N*n, 1, N*n, A.data(), N*n, beta.data(), b.data(), N*n);
                
                R.solveInPlace(b);
                double error = (b - x).norm() / x.norm();
                cout << "Error solve: " << error << endl;
                assert(error<=1e-8);
            }
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
