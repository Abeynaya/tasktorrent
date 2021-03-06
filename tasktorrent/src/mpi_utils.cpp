#ifndef TTOR_SHARED

#include "mpi_utils.hpp"

namespace ttor {

int comm_rank(MPI_Comm comm) {
    int rank;
    TASKTORRENT_MPI_CHECK(MPI_Comm_rank(comm, &rank));
    return rank;
}

int comm_size(MPI_Comm comm) {
    int size;
    TASKTORRENT_MPI_CHECK(MPI_Comm_size(comm, &size));
    return size;
}

std::string processor_name()
{
    char name[MPI_MAX_PROCESSOR_NAME];
    int size;
    TASKTORRENT_MPI_CHECK(MPI_Get_processor_name(name, &size));
    return std::string(name);
}

}

#endif