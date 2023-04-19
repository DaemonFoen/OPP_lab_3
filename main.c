#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DIMENSION 2
#define P1 2
#define P2 12
#define TRUE 1
#define FALSE 0
#define SEND_COLUMN 5

#define N1 9000
#define N2 1000
#define N3 9000

typedef struct coords {
    int row;
    int col;
} ProcCoords;

void fill_matrix(double *matrix, int row_dim, int col_dim) {
    int tmp = 0;
    for (int i = 0; i < row_dim; i++) {
        for (int j = 0; j < col_dim; j++) {
            matrix[i * col_dim + j] = (col_dim == N2) ? tmp % N2 : tmp % N3;
            tmp += 1;
        }
    }
}


void
send_A_on_main_column(ProcCoords current_process_coords, MPI_Comm columns_communicator, double *A, double *part_of_A) {
    if (current_process_coords.col == 0) {
        MPI_Scatter(A, (N1 / P1) * N2, MPI_DOUBLE, part_of_A, (N1 / P1) * N2, MPI_DOUBLE, 0, columns_communicator);
    }
}

void send_B_on_main_row(int current_process_rank, MPI_Comm rows_communicator, double *B, double *part_of_B,
                        ProcCoords current_process_coords) {

    if (current_process_rank == 0) {
        MPI_Datatype column_bypass;
        MPI_Type_vector(N2, N3 / P2, N3, MPI_DOUBLE, &column_bypass);
        MPI_Type_commit(&column_bypass);

        for (int i = 1; i < P2; i++) {
            MPI_Send(B + (N3 / P2) * i, 1, column_bypass, i, SEND_COLUMN, rows_communicator);
        }
        for (int i = 0, k = 0; i < N3 * N2;) {
            for (int j = 0; j < N3 / P2; j++) {
                part_of_B[k + j] = B[i + j];
            }
            i += N3;
            k += N3 / P2;
        }
        MPI_Type_free(&column_bypass);
    }

    if (current_process_coords.row == 0 && current_process_rank != 0) {
        MPI_Datatype cont_column_bypass;
        MPI_Type_contiguous(N2 * N3 / P2, MPI_DOUBLE, &cont_column_bypass);
        MPI_Type_commit(&cont_column_bypass);

        MPI_Status stat;
        MPI_Recv(part_of_B, 1, cont_column_bypass, 0, SEND_COLUMN, rows_communicator, &stat);

        MPI_Type_free(&cont_column_bypass);
    }
}

void matrix_mult(double *res_matrix, double *left_operand, double *right_operand, int res_row_dim, int res_col_dim,
                 int n2_dim) {
    for (int i = 0; i < res_row_dim; i++) {
        for (int j = 0; j < res_col_dim; j++) {
            res_matrix[i * res_col_dim + j] = 0;
            for (int k = 0; k < n2_dim; k++)
                res_matrix[i * res_col_dim + j] +=
                        left_operand[i * res_col_dim + k] * right_operand[k * res_col_dim + j];
        }
    }
}

void
collect_data(int current_process_rank, int processes_count, MPI_Comm grid_communicator, double *C, double *part_of_C) {

    MPI_Datatype part_col_C;
    MPI_Type_vector(N1 / P1, N3 / P2, N3 / P2, MPI_DOUBLE, &part_col_C);
    MPI_Type_commit(&part_col_C);

    if (current_process_rank != 0) {
        MPI_Send(part_of_C, 1, part_col_C, 0, 0, MPI_COMM_WORLD);
    }

    if (current_process_rank == 0) {
        MPI_Status stat;

        MPI_Datatype C_rec;
        MPI_Type_vector(N1 / P1, N3 / P2, N3, MPI_DOUBLE, &C_rec);
        MPI_Type_commit(&C_rec);

        for (int i = 1; i < processes_count; i++) {

            int coords[DIMENSION];
            MPI_Cart_coords(grid_communicator, i, DIMENSION, coords);
            ProcCoords process_coords;
            process_coords.col = coords[1];
            process_coords.row = coords[0];

            int offset = process_coords.col * (N3 / P2) + process_coords.row * N3 * (N1 / P1);
            MPI_Recv(C + offset, 1, C_rec, i, 0, MPI_COMM_WORLD, &stat);
        }

        MPI_Type_free(&C_rec);
        for (int i = 0; i < N1 / P1;) {
            for (int j = 0; j < N3 / P2; j++) {
                C[i * N3 + j] = part_of_C[i * (N3 / P2) + j];
            }
            i += 1;
        }
    }
    MPI_Type_free(&part_col_C);
}

int main(int argc, char **argv) {

    int current_process_rank = 0;
    int processes_count = 0;
    double time_start = 0;
    double time_end = 0;

    MPI_Init(&argc, &argv);
    time_start = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &processes_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process_rank);

    int dims[DIMENSION] = {P1, P2};
    int periods[DIMENSION] = {FALSE, FALSE};
    int reorder = FALSE;

    MPI_Comm grid_communicator;
    MPI_Cart_create(MPI_COMM_WORLD, DIMENSION, dims, periods, reorder, &grid_communicator);

    int coord_of_process[DIMENSION];
    MPI_Cart_coords(grid_communicator, current_process_rank, DIMENSION, coord_of_process);
    ProcCoords current_process_coords;
    current_process_coords.col = coord_of_process[1];
    current_process_coords.row = coord_of_process[0];

    MPI_Comm rows_communicator;
    int varying_coords_rows[DIMENSION] = {FALSE, TRUE}; // фиксируем первую координату, вторая меняется
    MPI_Cart_sub(grid_communicator, varying_coords_rows, &rows_communicator);

    MPI_Comm columns_communicator;
    int varying_coords_columns[DIMENSION] = {TRUE, FALSE}; // фиксируем вторую координату, первая меняется
    MPI_Cart_sub(grid_communicator, varying_coords_columns, &columns_communicator);

    double *A;
    double *B;
    double *C;
    double *part_of_A;
    double *part_of_B;
    double *part_of_C;
    part_of_A = (double *) malloc(sizeof(double) * (N1 / P1) * N2);
    part_of_B = (double *) malloc(sizeof(double) * (N3 / P2) * N2);
    part_of_C = (double *) malloc(sizeof(double) * (N3 / P2) * (N1 / P1));

    for (int i = 0; i < (N3 / P2) * N2; i++) part_of_B[i] = -1;
    for (int i = 0; i < (N1 / P1) * N2; i++) part_of_A[i] = -1;

    if (current_process_rank == 0) {
        A = (double *) malloc(sizeof(double) * N1 * N2);
        B = (double *) malloc(sizeof(double) * N2 * N3);
        C = (double *) malloc(sizeof(double) * N1 * N3);
        fill_matrix(A, N1, N2);
        fill_matrix(B, N2, N3);
        fill_matrix(C, N1, N3);
    }

    send_A_on_main_column(current_process_coords, columns_communicator, A, part_of_A);
    send_B_on_main_row(current_process_rank, rows_communicator, B, part_of_B, current_process_coords);

    MPI_Bcast(part_of_A, (N1 / P1) * N2, MPI_DOUBLE, 0, rows_communicator);
    MPI_Bcast(part_of_B, N3 / P2 * N2, MPI_DOUBLE, 0, columns_communicator);

    matrix_mult(part_of_C, part_of_A, part_of_B, N1 / P1, N3 / P2, N2);
    collect_data(current_process_rank, processes_count, grid_communicator, C, part_of_C);

    if (current_process_coords.row == 0 && current_process_coords.col == 0) {
        free(A);
        free(B);
        free(C);
    }

    free(part_of_A);
    free(part_of_B);
    free(part_of_C);

    time_end = MPI_Wtime();
    double time_spent = (double) (time_end - time_start);
    if (current_process_rank == 0) printf("EXECUTION TOOK %f\t SECONDS\n", time_spent);
    MPI_Finalize();
    return 0;
}
