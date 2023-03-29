#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#define DIMENSION 2

const int n1 = 600; //6000
const int n2 = 500; //5000
const int n3 = 400; //4000


void fillMatrix (double* matrix, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            matrix[i * M + j] = 1;
        }
    }
}


void mulMatrix (double* A, double* B, double* C, int row_count, int column_size, int column_count) {
    for (int i = 0; i < row_count; i++) {
        for (int j = 0; j < column_count; j++) {
            C[i * column_count + j] = 0;
            for (int k = 0; k < column_size; k++) {
                C[i * column_count + j] += A[i * column_size + k] * B[k * column_count + j];
            }
        }
    }
}


int main(int argc, char** argv) {
    int p1 = 2;
    int p2 = 2;
    int process_rank;
    int process_count;


    MPI_Init(&argc, &argv);
    double start = MPI_Wtime();

    //Общее число процессов
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    //Номер процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);



    //Создаем решетку процессов
    int ndims = DIMENSION; //размерность декартовой решетки
    int dims[DIMENSION] = {p1, p2}; //целочисленный массив, состоящий из ndims элементов, задающий количество процессов в каждом измерении
    int periodic[DIMENSION] = {0, 0};  //не замыкаем
    int reorder = 0; //не меняем порядок нумерации процессов в новом коммутаторе
    MPI_Comm comm_grid;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periodic, reorder, &comm_grid);


    int coord_of_process[DIMENSION]; // координаты процесса в декартовой решетке
    MPI_Cart_coords(comm_grid, process_rank, DIMENSION, coord_of_process);

    //коммуникатор строк решетки
    MPI_Comm comm_rows;
    int remain_dims_rows[DIMENSION] = {0, 1};
    MPI_Cart_sub(comm_grid, remain_dims_rows, &comm_rows);

    //коммутатор столбцов решетки
    MPI_Comm comm_columns;
    int remain_dims_columns[DIMENSION] = {1, 0};
    MPI_Cart_sub(comm_grid, remain_dims_columns, &comm_columns);



    double* A;
    double* B;

    if (process_rank == 0) {
        A = (double*)malloc(sizeof(double) * n1 * n2);
        B = (double*)malloc(sizeof(double) * n2 * n3);
        fillMatrix(A, n1, n2);
        fillMatrix(B, n2, n3);
    }

    //Количество строк в сообщении
    int row_count = n1 / p1;
    //Колонок в сообщении
    int column_count = n3 / p2;

    double* partA = (double*)malloc(sizeof(double) * row_count * n2);
    double* partB = (double*)malloc(sizeof(double) * n2 * column_count);
    double* partC = (double*)malloc(sizeof(double) * row_count * column_count);


    MPI_Datatype rowType;
    MPI_Type_contiguous(n2, MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);

    //Раздаем строки матрицы А по всему коммутатору строк
    if (coord_of_process[1] == 0) {
        MPI_Scatter(A, row_count, rowType, partA, row_count, rowType, 0, comm_columns);
    }
    MPI_Bcast(partA, row_count, rowType, 0, comm_rows);


//    if (process_rank == 0){
//        for (int i = 0; i < n2 * row_count; ++i) {
//            printf("[%f]",partA[i]);
//        }
//    }


    MPI_Datatype columnType;
    MPI_Datatype columnTypeResized;
    MPI_Type_vector(n2, 1, n3, MPI_DOUBLE, &columnType);
    MPI_Type_create_resized(columnType, 0, sizeof(double), &columnTypeResized);
    MPI_Type_commit(&columnType);
    MPI_Type_commit(&columnTypeResized);

    //Раздаем столбцы матрицы B по всему коммутатору столбцов
    if (coord_of_process[0] == 0) {
        MPI_Scatter(B, column_count, columnTypeResized, partB, column_count, columnTypeResized, 0, comm_rows);
    }
    MPI_Bcast(partB, column_count, columnTypeResized, 0, comm_columns);


//    if (process_rank == 0){
//        for (int i = 0; i < n2 * column_count; ++i) {
//            printf("[%f]",partB[i]);
//        }
//    }

    mulMatrix(partA, partB, partC, row_count, n2, column_count);


    MPI_Datatype partCType;
    MPI_Type_vector(row_count, column_count, n3, MPI_DOUBLE, &partCType);
    MPI_Type_commit(&partCType);

//    отправляем матрицу С нулевому процессу
    if (process_rank != 0) {
        MPI_Send(partC, row_count * column_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if (process_rank == 0) {
        double *C = (double *) calloc(n1 * n3, sizeof(double));
        for (int i = 0; i < row_count; i++) {
            for (int j = 0; j < column_count; j++) {
                C[i * n3 + j] = partC[i * column_count + j];
            }
        }

        for (int i = 0; i < p1; i++) {
            for (int j = 0; j < p2; j++) {
                if (!(i == 0 && j == 0)) {
                    MPI_Status status;
                    MPI_Recv(C + i * n3 * row_count + j * column_count,
                             n1 * n3, partCType, i * p2 + j, 0, MPI_COMM_WORLD, &status);
                }
            }
        }
        free(A);
        free(B);
        free(C);
    }


    double end = MPI_Wtime();
    if (process_rank == 0) {
        printf("p1 = %d p2 = %d time = %lf\n", p1, p2, end - start);
    }

    free(partA);
    free(partB);
    free(partC);
    MPI_Finalize();
    return 0;
}
