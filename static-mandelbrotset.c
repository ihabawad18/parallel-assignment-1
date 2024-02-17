#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));
    return iter;
}

void save_pgm(const char* filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg;
    int temp;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n"); // Writing Magic Number to the File   
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);  // Writing Width and Height
    fprintf(pgmimg, "255\n");  // Writing the maximum gray value 
    int count = 0;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            temp = image[i][j];
            fprintf(pgmimg, "%d ", temp); // Writing the gray values in the 2D array to the file 
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rows_per_process = HEIGHT / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank + 1) * rows_per_process;


    int(*image)[WIDTH] = (int(*)[WIDTH])malloc(sizeof(int[HEIGHT][WIDTH]));

    if (image == NULL) {
        // Error handling for failed memory allocation
        fprintf(stderr, "Failed to allocate memory for image array\n");
        return 1;
    }


    double AVG = 0;
    const int N = 10; // number of trials
    double total_time[N];
    struct complex c;


    //for (int k = 0; k < N; k++) {
    double start_time = clock(); // Start measuring time

    int i, j;
    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < WIDTH; j++) {
            c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
            c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
            image[i][j] = cal_pixel(c);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        for (int i = 1; i < size; i++) {

            MPI_Recv(&image[i * rows_per_process][0], rows_per_process * WIDTH, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        double end_time = clock(); // End measuring time
        printf("nb of proccessors %d\n", size);
        save_pgm("mandelbrot.pgm", image);
        printf("The execution time is: %f ms", (((double)(end_time - start_time)) / CLOCKS_PER_SEC) * 1000);
    }
    else {
        MPI_Send(&image[start_row][0], rows_per_process * WIDTH, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }


    MPI_Finalize();

    free(image);
    return 0;
}