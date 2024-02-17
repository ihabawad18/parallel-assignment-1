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

    int(*image)[WIDTH] = (int(*)[WIDTH])malloc(sizeof(int[HEIGHT][WIDTH]));
    if (image == NULL) {
        // Error handling for failed memory allocation
        fprintf(stderr, "Failed to allocate memory for image array\n");
        return 1;
    }

    double start_time = clock(); // Start measuring time

    if (rank == 0) {

        int count = 0, row = 0;

        for (int k = 1; k < size; k++) {
            MPI_Send(&row, 1, MPI_INT, k, 0, MPI_COMM_WORLD);
            count++;
            row++;
        }
        int temp_row[WIDTH]; // Temporary array to hold received row


        do {
            MPI_Status status;
            MPI_Recv(&temp_row, WIDTH, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int received_row = status.MPI_TAG; // Extract the received row number
            for (int j = 0; j < WIDTH; j++) {
                image[received_row][j] = temp_row[j]; // Copy received data to the image array
            }
            count--;
            int sender_rank = status.MPI_SOURCE; //get sender rank

            if (row < HEIGHT) {
                MPI_Send(&row, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD);
                row++;
                count++;
            }
            else {
                MPI_Send(&row, 1, MPI_INT, sender_rank, 9999, MPI_COMM_WORLD); // send invalid tag for termination
            }

        } while (count > 0);

        double end_time = clock(); // End measuring time
        printf("nb of proccessors %d\n", size);
        save_pgm("mandelbrot.pgm", image);
        printf("The execution time is: %f ms\n", (((double)(end_time - start_time)) / CLOCKS_PER_SEC) * 1000);
    }
    else {
        int y;
        MPI_Status status;
        MPI_Recv(&y, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        while (status.MPI_TAG == 0) {
            struct complex c;
            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;
                image[y][j] = cal_pixel(c);
            }
            MPI_Send(&image[y][0], WIDTH, MPI_INT, 0, y, MPI_COMM_WORLD);
            MPI_Status status2;
            MPI_Recv(&y, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status2);
            if (status2.MPI_TAG == 9999) {
                break;
            }
        }

    }

    MPI_Finalize();
    free(image);
    return 0;
}