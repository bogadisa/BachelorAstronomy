#include "functions.h" 

#ifdef __MACH__
#include <stdlib.h>
#else 
#include <malloc.h>
#endif

void import_JPEG_file (const char* filename, unsigned char** image_chars,
                       int* image_height, int* image_width,
                       int* num_components);
void export_JPEG_file (const char* filename, const unsigned char* image_chars,
                       int image_height, int image_width,
                       int num_components, int quality);


int main(int argc, char *argv[]) {
    int m, n, c, iters;
    int my_m, my_n, my_rank, num_procs;
    float kappa;
    image u, u_bar, whole_image;
    unsigned char *image_chars, *my_image_chars;
    char *input_jpeg_filename, *output_jpeg_filename;

    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
    kappa = strtof(argv[1], NULL) ; iters = atoi(argv[2]) ; input_jpeg_filename = argv[3] ; output_jpeg_filename = argv[4];

    if (my_rank==0) {
        import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
        printf("Succeeded! vertical pixels: %d, horizontal pixels: %d, num components: %d\n", m, n, c);
        printf("Allocating images...\n");
        allocate_image (&whole_image, m, n);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast (&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    /* 2D decomposition of the m x n pixels evenly among the MPI processes */
    my_m = m/num_procs + (my_rank < m%num_procs) + 1 + ((my_rank != 0) && (my_rank != num_procs-1));
    my_n = n;

    allocate_image (&u, my_m, my_n);
    allocate_image (&u_bar, my_m, my_n);
    /* each process asks process 0 for a partitioned region */
    /* of image_chars and copy the values into u */
    /* ... */

    if (my_rank==0) {
        printf("Done! Now dividing work among processes...\n");
    }
    // Getting ready to scatter the picture
    int *sendcounts, *displs;
    int sum = 0;
    sendcounts = (int *)malloc(num_procs*sizeof(int));
    displs = (int *)malloc(num_procs*sizeof(int));
    for (int i=0; i<num_procs; i++) {
        sendcounts[i] = n * (m/num_procs + (i < m%num_procs) + (i != (num_procs-1)));
        displs[i] = sum - (i != 0);
        sum += sendcounts[i];
    }
    
    my_image_chars = (unsigned char *)malloc(my_m*my_n*sizeof(unsigned char));

    if (my_rank==0) {
        printf("Done! Sending out the picture...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(image_chars, sendcounts, displs, MPI_UNSIGNED_CHAR, my_image_chars, sendcounts[my_rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    
    if (my_rank==0) printf("Done! Proceeding with converting...\n");
    convert_jpeg_to_image (my_image_chars, &u);
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank==0) printf("Done! Now starting denoising...\n");
    iso_diffusion_denoising_parallel (&u, &u_bar, kappa, iters);
    // iso_diffusion_denoising (&u, &u_bar, kappa, iters);
    
    /* each process sends its resulting content of u_bar to process 0 */
    /* process 0 receives from each process incoming values and */
    /* copy them into the designated region of struct whole_image */
    /* ... */

    //if (my_rank!=0) whole_image.image_data = NULL;
    if (my_rank==0) printf("Complete! Patching the picture back together\n");
    
    image u_cropped;
    int crop_m, crop_n;
    crop_m = m/num_procs + (my_rank < m%num_procs);
    crop_n = n;
    allocate_image(&u_cropped, crop_m, crop_n);
    for (int i=0; i<crop_m; i++) {
        u_cropped.image_data[i] = u_bar.image_data[i + (my_rank!=0)];
    }
    int *recvcounts;
    sum = 0;
    recvcounts = (int *)malloc(num_procs*sizeof(int));
    for (int i=0; i<num_procs; i++) {
        recvcounts[i] = (m/num_procs + (i < m%num_procs))*n;
        displs[i] = sum;
        sum += recvcounts[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Datatype stype;
    MPI_Type_vector(my_m, n, n, MPI_FLOAT, &stype);
    MPI_Type_commit(&stype);
    // doesnt work but shows promise
    // MPI_Gatherv(u_cropped.image_data, 1, stype, whole_image.image_data, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(u_cropped.image_data, recvcounts[my_rank], MPI_FLOAT, whole_image.image_data, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if (my_rank==0) {
        printf("Done! Converting back to jpeg format...\n");
        convert_image_to_jpeg(&whole_image, image_chars);
        printf("Done! Creating new file\n");
        export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
        printf("Success! Cleaning up before exiting");
        deallocate_image (&whole_image);
    }
    deallocate_image(&u);
    deallocate_image(&u_bar);

    MPI_Finalize();
    return 0;
}