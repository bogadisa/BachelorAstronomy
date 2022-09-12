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


int main (int argc, char *argv[]) {
    int m, n, c, iters;
    float kappa;
    image u, u_bar;
    unsigned char *image_chars;
    char *input_jpeg_filename, *output_jpeg_filename;

    kappa = strtof(argv[1], NULL) ; iters = atoi(argv[2]) ; input_jpeg_filename = argv[3] ; output_jpeg_filename = argv[4];

    import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
    printf("Succeeded! vertical pixels: %d, horizontal pixels: %d, num components: %d\n", m, n, c);
    printf("Allocating images...\n");
    allocate_image (&u, m, n);
    allocate_image (&u_bar, m, n);
    printf("Done!\nProceeding with converting...\n");
    convert_jpeg_to_image (image_chars, &u);
    printf("Done!\nNow starting denoising");
    iso_diffusion_denoising (&u, &u_bar, kappa, iters);
    printf("\nDone!\nConverting back to jpeg format...\n");
    convert_image_to_jpeg (&u_bar, image_chars);
    printf("Done!\nCreating new file\n");
    export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
    printf("Success! Cleaning up before exiting");
    deallocate_image (&u);
    deallocate_image (&u_bar);

    return 0;
}