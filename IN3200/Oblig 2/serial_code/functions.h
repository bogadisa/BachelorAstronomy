#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void allocate_image(image *u, int m, int n) {
    float **tmp;
    tmp = (float **)malloc(m*sizeof(float *));

    for (int i = 0; i < m; i ++) {
        tmp[i] = (float *)malloc(n*sizeof(float));
    }

    (*u).image_data = tmp;
    (*u).m = m;
    (*u).n = n;
}

void deallocate_image(image *u) {
    free((*u).image_data);
}

void convert_jpeg_to_image(const unsigned char* image_chars, image *u) {
    int i, j, m, n, c;
    m = (*u).m; n = (*u).n;
    for (i=0; i<m; i++) {
        for (j=0; j<n; j++) {
            (*u).image_data[i][j] = (float)image_chars[i*n +j];
        }
    }
    
}
void convert_image_to_jpeg(const image *u, unsigned char* image_chars) {
    int i, j, m, n, c;
    m = (*u).m; n = (*u).n;
    for (i=0; i<m; i++) {
        for (j=0; j<n; j++) {
            image_chars[i*n + j] = (unsigned char)(*u).image_data[i][j];
        }
    }
}

void iso_diffusion_denoising(image *u, image *u_bar, float kappa, int iters) {
    int i, j, k, m, n, c;
    int blink = iters/10;
    image tmp;
    m = (*u).m; n = (*u).n;
    for (i=0; i<m; i++) {
        for (j=0; j<n; j++) {
            (*u_bar).image_data[i][j] = (*u).image_data[i][j];
        }
    }
    for (k=0; k<iters; k++) {
        if ((k+1)%blink == 0) {
            printf(".");
        }
        for (i=0; i<m; i++) {
            for (j=0; j<n; j++) {
                if (((i == 0) || (i == (m-1))) || ((j == 0) || (j == (n-1)))) {
                    (*u_bar).image_data[i][j] = (*u).image_data[i][j];
                } else {
                    (*u_bar).image_data[i][j] = (*u).image_data[i][j] + kappa*((*u).image_data[i - 1][j]
                                                                                + (*u).image_data[i][j - 1]
                                                                                - 4*(*u).image_data[i][j]
                                                                                + (*u).image_data[i][j + 1]
                                                                                + (*u).image_data[i + 1][j]);
                }
            }
        }
        tmp = *u_bar;
        *u_bar = *u;
        *u = tmp;
    }
}