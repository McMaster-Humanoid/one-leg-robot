#include "kalman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Minimal matrix helpers (row-major)
 * Matrices are stored as flat arrays length = r*c
 */

/* allocate a zeroed matrix */
double *mat_alloc(int r, int c) {
    double *m = (double*)calloc(r*c, sizeof(double));
    if (!m) { fprintf(stderr, "alloc fail\n"); exit(1); }
    return m;
}

/* copy matrix */
double *mat_copy(const double *a, int r, int c) {
    double *b = mat_alloc(r,c);
    memcpy(b, a, sizeof(double)*r*c);
    return b;
}

/* print matrix (for debug) */
void mat_print(const double *a, int r, int c, const char *name) {
    printf("%s (%dx%d):\n", name, r, c);
    for (int i=0;i<r;i++){
        for (int j=0;j<c;j++){
            printf("% .6f ", a[i*c + j]);
        }
        printf("\n");
    }
}

/* C = A * B */
double *mat_mul(const double *A, int rA, int cA, const double *B, int rB, int cB) {
    if (cA != rB) { fprintf(stderr,"mat_mul dim mismatch\n"); exit(1); }
    double *C = mat_alloc(rA, cB);
    for (int i=0;i<rA;i++){
        for (int k=0;k<cA;k++){
            double aik = A[i*cA + k];
            for (int j=0;j<cB;j++){
                C[i*cB + j] += aik * B[k*cB + j];
            }
        }
    }
    return C;
}

/* C = A + B */
double *mat_add(const double *A, const double *B, int r, int c) {
    double *C = mat_alloc(r,c);
    for (int i=0;i<r*c;i++) C[i] = A[i] + B[i];
    return C;
}

/* C = A - B */
double *mat_sub(const double *A, const double *B, int r, int c) {
    double *C = mat_alloc(r,c);
    for (int i=0;i<r*c;i++) C[i] = A[i] - B[i];
    return C;
}

/* transpose */
double *mat_transpose(const double *A, int r, int c) {
    double *T = mat_alloc(c, r);
    for (int i=0;i<r;i++) for (int j=0;j<c;j++) T[j*r + i] = A[i*c + j];
    return T;
}

/* In-place Gauss-Jordan inversion for square matrix. Returns new matrix (copy of inverse). [consider optimizing this by using LU decomposition instead] */
double *mat_inv(const double *A, int n) {
    // We'll form augmented [A | I] and row-reduce.
    double *aug = mat_alloc(n, 2*n);
    // copy A into left
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) aug[i*(2*n) + j] = A[i*n + j];
    // make I on right
    for (int i=0;i<n;i++) aug[i*(2*n) + (n + i)] = 1.0;

    for (int col=0; col<n; col++) {
        // find pivot
        int pivot = col;
        double maxv = fabs(aug[pivot*(2*n) + col]);
        for (int r=col+1;r<n;r++) {
            double v = fabs(aug[r*(2*n) + col]);
            if (v > maxv) { maxv = v; pivot = r; }
        }
        if (maxv < 1e-12) { fprintf(stderr,"Singular matrix (inv)\n"); free(aug); return NULL; }
        // swap rows if needed
        if (pivot != col) {
            for (int j=0;j<2*n;j++){
                double tmp = aug[col*(2*n) + j];
                aug[col*(2*n) + j] = aug[pivot*(2*n) + j];
                aug[pivot*(2*n) + j] = tmp;
            }
        }
        // normalize pivot row
        double pivval = aug[col*(2*n) + col];
        for (int j=0;j<2*n;j++) aug[col*(2*n) + j] /= pivval;
        // eliminate other rows
        for (int r=0;r<n;r++){
            if (r==col) continue;
            double factor = aug[r*(2*n) + col];
            if (factor != 0.0) {
                for (int j=0;j<2*n;j++){
                    aug[r*(2*n) + j] -= factor * aug[col*(2*n) + j];
                }
            }
        }
    }
    // extract right half as inverse
    double *inv = mat_alloc(n,n);
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) inv[i*n + j] = aug[i*(2*n) + (n + j)];
    free(aug);
    return inv;
}

/* free matrix */
void mat_free(double *m) { free(m); }

/* A helper to set identity */
void mat_set_identity(double *A, int n) {
    for (int i=0;i<n*n;i++) A[i] = 0.0;
    for (int i=0;i<n;i++) A[i*n + i] = 1.0;
}

/* Kalman filter struct */
typedef struct {
    int n; // state dim
    int m; // meas dim
    double *x; // n x 1 state
    double *P; // n x n covariance
    double *F; // n x n state transition
    double *Q; // n x n process noise cov
    double *H; // m x n observation matrix
    double *R; // m x m measurement noise cov
} Kalman;

/* allocate and initialize Kalman with provided matrices (copied) */
Kalman *kf_create(int n, int m,
                  const double *x0, const double *P0,
                  const double *F, const double *Q,
                  const double *H, const double *R) {
    Kalman *kf = (Kalman*)malloc(sizeof(Kalman));
    kf->n = n; kf->m = m;
    kf->x = mat_alloc(n, 1);
    kf->P = mat_alloc(n, n);
    kf->F = mat_alloc(n, n);
    kf->Q = mat_alloc(n, n);
    kf->H = mat_alloc(m, n);
    kf->R = mat_alloc(m, m);
    memcpy(kf->x, x0, sizeof(double)*n);
    memcpy(kf->P, P0, sizeof(double)*n*n);
    memcpy(kf->F, F, sizeof(double)*n*n);
    memcpy(kf->Q, Q, sizeof(double)*n*n);
    memcpy(kf->H, H, sizeof(double)*m*n);
    memcpy(kf->R, R, sizeof(double)*m*m);
    return kf;
}

/* free */
void kf_free(Kalman *kf) {
    if (!kf) return;
    mat_free(kf->x); mat_free(kf->P);
    mat_free(kf->F); mat_free(kf->Q); mat_free(kf->H); mat_free(kf->R);
    free(kf);
}

/* predict:
 * x = F * x
 * P = F * P * F^T + Q
 */
void kf_predict(Kalman *kf) {
    int n = kf->n;
    double *x_new = mat_mul(kf->F, n, n, kf->x, n, 1); // n x 1
    double *FP = mat_mul(kf->F, n, n, kf->P, n, n);    // n x n
    double *FPT = mat_mul(FP, n, n, mat_transpose(kf->F, n, n), n, n); // FP*F^T
    mat_free(FP);
    double *P_new = mat_add(FPT, kf->Q, n, n);
    mat_free(FPT);
    // swap
    mat_free(kf->x); mat_free(kf->P);
    kf->x = x_new;
    kf->P = P_new;
}

/* update with measurement z (m x 1 vector)
 * y = z - H * x
 * S = H * P * H^T + R
 * K = P * H^T * S^{-1}
 * x = x + K * y
 * P = (I - K * H) * P
 */
void kf_update(Kalman *kf, const double *z) {
    int n = kf->n, m = kf->m;
    // Hx
    double *Hx = mat_mul(kf->H, m, n, kf->x, n, 1); // m x 1
    double *y = mat_sub(z, Hx, m, 1); // m x 1
    mat_free(Hx);

    double *HP = mat_mul(kf->H, m, n, kf->P, n, n); // m x n
    double *HT = mat_transpose(kf->H, m, n); // n x m
    double *HPHT = mat_mul(HP, m, n, HT, n, m); // m x m
    mat_free(HP);
    mat_free(HT);

    double *S = mat_add(HPHT, kf->R, m, m);
    mat_free(HPHT);

    double *S_inv = mat_inv(S, m);
    if (!S_inv) { mat_free(S); mat_free(y); fprintf(stderr,"S inversion failed\n"); return; }
    // K = P * H^T * S_inv
    double *H_t = mat_transpose(kf->H, m, n); // n x m
    double *PHt = mat_mul(kf->P, n, n, H_t, n, m); // n x m
    double *K = mat_mul(PHt, n, m, S_inv, m, m); // n x m

    // x = x + K * y
    double *Ky = mat_mul(K, n, m, y, m, 1); // n x 1
    double *x_new = mat_add(kf->x, Ky, n, 1);

    // P = (I - K*H) * P
    double *KH = mat_mul(K, n, m, kf->H, m, n); // n x n
    double *I = mat_alloc(n,n); mat_set_identity(I, n);
    double *IminusKH = mat_sub(I, KH, n, n);
    double *P_new = mat_mul(IminusKH, n, n, kf->P, n, n);

    // cleanup and swap
    mat_free(kf->x); mat_free(kf->P);
    kf->x = x_new;
    kf->P = P_new;

    mat_free(K); mat_free(PHt); mat_free(H_t); mat_free(S); mat_free(S_inv);
    mat_free(Ky); mat_free(y); mat_free(KH); mat_free(I); mat_free(IminusKH);
}

/* Example: 1D constant-velocity model
 * State vector x = [position; velocity]  (n = 2)
 * Measurement z = [position]             (m = 1)
 * F = [1 dt; 0 1]
 * H = [1 0]
 */
int main(void) {
    double dt = 1.0; // time step
    int n = 2, m = 1;

    double x0[2] = {0.0, 1.0}; // start at 0, velocity 1
    double P0[4] = {1.0, 0.0,
                    0.0, 1.0};
    double F[4] = {1.0, dt,
                   0.0, 1.0};
    double Q[4] = {0.01, 0.0,
                   0.0, 0.01};
    double H[1*2] = {1.0, 0.0};
    double R[1] = {0.5}; // measurement noise var

    Kalman *kf = kf_create(n, m, x0, P0, F, Q, H, R);

    /* simulate measurements for 10 steps: true position increases by 1 each step
       but measurements have noise */
    double true_x = 0.0, true_v = 1.0;
    printf("step\tmeas\tpos_est\tvel_est\n");
    for (int step=0; step<15; step++) {
        // simulate true dynamics
        true_x += true_v * dt;
        // simulate noisy measurement (simple pseudo-random)
        double noise = ((rand() / (double)RAND_MAX) - 0.5) * sqrt(R[0]) * 2.0;
        double z[1] = { true_x + noise };

        // predict
        kf_predict(kf);
        // update with measurement
        kf_update(kf, z);

        printf("%2d\t% .3f\t% .6f\t% .6f\n",
               step, z[0], kf->x[0], kf->x[1]);
    }

    /* cleanup */
    kf_free(kf);
    return 0;
}
