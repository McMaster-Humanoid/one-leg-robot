

Kalman *kf_create(int n, int m,
                  const double *x0, const double *P0,
                  const double *F, const double *Q,
                  const double *H, const double *R)

void kf_free(Kalman *kf);

void kf_predict(Kalman *kf);

void kf_update(Kalman *kf, const double *z);

