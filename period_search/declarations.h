void trifac(int nrows, int **ifp);
void areanorm(double t[], double f[], int ndir, int nfac, int **ifp, 
              double at[], double af[]);
void sphfunc(int ndir, double at[], double af[]);
void ellfit(double r[], double a, double b, double c, 
            int ndir, int ncoef, double at[], double af[]);
void lubksb(double **a, int n, int indx[], double b[]);
void ludcmp(double **a, int n, int indx[], double d[]);
int mrqmin(double **x1, double **x2, double x3[], double y[], 
            double sig[], double a[], int ia[], int ma, 
	    double **covar, double **alpha);
double mrqcof(double **x1, double **x2, double x3[], double y[], 
              double sig[], double a[], int ia[], int ma, 
	      double **alpha, double beta[]);
void curv(double cg[]);
void blmatrix(double bet, double lam);
double conv(int nc, double dres[], int ma);
void gauss_1(double **aa, int n, double b[]);
void covsrt(double **covar, int ma, int ia[], int mfit);
void phasec(double dcdp[], double alpha, double p[]);
void matrix(double omg, double t, double tmat[][4], double dtm[][4][4]);
double bright(double ee[], double ee0[], double t, double cg[], 
            double dyda[], int ncoef);
void shell(int n, double a[], int index[]);

double *vector_double(int length);
int *vector_int(int length);
double **matrix_double(int rows, int columns);
int **matrix_int(int rows, int columns);
double ***matrix_3_double(int n_1, int n_2, int n_3);
void deallocate_vector(void *p_x);
void deallocate_matrix(void *p_x, int rows);
void deallocate_matrix_3(void ***p_x, int n_1, int n_2);

double dot_product(double a[], double b[]);
double hapke(double mi0, double mi, double alfa, double sc_param[]);

void sph2cart(double *vektor);
void rotation(double vector[], char *axis, double angle, char *direction);
void inverze(double **a, int n);
void cross_product(double a[], double b[], double c[]);
double norm(double a[]);
double ran1(long *idum);
double gasdev(long *idum);

double raytracing(double sl[], double poz[], int n_fac,
                  double **d, double **e, double **f, double **o, 
		  double **normal, double ds[],
	          int n_over_horiz[], int **fac_list, char *sc_law,
		  double sc_param[]);
void precomp(int n_fac, double *x, double *y, double *z, int **fac,
             double **d, double **e, double **f, double **o, 
	     double **normal, double *ds,
	     int *n_over_horiz, int **fac_list);

void matrix_YORP(double omg, double yorp, double t, double tmat[][4], double dtm[][4][4]);
double bright_YORP(double ee[], double ee0[], double t, double cg[], 
            double dyda[], int ncoef);

int mrqmin_ell(double **x1, double **x2, double x3[], double y[], 
            double sig[], double a[], int ia[], int ma, 
	    double **covar, double **alpha, double (*funcs)());
double mrqcof_ell(double **x1, double **x2, double x3[], double y[], 
              double sig[], double a[], int ia[], int ma, 
	      double **alpha, double beta[], double (*funcs)());
int gauss_errc(double **aa, int n, double b[]);
void matrix_ell(double omg, double fi0, double t, double tmat[][4], double dtm[][4][4]);
double bright_ell(double ee[], double ee0[], double t, double cg[], 
            double dyda[], int ncoef);

double bright_ell_YORP(double ee[], double ee0[], double t, double cg[], 
            double dyda[], int ncoef);
void matrix_ell_YORP(double omg, double fi0, double yorp, double t, double tmat[][4], double dtm[][4][4]);
