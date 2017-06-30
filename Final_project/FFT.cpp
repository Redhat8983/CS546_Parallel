#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <cstdlib>
#include <time.h> 

/*
------------------------------------------------------------------------
FFT1D            c_fft1d(r,i,-1)
Inverse FFT1D    c_fft1d(r,i,+1)
------------------------------------------------------------------------
*/
/* ---------- FFT 1D
This computes an in-place complex-to-complex FFT
r is the real and imaginary arrays of n=2^m points.
isign = -1 gives forward transform
isign =  1 gives inverse transform
*/

//#define const int ARRAYSIZE = 10;
int NSIZE = 512;

typedef struct { float r; float i; } complex;
static complex ctmp;

#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}

void c_fft1d(complex *r, int      n, int      isign)
{
	int     m, i, i1, j, k, i2, l, l1, l2;
	float   c1, c2, z;
	complex t, u;

	if (isign == 0) return;

	/* Do the bit reversal */
	i2 = n >> 1;
	j = 0;
	for (i = 0; i<n - 1; i++) {
		if (i < j)
			C_SWAP(r[i], r[j]);
		k = i2;
		while (k <= j) {
			j -= k;
			k >>= 1;
		}
		j += k;
	}

	/* m = (int) log2((double)n); */
	for (i = n, m = 0; i>1; m++, i /= 2);

	/* Compute the FFT */
	c1 = -1.0;
	c2 = 0.0;
	l2 = 1;
	for (l = 0; l<m; l++) {
		l1 = l2;
		l2 <<= 1;
		u.r = 1.0;
		u.i = 0.0;
		for (j = 0; j<l1; j++) {
			for (i = j; i<n; i += l2) {
				i1 = i + l1;

				/* t = u * r[i1] */
				t.r = u.r * r[i1].r - u.i * r[i1].i;
				t.i = u.r * r[i1].i + u.i * r[i1].r;

				/* r[i1] = r[i] - t */
				r[i1].r = r[i].r - t.r;
				r[i1].i = r[i].i - t.i;

				/* r[i] = r[i] + t */
				r[i].r += t.r;
				r[i].i += t.i;
			}
			z = u.r * c1 - u.i * c2;

			u.i = u.r * c2 + u.i * c1;
			u.r = z;
		}
		c2 = sqrt((1.0 - c1) / 2.0);
		if (isign == -1) /* FWD FFT */
			c2 = -c2;
		c1 = sqrt((1.0 + c1) / 2.0);
	}

	/* Scaling for inverse transform */
	if (isign == 1) {       /* IFFT*/
		for (i = 0; i<n; i++) {
			r[i].r /= n;
			r[i].i /= n;
		}
	}
} // c_fft1d()

/////////////////////////////////////////////////////////

void ArrayCreater( complex *A )
{

	for (int i = 0; i < NSIZE; i++) {
		for (int j = 0; j < NSIZE; j++) {
			A[i*NSIZE + j].r = 1;
			A[i*NSIZE + j].i = 0;
		} // for j
		
	} // for i

} // ArrayCreater

//////////////////////////////////////////////////////////
void ArrayInverter( complex *A) {
	complex * TempA;
	TempA = (complex*)malloc(NSIZE*NSIZE*sizeof(complex));
	int row, col;

	// invert A to TempA
	for ( row = 0; row < NSIZE; row++)
		for ( col = 0; col < NSIZE; col++) 
			TempA[row + col*NSIZE] = A[row*NSIZE + col];
	// put TempA back to A
	for (row = 0; row < NSIZE; row++)
		for (col = 0; col < NSIZE; col++)
			A[row*NSIZE + col] = TempA[row*NSIZE + col];


} // ArrayInverter
//////////////////////////////////////////////////////////
/* MMuti -> mutipy A , B and put result in C
*/
void MMuti( complex*C, complex*A, complex *B )
{
	for (int row = 0; row < NSIZE; row++) {
		for (int col = 0; col < NSIZE; col++) {
			C[row*NSIZE + col].r = A[row*NSIZE + col].r * B[row*NSIZE + col].r;
			C[row*NSIZE + col].i = A[row*NSIZE + col].i * B[row*NSIZE + col].i;
		} // for j
	} // for

} // MMuti()

int main(int argc, char **argv) {


	complex *A, *B, *C, *D;

	A = (complex*)malloc(NSIZE*NSIZE*sizeof(complex));
	B = (complex*)malloc(NSIZE*NSIZE*sizeof(complex));
	C = (complex*)malloc(NSIZE*NSIZE*sizeof(complex));

	ArrayCreater(A);
	ArrayCreater(B);
	clock_t t1, t2;

	t1 = clock();

	int rowNumber;

	// A=2D-FFT(Im1) (task1)
	// row-FFT
	for (rowNumber = 0; rowNumber < NSIZE;rowNumber++)
	    c_fft1d(&A[rowNumber*NSIZE], NSIZE, -1);
	// col-FFT
	ArrayInverter(A);
	for (rowNumber = 0; rowNumber < NSIZE; rowNumber++)
		c_fft1d(&A[rowNumber*NSIZE], NSIZE, -1);
	ArrayInverter(A);
	// EOF  A=2D-FFT(Im1) (task1)

	// B = 2D - FFT(Im2) (task2)
	// row-FFT
	for (rowNumber = 0; rowNumber < NSIZE; rowNumber++)
		c_fft1d(&B[rowNumber*NSIZE], NSIZE, -1);
	// col-FFT
	ArrayInverter(B);
	for (rowNumber = 0; rowNumber < NSIZE; rowNumber++)
		c_fft1d(&B[rowNumber*NSIZE], NSIZE, -1);
	ArrayInverter(B);
	// EOF B = 2D - FFT(Im2) (task2)

	// C=MM_Point(A,B) (task3)
	MMuti(C, A, B);


	// D=Inverse-2DFFT(C ) (task4)
	for (rowNumber = 0; rowNumber < NSIZE; rowNumber++)
		c_fft1d(&C[rowNumber*NSIZE], NSIZE, +1);
	// col-FFT
	ArrayInverter(C);
	for (rowNumber = 0; rowNumber < NSIZE; rowNumber++)
		c_fft1d(&C[rowNumber*NSIZE], NSIZE, +1);
	ArrayInverter(C);
	// EOF D=Inverse-2DFFT(C ) (task4)

	t2 = clock();
	printf("%lf\n secs.", (t2 - t1) / (double)(CLOCKS_PER_SEC));

	if (NSIZE < 16) {
		for (int i = 0; i < NSIZE; i++) {
			for (int j = 0; j < NSIZE; j++) {
				printf("%f,  ", C[i*NSIZE + j].r);
			} // for j
			printf("\n");
		} // for i
	} // 
	
		

	//system("pause");
	return 0;
} //main
