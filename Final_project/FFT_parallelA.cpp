#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <cstdlib>
#include <windows.h>
#include <stdio.h>
#include <time.h> 
#include <iostream>

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
int RANGE;

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

void ArrayCreater(complex *A)
{

	for (int i = 0; i < NSIZE; i++) {
		for (int j = 0; j < NSIZE; j++) {
			A[i*NSIZE + j].r = 1;
			A[i*NSIZE + j].i = 0;
		} // for j

	} // for i

} // ArrayCreater

  //////////////////////////////////////////////////////////
void ArrayInverter(complex *A) {
	complex * TempA;
	TempA = (complex*)malloc(NSIZE*NSIZE*sizeof(complex));
	int row, col;

	// invert A to TempA
	for (row = 0; row < NSIZE; row++)
		for (col = 0; col < NSIZE; col++)
			TempA[row + col*NSIZE] = A[row*NSIZE + col];
	// put TempA back to A
	for (row = 0; row < NSIZE; row++)
		for (col = 0; col < NSIZE; col++)
			A[row*NSIZE + col] = TempA[row*NSIZE + col];


} // ArrayInverter
  //////////////////////////////////////////////////////////
  /* MMuti -> mutipy A , B and put result in C
  */
void MMuti(complex*C, complex*A, complex *B)
{
	for (int row = 0; row < RANGE; row++) {
		for (int col = 0; col < NSIZE; col++) {
			C[row*NSIZE + col].r = A[row*NSIZE + col].r * B[row*NSIZE + col].r;
			C[row*NSIZE + col].i = A[row*NSIZE + col].i * B[row*NSIZE + col].i;
		} // for j
	} // for

} // MMuti()

///////////////////////////////////////////////////////////
/* Task1
   FFT_2D A. for row Do 0->N
             for col Invert() Do0->N Invert()
   Task1 is finish.
   **perameter**( *array )
*/
void FFT2D_InputA( complex *A, int nprocs, int myrank) 
{
	int rowNumber;
	int proc;
	int local_row;
	MPI_Status status;

	// MPI_partition
	if ( myrank == 0 )
	{
		for (proc = 1; proc < nprocs; proc++) {
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Send(&A[proc*RANGE*NSIZE + local_row*NSIZE], NSIZE, MPI_COMPLEX,proc, 0, MPI_COMM_WORLD);
			} // for send col			
		} // for
	} // myrank ==0
	else {
		// Recv
		for (local_row = 0; local_row < RANGE; local_row++) {
			MPI_Recv(&A[local_row*NSIZE], NSIZE, MPI_COMPLEX,0, 0, MPI_COMM_WORLD, &status);
		} // for recv col
	} // else 
	// A=2D-FFT(Im1) (task1)
	// row-FFT 1st time c_fft1d
	for (rowNumber = 0; rowNumber < RANGE; rowNumber++)
		c_fft1d(&A[rowNumber*NSIZE], NSIZE, -1);
	// Recv after 1st time c_fft1d;
	if (myrank > 0) {
		for (local_row = 0; local_row <RANGE; local_row++) {
			MPI_Send(&A[local_row*NSIZE], NSIZE, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD);
		} // for send col				
	} // if myrnk>0
	else {
		for (proc = 1; proc < nprocs; proc++)
		{
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Recv(&A[proc*RANGE*NSIZE + local_row*NSIZE], NSIZE, MPI_COMPLEX, proc, 0, MPI_COMM_WORLD, &status);
			} // for; recv for each row 
		} // for recv from rank 1 to n
	} // else
	// col-FFT
	ArrayInverter(A);
	if (myrank == 0)
	{
		for (proc = 1; proc < nprocs; proc++) {
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Send(&A[proc*RANGE*NSIZE + local_row*NSIZE], NSIZE, MPI_COMPLEX, proc, 0, MPI_COMM_WORLD);
			} // for send col			
		} // for
	} // myrank ==0
	else {
		// Recv
		for (local_row = 0; local_row < RANGE; local_row++) {
			MPI_Recv(&A[local_row*NSIZE], NSIZE, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &status);
		} // for recv col
	} // else 
	
	for (rowNumber = 0; rowNumber < RANGE; rowNumber++)
		c_fft1d(&A[rowNumber*NSIZE], NSIZE, -1);

	if (myrank > 0) {
		for (local_row = 0; local_row < RANGE; local_row++) {
			MPI_Send(&A[local_row*NSIZE], NSIZE, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD);
		} // for send col				
	} // if myrnk>0
	else {
		for (proc = 1; proc < nprocs; proc++)
		{
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Recv(&A[proc*RANGE*NSIZE + local_row*NSIZE], NSIZE, MPI_COMPLEX, proc, 0, MPI_COMM_WORLD, &status);
			} // for; recv for each row 
		} // for recv from rank 1 to n
	} // else
	ArrayInverter(A);
	// EOF  A=2D-FFT(Im1) (task1)
	
} // FFT2D_InputA()
//////////////////////////////////////////////////////////
/*Same as function FFT_InputA */
void FFT2D_InputB(complex *A, int nprocs, int myrank)
{
	int rowNumber;
	int proc;
	int local_row;
	MPI_Status status;

	// MPI_partition
	if (myrank == 0)
	{
		for (proc = 1; proc < nprocs; proc++) {
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Send(&A[proc*RANGE*NSIZE + local_row*NSIZE], NSIZE, MPI_COMPLEX, proc, 0, MPI_COMM_WORLD);
			} // for send col			
		} // for
	} // myrank ==0
	else {
		// Recv
		for (local_row = 0; local_row < RANGE; local_row++) {
			MPI_Recv(&A[local_row*NSIZE], NSIZE, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &status);
		} // for recv col
	} // else 
	  // B=2D-FFT(Im1) (task2)
	  // row-FFT 1st time c_fft1d
	for (rowNumber = 0; rowNumber < RANGE; rowNumber++)
		c_fft1d(&A[rowNumber*NSIZE], NSIZE, -1);
	// Recv after 1st time c_fft1d;
	if (myrank > 0) {
		for (local_row = 0; local_row <RANGE; local_row++) {
			MPI_Send(&A[local_row*NSIZE], NSIZE, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD);
		} // for send col				
	} // if myrnk>0
	else {
		for (proc = 1; proc < nprocs; proc++)
		{
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Recv(&A[proc*RANGE*NSIZE + local_row*NSIZE], NSIZE, MPI_COMPLEX, proc, 0, MPI_COMM_WORLD, &status);
			} // for; recv for each row 
		} // for recv from rank 1 to n
	} // else
	  // col-FFT
	ArrayInverter(A);
	if (myrank == 0)
	{
		for (proc = 1; proc < nprocs; proc++) {
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Send(&A[proc*RANGE*NSIZE + local_row*NSIZE], NSIZE, MPI_COMPLEX, proc, 0, MPI_COMM_WORLD);
			} // for send col			
		} // for
	} // myrank ==0
	else {
		// Recv
		for (local_row = 0; local_row < RANGE; local_row++) {
			MPI_Recv(&A[local_row*NSIZE], NSIZE, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &status);
		} // for recv col
	} // else 

	for (rowNumber = 0; rowNumber < RANGE; rowNumber++)
		c_fft1d(&A[rowNumber*NSIZE], NSIZE, -1);

	if (myrank > 0) {
		for (local_row = 0; local_row < RANGE; local_row++) {
			MPI_Send(&A[local_row*NSIZE], NSIZE, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD);
		} // for send col				
	} // if myrnk>0
	else {
		for (proc = 1; proc < nprocs; proc++)
		{
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Recv(&A[proc*RANGE*NSIZE + local_row*NSIZE], NSIZE, MPI_COMPLEX, proc, 0, MPI_COMM_WORLD, &status);
			} // for; recv for each row 
		} // for recv from rank 1 to n
	} // else
	ArrayInverter(A);
	// EOF  B=2D-FFT(Im1) (task2)
} // FFT2D_InputB()
///////////////////////////////////////////////////////
void FFT2D_Inverse( complex *C, int nprocs, int myrank) {
	int rowNumber;
	int proc;
	int local_row;
	MPI_Status status;

	// MPI_partition
	if (myrank == 0)
	{
		for (proc = 1; proc < nprocs; proc++) {
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Send(&C[proc*NSIZE*RANGE + local_row*NSIZE], NSIZE, MPI_COMPLEX,proc, 0, MPI_COMM_WORLD);
			} // for send col			
		} // for
	} // myrank ==0
	else {
		// Recv
		for (local_row = 0; local_row < RANGE; local_row++) {
			MPI_Recv(&C[local_row*NSIZE], NSIZE, MPI_COMPLEX,0, 0, MPI_COMM_WORLD, &status);
		} // for recv col
	} // else 
	// D=Inverse-2DFFT(C ) (task4)
	for (rowNumber = 0; rowNumber < NSIZE; rowNumber++)
		c_fft1d(&C[rowNumber*NSIZE], NSIZE, +1);

	if (myrank > 0) {
		for (local_row = 0; local_row < RANGE; local_row++) {
			MPI_Send(&C[local_row*NSIZE], NSIZE, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD);
		} // for send col				
	} // if myrnk>0
	else {
		for (proc = 1; proc < nprocs; proc++)
		{
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Recv(&C[proc*RANGE*NSIZE + local_row*NSIZE], NSIZE, MPI_COMPLEX, proc, 0, MPI_COMM_WORLD, &status);
			} // for; recv for each row 
		} // for recv from rank 1 to n
	} // else

	// col-FFT
	ArrayInverter(C);

	if (myrank == 0)
	{
		for (proc = 1; proc < nprocs; proc++) {
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Send(&C[proc*NSIZE*RANGE + local_row*NSIZE], NSIZE, MPI_COMPLEX, proc, 0, MPI_COMM_WORLD);
			} // for send col			
		} // for
	} // myrank ==0
	else {
		// Recv
		for (local_row = 0; local_row < RANGE; local_row++) {
			MPI_Recv(&C[local_row*NSIZE], NSIZE, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &status);
		} // for recv col
	} // else 

	for (rowNumber = 0; rowNumber < NSIZE; rowNumber++)
		c_fft1d(&C[rowNumber*NSIZE], NSIZE, +1);

	if (myrank > 0) {
		for (local_row = 0; local_row < RANGE; local_row++) {
			MPI_Send(&C[local_row*NSIZE], NSIZE, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD);
		} // for send col				
	} // if myrnk>0
	else {
		for (proc = 1; proc < nprocs; proc++)
		{
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Recv(&C[proc*RANGE*NSIZE + local_row*NSIZE], NSIZE, MPI_COMPLEX, proc, 0, MPI_COMM_WORLD, &status);
			} // for; recv for each row 
		} // for recv from rank 1 to n
	} // else
	ArrayInverter(C);
	// EOF D=Inverse-2DFFT(C ) (task4)
} // void Inverse_2DFFT
/////////////////////////////////////////////////////////////
void MM_Point(complex *C, complex *A, complex *B, int nprocs, int myrank)
{
	int rowNumber;
	int proc;
	int local_row;
	MPI_Status status;

	// MPI_partition
	if (myrank == 0)
	{
		for (proc = 1; proc < nprocs; proc++) {
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Send(&A[proc*NSIZE*RANGE + local_row*NSIZE], NSIZE, MPI_COMPLEX,
					proc, 0, MPI_COMM_WORLD);
				MPI_Send(&B[proc*NSIZE*RANGE + local_row*NSIZE], NSIZE, MPI_COMPLEX,
					proc, 1, MPI_COMM_WORLD);
			} // for send col			
		} // for
	} // myrank ==0
	else {
		// Recv
		for (local_row = 0; local_row < RANGE; local_row++) {
			MPI_Recv(&A[local_row*NSIZE], NSIZE, MPI_COMPLEX,
				0, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(&B[local_row*NSIZE], NSIZE, MPI_COMPLEX,
				0, 1, MPI_COMM_WORLD, &status);
		} // for recv col
	} // else 

	// C=MM_Point(A,B) (task3)
	MMuti(C, A, B);
	// EOF(task3) 
	if (myrank > 0) {
		for (local_row = 0; local_row < RANGE; local_row++) {
			MPI_Send(&C[local_row*NSIZE], NSIZE, MPI_COMPLEX,
				0, 2, MPI_COMM_WORLD);
		} // for send col				
	} // if myrnk>0
	else {
		for (proc = 1; proc < nprocs; proc++)
		{
			for (local_row = 0; local_row < RANGE; local_row++) {
				MPI_Recv(&C[proc*NSIZE*RANGE + local_row*NSIZE], NSIZE, MPI_COMPLEX, proc, 2, MPI_COMM_WORLD, &status);
			} // for; recv for each row 
		} // for recv from rank 1 to n
	} // else, myrank ==0

} // MM_Point


int main(int argc, char **argv) {

	clock_t t1, t2;

	complex *A, *B, *C ;

	A = (complex*)malloc(NSIZE*NSIZE*sizeof(complex));
	B = (complex*)malloc(NSIZE*NSIZE*sizeof(complex));
	C = (complex*)malloc(NSIZE*NSIZE*sizeof(complex));

	// MPI initialize
	int nprocs, myrank;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size( MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	RANGE = NSIZE / nprocs; // get partition

	if (myrank == 0) {
		ArrayCreater(A);
		ArrayCreater(B);
	} // if read input

	t1 = clock();

	FFT2D_InputA(A, nprocs,myrank);

	FFT2D_InputB(B, nprocs,myrank);

	MM_Point(C, A, B, nprocs, myrank);

	FFT2D_Inverse(C, nprocs, myrank);

	t2 = clock();

	if ( myrank == 0 )
	   printf("%lf\n secs.", (t2 - t1) / (double)(CLOCKS_PER_SEC));

	if (myrank == 0 && NSIZE < 16) {
		printf("Ans:\n");
		for (int i = 0; i < NSIZE; i++) {
			for (int j = 0; j < NSIZE; j++) {
				printf("%f,  ", C[i*NSIZE + j].r);
			} // for j
			printf("\n");
		} // for i
	} // if myrank=0

	MPI_Finalize();
	//system("pause");
	return 0;
} //main

