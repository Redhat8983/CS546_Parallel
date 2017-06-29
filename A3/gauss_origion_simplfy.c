/* Gaussian elimination without pivoting.
* Compile with "gcc gauss.c"
*/

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
* You need not submit the provided code.
*/

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
int N = 9;  /* Matrix size */

			/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */


/* Prototype */
void gauss();  /* The function you will provide.
			   * It is this routine that is timed.
			   * It is called only on the parent.
			   */
			   /* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
	int seed = 0;  /* Random seed */
	char uid[32]; /*User name */
	argc = 3;

				  /* Read command-line arguments */
	srand(0);  /* Randomize */

	if (argc == 3) {
		srand(seed);
		printf("Random seed = %i\n", seed);
	}
	if (argc >= 2) {
		if (N < 1 || N > MAXN) {
			printf("N = %i is out of range.\n", N);
			exit(0);
		}
	}
	else {
		printf("Usage: %s <matrix_dimension> [random seed]\n",
			argv[0]);
		exit(0);
	}

	/* Print parameters */
	printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
	int row, col;

	printf("\nInitializing...\n");
	for (col = 0; col < N; col++) {
		for (row = 0; row < N; row++) {
			A[row][col] = (float)rand() / 32768.0;
		}
		B[col] = (float)rand() / 32768.0;
		X[col] = 0.0;
	}

}

/* Print input matrices */
void print_inputs() {
	int row, col;

	if (N < 10) {
		printf("\nA =\n\t");
		for (row = 0; row < N; row++) {
			for (col = 0; col < N; col++) {
				printf("%5.2f%s", A[row][col], (col < N - 1) ? ", " : ";\n\t");
			}
		}
		printf("\nB = [");
		for (col = 0; col < N; col++) {
			printf("%5.2f%s", B[col], (col < N - 1) ? "; " : "]\n");
		}
	}
}

void print_X() {
	int row;

	if (N < 100) {
		printf("\nX = [");
		for (row = 0; row < N; row++) {
			printf("%5.2f%s", X[row], (row < N - 1) ? "; " : "]\n");
		}
	}
}

int main(int argc, char **argv) {

	/* Process program parameters */
	parameters(argc, argv);

	/* Initialize A and B */
	initialize_inputs();

	/* Print input matrices */
	print_inputs();

	/* Gaussian Elimination */
	gauss();


	/* Display output */
	print_X();


	printf("--------------------------------------------\n");

	system("pause");
	exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
* defined in the beginning of this code.  X[] is initialized to zeros.
*/
void gauss() {
	int norm, row, col;  /* Normalization row, and zeroing
						 * element row and col */
	float multiplier;

	printf("Computing Serially.\n");

	/* Gaussian elimination */
	for (norm = 0; norm < N - 1; norm++) {
		for (row = norm + 1; row < N; row++) {
			multiplier = A[row][norm] / A[norm][norm];
			for (col = norm; col < N; col++) {
				A[row][col] -= A[norm][col] * multiplier;
			}
			B[row] -= B[norm] * multiplier;
		}
	}
	/* (Diagonal elements are not normalized to 1.  This is treated in back
	* substitution.)
	*/


	/* Back substitution */
	for (row = N - 1; row >= 0; row--) {
		X[row] = B[row];
		for (col = N - 1; col > row; col--) {
			X[row] -= A[row][col] * X[col];
		}
		X[row] /= A[row][row];
	}
}
