/*********************************************************************************
 * Smith–Waterman algorithm
 * Purpose:     Local alignment of nucleotide or protein sequences
 * Authors:     Daniel Holanda, Hanoch Griner, Taynara Pinheiro
 *
 * Compilation: gcc omp_smithW.c -o omp_smithW -fopenmp -DDEBUG // debugging mode
 *              gcc omp_smithW.c -O3 -o omp_smithW -fopenmp // production run
 * Execution:	./omp_smithW <number_of_col> <number_of_rows>
 *
 * Updated by C. Liao, Jan 2nd, 2019
 *********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h> // C99 does not support the boolean data type

#include "parameters.h"

/*--------------------------------------------------------------------
 * Text Tweaks
 */
#define RESET   "\033[0m"
#define BOLDRED "\033[1m\033[31m"      /* Bold Red */
/* End of text tweaks */

/*--------------------------------------------------------------------
 * Constants
 */
#define PATH -1
#define NONE 0
#define UP 1
#define LEFT 2
#define DIAGONAL 3
/* End of constants */

/*--------------------------------------------------------------------
* Helpers
*/
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define max(a,b) ((a) > (b) ? a : b)

// #define DEBUG
/* End of Helpers */


/*--------------------------------------------------------------------
 * Functions Prototypes
 */
void similarityScore(long long int i, long long int j, int* H, int* P, long long int* maxPos);
int matchMissmatchScore(long long int i, long long int j);
void backtrack(int* P, long long int maxPos);
void printMatrix(int* matrix);
void printPredecessorMatrix(int* matrix);
void generate(void);
long long int nElement(long long int i);
void calcFirstDiagElement(long long int i, long long int *si, long long int *sj);

/* End of prototypes */


/*--------------------------------------------------------------------
 * Global Variables
 */
bool useBuiltInData=true;

//Defines size of strings to be compared
long long int m = 8 ; //Columns - Size of string a
long long int n = 9;  //Lines - Size of string b
// the generated scoring matrix's size is m++ and n++ later to have the first row/column as 0s.

//Defines scores
int matchScore = 3;
int missmatchScore = -3;
int gapScore = -2;

//Strings over the Alphabet Sigma
char *a, *b;

/* End of global variables */

/*--------------------------------------------------------------------
 * Function:    main
 */
int main(int argc, char* argv[]) {
  // thread_count is no longer used
  int thread_count;

  if (argc==3)
  {
    m = strtoll(argv[1], NULL, 10);
    n = strtoll(argv[2], NULL, 10);
    useBuiltInData = false;
  }

//#ifdef DEBUG
  if (useBuiltInData)
    printf ("Using built-in data for testing ..\n");
  printf("Problem size: Matrix[%lld][%lld], FACTOR=%d CUTOFF=%d\n", n, m, FACTOR, CUTOFF);
//#endif

    //Allocates a and b
    a = (char*) malloc(m * sizeof(char));
    b = (char*) malloc(n * sizeof(char));

    //Because now we have zeros
    m++;
    n++;

    //Allocates similarity matrix H
    int *H;
    H = (int *) calloc(m * n, sizeof(int));

    //Allocates predecessor matrix P
    int *P;
    P = (int *)calloc(m * n, sizeof(int));


    if (useBuiltInData)
    {
      //Uncomment this to test the sequence available at 
      //http://vlab.amrita.edu/?sub=3&brch=274&sim=1433&cnt=1
      // OBS: m=11 n=7
      // a[0] =   'C';
      // a[1] =   'G';
      // a[2] =   'T';
      // a[3] =   'G';
      // a[4] =   'A';
      // a[5] =   'A';
      // a[6] =   'T';
      // a[7] =   'T';
      // a[8] =   'C';
      // a[9] =   'A';
      // a[10] =  'T';

      // b[0] =   'G';
      // b[1] =   'A';
      // b[2] =   'C';
      // b[3] =   'T';
      // b[4] =   'T';
      // b[5] =   'A';
      // b[6] =   'C';
      // https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm#Example
      // Using the wiki example to verify the results
      b[0] =   'G';
      b[1] =   'G';
      b[2] =   'T';
      b[3] =   'T';
      b[4] =   'G';
      b[5] =   'A';
      b[6] =   'C';
      b[7] =   'T';
      b[8] =   'A';

      a[0] =   'T';
      a[1] =   'G';
      a[2] =   'T';
      a[3] =   'T';
      a[4] =   'A';
      a[5] =   'C';
      a[6] =   'G';
      a[7] =   'G';
    }
    else
    {
      //Gen random arrays a and b
      generate();
    }

    //Start position for backtrack
    long long int maxPos = 0;
    //Calculates the similarity matrix
    long long int i, j;


    // The way to generate all wavefront is to go through the top edge elements
    // starting from the left top of the matrix, go to the bottom top -> down, then left->right
    // total top edge element count =  dim1_size + dim2_size -1 
    //Because now we have zeros ((m-1) + (n-1) - 1)
    long long int nDiag = m + n - 3;

#ifdef DEBUG
    printf("nDiag=%d\n", nDiag);
    printf("Number of wavefront lines and their first element positions:\n");
#endif

#pragma omp parallel 
    {
#pragma omp master	    
      {
        thread_count = omp_get_num_threads();
        printf ("if-clause() Using %d out of max %d threads...", thread_count, omp_get_max_threads());
      }
    }

    //Gets Initial time
    double initialTime = omp_get_wtime();

//  #pragma omp parallel default(none) shared(H, P, maxPos, nDiag, j) private(i)
    {
      for (i = 1; i <= nDiag; ++i) // start from 1 since 0 is the boundary padding
      {
        long long int nEle, si, sj;
        nEle = nElement(i);
        calcFirstDiagElement(i, &si, &sj);

 #pragma omp parallel for private(j) shared (nEle, si, sj, H, P, maxPos) if (nEle>=CUTOFF)
          for (j = 0; j < nEle; ++j) 
          {  // going upwards : anti-diagnol direction
            long long int ai = si - j ; // going up vertically
            long long int aj = sj + j;  //  going right in horizontal
            similarityScore(ai, aj, H, P, &maxPos); // a critical section is used inside
          }
     } // for end nDiag
    } // end omp parallel

  double finalTime = omp_get_wtime();
  printf("\nElapsed time for scoring matrix computation: %f\n", finalTime - initialTime);

  initialTime = omp_get_wtime();
  backtrack(P, maxPos);
  finalTime = omp_get_wtime();

  //Gets backtrack time
  finalTime = omp_get_wtime();
  printf("Elapsed time for backtracking: %f\n", finalTime - initialTime);

    if (useBuiltInData)
    {
      printf ("Verifying results using the builtinIn data: %s\n", (H[n*m-1]==7)?"true":"false");
      assert (H[n*m-1]==7);
    }

#ifdef DEBUG
    printf("\nSimilarity Matrix:\n");
    printMatrix(H);

    printf("\nPredecessor Matrix:\n");
    printPredecessorMatrix(P);
#endif

    //Frees similarity matrixes
    free(H);
    free(P);

    //Frees input arrays
    free(a);
    free(b);

    return 0;
}  /* End of main */

/*--------------------------------------------------------------------
 * Function:    nElement
 * Purpose:     Calculate the number of i-diagonal's elements
 * i value range 1 to nDiag.  we inclulde the upper bound value. 0 is for the padded wavefront, which is ignored.
 */
long long int nElement(long long int i) {
    if (i < m && i < n) { // smaller than both directions
        //Number of elements in the diagonal is increasing
        return i;
    }
    else if (i < max(m, n)) { // smaller than only one direction
        //Number of elements in the diagonal is stable
        long int min = min(m, n);  // the longer direction has the edge elements, the number is the smaller direction's size
        return min - 1;
    }
    else {
        //Number of elements in the diagonal is decreasing
        long int min = min(m, n);
        return 2 * min - i + llabs(m - n) - 2;
    }
}

/*--------------------------------------------------------------------
 * Function:    calcElement: expect valid i value is from 1 to nDiag. since the first one is 0 padding
 * Purpose:     Calculate the position of (si, sj)-element
 * n rows, m columns: we sweep the matrix on the left edge then bottom edge to get the wavefront
 */
void calcFirstDiagElement(long long int i, long long int *si, long long int *sj) {
    // Calculate the first element of diagonal
    if (i < n) { // smaller than row count
        *si = i;
        *sj = 1; // start from the j==1 since j==0 is the padding
    } else {  // now we sweep horizontally at the bottom of the matrix
        *si = n - 1;  // i is fixed
        *sj = i - n + 2; // j position is the nDiag (id -n) +1 +1 // first +1 
    }
}

/*
 // understanding the calculation by an example
 n =6 // row
 m =2  // col

 padded scoring matrix
 n=7
 m=3

   0 1 2
 -------
 0 x x x
 1 x x x
 2 x x x
 3 x x x
 4 x x x
 5 x x x
 6 x x x

 We should peel off top row and left column since they are the padding
 the remaining 6x2 sub matrix is what is interesting for us
 Now find the number of wavefront lines and their first element's position in the scoring matrix

total diagnol frontwave = (n-1) + (m-1) -1 // submatrix row+column -1
We use the left most element in each wavefront line as its first element.
Then we have the first elements like
(1,1),
(2,1)
(3,1)
..
(6,1) (6,2)
 
 */
/*--------------------------------------------------------------------
 * Function:    SimilarityScore
 * Purpose:     Calculate  value of scoring matrix element H(i,j) : the maximum Similarity-Score H(i,j)
 *             int *P; the predecessor array,storing which of the three elements is picked with max value
 */
void similarityScore(long long int i, long long int j, int* H, int* P, long long int* maxPos) {

    int up, left, diag;

    //Stores index of element
    long long int index = m * i + j;

    //Get element above
    up = H[index - m] + gapScore;

    //Get element on the left
    left = H[index - 1] + gapScore;

    //Get element on the diagonal
    diag = H[index - m - 1] + matchMissmatchScore(i, j);

    //Calculates the maximum
    int max = NONE;
    int pred = NONE;
    /* === Matrix ===
     *      a[0] ... a[n]
     * b[0]
     * ...
     * b[n]
     *
     * generate 'a' from 'b', if '←' insert e '↑' remove
     * a=GAATTCA
     * b=GACTT-A
     *
     * generate 'b' from 'a', if '←' insert e '↑' remove
     * b=GACTT-A
     * a=GAATTCA
    */

    if (diag > max) { //same letter ↖
        max = diag;
        pred = DIAGONAL;
    }

    if (up > max) { //remove letter ↑
        max = up;
        pred = UP;
    }

    if (left > max) { //insert letter ←
        max = left;
        pred = LEFT;
    }
    //Inserts the value in the similarity and predecessor matrixes
    H[index] = max;
    P[index] = pred;

    //Updates maximum score to be used as seed on backtrack
    #pragma omp critical
    if (max > H[*maxPos]) {
        *maxPos = index;
    }
}  /* End of similarityScore */

/*--------------------------------------------------------------------
 * Function:    matchMissmatchScore
 * Purpose:     Similarity function on the alphabet for match/missmatch
 */
int matchMissmatchScore(long long int i, long long int j) {
    if (a[j - 1] == b[i - 1])
        return matchScore;
    else
        return missmatchScore;
}  /* End of matchMissmatchScore */

/*--------------------------------------------------------------------
 * Function:    backtrack
 * Purpose:     Modify matrix to print, path change from value to PATH
 */
void backtrack(int* P, long long int maxPos) {
    //hold maxPos value
    long long int predPos;

    //backtrack from maxPos to startPos = 0
    do {
        if (P[maxPos] == DIAGONAL)
            predPos = maxPos - m - 1;
        else if (P[maxPos] == UP)
            predPos = maxPos - m;
        else if (P[maxPos] == LEFT)
            predPos = maxPos - 1;
        P[maxPos] *= PATH;
        maxPos = predPos;
    } while (P[maxPos] != NONE);
}  /* End of backtrack */

/*--------------------------------------------------------------------
 * Function:    printMatrix
 * Purpose:     Print Matrix
 */
void printMatrix(int* matrix) {
    long long int i, j;
    printf("-\t-\t");
    for (j = 0; j < m-1; j++) {
    	printf("%c\t", a[j]);
    }
    printf("\n-\t");
    for (i = 0; i < n; i++) { //Lines
        for (j = 0; j < m; j++) {  
        	if (j==0 && i>0) printf("%c\t", b[i-1]);
            printf("%d\t", matrix[m * i + j]);
        }
        printf("\n");
    }

}  /* End of printMatrix */

/*--------------------------------------------------------------------
 * Function:    printPredecessorMatrix
 * Purpose:     Print predecessor matrix
 */
void printPredecessorMatrix(int* matrix) {
    long long int i, j, index;
    printf("    ");
    for (j = 0; j < m-1; j++) {
    	printf("%c ", a[j]);
    }
    printf("\n  ");
    for (i = 0; i < n; i++) { //Lines
        for (j = 0; j < m; j++) {
        	if (j==0 && i>0) printf("%c ", b[i-1]);
            index = m * i + j;
            if (matrix[index] < 0) {
                printf(BOLDRED);
                if (matrix[index] == -UP)
                    printf("↑ ");
                else if (matrix[index] == -LEFT)
                    printf("← ");
                else if (matrix[index] == -DIAGONAL)
                    printf("↖ ");
                else
                    printf("- ");
                printf(RESET);
            } else {
                if (matrix[index] == UP)
                    printf("↑ ");
                else if (matrix[index] == LEFT)
                    printf("← ");
                else if (matrix[index] == DIAGONAL)
                    printf("↖ ");
                else
                    printf("- ");
            }
        }
        printf("\n");
    }

}  /* End of printPredecessorMatrix */

/*--------------------------------------------------------------------
 * Function:    generate
 * Purpose:     Generate arrays a and b
 */
void generate() {
    //Random seed
    srand(time(NULL));

    //Generates the values of a
    long long int i;
    for (i = 0; i < m; i++) {
        int aux = rand() % 4;
        if (aux == 0)
            a[i] = 'A';
        else if (aux == 2)
            a[i] = 'C';
        else if (aux == 3)
            a[i] = 'G';
        else
            a[i] = 'T';
    }

    //Generates the values of b
    for (i = 0; i < n; i++) {
        int aux = rand() % 4;
        if (aux == 0)
            b[i] = 'A';
        else if (aux == 2)
            b[i] = 'C';
        else if (aux == 3)
            b[i] = 'G';
        else
            b[i] = 'T';
    }
} /* End of generate */


/*--------------------------------------------------------------------
 * External References:
 * http://vlab.amrita.edu/?sub=3&brch=274&sim=1433&cnt=1
 * http://pt.slideshare.net/avrilcoghlan/the-smith-waterman-algorithm
 * http://baba.sourceforge.net/
 */
