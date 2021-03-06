/***********************************************************************
 * Smith–Waterman algorithm
 * Purpose:     Local alignment of nucleotide or protein sequences
 * Authors:     Daniel Holanda, Hanoch Griner, Taynara Pinheiro
 ***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <assert.h>

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

// #define DEBUG

/* End of constants */


/*--------------------------------------------------------------------
 * Functions Prototypes
 */
void similarityScore(long long int i, long long int j, int* H, int* P, long long int* maxPos);
int matchMissmatchScore(long long int i, long long int j);
void backtrack(int* P, long long int maxPos);
void printMatrix(int* matrix);
void printPredecessorMatrix(int* matrix);

// Generate random sequences for a and b
void generate(void);
/* End of prototypes */


/*--------------------------------------------------------------------
 * Global Variables
 */
bool useBuiltInData=true; 

//Defines size of strings to be compared
long long int n = 9;  //Rows- Size of string b
long long int m = 8; //Columns - Size of string a

//Defines scores
// Follow https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm#Example
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
  if (argc==3)
  {
    m = strtoll(argv[1], NULL, 10);
    n = strtoll(argv[2], NULL, 10);
    useBuiltInData = false;
  }
  // if no arguments, use builtin data and size 9 x 8 

#ifdef DEBUG
  if (useBuiltInData)
    printf ("\n Using built-in data for testing ..");
  printf("\nMatrix[%lld][%lld]\n", n, m);
#endif

  //Allocates a and b
  a = (char*) malloc(m * sizeof(char));
  b = (char*) malloc(n * sizeof(char));

  //Because now we have zeros
  m++;
  n++;

  //Allocates similarity matrix H: scoring matrix
  // this matrix is linearized. Need to convert i, j to linearized offset when using it
  int *H;
  H = (int *) calloc(m * n, sizeof(int));

  //Allocates predecessor matrix P
  // This is used to keep track of the max elements found along the path
  // Later backtracking can restore them.
  int *P;
  P = (int *) calloc(m * n, sizeof(int));

  if (useBuiltInData)
  {
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

  double initialTime = omp_get_wtime();

  // original algorithm: sweep the matrix row by row, column by column
  // all three neighbors (-1,-1), (-1,0) and (0,-1), are visited before the current element is visited in this sweep.
  for (i = 1; i < n; i++) { //Rows
    for (j = 1; j < m; j++) { //Columns
      similarityScore(i, j, H, P, &maxPos);
    }
  }

  double finalTime = omp_get_wtime();
  printf("\nElapsed time for scoring matrix computation: %f\n\n", finalTime - initialTime);

  initialTime = omp_get_wtime();
  backtrack(P, maxPos);
  finalTime = omp_get_wtime();

  //Gets backtrack time
  finalTime = omp_get_wtime();
  printf("\nElapsed time for backtracking: %f\n\n", finalTime - initialTime);

#ifdef DEBUG
  printf("\nSimilarity Matrix:\n");
  printMatrix(H);

  if (useBuiltInData)
  {
    printf("Verifying correctness using builtin data =%d",H[m*n-1]==7 );
    assert (H[m*n-1]==7);
  }
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
 * Function:    SimilarityScore
 * Purpose:     Calculate  the maximum Similarity-Score H(i,j)
 */
void similarityScore(long long int i, long long int j, int* H, int* P, long long int* maxPos) {

    int up, left, diag;

    //Stores index of element
    long long int index = m * i + j;

    //Get element above: (i, j-1)
    up = H[index - m] + gapScore;

    //Get element on the left: (i-1, j)
    left = H[index - 1] + gapScore;

    //Get element on the diagonal: (i-1, j-1)
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
    if (max > H[*maxPos]) {
        *maxPos = index;
    }

}  /* End of similarityScore */


/*--------------------------------------------------------------------
 * Function:    matchMissmatchScore
 * Purpose:     Similarity function on the alphabet for match/missmatch
 */
int matchMissmatchScore(long long int i, long long int j) {
    if (a[j-1] == b[i-1])
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
        if(P[maxPos] == DIAGONAL)
            predPos = maxPos - m - 1;
        else if(P[maxPos] == UP)
            predPos = maxPos - m;
        else if(P[maxPos] == LEFT)
            predPos = maxPos - 1;
        P[maxPos]*=PATH;
        maxPos = predPos;
    } while(P[maxPos] != NONE);
}  /* End of backtrack */

/*--------------------------------------------------------------------
 * Function:    printMatrix
 * Purpose:     Print Matrix
 */
void printMatrix(int* matrix) {
    long long int i, j;
    for (i = 0; i < n; i++) { //Lines
        for (j = 0; j < m; j++) {
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
    for (i = 0; i < n; i++) { //Lines
        for (j = 0; j < m; j++) {
            index = m * i + j;
            if(matrix[index] < 0) {
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
 void generate(){
    //Generates the values of a
    long long int i;
    for(i=0;i<m;i++){
        int aux=rand()%4;
        if(aux==0)
            a[i]='A';
        else if(aux==2)
            a[i]='C';
        else if(aux==3)
            a[i]='G';
        else
            a[i]='T';
    }

    //Generates the values of b
    for(i=0;i<n;i++){
        int aux=rand()%4;
        if(aux==0)
            b[i]='A';
        else if(aux==2)
            b[i]='C';
        else if(aux==3)
            b[i]='G';
        else
            b[i]='T';
    }
} /* End of generate */


/*--------------------------------------------------------------------
 * External References:
 * http://vlab.amrita.edu/?sub=3&brch=274&sim=1433&cnt=1
 * http://pt.slideshare.net/avrilcoghlan/the-smith-waterman-algorithm
 * http://baba.sourceforge.net/
 */
