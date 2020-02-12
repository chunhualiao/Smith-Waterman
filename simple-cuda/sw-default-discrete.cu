/*********************************************************************************
 * Smith–Waterman algorithm
 * Purpose:     Local alignment of nucleotide or protein sequences
 * Authors:     Daniel Holanda, Hanoch Griner, Taynara Pinheiro
 * Compilation: nvcc -std=c++11 -O3 -DNDEBUG=1 cuda_global_smithW.cu -o cuda_gm_smithW
 *              nvcc -std=c++11 -O0 -DDEBUG -g -G cuda_shared_smithW.cu -o dbg_cuda_smithW
 * Execution:   ./cuda_smithW <number_of_col> <number_of_rows>
 *********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//~ #include <time.h>
//~ #include <omp.h>

#include <cassert>
#include <chrono>
#include <iostream>

#ifndef NDEBUG
static constexpr bool DEBUG_MODE = true;
#else
static constexpr bool DEBUG_MODE = false;
#endif /* NDEBUG */


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
/* End of constants */

/*--------------------------------------------------------------------
* Helpers
*/
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define max(a,b) ((a) > (b) ? a : b)

// #define DEBUG
/* End of Helpers */

// my types
// \note changed type to unsigned to make it collaborate with CUDA atomicCAS
// \todo maybe rename it to index_t and change all long longs to index_t
typedef unsigned long long maxpos_t;

/// defines type for indices into arrays and matrices
///    (needs to be a signed type)
typedef long long int index_t;

/// defines data type for scoring
typedef int           score_t;

/// defines data type for linking paths
enum link_t { UNDEF = -1, NOLINK = 0, UP = 1, LEFT = 2, DIAGONAL = 3 };



/*--------------------------------------------------------------------
 * Functions Prototypes
 */
int backtrack(link_t* P, maxpos_t maxPos);
void printMatrix(score_t* matrix);
void printPredecessorMatrix(link_t* matrix);
void generate(void);
long long int nElement(long long int i);

// \pp modified to pass i (a induction variable) by value
void calcFirstDiagElement(long long int i, long long int *si, long long int *sj);

/* End of prototypes */

/*--------------------------------------------------------------------
 * Global Variables
 */
// Defines size of strings to be compared
index_t m = 8; // Columns - Size of string a
index_t n = 9; // Rows    - Size of string b

// Defines scores
static const int       MATCH_SCORE     =  3; //  5 in omp_smithW_orig
static const int       MISSMATCH_SCORE = -3; // -3
static const int       GAP_SCORE       = -2; // -4

// GPU THREADS PER BLOCK
static const long long THREADS_PER_BLOCK = 1024;

// Strings over the Alphabet Sigma
char *a, *b;

/* End of global variables */

static inline
void check_cuda_success(cudaError_t err)
{
  if (err == cudaSuccess) return;

  std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  exit(0);
}


template <class T>
T* shared_alloc_only(size_t numelems)
{
  void*       ptr /* = NULL*/;
  cudaError_t err = cudaMalloc(&ptr, numelems * sizeof(T));
  check_cuda_success(err);

  return reinterpret_cast<T*>(ptr);
}

template <class T>
T* shared_alloc(T*, size_t numelems)
{
  void*       ptr /* = NULL*/;
  cudaError_t err = cudaMalloc(&ptr, numelems * sizeof(T));
  check_cuda_success(err);

  return reinterpret_cast<T*>(ptr);
}

template <class T>
T* shared_alloc_transfer(T* src, size_t numelems = 1)
{
  T*          dst = shared_alloc_only<T>(numelems);
  cudaError_t err = cudaMemcpyAsync(dst, src, numelems * sizeof(T), cudaMemcpyHostToDevice);
  check_cuda_success(err);

  return dst;
}

template <class T>
T* shared_alloc_zero(size_t numelems = 1)
{
  T* ptr = shared_alloc_only<T>(numelems);

  cudaError_t err = cudaMemsetAsync(ptr, 0, numelems*sizeof(T), 0);
  check_cuda_success(err);

  return ptr;
}


void shared_free(void* p)
{
  cudaError_t err = cudaFree(p);
  check_cuda_success(err);
}

template <class T>
void shared_to_host_free(T* dst, T* src, size_t numelems = 1)
{
  cudaError_t err = cudaMemcpyAsync(dst, src, numelems * sizeof(T), cudaMemcpyDeviceToHost, 0);
  check_cuda_success(err);

  shared_free(src);
}


/*--------------------------------------------------------------------
 * Function:    matchMissmatchScore
 * Purpose:     Similarity function on the alphabet for match/missmatch
 */
__device__
score_t 
matchMissmatchScore_cuda(index_t i, index_t j, const char* seqa, const char* seqb)
{
    if (seqa[j - 1] == seqb[i - 1])
        return MATCH_SCORE;

    return MISSMATCH_SCORE;
}  /* End of matchMissmatchScore_cuda */


/*--------------------------------------------------------------------
 * Function:    SimilarityScore
 * Purpose:     Calculate  the maximum Similarity-Score H(i,j)
 */
__global__
void similarityScore_kernel( index_t si,
                             index_t sj,
                             index_t j_upper_bound,
                             score_t* H,
                             link_t* P,
                             maxpos_t* maxPos,
                             const char* seqa,
                             const char* seqb,
                             index_t cols
                           )
{
    // compute the second loop index j
    const index_t loopj = blockIdx.x * blockDim.x + threadIdx.x;

    if (loopj >= j_upper_bound) return;

    // compute original i and j
    index_t i = si - loopj;
    index_t j = sj + loopj;

    // bounds test for matchMissmatchScore_cuda
    assert(i > 0); // was: assert(i > 0 && i <= n); -- n currently not passed in
    assert(j > 0 && j <= cols);

    // Stores index of element
    maxpos_t index = cols * i + j;

    assert(index >= cols);
    // Get element above
    score_t up = H[index - cols] + GAP_SCORE;

    assert(index > 0);
    // Get element on the left
    score_t left = H[index - 1] + GAP_SCORE;

    assert(index > cols);
    // Get element on the diagonal
    score_t diag = H[index - cols - 1] + matchMissmatchScore_cuda(i, j, seqa, seqb);

    // Calculates the maximum
    score_t max  = NONE;
    link_t  pred = NOLINK;
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

    // same letter ↖
    if (diag > max) {
        max = diag;
        pred = DIAGONAL;
    }

    // remove letter ↑
    if (up > max) {
        max = up;
        pred = UP;
    }

    //insert letter ←
    if (left > max) {
        max = left;
        pred = LEFT;
    }

    //Inserts the value in the similarity and predecessor matrixes
    H[index] = max;
    P[index] = pred;

    // Updates maximum score to be used as seed on backtrack
    {
      // \note \pp
      //   locks seem to be a NOGO in CUDA warps,
      //   thus the update to set the maximum is made nonblocking.
      maxpos_t current = *maxPos;
      maxpos_t assumed = current+1;

      while (assumed != current && max > H[current])
      {
        assumed = current;

        // \note consider atomicCAS_system for multi GPU systems
        current = atomicCAS(maxPos, assumed, index);
      }
    }
}  /* End of similarityScore_kernel */



/*--------------------------------------------------------------------
 * Function:    main
 */
int main(int argc, char* argv[])
{
  typedef std::chrono::time_point<std::chrono::system_clock> time_point;

  bool     useBuiltInData = true;
  maxpos_t maxPos = 0;

  if (argc==3)
  {
    m = strtoll(argv[1], NULL, 10);
    n = strtoll(argv[2], NULL, 10);
    useBuiltInData = false;
  }

//#ifdef DEBUG
  if (useBuiltInData)
    printf ("Using built-in data for testing ..\n");

  // Allocates a and b
  a = (char*)malloc((m+1) * sizeof(char));
  b = (char*)malloc((n+1) * sizeof(char));
  //~ a = unified_alloc<char>(m);
  //~ b = unified_alloc<char>(n);

  std::cerr << "a,b allocated: " << m << "/" << n << std::endl;

  // Because now we have zeros
  m++;
  n++;

  if (useBuiltInData)
  {
    //Uncomment this to test the sequence available at
    //http://vlab.amrita.edu/?sub=3&brch=274&sim=1433&cnt=1
    // assert(m=11 && n=7);
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
    assert(m>=8 && n>=9);

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
  
  

  // Allocates similarity matrix H
  score_t* H = (score_t*)malloc(m * n * sizeof(score_t));

  //Allocates predecessor matrix P
  link_t*  P = (link_t*)malloc(m * n * sizeof(link_t));

  time_point     starttime = std::chrono::system_clock::now();
  
  
  // \todo \pp
  //   gpuA and gpuB could be allocated in constant memory (if small enough)
  char*       gpuA       = shared_alloc_transfer(a, m);
  char*       gpuB       = shared_alloc_transfer(b, n);
  score_t*    gpuH       = shared_alloc_zero<score_t>(m*n);
  link_t*     gpuP       = shared_alloc_zero<link_t>(m*n);
  maxpos_t*   gpuMaxPos  = shared_alloc_zero<maxpos_t>();

  // Because now we have zeros ((m-1) + (n-1) - 1)
  long long int nDiag = m + n - 3;

  for (int i = 1; i <= nDiag; ++i)
  {
      long long nEle = nElement(i);
      long long si /* uninitialized */;
      long long sj /* uninitialized */;

      calcFirstDiagElement(i, &si, &sj);

      {
        // CUDA, here we go

        // \note
        //   * MAKE SURE THAT a,b,H,P,maxPos are ACCESSIBLE from GPU.
        //     This prototype allocates a,b,H,P in unified memory space, thus
        //     we just copy the pointers. If the allocation policy changes,
        //     memory referenced by a,b,H,P has to be transferred to the GPU,
        //     and memory referenced by H and P has to be transferred back.
        //   * a and b do not change, thus they only need to be transferred
        //     initially.
        //   * transfers of H and P could probably be optimized to only
        //     include data along the wavefront.
        // \todo
        //   * study amount of data transfer for H and P
        //~ const long long ITER_SPACE = ceil(nEle/THREADS_PER_BLOCK);
        const long long ITER_SPACE = (nEle+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

        // comp. of ai and aj moved into CUDA kernel
        similarityScore_kernel
            <<<ITER_SPACE, THREADS_PER_BLOCK, 0, 0>>>
            (si, sj, nEle, gpuH, gpuP, gpuMaxPos, gpuA, gpuB, m);
      }
  }
  
  // data transfer
  //   P,H,maxPos
  shared_to_host_free(P,       gpuP, m*n);
  if (DEBUG_MODE || useBuiltInData)
    shared_to_host_free(H,       gpuH, m*n);
  else
    shared_free(gpuH);

  shared_to_host_free(&maxPos, gpuMaxPos);
  shared_free(gpuA);
  shared_free(gpuB);
  
  check_cuda_success( cudaStreamSynchronize(0) );

  time_point     endtime = std::chrono::system_clock::now();
  int len = backtrack(P, maxPos);

  if (DEBUG_MODE)
  {
    printf("\nSimilarity Matrix:\n");
    printMatrix(H);
  
    printf("\nPredecessor Matrix:\n");
    printPredecessorMatrix(P);
  }

  if (useBuiltInData)
  { 
    const bool correct = H[maxPos] == 13;
       
    std::cerr << "Max(builtin data): " << H[maxPos] << " == 13? " << correct
              << " " << maxPos 
              << std::endl;
    
    if (!correct) throw std::logic_error("Invalid result"); 
  }

  int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endtime-starttime).count();

  std::cout << "\nElapsed time: " << elapsed << " ms"
            << "\nPath length: " << len 
            << "\nScore: " << (DEBUG_MODE || useBuiltInData ? H[maxPos] : -1) 
            << std::endl;

  // Frees similarity matrixes
  free(H);
  free(P);

  //Frees input arrays
  free(a);
  free(b);

  return 0;
}  /* End of main */

/*--------------------------------------------------------------------
 * Function:    nElement
 * Purpose:     Calculate the number of i-diagonal elements
 */
long long int nElement(long long int i) {
    if (i < m && i < n) {
        // Number of elements in the diagonal is increasing
        return i;
    }
    else if (i < max(m, n)) {
        //Number of elements in the diagonal is stable
        long int min_mn = min(m, n);
        return min_mn - 1;
    }
    else {
        //Number of elements in the diagonal is decreasing
        long int min_mn = min(m, n);
        return 2 * min_mn - i + abs(m - n) - 2;
    }
}

/*--------------------------------------------------------------------
 * Function:    calcElement
 * Purpose:     Calculate the position of (si, sj)-element
 */
void calcFirstDiagElement(long long int i, long long int *si, long long int *sj) {
    // Calculate the first element of diagonal
    if (i < n) {
        *si = i;
        *sj = 1;
    } else {
        *si = n - 1;
        *sj = i - n + 2;
    }
}



/*--------------------------------------------------------------------
 * Function:    backtrack
 * Purpose:     Modify matrix to print, path change from value to PATH
 */
int backtrack(link_t* P, maxpos_t maxPos) {
    //hold maxPos value
    long long int predPos = 0;
    int        len = 0;

#ifdef DEBUG
    std::cerr << "maxpos = " << maxPos << std::endl;
#endif

    //backtrack from maxPos to startPos = 0
    do {
#ifdef DEBUG
        std::cerr << "P[" << maxPos << "] = "
                  << std::flush
                  << P[maxPos]
                  << std::endl;
#endif

        switch (P[maxPos])
        {
          case DIAGONAL:
            predPos = maxPos - m - 1;
            break;

          case UP:
            predPos = maxPos - m;
            break;

          case LEFT:
            predPos = maxPos - 1;
            break;

          default:
            assert(false);
        }

#ifdef DEBUG
        P[maxPos] *= PATH;
#endif
        maxPos = predPos;
        ++len;
    } while (P[maxPos] != NONE);

    return len;
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
          
          if ((m * i + j) == 69) printf("*");
          printf("%d\t", matrix[m * i + j]);
        }
        printf("\n");
    }

}  /* End of printMatrix */

/*--------------------------------------------------------------------
 * Function:    printPredecessorMatrix
 * Purpose:     Print predecessor matrix
 */
void printPredecessorMatrix(link_t* matrix) {
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
