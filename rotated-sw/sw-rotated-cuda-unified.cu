
/// \brief
/// This implements the Smith-Waterman algorithm based on a 45 degree 
/// rotated dynamic programming matrix. The benefit of the
/// rotation is:
///   - each diagonal can be represented by contiguous memory
///     --> reduces page faults
///     --> reduces resident memory use in the core algorithm
///     --> enables the algorithm to run more efficiently on large
///         inputs.
///   
/// The disadvantages of the rotation is:
///   - if the original output matrix needs to be maintained, copying
///     back may lead to costly page faults, outweighing the benefits of 
///     the rotation.
///     see @todo_1 for details.
///
/// \email pirkelbauer2@llnl.gov

/*
 * Compilation: nvcc -std=c++11 -O3 -DNDEBUG=1 sw-rotated-cuda-unified.cu -o smithW-cuda
 *              nvcc -std=c++11 -O0 -G -g sw-rotated-cuda-unified.cu -o dbg-smithW-cuda
 */

#include <vector>
#include <limits>
#include <cassert>
#include <algorithm>
#include <utility>
#include <iostream>
#include <chrono>

#include "parameters.h"

static const bool DEBUG_MODE = false;

/*--------------------------------------------------------------------
 * Text Tweaks
 */
#define RESET   "\033[0m"
#define BOLDRED "\033[1m\033[31m"      /* Bold Red */
/* End of text tweaks */

/// defines type for indices into arrays and matrices
///    (needs to be a signed type)
typedef long long int index_t;

/// defines data type for scoring
typedef int           score_t;

/// defines data type for linking paths
enum link_t { UNDEF = -1, NOLINK = 0, UP = 1, LEFT = 2, DIAGONAL = 3 };

// global constants
static const score_t PATH            = -1;
static const score_t NONE            =  0; // -4
static const score_t MATCH_SCORE     =  3; //  5 in omp_smithW_orig
static const score_t MISSMATCH_SCORE = -3; // -3
static const score_t GAP_SCORE       = -2; // -4

typedef std::vector<char> char_seq;

template <class T>
static inline
void rotate3(T& a, T& b, T& c)
{
  T tmp = a;

  a = c; c = b; b = tmp;
}


static inline
__device__
int matchMissmatchScore(const char* a, const char* b, size_t ai, size_t bi)
{
  return a[ai] == b[bi] ? MATCH_SCORE : MISSMATCH_SCORE;
}  /* End of matchMissmatchScore */


static
__global__
void similarityScore_kernel( index_t iterspace_lb,
                             index_t iterspace_ub,
                             index_t i,
                             score_t* M_0,
                             link_t* P_0,
                             const score_t* M_1,
                             const score_t* M_2,
                             const char* a,
                             const char* b,
                             const score_t** maxpos
                           )
{
  const index_t loop_j = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (loop_j >= iterspace_ub - iterspace_lb) return;
   
  const index_t ai   = (iterspace_ub - loop_j) - 1;
  const index_t bi   = i - ai;  
  const index_t j    = iterspace_lb + loop_j;    
  
  assert(!DEBUG_MODE || (M_1[j] >= 0 && M_1[j-1] >= 0 && M_2[j-1] >= 0));
  
  const index_t up   = M_1[j]   + GAP_SCORE;
  const index_t lft  = M_1[j-1] + GAP_SCORE;
  const index_t diag = M_2[j-1] + matchMissmatchScore(a, b, ai-1, bi-1);
  
  score_t       max  = NONE;
  link_t        pred = NOLINK;

  if (up > max)
  {
    max  = up;
    pred = UP;
  }

  if (lft > max)
  {
    max  = lft;
    pred = LEFT;
  }

  if (diag > max)
  {
    max  = diag;
    pred = DIAGONAL;
  }

  assert(!DEBUG_MODE || (M_0[j] < 0));
  M_0[j] = max;
  P_0[j] = pred;
  
  // Updates maximum score to be used as seed on backtrack
  {   
    // \note \pp
    //   locks seem to be a NOGO in CUDA warps,
    //   thus the update to set the maximum is made nonblocking.
    const score_t* assumed = nullptr;
    const score_t* current = *maxpos;
  
    while ((current != assumed) && max > *current)
    {
      // \note consider atomicCAS_system for multi GPU systems
      assumed = current;      
      current = (const score_t*) atomicCAS( (unsigned long long int*) maxpos, 
                                            (unsigned long long int) assumed, 
                                            (unsigned long long int) (M_0 + j)
                                          );
    }
  }
}

/*--------------------------------------------------------------------
 * Function:    calcElement
 * Purpose:     Calculate the first element of a given diagonal
 */
index_t diagonalBasePoint(index_t i, index_t w)
{
  // base point is on the first row
  if (i-1 <= w) return i-1;

  // base point is the last element on the (i-w)+2 th row
  return (w+1)*(i-w)-1;
}


static inline
void check_cuda_success(cudaError_t err)
{
  if (err == cudaSuccess) return;

  std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  exit(0);
}

/// malloc replacement
template<class T>
static
T* unified_alloc(size_t numelems)
{
  void*       ptr /* = NULL*/;
  cudaError_t err = cudaMallocManaged(&ptr, numelems * sizeof(T), cudaMemAttachGlobal);

  check_cuda_success(err);
  return reinterpret_cast<T*>(ptr);
}

static
void unified_free(void* ptr)
{
  cudaError_t err = cudaFree(ptr);

  check_cuda_success(err);
}


/// \brief computes smith-waterman
/// \param a input sequence of length w
/// \param b input sequence of length h
/// \param w length of input sequence a
/// \param h length of input sequence b
/// \param H output matrix (size == (w+1) * (h+1)) representing all scores
/// \param P output matrix (size == (w+1) * (h+1)) to link longest sequences
/// \param maxscore output score of longest matching sequence in H and P
/// \param maxloc output position of longest matching sequence in H and P
/// \note output data does not need to be initialized
void smithWaterman( const char* a,
                    const char* b,
                    index_t w,
                    index_t h,
                    score_t* H,
                    link_t* P,
                    score_t** maxloc
                  )
{
  // Size is important for pointer CAS in CUDA Kernel
  static_assert( sizeof(maxloc) == sizeof(unsigned long long int), 
                 "pointer/int size mismatch (req. for CUDA atomicCAS)!"
               );

  const index_t  MAXITER  = 2 + w + h - 1;

  // wavefront arrays for three iterations
  score_t* const        wavefronts = unified_alloc<score_t>(3*MAXITER);
  link_t*               pred_0     = unified_alloc<link_t>(MAXITER);
  link_t*               pred_1     = unified_alloc<link_t>(MAXITER);
  score_t* const        maxscr     = unified_alloc<score_t>(1);
  const score_t** const maxpos     = unified_alloc<const score_t*>(1);
  
  // wavefront representation _time
  score_t*       M_2 = wavefronts;
  score_t*       M_1 = wavefronts + MAXITER;

  // wavefront output
  score_t*       M_0 = wavefronts + 2*MAXITER;

  // initialize t == 0
  M_1[0] = NONE;

  // set maxloc to origin, and origin to 0
  *maxloc  = H;
  **maxloc = 0;
  *maxscr  = 0;

  // smith waterman
  for (index_t i = 1; i <= MAXITER; ++i)
  {
    const index_t  lb       = (i<=h) ? (M_0[0] = NONE, 1) : i - h;
    const index_t  ub       = (i<=w) ? (M_0[i] = NONE, i) : w + 1;
    
    *maxpos = maxscr;
    
    assert((ub - lb >= 0) && (ub - lb <= h));
    
    const index_t THREADS_PER_BLOCK = 1024;
    const index_t ITER_SPACE = (ub-lb+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    
    similarityScore_kernel
      <<<ITER_SPACE, THREADS_PER_BLOCK>>>
      (lb, ub, i, M_0, pred_0, M_1, M_2, a, b, maxpos);      

    // rotate wavefront vectors
    rotate3(M_0, M_1, M_2);

    // swap output vectors
    std::swap(pred_0, pred_1);

    cudaDeviceSynchronize();

    // for debugging purposes clear all M_0
    if (DEBUG_MODE)
      std::fill(M_0, M_0+MAXITER, std::numeric_limits<score_t>::min());

    // \note
    // The problem with the CUDA algorithm is that getting the
    // data off of the GPU produces an enormous amount of page faults.
    // e.g., on Lassen with a 100000x10000 data set copying the data
    //       out from the GPU results in a 20x performance hit.
    //
    // \todo @todo_1
    // Can we hide the data transfer behind the computation of the
    // next iteration?
    // - in principle, this should be possible. The next round of
    //   computation only changes M_0, and pred_0, thus we could
    //   overlap the computation with data transfer back to the CPU.
    // - in a first attempt, CUDA streams were attempted. In this
    //   versions there existed a computestream and a transferstream.
    //   the transferstream synchronized one time step after the
    //   computation. However, the result was a minor slow down to
    //   the version in this file.
    //   see file sw-rotated-cuda-global-stream.cu
    //
    // - TRY to copy the data back into the H,P arrays using OpenMP
    //   ...
    // - Alternatively, the interface to SmithWaterman could be changed.
    //   The H,P representations are artifacts from the standard
    //   algorithm. By moving towards a 45 degree rotated base version
    //   copying back could be entirely avoided at the expense that
    //   the data representation becomes twice is big (though this
    //   could possibly be remedied by using a clever data layout).
    if (0)
    {
      // results from this iteration are in M_1
      // -> strided copies back to H and P
      index_t       ofs  = diagonalBasePoint(i, w) + w + 1;      
      cudaError_t   errH = cudaMemcpy2D( H+ofs, 
                                         w*sizeof(*H), 
                                         M_1 + lb, 
                                         sizeof(*M_1), 
                                         sizeof(*M_1), 
                                         ub-lb, 
                                         cudaMemcpyDefault
                                       );      
      check_cuda_success(errH);
      
      cudaError_t errP  = cudaMemcpy2D( P+ofs, 
                                        w*sizeof(*P), 
                                        pred_1 + lb, 
                                        sizeof(*pred_1), 
                                        sizeof(*pred_1), 
                                        ub-lb, 
                                        cudaMemcpyDefault
                                      );      
      check_cuda_success(errP);
    }
    

    {
      const score_t* maxx = *maxpos;
            
      // update maxscore, if maxpos points to an improved location
      if (maxx != maxscr)
      {
        index_t       j   = maxx - M_1;
        index_t       ofs = diagonalBasePoint(i, w) + w + 1;

        ofs += (j - lb) * w;

        *maxloc = H + ofs;
        *maxscr = *maxx;
      }
    }
  }


  unified_free(wavefronts);
  unified_free(pred_0);
  unified_free(pred_1);
  unified_free(maxscr);
  unified_free(maxpos);
}


/*--------------------------------------------------------------------
 * Function:    backtrack
 * Purpose:     Modify matrix to print, path change from value to PATH
 */
void backtrack(link_t* P, index_t maxPos, index_t m) {
    //hold maxPos value
    index_t predPos = 0;

    //backtrack from maxPos to startPos = 0
    do {
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

        P[maxPos] = static_cast<link_t>(P[maxPos] * PATH);
        maxPos = predPos;
    } while (P[maxPos] != NONE);
}  /* End of backtrack */

/*--------------------------------------------------------------------
 * Function:    printMatrix
 * Purpose:     Print Matrix
 */
void printMatrix(score_t* matrix, const char* a, const char* b, index_t m, index_t n) {
    printf("-\t-\t");
    for (index_t j = 0; j < m; j++) {
      printf("%c\t", a[j]);
    }
    printf("\n-\t");
    for (index_t i = 0; i < n+1; i++) { // Lines
        for (index_t j = 0; j < m+1; j++) {
          if (j==0 && i>0) printf("%c\t", b[i-1]);
            printf("%d\t", std::max(0, matrix[(m+1) * i + j]));
        }
        printf("\n");
    }

}  /* End of printMatrix */

/*--------------------------------------------------------------------
 * Function:    printPredecessorMatrix
 * Purpose:     Print predecessor matrix
 */
void printPredecessorMatrix(link_t* matrix, const char* a, const char* b, index_t m, index_t n) {
    printf("    ");
    for (index_t j = 0; j < m; j++) {
      printf("%c ", a[j]);
    }
    printf("\n  ");
    for (index_t i = 0; i < n+1; i++) { //Lines
        for (index_t j = 0; j < m+1; j++) {
          if (j==0 && i>0) printf("%c ", b[i-1]);
            index_t index = m * i + j;
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
void generate(char* a, char* b, index_t m, index_t n) {
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
 * Function:    main
 */
int main(int argc, char* argv[])
{
  typedef std::chrono::time_point<std::chrono::system_clock> time_point;

  bool     useBuiltInData = true;
  index_t  m = 8;
  index_t  n = 9;

  if (argc==3)
  {
    m = strtoll(argv[1], NULL, 10);
    n = strtoll(argv[2], NULL, 10);
    useBuiltInData = false;
  }

  if (useBuiltInData)
    printf ("Using built-in data for testing ..\n");

  printf("Problem size: Matrix[%lld][%lld], FACTOR=%d CUTOFF=%d\n", n, m, FACTOR, CUTOFF);

  // Allocates a and b
  // \pp \note m (instead of m+1), b/c end marker is not needed
  //~ char* a = (char*)malloc(m * sizeof(char));
  //~ char* b = (char*)malloc(n * sizeof(char));
  char* a = unified_alloc<char>(m);
  char* b = unified_alloc<char>(n);

  std::cerr << "a,b allocated: " << m << "/" << n << std::endl;

  //~ // Because now we have zeros
  // \pp m and n are the lengths of input strings ..
  //~ m++;
  //~ n++;

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
    assert(m==8 && n==9);

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
    // Gen random arrays a and b
    generate(a, b, m, n);
  }

  time_point     starttime = std::chrono::system_clock::now();

  // Allocates similarity matrix H
  score_t* H = (score_t*) calloc((m+1) * (n+1), sizeof(score_t));

  // Allocates predecessor matrix P
  link_t*  P = (link_t*)  calloc((m+1) * (n+1), sizeof(link_t));
  score_t* maxloc = nullptr;

  smithWaterman(a, b, m, n, H, P, &maxloc);

  time_point     endtime = std::chrono::system_clock::now();

  if (DEBUG_MODE)
  {
    printf("\nSimilarity Matrix:\n");
    printMatrix(H, a, b, m, n);

    printf("\nPredecessor Matrix:\n");
    printPredecessorMatrix(P, a, b, m, n);
  }

  if (useBuiltInData)
  {
    printf ("Verifying results using the builtinIn data: %s\n", (H[(n+1)*(m+1)-1]==7)?"true":"false");
    assert (H[(n+1)*(m+1)-1]==7);
  }

  backtrack(P, maxloc - H, m+1);

  int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endtime-starttime).count();

  printf("\nElapsed time: %d ms\n\n", elapsed);

  // Frees similarity matrixes
  free(H);
  free(P);

  //Frees input arrays
  //~ free(a);
  //~ free(b);  
  unified_free(a);
  unified_free(b);

  return 0;
}  /* End of main */
