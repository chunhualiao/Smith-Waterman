
/// \brief
/// This implements the Smith-Waterman algorithm based on a 45 degree
/// rotated dynamic programming matrix. The benefit of the
/// rotation is:
///   - each diagonal can be represented by contiguous memory
///     --> reduces page faults
///     --> reduces resident memory use in the core algorithm
///     --> enables the algorithm to run more efficiently on very large
///         inputs.
///
/// The algorithm takes advantage of CUDA unified memory in the following way:
///   - With very large data sizes the best common subsequence  
///     automatically kept resident in the GPU. With discrete memory,
///     this would be very hard to implement efficiently (PP). 
///
/// The disadvantages of the rotation is:
///   - if the original output matrix needs to be maintained, copying
///     back may lead to costly page faults, outweighing the benefits of
///     the rotation.
///
/// \email pirkelbauer2@llnl.gov

/*
 * Compilation: nvcc -std=c++11 -O3 -DNDEBUG=1 sw-rotated.cu -o smithW-cuda
 *              nvcc -std=c++11 -O0 -G -g sw-rotated.cu -o dbg-smithW-cuda
 */

#include <iostream>
#include <chrono>
#include <cassert>
#include <iomanip>

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


//
// CUDA wrappers

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
  cudaError_t err = cudaMallocManaged(&ptr, numelems * sizeof(T));
  check_cuda_success(err);

  //~ err = cudaMemAdvise(ptr, numelems * sizeof(T), cudaMemAdviseSetPreferredLocation, 0);
  //~ check_cuda_success(err);

  return reinterpret_cast<T*>(ptr);
}

/// malloc replacement
template<class T>
static
T* unified_alloc_zero(size_t numelems)
{
  T* ptr = unified_alloc<T>(numelems);

  // since we only use default streams, we can do async
  cudaError_t err = cudaMemsetAsync(ptr, 0, numelems*sizeof(T));

  check_cuda_success(err);
  return ptr;
}

// free replacement
static
void unified_free(void* ptr)
{
  cudaError_t err = cudaFree(ptr);

  check_cuda_success(err);
}



//
// Smith-Waterman Kernel

static inline
__device__
int matchMissmatchScore(const char* a, const char* b, size_t ai, size_t bi)
{
  return a[ai] == b[bi] ? MATCH_SCORE : MISSMATCH_SCORE;
}  /* End of matchMissmatchScore */

__managed__ __device__
unsigned long long int maxpos = 0;

__managed__ __device__
unsigned long long int maxiter = 0;

static
__global__
void similarityScore_kernel( score_t* const H,
                             link_t* const P,
                             const index_t ofs_0,
                             const index_t ofs_1,
                             const index_t ofs_2,
                             const char* const a,
                             const char* const b,
                             int rightShift, /* shift start idx right, if left-most cell is boundary */
                             int iteration,
                             int diaglen,
                             int abase
                           )
{
  const index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= diaglen) return; /* out of bounds threads */
  
  const index_t ai   = abase - idx;
  const index_t bi   = iteration - ai;
  
  assert(ai >= 0);  
  assert(bi >= 0);  
  const index_t j    = idx + rightShift;
  
  const score_t up   = H[ofs_1 + idx]     + GAP_SCORE;
  const score_t lft  = H[ofs_1 + idx + 1] + GAP_SCORE;
  const score_t diag = H[ofs_2 + idx]     + matchMissmatchScore(a, b, ai, bi);

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

  H[ofs_0 + j] = max;
  P[ofs_0 + j] = pred;

  // Updates maximum score to be used as seed on backtrack
  {
    // nonblocking update of maxpos.
    unsigned long long curridx = maxpos;
    
    assert(curridx < ofs_0 + diaglen);
    score_t                currmax = H[curridx];

    while (max > currmax)
    {
      unsigned long long int assumed = curridx;
      
      curridx = atomicCAS( &maxpos, assumed, ofs_0 + j );
      
      if (curridx == assumed)
      { 
        // also store iteration number to simplify backtracking
        maxiter = iteration;
        break;
      }
      
      currmax = H[curridx];
    }
  }  
}

std::pair<score_t*, link_t*>
smithWaterman(const char* a, const char* b, int alen, int blen)
{
  static constexpr index_t THREADS_PER_BLOCK = 1024;
  
  const size_t  datalen = size_t(alen+1) * size_t(blen+1);
  score_t*      H       = unified_alloc_zero<score_t>(datalen);
  link_t*       P       = unified_alloc_zero<link_t>(datalen);
  int           numdiag = alen+blen+1;
  int           diaglen = 1;
  
  // initialize the first three values (M_2, M_1) to zero
  // cudaMemsetAsync(H, 0, sizeof(score_t) * 3, 0);
  // cudaMemsetAsync(P, 0, sizeof(link_t) * 3, 0);
   
  index_t       ofs_2     = 0;
  index_t       ofs_1     = ofs_2 + 1;
  index_t       ofs_0     = ofs_1 + 2;
  
  for (int i = 0; i < numdiag-2; ++i)
  {
    index_t    abase              = alen - 1;
    const bool leftCellIsBoundary = (i < abase);
    int        numBoundaryCells   = 0;
  
    if (leftCellIsBoundary)
    {
      //~ cudaMemsetAsync(H + M_0, 0, sizeof(score_t), 0);
      ++diaglen;
      ++numBoundaryCells;
      abase = i;
    }
    else if (i >= alen)
    {
      // two iterations after the lower left corner, the offset needs 
      //   to be adjusted.
      ++ofs_2;
    }
        
    if (i < blen-1 /* right cell is boundary */)
    {
      //~ cudaMemsetAsync(H + M_0 + diaglen - 1, 0, sizeof(score_t), 0);
      ++diaglen;
      ++numBoundaryCells;
    }
    
    const long long ITER_SPACE = ((diaglen-numBoundaryCells)+(THREADS_PER_BLOCK-1))/THREADS_PER_BLOCK;
        
    similarityScore_kernel
        <<<ITER_SPACE, THREADS_PER_BLOCK, 0, 0>>>
        (H, P, ofs_0, ofs_1, ofs_2, a, b, leftCellIsBoundary, i, diaglen-numBoundaryCells, abase)
        ;
    
    ofs_2 = ofs_1;
    ofs_1 = ofs_0;
    ofs_0 = ofs_0 + diaglen;
    
    --diaglen;
  }
  
  check_cuda_success( cudaStreamSynchronize(0) );
  return std::make_pair(H, P);
}

//
// Other functions

/*--------------------------------------------------------------------
 * Function:    backtrack
 * Purpose:     Modify matrix to print, path change from value to PATH
 */
//~ void backtrack(link_t* P, index_t maxPos, index_t m) 
void backtrack(link_t*, index_t, index_t) 
{
  /* TODO */
  
#if OLD_CODE  
  
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
#endif /* OLD_CODE */    
}  /* End of backtrack */



index_t calcIdx(index_t i, index_t j, index_t alen, index_t blen)
{
  /* TODO: this is simple code to verify;
           actual code needs to become more efficient
  */
  
  int n = 0;
  
  for (index_t d = 0; d <= alen+blen+1; ++d)
    for (index_t x = std::min(d, alen); x >= 0; --x)
    {
      int y = d - x;
      
      if (y > blen) continue;
      
      if ((x == i) && (y == j))
        return n;
        
      ++n;
    }
    
  assert(false);
  std::cerr << "oops" << std::endl;
  return -1;
}



/*--------------------------------------------------------------------
 * Function:    printMatrix
 * Purpose:     Print Matrix
 */
void printMatrix(score_t* M, const char* a, const char* b, index_t alen, index_t blen) 
{  
  const index_t numdiag = alen+blen+1;
  const index_t minlen  = std::min(alen, blen)+1;
  
  std::cerr << minlen << " -> " << numdiag << std::endl;
  std::cout << std::setw(4) << "" << std::setw(4) << "" << std::setw(4) << '-';
  
  for (int j = 0; j < blen; ++j)
  {
    std::cout << std::setw(4) << b[j];
  }
  
  //~ std::cout << std::setw(4) << calcIdx(2, 4, alen, blen);;
  
  {
    // first empty row
      
    std::cout << std::endl;
    std::cout << std::setw(4) << "" << std::setw(4) << '-';
    
    for (int j = 0; j <= blen; ++j)
    {
      std::cout << std::setw(4) << M[calcIdx(0, j, alen, blen)];
    }
    
    std::cout << "." << std::endl;
  }
  
  {
    for (int i = 1; i <= alen; ++i)
    {
      std::cout << std::setw(4) << (i) << std::setw(4) << a[i-1];
      
      for (int j = 0; j <= blen; ++j)
      {        
        std::cout << std::setw(4) << M[calcIdx(i, j, alen, blen)];
      }
      
      std::cout << std::endl;
    }
  }
}  /* End of printMatrix */

/*--------------------------------------------------------------------
 * Function:    printPredecessorMatrix
 * Purpose:     Print predecessor matrix
 */
//~ void printPredecessorMatrix(link_t* matrix, const char* a, const char* b, index_t m, index_t n) 
void printPredecessorMatrix(link_t*, const char*, const char*, index_t, index_t) 
{
    /* TODO */
    
#if OLD_CODE    
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
#endif /* OLD_CODE */    
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
    std::cout << ("Using built-in data for testing ..\n");

  //~ std::cout << "Problem size: Matrix[" << n << "][" << m << "], FACTOR=" << FACTOR<< " CUTOFF=" << CUTOFF
            //~ << std::endl;

  // Allocates a and b
  // \pp \note m (instead of m+1), b/c end marker is not needed
  char* a = unified_alloc<char>(m);
  char* b = unified_alloc<char>(n);

  std::cerr << "a,b allocated: " << m << "/" << n << std::endl;
  
  int alen = m;
  int blen = n;

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
    assert(alen==8 && blen==9);

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
    generate(a, b, alen, blen);
  }
  
  time_point     starttime = std::chrono::system_clock::now();

  std::pair<score_t*, link_t*> res = smithWaterman(a, b, alen, blen);

  time_point     endtime = std::chrono::system_clock::now();
  int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endtime-starttime).count();
  
  std::cout << "\nElapsed time: " << elapsed << " ms"
            << "\npos " << maxpos << " @" << maxiter
            << "\nScore: " << res.first[maxpos] 
            << std::endl;

  if (DEBUG_MODE)
  {
    printMatrix(res.first, a, b, alen, blen);
    // res.first = static_cast<score_t*>(calloc((alen+1)*(blen+1), sizeof(score_t)));

    //~ printf("\nPredecessor Matrix:\n");
    //~ printPredecessorMatrix(P, a, b, m, n);
  }

  if (useBuiltInData)
  { 
    const bool correct = res.first[maxpos] == 13;
       
    std::cerr << "Max(builtin data): " << res.first[maxpos] << " == 13? " << correct
              << std::endl;
    
    if (!correct) throw std::logic_error("Invalid result"); 
  }

  //~ backtrack(P, maxloc - H, m+1);

  // frees matrices
  unified_free(res.first);
  unified_free(res.second);
  unified_free(a);
  unified_free(b);

  return 0;
}  /* End of main */
