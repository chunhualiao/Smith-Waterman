//https://lc.llnl.gov/confluence/display/LC/Clang+OpenMP+4.5+with+GPU+support#space-menu-link-content
// Revised a bit
#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp parallel 
  {
#pragma omp master          
    { 
      int thread_count = omp_get_num_threads();
      printf ("Using %d out of max %d threads...\n", thread_count, omp_get_max_threads());
    }
  }

  int runningOnGPU = 0;

  printf ("The number of target devices =%d\n", omp_get_num_devices());
  /* Test if GPU is available using OpenMP4.5 */
#pragma omp target map(from:runningOnGPU)
  {
    // This function returns true if currently running on the host device, false otherwise.
    if (!omp_is_initial_device())
      runningOnGPU = 1;
  }
  /* If still running on CPU, GPU must not be available */
  if (runningOnGPU == 1)
    printf("### Able to use the GPU! ### \n");
  else
    printf("### Unable to use the GPU, using CPU! ###\n");

  return 0;
}
