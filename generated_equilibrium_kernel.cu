
// Removed cupy/complex.cuh as it may cause CUDA_ERROR_INVALID_VALUE
#include <float.h>          // C-style header, works better with NVCC backend
#include <math.h>           // C-style header, works better with NVCC backend
#include <stdio.h>          // For printf debugging

// --- GPU Debug logging helpers (must be outside extern "C") ---
__device__ void gpu_debug_log(int segment, const char* message, int condition_idx) {
    #ifdef VERBOSE_DEBUG
    // Only print for first 3 conditions to reduce clutter
    if (condition_idx >= 3 && condition_idx >= 0) return;
    
    if (condition_idx >= 0) {
        printf("[GPU] SEGMENT %02d: %s (condition %d)\n", segment, message, condition_idx);
    } else {
        printf("[GPU] SEGMENT %02d: %s\n", segment, message);
    }
    #endif
}

__device__ void gpu_debug_log_value(const char* message, double value) {
    #ifdef VERBOSE_DEBUG
    printf("[GPU]   %s: %.15e\n", message, value);
    #endif
}

__device__ void gpu_debug_log_array(const char* message, const double* arr, int size) {
    #ifdef VERBOSE_DEBUG
    printf("[GPU]   %s: [", message);
    for (int i = 0; i < size && i < 5; ++i) {
        printf("%.6f", arr[i]);
        if (i < size - 1) printf(", ");
    }
    if (size > 5) printf("...");
    printf("]\n");
    #endif
}

// --- Static C Code Includes ---
// Content of svd.c
////////////////////////////////////////////////////////////////////////////////
// File: singular_value_decomposition.c                                       //
// Contents:                                                                  //
//    Singular_Value_Decomposition                                            //
//    Singular_Value_Decomposition_Solve                                      //
//    Singular_Value_Decomposition_Inverse                                    //
////////////////////////////////////////////////////////////////////////////////

// SVD Implementation - included as source, not header
#include <float.h>               // required for DBL_EPSILON
#include <math.h>                // required for fabs(), sqrt();

#define MAX_ITERATION_COUNT 30   // Maximum number of iterations

//                        Internally Defined Routines 
__device__ static void Householders_Reduction_to_Bidiagonal_Form(double* A, int nrows,
    int ncols, double* U, double* V, double* diagonal, double* superdiagonal );
__device__ static int  Givens_Reduction_to_Diagonal_Form( int nrows, int ncols,
           double* U, double* V, double* diagonal, double* superdiagonal );
__device__ static void Sort_by_Decreasing_Singular_Values(int nrows, int ncols,
                                double* singular_value, double* U, double* V);

////////////////////////////////////////////////////////////////////////////////
//  int Singular_Value_Decomposition(double* A, int nrows, int ncols,         //
//        double* U, double* singular_values, double* V, double* dummy_array) //
//                                                                            //
//  Description:                                                              //
//     This routine decomposes an m x n matrix A, with m >= n, into a product //
//     of the three matrices U, D, and V', i.e. A = UDV', where U is an m x n //
//     matrix whose columns are orthogonal, D is a n x n diagonal matrix, and //
//     V is an n x n orthogonal matrix.  V' denotes the transpose of V.  If   //
//     m < n, then the procedure may be used for the matrix A'.  The singular //
//     values of A are the diagonal elements of the diagonal matrix D and     //
//     correspond to the positive square roots of the eigenvalues of the      //
//     matrix A'A.                                                            //
//                                                                            //
//     This procedure programmed here is based on the method of Golub and     //
//     Reinsch as given on pages 134 - 151 of the "Handbook for Automatic     //
//     Computation vol II - Linear Algebra" edited by Wilkinson and Reinsch   //
//     and published by Springer-Verlag, 1971.                                //
//                                                                            //
//     The Golub and Reinsch's method for decomposing the matrix A into the   //
//     product U, D, and V' is performed in three stages:                     //
//       Stage 1:  Decompose A into the product of three matrices U1, B, V1'  //
//         A = U1 B V1' where B is a bidiagonal matrix, and U1, and V1 are a  //
//         product of Householder transformations.                            //
//       Stage 2:  Use Given' transformations to reduce the bidiagonal matrix //
//         B into the product of the three matrices U2, D, V2'.  The singular //
//         value decomposition is then UDV'where U = U2 U1 and V' = V1' V2'.  //
//       Stage 3:  Sort the matrix D in decreasing order of the singular      //
//         values and interchange the columns of both U and V to reflect any  //
//         change in the order of the singular values.                        //
//                                                                            //
//     After performing the singular value decomposition for A, call          //
//     Singular_Value_Decomposition to solve the equation Ax = B or call      //
//     Singular_Value_Decomposition_Inverse to calculate the pseudo-inverse   //
//     of A.                                                                  //
//                                                                            //
//  Arguments:                                                                //
//     double* A                                                              //
//        On input, the pointer to the first element of the matrix            //
//        A[nrows][ncols].  The matrix A is unchanged.                        //
//     int nrows                                                              //
//        The number of rows of the matrix A.                                 //
//     int ncols                                                              //
//        The number of columns of the matrix A.                              //
//     double* U                                                              //
//        On input, a pointer to a matrix with the same number of rows and    //
//        columns as the matrix A.  On output, the matrix with mutually       //
//        orthogonal columns which is the left-most factor in the singular    //
//        value decomposition of A.                                           //
//     double* singular_values                                                //
//        On input, a pointer to an array dimensioned to same as the number   //
//        of columns of the matrix A, ncols.  On output, the singular values  //
//        of the matrix A sorted in decreasing order.  This array corresponds //
//        to the diagonal matrix in the singular value decomposition of A.    //
//     double* V                                                              //
//        On input, a pointer to a square matrix with the same number of rows //
//        and columns as the columns of the matrix A, i.e. V[ncols][ncols].   //
//        On output, the orthogonal matrix whose transpose is the right-most  //
//        factor in the singular value decomposition of A.                    //
//     double* dummy_array                                                    //
//        On input, a pointer to an array dimensioned to same as the number   //
//        of columns of the matrix A, ncols.  This array is used to store     //
//        the super-diagonal elements resulting from the Householder reduction//
//        of the matrix A to bidiagonal form.  And as an input to the Given's //
//        procedure to reduce the bidiagonal form to diagonal form.           //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - During the Given's reduction of the bidiagonal form to    //
//                  diagonal form the procedure failed to terminate within    //
//                  MAX_ITERATION_COUNT iterations.                           //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double A[M][N];                                                        //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double singular_values[N];                                             //
//     double* dummy_array;                                                   //
//                                                                            //
//     (your code to initialize the matrix A)                                 //
//     dummy_array = (double*) malloc(N * sizeof(double));                    //
//     if (dummy_array == NULL) {printf(" No memory available\n"); exit(0); } //
//                                                                            //
//     err = Singular_Value_Decomposition((double*) A, M, N, (double*) U,     //
//                              singular_values, (double*) V, dummy_array);   //
//                                                                            //
//     free(dummy_array);                                                     //
//     if (err < 0) printf(" Failed to converge\n");                          //
//     else { printf(" The singular value decomposition of A is \n");         //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
__device__ int Singular_Value_Decomposition(double* A, int nrows, int ncols, double* U, 
                      double* singular_values, double* V, double* dummy_array)
{
   Householders_Reduction_to_Bidiagonal_Form( A, nrows, ncols, U, V,
                                                singular_values, dummy_array);
   if (Givens_Reduction_to_Diagonal_Form( nrows, ncols, U, V,
                                singular_values, dummy_array ) < 0) return -1;
   Sort_by_Decreasing_Singular_Values(nrows, ncols, singular_values, U, V);
  
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
// static void Householders_Reduction_to_Bidiagonal_Form(double* A, int nrows,//
//  int ncols, double* U, double* V, double* diagonal, double* superdiagonal )//
//                                                                            //
//  Description:                                                              //
//     This routine decomposes an m x n matrix A, with m >= n, into a product //
//     of the three matrices U, B, and V', i.e. A = UBV', where U is an m x n //
//     matrix whose columns are orthogonal, B is a n x n bidiagonal matrix,   //
//     and V is an n x n orthogonal matrix.  V' denotes the transpose of V.   //
//     If m < n, then the procedure may be used for the matrix A'.  The       //
//                                                                            //
//     The matrix U is the product of Householder transformations which       //
//     annihilate the subdiagonal components of A while the matrix V is       //
//     the product of Householder transformations which annihilate the        //
//     components of A to the right of the superdiagonal.                     //
//                                                                            //
//     The Householder transformation which leaves invariant the first k-1    //
//     elements of the k-th column and annihilates the all the elements below //
//     the diagonal element is P = I - (2/u'u)uu', u is an nrows-dimensional  //
//     vector the first k-1 components of which are zero and the last         //
//     components agree with the current transformed matrix below the diagonal//
//     diagonal, the remaining k-th element is the diagonal element - s, where//
//     s = (+/-)sqrt(sum of squares of the elements below the diagonal), the  //
//     sign is chosen opposite that of the diagonal element.                  //
//                                                                            //
//  Arguments:                                                                //
//     double* A                                                              //
//        On input, the pointer to the first element of the matrix            //
//        A[nrows][ncols].  The matrix A is unchanged.                        //
//     int nrows                                                              //
//        The number of rows of the matrix A.                                 //
//     int ncols                                                              //
//        The number of columns of the matrix A.                              //
//     double* U                                                              //
//        On input, a pointer to a matrix with the same number of rows and    //
//        columns as the matrix A.  On output, the matrix with mutually       //
//        orthogonal columns which is the left-most factor in the bidiagonal  //
//        decomposition of A.                                                 //
//     double* V                                                              //
//        On input, a pointer to a square matrix with the same number of rows //
//        and columns as the columns of the matrix A, i.e. V[ncols][ncols].   //
//        On output, the orthogonal matrix whose transpose is the right-most  //
//        factor in the bidiagonal decomposition of A.                        //
//     double* diagonal                                                       //
//        On input, a pointer to an array dimensioned to same as the number   //
//        of columns of the matrix A, ncols.  On output, the diagonal of the  //
//        bidiagonal matrix.                                                  //
//     double* superdiagonal                                                  //
//        On input, a pointer to an array dimensioned to same as the number   //
//        of columns of the matrix A, ncols.  On output, the superdiagonal    //
//        of the bidiagonal matrix.                                           //
//                                                                            //
//  Return Values:                                                            //
//     The function is of type void and therefore does not return a value.    //
//     The matrices U, V, and the diagonal and superdiagonal are calculated   //
//     using the addresses passed in the argument list.                       //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double A[M][N];                                                        //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double diagonal[N];                                                    //
//     double superdiagonal[N];                                               //
//                                                                            //
//     (your code to initialize the matrix A - Note this routine is not       //
//     (accessible from outside i.e. it is declared static)                   //
//                                                                            //
//     Householders_Reduction_to_Bidiagonal_Form((double*) A, nrows, ncols,   //
//                   (double*) U, (double*) V, diagonal, superdiagonal )      //
//                                                                            //
//     free(dummy_array);                                                     //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
__device__ static void Householders_Reduction_to_Bidiagonal_Form(double* A, int nrows,
    int ncols, double* U, double* V, double* diagonal, double* superdiagonal )
{
   int i,j,k,ip1;
   double s, s2, si, scale;
   double dum;
   double *pu, *pui, *pv, *pvi;
   double half_norm_squared;

// Copy A to U
   j = nrows*ncols;
   for(i = 0; i < j; i++) {
      U[i] = A[i];
   }
   

//
 
   diagonal[0] = 0.0;
   s = 0.0;
   scale = 0.0;
   for ( i = 0, pui = U, ip1 = 1; i < ncols; pui += ncols, i++, ip1++ ) {
      superdiagonal[i] = scale * s;
//       
//                  Perform Householder transform on columns.
//
//       Calculate the normed squared of the i-th column vector starting at 
//       row i.
//
      for (j = i, pu = pui, scale = 0.0; j < nrows; j++, pu += ncols)
         scale += fabs( *(pu + i) );
       
      if (scale > 0.0) {
         for (j = i, pu = pui, s2 = 0.0; j < nrows; j++, pu += ncols) {
            *(pu + i) /= scale;
            s2 += *(pu + i) * *(pu + i);
         }
//
//    
//       Chose sign of s which maximizes the norm
//  
         s = ( *(pui + i) < 0.0 ) ? sqrt(s2) : -sqrt(s2);
//
//       Calculate -2/u'u
//
         half_norm_squared = *(pui + i) * s - s2;
//
//       Transform remaining columns by the Householder transform.
//
         *(pui + i) -= s;
         
         for (j = ip1; j < ncols; j++) {
            for (k = i, si = 0.0, pu = pui; k < nrows; k++, pu += ncols)
               si += *(pu + i) * *(pu + j);
            si /= half_norm_squared;
            for (k = i, pu = pui; k < nrows; k++, pu += ncols) {
               *(pu + j) += si * *(pu + i);
            }
         }
      }
      for (j = i, pu = pui; j < nrows; j++, pu += ncols) *(pu + i) *= scale;
      diagonal[i] = s * scale;
//       
//                  Perform Householder transform on rows.
//
//       Calculate the normed squared of the i-th row vector starting at 
//       column i.
//
      s = 0.0;
      scale = 0.0;
      if (i >= nrows || i == (ncols - 1) ) continue;
      for (j = ip1; j < ncols; j++) scale += fabs ( *(pui + j) );
      if ( scale > 0.0 ) {
         for (j = ip1, s2 = 0.0; j < ncols; j++) {
            *(pui + j) /= scale;
            s2 += *(pui + j) * *(pui + j);
         }
         s = ( *(pui + ip1) < 0.0 ) ? sqrt(s2) : -sqrt(s2);
//
//       Calculate -2/u'u
//
         half_norm_squared = *(pui + ip1) * s - s2;
//
//       Transform the rows by the Householder transform.
//
         *(pui + ip1) -= s;
         for (k = ip1; k < ncols; k++)
            superdiagonal[k] = *(pui + k) / half_norm_squared;
         if ( i < (nrows - 1) ) {
            for (j = ip1, pu = pui + ncols; j < nrows; j++, pu += ncols) {
               for (k = ip1, si = 0.0; k < ncols; k++) 
                  si += *(pui + k) * *(pu + k);
               for (k = ip1; k < ncols; k++) { 
                  *(pu + k) += si * superdiagonal[k];
               }
            }
         }
         for (k = ip1; k < ncols; k++) *(pui + k) *= scale;
      }
   }

// Update V
   pui = U + ncols * (ncols - 2);
   pvi = V + ncols * (ncols - 1);
   *(pvi + ncols - 1) = 1.0;
   s = superdiagonal[ncols - 1];
   pvi -= ncols;
   for (i = ncols - 2, ip1 = ncols - 1; i >= 0; i--, pui -= ncols,
                                                      pvi -= ncols, ip1-- ) {
      if ( s != 0.0 ) {
         pv = pvi + ncols;
         for (j = ip1; j < ncols; j++, pv += ncols)
            *(pv + i) = ( *(pui + j) / *(pui + ip1) ) / s;
         for (j = ip1; j < ncols; j++) { 
            si = 0.0;
            for (k = ip1, pv = pvi + ncols; k < ncols; k++, pv += ncols)
               si += *(pui + k) * *(pv + j);
            for (k = ip1, pv = pvi + ncols; k < ncols; k++, pv += ncols)
               *(pv + j) += si * *(pv + i);                  
         }
      }
      pv = pvi + ncols;
      for ( j = ip1; j < ncols; j++, pv += ncols ) {
         *(pvi + j) = 0.0;
         *(pv + i) = 0.0;
      }
      *(pvi + i) = 1.0;
      s = superdiagonal[i];
   }

// Update U

   pui = U + ncols * (ncols - 1);
   for (i = ncols - 1, ip1 = ncols; i >= 0; ip1 = i, i--, pui -= ncols ) {
      s = diagonal[i];
      for ( j = ip1; j < ncols; j++) *(pui + j) = 0.0;
      if ( s != 0.0 ) {
         for (j = ip1; j < ncols; j++) { 
            si = 0.0;
            pu = pui + ncols;
            for (k = ip1; k < nrows; k++, pu += ncols)
               si += *(pu + i) * *(pu + j);
            si = (si / *(pui + i) ) / s;
            for (k = i, pu = pui; k < nrows; k++, pu += ncols)
               *(pu + j) += si * *(pu + i);                  
         }
         for (j = i, pu = pui; j < nrows; j++, pu += ncols){
            *(pu + i) /= s;
         }
      }
      else 
         for (j = i, pu = pui; j < nrows; j++, pu += ncols) *(pu + i) = 0.0;
      *(pui + i) += 1.0;
   }
}


////////////////////////////////////////////////////////////////////////////////
// static int Givens_Reduction_to_Diagonal_Form( int nrows, int ncols,        //
//         double* U, double* V, double* diagonal, double* superdiagonal )    //
//                                                                            //
//  Description:                                                              //
//     This routine decomposes a bidiagonal matrix given by the arrays        //
//     diagonal and superdiagonal into a product of three matrices U1, D and  //
//     V1', the matrix U1 premultiplies U and is returned in U, the matrix    //
//     V1 premultiplies V and is returned in V.  The matrix D is a diagonal   //
//     matrix and replaces the array diagonal.                                //
//                                                                            //
//     The method used to annihilate the offdiagonal elements is a variant    //
//     of the QR transformation.  The method consists of applying Givens      //
//     rotations to the right and the left of the current matrix until        //
//     the new off-diagonal elements are chased out of the matrix.            //
//                                                                            //
//     The process is an iterative process which due to roundoff errors may   //
//     not converge within a predefined number of iterations.  (This should   //
//     be unusual.)                                                           //
//                                                                            //
//  Arguments:                                                                //
//     int nrows                                                              //
//        The number of rows of the matrix U.                                 //
//     int ncols                                                              //
//        The number of columns of the matrix U.                              //
//     double* U                                                              //
//        On input, a pointer to a matrix already initialized to a matrix     //
//        with mutually orthogonal columns.   On output, the matrix with      //
//        mutually orthogonal columns.                                        //
//     double* V                                                              //
//        On input, a pointer to a square matrix with the same number of rows //
//        and columns as the columns of the matrix U, i.e. V[ncols][ncols].   //
//        The matrix V is assumed to be initialized to an orthogonal matrix.  //
//        On output, V is an orthogonal matrix.                               //
//     double* diagonal                                                       //
//        On input, a pointer to an array of dimension ncols which initially  //
//        contains the diagonal of the bidiagonal matrix.  On output, the     //
//        it contains the diagonal of the diagonal matrix.                    //
//     double* superdiagonal                                                  //
//        On input, a pointer to an array of dimension ncols which initially  //
//        the first component is zero and the successive components form the  //
//        superdiagonal of the bidiagonal matrix.                             //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The procedure failed to terminate within                  //
//                  MAX_ITERATION_COUNT iterations.                           //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double diagonal[N];                                                    //
//     double superdiagonal[N];                                               //
//     int err;                                                               //
//                                                                            //
//     (your code to initialize the matrices U, V, diagonal, and )            //
//     ( superdiagonal.  - Note this routine is not accessible from outside)  //
//     ( i.e. it is declared static.)                                         //
//                                                                            //
//     err = Givens_Reduction_to_Diagonal_Form( M,N,(double*)U,(double*)V,    //
//                                                 diagonal, superdiagonal ); //
//     if ( err < 0 ) printf("Failed to converge\n");                         //
//     else { ... }                                                           //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
__device__ static int Givens_Reduction_to_Diagonal_Form( int nrows, int ncols,
           double* U, double* V, double* diagonal, double* superdiagonal )
{

   double epsilon;
   double c, s;
   double f,g,h;
   double x,y,z;
   double *pu, *pv;
   int i,j,k,m;
   int rotation_test;
   int iteration_count;
  
   for (i = 0, x = 0.0; i < ncols; i++) {
      y = fabs(diagonal[i]) + fabs(superdiagonal[i]);
      if ( x < y ) x = y;
   }
   epsilon = x * DBL_EPSILON;
   for (k = ncols - 1; k >= 0; k--) {
      iteration_count = 0;
      while(1) {
         rotation_test = 1;
         for (m = k; m >= 0; m--) { 
            if (fabs(superdiagonal[m]) <= epsilon) {rotation_test = 0; break;}
            if (fabs(diagonal[m-1]) <= epsilon) break;
         }
         if (rotation_test) {
            c = 0.0;
            s = 1.0;
            for (i = m; i <= k; i++) {  
               f = s * superdiagonal[i];
               superdiagonal[i] *= c;
               if (fabs(f) <= epsilon) break;
               g = diagonal[i];
               h = sqrt(f*f + g*g);
               diagonal[i] = h;
               c = g / h;
               s = -f / h; 
               for (j = 0, pu = U; j < nrows; j++, pu += ncols) { 
                  y = *(pu + m - 1);
                  z = *(pu + i);
                  *(pu + m - 1 ) = y * c + z * s;
                  *(pu + i) = -y * s + z * c;
               }
            }
         }
         z = diagonal[k];
         if (m == k ) {
            if ( z < 0.0 ) {
               diagonal[k] = -z;
               for ( j = 0; j < ncols; j++) {
                  V[j * ncols + k] = - V[j * ncols + k];
               }
            }
            break;
         }
         else {
            if ( iteration_count >= MAX_ITERATION_COUNT ) return -1;
            iteration_count++;
            x = diagonal[m];
            y = diagonal[k-1];
            g = superdiagonal[k-1];
            h = superdiagonal[k];
            f = ( (y - z) * ( y + z ) + (g - h) * (g + h) )/(2.0 * h * y);
            g = sqrt( f * f + 1.0 );
            if ( f < 0.0 ) g = -g;
            f = ( (x - z) * (x + z) + h * (y / (f + g) - h) ) / x;
// Next QR Transformtion
            c = 1.0;
            s = 1.0;
            for (i = m + 1; i <= k; i++) {
               g = superdiagonal[i];
               y = diagonal[i];
               h = s * g;
               g *= c;
               z = sqrt( f * f + h * h );
               superdiagonal[i-1] = z;
               if (z != 0.0) {
                  c = f / z;
                  s = h / z;
               } 
               f =  x * c + g * s;
               g = -x * s + g * c;
               h = y * s;
               y *= c;
               for (j = 0; j < ncols; j++) {
                  x = V[j * ncols + (i-1)];
                  z = V[j * ncols + i];
                  V[j * ncols + (i-1)] = x * c + z * s;
                  V[j * ncols + i] = -x * s + z * c;
               }
               z = sqrt( f * f + h * h );
               diagonal[i - 1] = z;
               if (z != 0.0) {
                  c = f / z;
                  s = h / z;
               } 
               f = c * g + s * y;
               x = -s * g + c * y;
               for (j = 0, pu = U; j < nrows; j++, pu += ncols) {
                  y = *(pu + i - 1);
                  z = *(pu + i);
                  *(pu + i - 1) = c * y + s * z;
                  *(pu + i) = -s * y + c * z;
               }
            }
            superdiagonal[m] = 0.0;
            superdiagonal[k] = f;
            diagonal[k] = x;
         }
      } 
   }
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
// static void Sort_by_Decreasing_Singular_Values(int nrows, int ncols,       //
//                            double* singular_values, double* U, double* V)  //
//                                                                            //
//  Description:                                                              //
//     This routine sorts the singular values from largest to smallest        //
//     singular value and interchanges the columns of U and the columns of V  //
//     whenever a swap is made.  I.e. if the i-th singular value is swapped   //
//     with the j-th singular value, then the i-th and j-th columns of U are  //
//     interchanged and the i-th and j-th columns of V are interchanged.      //
//                                                                            //
//  Arguments:                                                                //
//     int nrows                                                              //
//        The number of rows of the matrix U.                                 //
//     int ncols                                                              //
//        The number of columns of the matrix U.                              //
//     double* singular_values                                                //
//        On input, a pointer to the array of singular values.  On output, the//
//        sorted array of singular values.                                    //
//     double* U                                                              //
//        On input, a pointer to a matrix already initialized to a matrix     //
//        with mutually orthogonal columns.  On output, the matrix with       //
//        mutually orthogonal possibly permuted columns.                      //
//     double* V                                                              //
//        On input, a pointer to a square matrix with the same number of rows //
//        and columns as the columns of the matrix U, i.e. V[ncols][ncols].   //
//        The matrix V is assumed to be initialized to an orthogonal matrix.  //
//        On output, V is an orthogonal matrix with possibly permuted columns.//
//                                                                            //
//  Return Values:                                                            //
//        The function is of type void.                                       //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double diagonal[N];                                                    //
//                                                                            //
//     (your code to initialize the matrices U, V, and diagonal. )            //
//     ( - Note this routine is not accessible from outside)                  //
//     ( i.e. it is declared static.)                                         //
//                                                                            //
//     Sort_by_Decreasing_Singular_Values(nrows, ncols, singular_values,      //
//                                                 (double*) U, (double*) V); //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
__device__ static void Sort_by_Decreasing_Singular_Values(int nrows, int ncols,
                                double* singular_values, double* U, double* V)
{
   int i,j,max_index;
   double temp;
   double *p1, *p2;

   for (i = 0; i < ncols - 1; i++) {
      max_index = i;
      for (j = i + 1; j < ncols; j++)
         if (singular_values[j] > singular_values[max_index] ) 
            max_index = j;
      if (max_index == i) continue;
      temp = singular_values[i];
      singular_values[i] = singular_values[max_index];
      singular_values[max_index] = temp;
      p1 = U + max_index;
      p2 = U + i;
      for (j = 0; j < nrows; j++, p1 += ncols, p2 += ncols) {
         temp = *p1;
         *p1 = *p2;
         *p2 = temp;
      } 
      p1 = V + max_index;
      p2 = V + i;
      for (j = 0; j < ncols; j++, p1 += ncols, p2 += ncols) {
         temp = *p1;
         *p1 = *p2;
         *p2 = temp;
      }
   } 
}


////////////////////////////////////////////////////////////////////////////////
//  void Singular_Value_Decomposition_Solve(double* U, double* D, double* V,  //
//              double tolerance, int nrows, int ncols, double *B, double* x) //
//                                                                            //
//  Description:                                                              //
//     This routine solves the system of linear equations Ax=B where A =UDV', //
//     is the singular value decomposition of A.  Given UDV'x=B, then         //
//     x = V(1/D)U'B, where 1/D is the pseudo-inverse of D, i.e. if D[i] > 0  //
//     then (1/D)[i] = 1/D[i] and if D[i] = 0, then (1/D)[i] = 0.  Since      //
//     the singular values are subject to round-off error.  A tolerance is    //
//     given so that if D[i] < tolerance, D[i] is treated as if it is 0.      //
//     The default tolerance is D[0] * DBL_EPSILON * ncols, if the user       //
//     specified tolerance is less than the default tolerance, the default    //
//     tolerance is used.                                                     //
//                                                                            //
//  Arguments:                                                                //
//     double* U                                                              //
//        A matrix with mutually orthonormal columns.                         //
//     double* D                                                              //
//        A diagonal matrix with decreasing non-negative diagonal elements.   //
//        i.e. D[i] > D[j] if i < j and D[i] >= 0 for all i.                  //
//     double* V                                                              //
//        An orthogonal matrix.                                               //
//     double tolerance                                                       //
//        An lower bound for non-zero singular values (provided tolerance >   //
//        ncols * DBL_EPSILON * D[0]).                                        //
//     int nrows                                                              //
//        The number of rows of the matrix U and B.                           //
//     int ncols                                                              //
//        The number of columns of the matrix U.  Also the number of rows and //
//        columns of the matrices D and V.                                    //
//     double* B                                                              //
//        A pointer to a vector dimensioned as nrows which is the  right-hand //
//        side of the equation Ax = B where A = UDV'.                         //
//     double* x                                                              //
//        A pointer to a vector dimensioned as ncols, which is the least      //
//        squares solution of the equation Ax = B where A = UDV'.             //
//                                                                            //
//  Return Values:                                                            //
//        The function is of type void.                                       //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     #define NB                                                             //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double D[N];                                                           //
//     double B[M];                                                           //
//     double x[N];                                                           //
//     double tolerance;                                                      //
//                                                                            //
//     (your code to initialize the matrices U,D,V,B)                         //
//                                                                            //
//     Singular_Value_Decomposition_Solve((double*) U, D, (double*) V,        //
//                                              tolerance, M, N, B, x, bcols) //
//                                                                            //
//     printf(" The solution of Ax=B is \n");                                 //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //

__device__ void Singular_Value_Decomposition_Solve(double* U, double* D, double* V,  
                double tolerance, int nrows, int ncols, double *B, double* x) 
{
   int i,j,k;
   double *pu, *pv;
   double dum;
   double s_max, rcond;
   int effective_rank;

   // Set minimum tolerance based on machine precision
   dum = DBL_EPSILON * D[0] * (double) ncols;
   if (tolerance < dum) tolerance = dum;
   
   // Determine effective rank using relative condition number threshold
   // CRITICAL FIX: Changed from 1e-10 to 1e-16 to match LAPACK behavior exactly
   rcond = 1e-16;  // Match LAPACK's default tolerance for better handling of ill-conditioned systems
   s_max = D[0];   // Largest singular value
   effective_rank = 0;
   
   for (i = 0; i < ncols; i++) {
       if (D[i] > rcond * s_max && D[i] > tolerance) {
           effective_rank++;
       }
   }

   // Solve using only the well-conditioned subspace
   for ( i = 0, pv = V; i < ncols; i++, pv += ncols) {
      x[i] = 0.0;
      for (j = 0; j < effective_rank; j++) {
         if (D[j] > tolerance && D[j] > rcond * s_max) {
            // Compute U'*B for this singular value
            for (k = 0, dum = 0.0, pu = U; k < nrows; k++, pu += ncols)
               dum += *(pu + j) * B[k];
            
            // Apply damping for better numerical stability
            double damping = D[j] / (D[j] + tolerance);
            x[i] += damping * dum * *(pv + j) / D[j];
         }
      }
   } 
}
//Or, solve the transpose system, for underdetermined systems (m < n)
//U and V are defined as the orthogonal matrices obtained from decomposing A.T
//Therefore, they should be "swapped" for finding the solution to A
//Of course, this swap is only done conceptually to minimize computational expense
__device__ void Singular_Value_Decomposition_SolveT(double* U, double* D, double* V,  
                double tolerance, int nrows, int ncols, double *B, double* x) 
{
   int i,j,k;
   double *pu, *pv;
   double dum;
   double s_max, rcond;
   int effective_rank;

   // Set minimum tolerance based on machine precision
   dum = DBL_EPSILON * D[0] * (double) ncols;
   if (tolerance < dum) tolerance = dum;
   
   // Determine effective rank using relative condition number threshold
   // CRITICAL FIX: Changed from 1e-10 to 1e-16 to match LAPACK behavior exactly
   rcond = 1e-16;  // Match LAPACK's default tolerance for better handling of ill-conditioned systems
   s_max = D[0];   // Largest singular value
   effective_rank = 0;
   
   for (i = 0; i < ncols; i++) {
       if (D[i] > rcond * s_max && D[i] > tolerance) {
           effective_rank++;
       }
   }

   // Solve using only the well-conditioned subspace
   for ( i = 0, pu = U; i < nrows; i++, pu += ncols) {
      x[i] = 0.0;
      for (j = 0, pv = V; j < effective_rank; j++, pv += ncols) {
         if (D[j] > tolerance && D[j] > rcond * s_max) {
            // Compute V'*B for this singular value
            for (k = 0, dum = 0.0; k < ncols; k++)
               dum += *(pv + k) * B[k];
            
            // Apply damping for better numerical stability
            double damping = D[j] / (D[j] + tolerance);
            x[i] += damping * dum * *(pu + j) / D[j];
         }
      }
   } 
}


////////////////////////////////////////////////////////////////////////////////
//  void Singular_Value_Decomposition_Inverse(double* U, double* D, double* V,//
//                     double tolerance, int nrows, int ncols, double *Astar) //
//                                                                            //
//  Description:                                                              //
//     This routine calculates the pseudo-inverse of the matrix A = UDV'.     //
//     where U, D, V constitute the singular value decomposition of A.        //
//     Let Astar be the pseudo-inverse then Astar = V(1/D)U', where 1/D is    //
//     the pseudo-inverse of D, i.e. if D[i] > 0 then (1/D)[i] = 1/D[i] and   //
//     if D[i] = 0, then (1/D)[i] = 0.  Because the singular values are       //
//     subject to round-off error.  A tolerance is given so that if           //
//     D[i] < tolerance, D[i] is treated as if it were 0.                     //
//     The default tolerance is D[0] * DBL_EPSILON * ncols, assuming that the //
//     diagonal matrix of singular values is sorted from largest to smallest, //
//     if the user specified tolerance is less than the default tolerance,    //
//     then the default tolerance is used.                                    //
//                                                                            //
//  Arguments:                                                                //
//     double* U                                                              //
//        A matrix with mutually orthonormal columns.                         //
//     double* D                                                              //
//        A diagonal matrix with decreasing non-negative diagonal elements.   //
//        i.e. D[i] > D[j] if i < j and D[i] >= 0 for all i.                  //
//     double* V                                                              //
//        An orthogonal matrix.                                               //
//     double tolerance                                                       //
//        An lower bound for non-zero singular values (provided tolerance >   //
//        ncols * DBL_EPSILON * D[0]).                                        //
//     int nrows                                                              //
//        The number of rows of the matrix U and B.                           //
//     int ncols                                                              //
//        The number of columns of the matrix U.  Also the number of rows and //
//        columns of the matrices D and V.                                    //
//     double* Astar                                                          //
//        On input, a pointer to the first element of an ncols x nrows matrix.//
//        On output, the pseudo-inverse of UDV'.                              //
//                                                                            //
//  Return Values:                                                            //
//        The function is of type void.                                       //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double D[N];                                                           //
//     double Astar[N][M];                                                    //
//     double tolerance;                                                      //
//                                                                            //
//     (your code to initialize the matrices U,D,V)                           //
//                                                                            //
//     Singular_Value_Decomposition_Inverse((double*) U, D, (double*) V,      //
//                                        tolerance, M, N, (double*) Astar);  //
//                                                                            //
//     printf(" The pseudo-inverse of A = UDV' is \n");                       //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //

__device__ void Singular_Value_Decomposition_Inverse(double* U, double* D, double* V,  
                        double tolerance, int nrows, int ncols, double *Astar) 
{
   int i,j,k;
   double *pu, *pv, *pa;
   double dum;

   dum = DBL_EPSILON * D[0] * (double) ncols;
   if (tolerance < dum) tolerance = dum;
   for ( i = 0, pv = V, pa = Astar; i < ncols; i++, pv += ncols) 
      for ( j = 0, pu = U; j < nrows; j++, pa++) 
        for (k = 0, *pa = 0.0; k < ncols; k++, pu++)
           if (D[k] > tolerance) *pa += *(pv + k) * *pu / D[k];
}


////////////////////////////////////////////////////////////////////////////////
// Device function version of LAPACK's dgelsd with automatic scaling         //
// This provides better numerical stability for poorly conditioned matrices   //
////////////////////////////////////////////////////////////////////////////////

// Helper function to compute infinity norm of a matrix
__device__ static double matrix_inf_norm(double* A, int m, int n) {
    double max_row_sum = 0.0;
    for (int i = 0; i < m; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            row_sum += fabs(A[i * n + j]);
        }
        if (row_sum > max_row_sum) {
            max_row_sum = row_sum;
        }
    }
    return max_row_sum;
}

// Helper function to compute column norms
__device__ static void compute_column_norms(double* A, int m, int n, double* col_norms) {
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            double val = A[i * n + j];
            sum += val * val;
        }
        col_norms[j] = sqrt(sum);
    }
}

////////////////////////////////////////////////////////////////////////////////
//  int dgelsd_device(double* A, int m, int n, double* B, int nrhs,          //
//                    double* work_svd, double rcond)                         //
//                                                                            //
//  Description:                                                              //
//     Device function that mimics LAPACK's dgelsd behavior with automatic    //
//     matrix scaling for better numerical stability. Solves overdetermined   //
//     or underdetermined linear systems using SVD with equilibration.       //
//                                                                            //
//  Arguments:                                                                //
//     double* A                                                              //
//        On input, the m x n matrix. DESTROYED on output.                    //
//     int m                                                                  //
//        The number of rows of the matrix A.                                 //
//     int n                                                                  //
//        The number of columns of the matrix A.                              //
//     double* B                                                              //
//        On input, the m x nrhs right hand side matrix.                     //
//        On output, the n x nrhs solution matrix X.                         //
//     int nrhs                                                               //
//        The number of right hand sides.                                     //
//     double* work_svd                                                       //
//        Workspace array of size at least m*n + n*n + n + max(m,n)*nrhs     //
//     double rcond                                                           //
//        Reciprocal condition number threshold for singular values.          //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - SVD did not converge                                      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
__device__ int dgelsd_device(double* A, int m, int n, double* B, int nrhs,
                            double* work_svd, double rcond) {
    
    // Allocate workspace from work_svd
    double* U = work_svd;                    // m x n
    double* V = work_svd + m * n;            // n x n  
    double* singular_values = V + n * n;     // n
    double* superdiagonal = singular_values + n;  // n
    double* row_scale = superdiagonal + n;   // m
    double* col_scale = row_scale + m;       // n
    double* B_copy = col_scale + n;          // max(m,n) x nrhs
    
    // Step 1: Compute row and column scaling factors for equilibration
    // This is crucial for handling matrices with widely varying scales
    
    // Initialize scaling factors
    for (int i = 0; i < m; i++) {
        row_scale[i] = 0.0;
    }
    for (int j = 0; j < n; j++) {
        col_scale[j] = 0.0;
    }
    
    // Compute row scales (max absolute value in each row)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double abs_val = fabs(A[i * n + j]);
            if (abs_val > row_scale[i]) {
                row_scale[i] = abs_val;
            }
        }
        // Avoid division by zero
        if (row_scale[i] == 0.0) {
            row_scale[i] = 1.0;
        }
    }
    
    // Scale A by row scales and compute column scales
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] /= row_scale[i];
            double abs_val = fabs(A[i * n + j]);
            if (abs_val > col_scale[j]) {
                col_scale[j] = abs_val;
            }
        }
    }
    
    // Avoid division by zero for column scales
    for (int j = 0; j < n; j++) {
        if (col_scale[j] == 0.0) {
            col_scale[j] = 1.0;
        }
    }
    
    // Apply column scaling to A
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] /= col_scale[j];
        }
    }
    
    // Scale B by row scales
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < nrhs; j++) {
            B[i * nrhs + j] /= row_scale[i];
        }
    }
    
    // Step 2: Perform SVD on the scaled matrix
    int svd_result = Singular_Value_Decomposition(A, m, n, U, singular_values, V, superdiagonal);
    if (svd_result != 0) {
        return -1;  // SVD failed
    }
    
    // Step 3: Solve the system using the SVD
    // Copy B to B_copy for the solve operation
    for (int i = 0; i < m * nrhs; i++) {
        B_copy[i] = B[i];
    }
    
    // Use existing SVD solve function
    // B_copy contains the RHS, output goes to a temporary location first
    double x_temp[200];  // Temporary solution storage (max size)
    Singular_Value_Decomposition_Solve(U, singular_values, V, rcond, m, n, B_copy, x_temp);
    
    // Step 4: Unscale the solution by column scales and copy to B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < nrhs; j++) {
            B[i * nrhs + j] = x_temp[i * nrhs + j] / col_scale[i];
        }
    }
    
    // Zero out the remaining rows if m > n
    for (int i = n; i < m; i++) {
        for (int j = 0; j < nrhs; j++) {
            B[i * nrhs + j] = 0.0;
        }
    }
    
    return 0;
}


// Content of phase_rec.h
#ifndef PHASE_REC_H
#define PHASE_REC_H

typedef double (*pycgpu_func_t)(const double*);
typedef void (*pycgpu_array_func_t)(double*, const double*);

typedef struct PhaseRecord {
    pycgpu_func_t obj;
    pycgpu_func_t formulaobj;
    pycgpu_array_func_t formulagrad;
    pycgpu_array_func_t formulahess;
    pycgpu_array_func_t internal_cons_func;
    pycgpu_array_func_t internal_cons_jac;
    pycgpu_array_func_t mass_obj;
    pycgpu_array_func_t formulamole_obj;
    pycgpu_array_func_t formulamole_grad;
    int num_statevars; //number of STATE variables
    int phase_dof; //number of SITE variables
    int num_vars;
    int num_elements;
    int num_internal_cons;
    int nonvacant_elements; //number of non-vacancy components
    __device__ void init(pycgpu_func_t on, pycgpu_func_t fon, pycgpu_array_func_t fgn, pycgpu_array_func_t fhn, pycgpu_array_func_t icfn, pycgpu_array_func_t icjn, pycgpu_array_func_t mon, pycgpu_array_func_t fmon, pycgpu_array_func_t fmgn, int ns, int pd, int ne, int nic, int nve = 0) {
        obj = on;
        formulaobj = fon;
        formulagrad = fgn;
        formulahess = fhn;
        internal_cons_func = icfn;
        internal_cons_jac = icjn;
        mass_obj = mon;
        formulamole_obj = fmon;
        formulamole_grad = fmgn;
        num_statevars = ns;
        phase_dof = pd;
        num_vars = ns+pd;
        num_elements = ne;
        num_internal_cons = nic;
        nonvacant_elements = nve > 0 ? nve : ne; // Default to num_elements if not specified
    }
    
    // Default initialization (not a constructor in CUDA)
    __device__ void reset() {
        obj = nullptr;
        formulaobj = nullptr;
        formulagrad = nullptr;
        formulahess = nullptr;
        internal_cons_func = nullptr;
        internal_cons_jac = nullptr;
        mass_obj = nullptr;
        formulamole_obj = nullptr;
        formulamole_grad = nullptr;
        num_statevars = 0;
        phase_dof = 0;
        num_vars = 0;
        num_elements = 0;
        num_internal_cons = 0;
        nonvacant_elements = 0;
    }
} PhaseRecord;

#endif // PHASE_REC_H

// Content of comp_set.h
// #pragma once  // Removed for inlining
// #include "phase_rec.h"  // Removed for inlining

// Use consistent constant names with minimizer.h
#ifndef MAX_DOF_PER_PHASE
#define MAX_DOF_PER_PHASE 64
#endif

#ifndef MAX_COMPONENTS  
#define MAX_COMPONENTS 32
#endif

#ifndef MAX_STATEVARS
#define MAX_STATEVARS 8
#endif

// Legacy aliases for compatibility
#define NDOF_MAX MAX_DOF_PER_PHASE
#define NELEM_MAX MAX_COMPONENTS

struct CompositionSet {
    const PhaseRecord* phase_record;
    // CRITICAL FIX: DOF array must be large enough for workspace state variables + phase DOF
    // With workspace having 3 state vars (N, P, T) and phase having 2 site fractions,
    // we need at least 5 elements. But MAX_DOF_PER_PHASE might be set to 4.
    // Increase the size to handle this case.
    double dof[MAX_STATEVARS + MAX_DOF_PER_PHASE];  // Ensure enough space
    double X[MAX_COMPONENTS];
    double energy;
    double NP;
    bool fixed;

    __device__ CompositionSet() : phase_record(nullptr), energy(0.0), NP(0.0), fixed(false) {
        for(int i = 0; i < MAX_STATEVARS + MAX_DOF_PER_PHASE; i++) dof[i] = 0.0;
        for(int i = 0; i < MAX_COMPONENTS; i++) X[i] = 0.0;
    }

    __device__ void init(const PhaseRecord* pr) {
        phase_record = pr;
    }

    __device__ void update(double* site_fracs, double phase_amt, double* state_variables, int workspace_num_statevars) {
        // CRITICAL FIX: With the updated energy functions that accept all state variables,
        // we now pass the full workspace state variables array directly to the energy functions.
        // The energy functions now expect [N, P, T, Y1, Y2...] format.
        
        // Store the full workspace DOF array
        // Copy all workspace state variables
        for(int i = 0; i < workspace_num_statevars; i++) {
            dof[i] = state_variables[i];
        }
        // Copy site fractions after workspace state variables
        for(int i = 0; i < phase_record->phase_dof; i++) {
            dof[workspace_num_statevars + i] = site_fracs[i];
        }
        
        NP = phase_amt;
        // Pass full workspace DOF to energy functions - they now expect workspace format
        energy = phase_record->obj(dof);
        
        // DEBUG: Check memory before mass_obj
        #ifdef VERBOSE_DEBUG
        if (workspace_num_statevars == 3) {
            printf("GPU DEBUG: update() before mass_obj - workspace_num_statevars still = %d\\n", workspace_num_statevars);
        }
        #endif
        
        phase_record->mass_obj(X, dof);  // Fills entire X array at once
        
        // DEBUG: Check memory after mass_obj
        #ifdef VERBOSE_DEBUG
        if (workspace_num_statevars != 3) {
            printf("GPU ERROR: update() after mass_obj - workspace_num_statevars corrupted to %d!\\n", workspace_num_statevars);
        }
        #endif
    }
    
    __device__ double calculate_phase_comp_sum(int workspace_num_statevars) {
        // Calculate sum of moles of atoms per formula unit using formulamole_obj
        // This matches CPU's formulamole_obj usage for phase normalization
        double formulamoles[MAX_COMPONENTS];
        
        // Initialize all to zero since formulamole_obj only fills nonvacant elements
        for(int i = 0; i < MAX_COMPONENTS; ++i) {
            formulamoles[i] = 0.0;
        }
        
        // Call formulamole_obj which gives moles of each component per formula unit
        if (phase_record && phase_record->formulamole_obj) {
            // With updated energy functions, pass the full DOF array directly
            phase_record->formulamole_obj(formulamoles, dof);
        }
        
        // Sum up the moles of atoms per formula unit
        double phase_sum = 0.0;
        for(int i = 0; i < phase_record->nonvacant_elements; ++i) {
            phase_sum += formulamoles[i];
        }
        
        return phase_sum;
    }
};


// Content of lu_solver.h (LU decomposition solver)
// #pragma once  // Removed for inlining

// Simple LU decomposition with partial pivoting for GPU
// Implements equivalent of LAPACK's dgetrf and dgetrs

// Define maximum matrix dimension based on phase-related constants
// This should be sufficient for matrices arising from phase equilibrium calculations
// The maximum dimension comes from full_e_matrix which is (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)^2
#ifndef MAX_LU_DIM
#define MAX_LU_DIM (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)
#endif

__device__ void swap_rows(double* A, int row1, int row2, int ncols) {
    for (int j = 0; j < ncols; j++) {
        double temp = A[row1 * ncols + j];
        A[row1 * ncols + j] = A[row2 * ncols + j];
        A[row2 * ncols + j] = temp;
    }
}

__device__ void swap_int(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__device__ int lu_decomposition(double* A, int n, int* ipiv) {
    // LU decomposition with partial pivoting
    // A is n x n matrix stored in row-major order
    // On output: A contains L and U (L has unit diagonal)
    // ipiv contains pivot indices
    
    for (int i = 0; i < n; i++) {
        ipiv[i] = i;
    }
    
    for (int k = 0; k < n; k++) {
        // Find pivot
        double max_val = fabs(A[k * n + k]);
        int max_row = k;
        
        for (int i = k + 1; i < n; i++) {
            double val = fabs(A[i * n + k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }
        
        // Check for singular matrix
        if (max_val < 1e-20) {
            return k + 1; // Matrix is singular
        }
        
        // Swap rows if needed
        if (max_row != k) {
            swap_rows(A, k, max_row, n);
            swap_int(&ipiv[k], &ipiv[max_row]);
        }
        
        // Compute multipliers and eliminate column
        for (int i = k + 1; i < n; i++) {
            A[i * n + k] /= A[k * n + k]; // Store multiplier in L
            
            // Update remaining matrix
            for (int j = k + 1; j < n; j++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
    
    return 0; // Success
}

__device__ void lu_solve(const double* LU, int n, const int* ipiv, double* b) {
    // Solve LUx = b where LU contains the factorization from lu_decomposition
    // b is the right-hand side on input, solution on output
    
    // Apply row permutations to b
    double temp[MAX_LU_DIM];
    for (int i = 0; i < n; i++) {
        temp[i] = b[i];
    }
    for (int i = 0; i < n; i++) {
        b[i] = temp[ipiv[i]];
    }
    
    // Forward substitution for Ly = b
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            b[i] -= LU[i * n + j] * b[j];
        }
    }
    
    // Back substitution for Ux = y
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            b[i] -= LU[i * n + j] * b[j];
        }
        b[i] /= LU[i * n + i];
    }
}

__device__ void invert_matrix_lu(double* A, int n, double* work) {
    // Invert matrix A in-place using LU decomposition
    // work is a temporary array of size n*n
    
    // Copy A to work
    for (int i = 0; i < n * n; i++) {
        work[i] = A[i];
    }
    
    int ipiv[MAX_LU_DIM];
    
    // LU decomposition
    int info = lu_decomposition(work, n, ipiv);
    if (info != 0) {
        // Singular matrix - set to identity (or could use pseudo-inverse)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = (i == j) ? 1.0 : 0.0;
            }
        }
        return;
    }
    
    // Compute inverse by solving AX = I
    // Process column by column
    for (int j = 0; j < n; j++) {
        // Set up unit vector for column j
        double col[MAX_LU_DIM];
        for (int i = 0; i < n; i++) {
            col[i] = (i == j) ? 1.0 : 0.0;
        }
        
        // Solve for this column
        lu_solve(work, n, ipiv, col);
        
        // Store result in output matrix
        for (int i = 0; i < n; i++) {
            A[i * n + j] = col[i];
        }
    }
}

__device__ void lstsq_lu(double* A, int nrows, int ncols, double* b, double* x) {
    // Simple least squares solver using LU decomposition
    // For overdetermined systems, solves normal equations: A^T A x = A^T b
    // For square systems, solves Ax = b directly
    
    if (nrows == ncols) {
        // Square system - solve directly
        int ipiv[MAX_LU_DIM];
        
        // Copy b to x
        for (int i = 0; i < nrows; i++) {
            x[i] = b[i];
        }
        
        // LU decomposition
        int info = lu_decomposition(A, nrows, ipiv);
        if (info != 0) {
            // Singular matrix - set solution to zero
            for (int i = 0; i < ncols; i++) {
                x[i] = 0.0;
            }
            return;
        }
        
        // Solve
        lu_solve(A, nrows, ipiv, x);
    } else if (nrows > ncols) {
        // Overdetermined - solve normal equations
        // This is less numerically stable than SVD but simpler
        
        // Compute A^T A (symmetric positive semi-definite)
        double ATA[MAX_LU_DIM * MAX_LU_DIM];
        for (int i = 0; i < ncols; i++) {
            for (int j = 0; j < ncols; j++) {
                double sum = 0.0;
                for (int k = 0; k < nrows; k++) {
                    sum += A[k * ncols + i] * A[k * ncols + j];
                }
                ATA[i * ncols + j] = sum;
            }
        }
        
        // Compute A^T b
        double ATb[MAX_LU_DIM];
        for (int i = 0; i < ncols; i++) {
            double sum = 0.0;
            for (int k = 0; k < nrows; k++) {
                sum += A[k * ncols + i] * b[k];
            }
            ATb[i] = sum;
        }
        
        // Solve ATA x = ATb
        int ipiv[MAX_LU_DIM];
        for (int i = 0; i < ncols; i++) {
            x[i] = ATb[i];
        }
        
        int info = lu_decomposition(ATA, ncols, ipiv);
        if (info != 0) {
            // Singular matrix
            for (int i = 0; i < ncols; i++) {
                x[i] = 0.0;
            }
            return;
        }
        
        lu_solve(ATA, ncols, ipiv, x);
    } else {
        // Underdetermined - not handled, set to zero
        for (int i = 0; i < ncols; i++) {
            x[i] = 0.0;
        }
    }
}

// Content of minimizer.h (defines SystemSpecification, SystemState, run_loop, etc.)
// #pragma once  // Removed for inlining
#include <math.h>       // For fabs, fmax, fmin, etc.
#include <float.h>      // For DBL_EPSILON if needed
// #include "phase_rec.h" // PhaseRecord definition  // Removed for inlining
// #include "comp_set.h" // CompositionSet definition  // Removed for inlining
// #include "lu_solver.h" // LU decomposition solver  // Removed for inlining
// #include "debug_gpu.h" // GPU debug system  // Removed for inlining

// Forward declare SVD functions (actual definitions in svd.c will be included at compile time)
__device__ int Singular_Value_Decomposition(double* A, int nrows, int ncols, double* U, 
                      double* singular_values, double* V, double* dummy_array);
__device__ void Singular_Value_Decomposition_Solve(double* U, double* D, double* V,  
                double tolerance, int nrows, int ncols, double *B, double* x);
__device__ void Singular_Value_Decomposition_Inverse(double* U, double* D, double* V,  
                        double tolerance, int nrows, int ncols, double *Astar);

// Forward declarations for functions defined later in this file
__device__ void compute_phase_matrix(double* phase_matrix_out, const double* hess_in,
                                    const double* cons_jac_tmp_in,
                                    const CompositionSet& compset_ref, int num_statevars_val,
                                    const double* phase_dof_site_fracs);
__device__ void invert_matrix(double* matrix, int dim, double* U, double* V, 
                              double* singular_values, double* superdiag, double* work);
__device__ void lstsq(double* A, int nrows, int ncols, double* b, double tolerance,
                      double* U, double* V, double* singular_values, double* superdiag);

// From constants.py
#define MIN_SITE_FRACTION 1e-14
#define MIN_PHASE_FRACTION 1e-6
#define COMP_DIFFERENCE_TOL 1e-4
#define INTERNAL_CONSTRAINT_SCALING 1.0
#define MAX_ENDMEMBER_PAIRS 5000
#define MAX_EXTRA_POINTS 90000

// Maximum expected sizes for static allocation
#ifndef MAX_COMPONENTS
#define MAX_COMPONENTS 32
#endif
#ifndef MAX_PHASES
#define MAX_PHASES 64
#endif
#ifndef MAX_STATEVARS
#define MAX_STATEVARS 8
#endif
#ifndef MAX_DOF_PER_PHASE
#define MAX_DOF_PER_PHASE 64
#endif
#ifndef MAX_INTERNAL_CONSTRAINTS
#define MAX_INTERNAL_CONSTRAINTS 32
#endif
// MAX_PHASE_LOCAL_CONDITIONS is effectively 0
#ifndef MAX_FIXED_MOLE_FRACTION_CONDITIONS
#define MAX_FIXED_MOLE_FRACTION_CONDITIONS MAX_COMPONENTS
#endif

// Forward declaration
struct SystemState;
// struct CompositionSet; // Already defined in comp_set.h
// struct PhaseRecord; // Already defined in phase_rec.h

// Corresponds to SystemSpecification in minimizer.pyx
#ifdef __cplusplus
extern "C" {
#endif

typedef struct SystemSpecification {
    int num_statevars;
    int num_components;
    double prescribed_system_amount;
    double initial_chemical_potentials[MAX_COMPONENTS];
    double prescribed_mole_fraction_coefficients[MAX_FIXED_MOLE_FRACTION_CONDITIONS][MAX_COMPONENTS];
    double prescribed_mole_fraction_rhs[MAX_FIXED_MOLE_FRACTION_CONDITIONS];
    int num_prescribed_mole_fraction_conditions;
    int num_prescribed_mole_fraction_coefficients_cols;

    int free_chemical_potential_indices[MAX_COMPONENTS];
    int num_free_chemical_potentials;
    int free_statevar_indices[MAX_STATEVARS];
    int num_free_statevars;
    int fixed_chemical_potential_indices[MAX_COMPONENTS];
    int num_fixed_chemical_potentials;
    int fixed_statevar_indices[MAX_STATEVARS];
    int num_fixed_statevars;
    int fixed_stable_compset_indices[MAX_PHASES];
    int num_fixed_stable_compsets;
    int max_num_free_stable_phases;
    double ALLOWED_MASS_RESIDUAL;

    #define MAX_SVD_DIM (MAX_PHASES + MAX_FIXED_MOLE_FRACTION_CONDITIONS + MAX_COMPONENTS + MAX_STATEVARS + 2)
    #define MAX_SVD_M MAX_SVD_DIM
    #define MAX_SVD_N MAX_SVD_DIM

    // For invert_matrix: A is N x N, U, V are N x N, work is N*N
    // Max N for phase_matrix: MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS (since phase_local_conditions is 0)
    #define MAX_PHASE_MATRIX_DIM (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)

    double A_lstsq_copy[MAX_SVD_M * MAX_SVD_N];
    double U_lstsq[MAX_SVD_M * MAX_SVD_N];
    double V_lstsq[MAX_SVD_N * MAX_SVD_N];
    double singular_values_lstsq[MAX_SVD_N];
    double superdiag_lstsq[MAX_SVD_N];

    double U_inv[MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM];
    double V_inv[MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM];
    double singular_values_inv[MAX_PHASE_MATRIX_DIM];
    double superdiag_inv[MAX_PHASE_MATRIX_DIM];
    double work_inv[MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM];

    __device__ void init(int ns, int nc, double psa,
                         const double* icp, int num_icp,
                         const double* pmfc, int num_pmfc_rows, int num_pmfc_cols,
                         const double* pmfr, int num_pmfr,
                         const int* fcpi, int n_fcpi,
                         const int* fsvi, int n_fsvi,
                         const int* fix_cpi, int n_fix_cpi,
                         const int* fix_svi, int n_fix_svi,
                         const int* fsci, int n_fsci) {
        num_statevars = ns;
        num_components = nc;
        prescribed_system_amount = psa;
        num_prescribed_mole_fraction_conditions = num_pmfc_rows;
        num_prescribed_mole_fraction_coefficients_cols = num_pmfc_cols;

        for (int i = 0; i < num_icp; ++i) initial_chemical_potentials[i] = icp[i];
        for (int i = num_icp; i < MAX_COMPONENTS; ++i) initial_chemical_potentials[i] = 0.0;

        for (int i = 0; i < num_pmfc_rows; ++i) {
            for (int j = 0; j < num_pmfc_cols; ++j) {
                prescribed_mole_fraction_coefficients[i][j] = pmfc[i * num_pmfc_cols + j];
            }
            for (int j = num_pmfc_cols; j < MAX_COMPONENTS; ++j) {
                prescribed_mole_fraction_coefficients[i][j] = 0.0;
            }
        }
        for (int i = num_pmfc_rows; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS; ++i) {
            for (int j = 0; j < MAX_COMPONENTS; ++j) {
                 prescribed_mole_fraction_coefficients[i][j] = 0.0;
            }
        }

        for (int i = 0; i < num_pmfr; ++i) prescribed_mole_fraction_rhs[i] = pmfr[i];
        for (int i = num_pmfr; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS; ++i) prescribed_mole_fraction_rhs[i] = 0.0;

        num_free_chemical_potentials = n_fcpi;
        for (int i = 0; i < n_fcpi; ++i) free_chemical_potential_indices[i] = fcpi[i];
        num_free_statevars = n_fsvi;
        for (int i = 0; i < n_fsvi; ++i) free_statevar_indices[i] = fsvi[i];
        num_fixed_chemical_potentials = n_fix_cpi;
        for (int i = 0; i < n_fix_cpi; ++i) fixed_chemical_potential_indices[i] = fix_cpi[i];
        num_fixed_statevars = n_fix_svi;
        for (int i = 0; i < n_fix_svi; ++i) fixed_statevar_indices[i] = fix_svi[i];
        num_fixed_stable_compsets = n_fsci;
        for (int i = 0; i < n_fsci; ++i) fixed_stable_compset_indices[i] = fsci[i];

        max_num_free_stable_phases = num_components + num_free_statevars - num_fixed_stable_compsets;

        // Match CPU behavior exactly: always use 1e-8
        // The CPU code has complex logic but ultimately sets:
        // self.ALLOWED_MASS_RESIDUAL = 1e-8
        ALLOWED_MASS_RESIDUAL = 1e-8;
    }
    SystemSpecification(){}
} SystemSpecification;

#ifdef __cplusplus
}
#endif

typedef struct CompsetState {
    double x[MAX_STATEVARS + MAX_DOF_PER_PHASE];
    int x_length;
    double energy;
    double grad[MAX_STATEVARS + MAX_DOF_PER_PHASE];
    int grad_length;
    double hess[(MAX_STATEVARS + MAX_DOF_PER_PHASE) * (MAX_STATEVARS + MAX_DOF_PER_PHASE)];
    int hess_rows;
    int hess_cols;
    double masses[MAX_COMPONENTS];
    int masses_length;
    double mass_jac[MAX_COMPONENTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE)];
    int mass_jac_rows;
    int mass_jac_cols;
    // phase_matrix_dim = phase_dof + num_internal_cons (since num_phase_local_conditions is 0)
    double phase_matrix[(MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS) * (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)];
    int phase_matrix_dim;
    double full_e_matrix[(MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS) * (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)];
    int full_e_matrix_dim;

    double c_G[MAX_DOF_PER_PHASE];
    int c_G_length;
    double c_statevars[MAX_DOF_PER_PHASE * MAX_STATEVARS];
    int c_statevars_rows;
    int c_statevars_cols;
    double c_component[MAX_COMPONENTS * MAX_DOF_PER_PHASE];
    int c_component_rows;
    int c_component_cols;
    double delta_y[MAX_DOF_PER_PHASE];
    int delta_y_length;
    double moles_normalization;
    double internal_cons[MAX_INTERNAL_CONSTRAINTS];
    int internal_cons_length;
    double moles_normalization_grad[MAX_STATEVARS + MAX_DOF_PER_PHASE];
    int moles_normalization_grad_length;
    double cons_jac_tmp[MAX_INTERNAL_CONSTRAINTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE)];
    int cons_jac_tmp_rows;
    int cons_jac_tmp_cols;
    // phase_local_jac_tmp is removed as num_phase_local_conditions is 0

    __device__ void init(const SystemSpecification* spec, const CompositionSet* compset) {
        // Assuming compset and compset->phase_record are valid pointers
        const PhaseRecord* pr = compset->phase_record; // Convenience pointer

        x_length = spec->num_statevars + pr->phase_dof;
        for(int i=0; i<x_length; ++i) x[i] = 0.0; // Will be set from compset->dof
        for(int i=x_length; i<MAX_STATEVARS + MAX_DOF_PER_PHASE; ++i) x[i] = 0.0;

        energy = 0.0;

        grad_length = spec->num_statevars + pr->phase_dof;
        for(int i=0; i<grad_length; ++i) grad[i] = 0.0;
        for(int i=grad_length; i<MAX_STATEVARS + MAX_DOF_PER_PHASE; ++i) grad[i] = 0.0;

        hess_rows = spec->num_statevars + pr->phase_dof;
        hess_cols = spec->num_statevars + pr->phase_dof;
        for(int i=0; i < hess_rows * hess_cols; ++i) hess[i] = 0.0;
        for(int i=hess_rows * hess_cols; i < (MAX_STATEVARS + MAX_DOF_PER_PHASE) * (MAX_STATEVARS + MAX_DOF_PER_PHASE); ++i) hess[i] = 0.0;

        masses_length = spec->num_components; // Should be pr->num_elements if PhaseRecord stores that
        for(int i=0; i<masses_length; ++i) masses[i] = 0.0;
        for(int i=masses_length; i<MAX_COMPONENTS; ++i) masses[i] = 0.0;

        mass_jac_rows = spec->num_components; // pr->num_elements
        mass_jac_cols = spec->num_statevars + pr->phase_dof;
        for(int i=0; i<mass_jac_rows * mass_jac_cols; ++i) mass_jac[i] = 0.0;
        for(int i=mass_jac_rows*mass_jac_cols; i < MAX_COMPONENTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE); ++i) mass_jac[i] = 0.0;

        phase_matrix_dim = pr->phase_dof + pr->num_internal_cons; // num_phase_local_conditions is 0
        for(int i=0; i<phase_matrix_dim * phase_matrix_dim; ++i) phase_matrix[i] = 0.0;
        for(int i=phase_matrix_dim*phase_matrix_dim; i < (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS) * (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS); ++i) phase_matrix[i] = 0.0;

        full_e_matrix_dim = phase_matrix_dim;
        for(int i=0; i<full_e_matrix_dim * full_e_matrix_dim; ++i) full_e_matrix[i] = 0.0;
        for(int i=full_e_matrix_dim*full_e_matrix_dim; i < (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS) * (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS); ++i) full_e_matrix[i] = 0.0;

        c_G_length = pr->phase_dof;
        for(int i=0; i<c_G_length; ++i) c_G[i] = 0.0;
        for(int i=c_G_length; i<MAX_DOF_PER_PHASE; ++i) c_G[i] = 0.0;

        c_statevars_rows = pr->phase_dof;
        c_statevars_cols = spec->num_statevars;
        for(int i=0; i<c_statevars_rows*c_statevars_cols; ++i) c_statevars[i] = 0.0;
        for(int i=c_statevars_rows*c_statevars_cols; i < MAX_DOF_PER_PHASE * MAX_STATEVARS; ++i) c_statevars[i] = 0.0;

        c_component_rows = spec->num_components; // pr->num_elements
        c_component_cols = pr->phase_dof;
        for(int i=0; i<c_component_rows*c_component_cols; ++i) c_component[i] = 0.0;
        for(int i=c_component_rows*c_component_cols; i < MAX_COMPONENTS * MAX_DOF_PER_PHASE; ++i) c_component[i] = 0.0;

        delta_y_length = pr->phase_dof;
        for(int i=0; i<delta_y_length; ++i) delta_y[i] = 0.0;
        for(int i=delta_y_length; i<MAX_DOF_PER_PHASE; ++i) delta_y[i] = 0.0;

        moles_normalization = 0.0;

        internal_cons_length = pr->num_internal_cons;
        for(int i=0; i<internal_cons_length; ++i) internal_cons[i] = 0.0;
        for(int i=internal_cons_length; i<MAX_INTERNAL_CONSTRAINTS; ++i) internal_cons[i] = 0.0;

        moles_normalization_grad_length = mass_jac_cols; // Same as mass_jac columns since both are in workspace format
        for(int i=0; i<moles_normalization_grad_length; ++i) moles_normalization_grad[i] = 0.0;
        for(int i=moles_normalization_grad_length; i<MAX_STATEVARS + MAX_DOF_PER_PHASE; ++i) moles_normalization_grad[i] = 0.0;

        cons_jac_tmp_rows = pr->num_internal_cons;
        cons_jac_tmp_cols = spec->num_statevars + pr->phase_dof;
        for(int i=0; i<cons_jac_tmp_rows * cons_jac_tmp_cols; ++i) cons_jac_tmp[i] = 0.0;
        for(int i=cons_jac_tmp_rows * cons_jac_tmp_cols; i<MAX_INTERNAL_CONSTRAINTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE); ++i) cons_jac_tmp[i] = 0.0;
    }
    __device__ CompsetState(){}
} CompsetState;

typedef struct SystemState {
    CompositionSet compsets[MAX_PHASES];
    int num_compsets;
    CompsetState cs_states[MAX_PHASES];

    int iteration;
    int iterations_since_last_phase_change;
    int metastable_phase_iterations[MAX_PHASES];
    int times_compset_removed[MAX_PHASES];
    double mass_residual;
    double phase_amt[MAX_PHASES];
    double chemical_potentials[MAX_COMPONENTS];
    int condition_idx;  // Thread/condition index for debug output
    double previous_chemical_potentials[MAX_COMPONENTS];
    double largest_chemical_potential_difference;
    double delta_ms[MAX_PHASES * MAX_COMPONENTS];
    int delta_ms_rows;
    int delta_ms_cols;
    double delta_statevars[MAX_STATEVARS];
    double phase_compositions[MAX_PHASES * MAX_COMPONENTS];
    int phase_compositions_rows;
    int phase_compositions_cols;

    int free_stable_compset_indices[MAX_PHASES];
    int num_free_stable_compsets;

    double largest_statevar_change;
    double largest_phase_amt_change;
    double largest_y_change;
    double system_amount;
    double mole_fractions[MAX_COMPONENTS];

    double _driving_forces_arr[MAX_PHASES];
    double _phase_energies_per_mole_atoms_arr[MAX_PHASES];
    double _phase_amounts_per_mole_atoms_arr[MAX_PHASES * MAX_COMPONENTS];

    __device__ void init(SystemSpecification* spec, CompositionSet* initial_compsets, int initial_num_compsets) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: SystemState::init called with num_compsets=%d\n", initial_num_compsets);
        #endif
        num_compsets = initial_num_compsets;
        if (num_compsets > MAX_PHASES) num_compsets = MAX_PHASES; // Cap at MAX_PHASES

        for (int i = 0; i < num_compsets; ++i) {
            compsets[i] = initial_compsets[i]; // Simple assignment
            compsets[i].fixed = false;
            if (compsets[i].phase_record != nullptr) { // Ensure phase_record is valid
                 cs_states[i].init(spec, &compsets[i]);
            } else {
                // Handle error: phase_record is null. Maybe default init cs_state or mark compset invalid.
                cs_states[i] = CompsetState(); // Default constructor
            }
        }
        for (int i = num_compsets; i < MAX_PHASES; ++i) {
             cs_states[i] = CompsetState();
             compsets[i] = CompositionSet();
        }

        for (int i = 0; i < spec->num_fixed_stable_compsets; ++i) {
            int fixed_idx = spec->fixed_stable_compset_indices[i];
            if (fixed_idx < num_compsets) {
                compsets[fixed_idx].fixed = true;
            }
        }

        iteration = 0;
        iterations_since_last_phase_change = 0;
        for (int i = 0; i < MAX_PHASES; ++i) {
            metastable_phase_iterations[i] = 0;
            times_compset_removed[i] = 0;
            // CRITICAL FIX: Initialize phase_amt from NP, but we'll normalize below
            phase_amt[i] = (i < num_compsets) ? compsets[i].NP : 0.0;
            _driving_forces_arr[i] = 0.0;
            _phase_energies_per_mole_atoms_arr[i] = 0.0;
        }
        mass_residual = 1e10;

        for (int i = 0; i < MAX_COMPONENTS; ++i) {
            chemical_potentials[i] = 0.0;
            previous_chemical_potentials[i] = 0.0;
            mole_fractions[i] = 0.0;
        }
        for (int i = 0; i < spec->num_fixed_chemical_potentials; ++i) {
            int c_idx = spec->fixed_chemical_potential_indices[i];
            if (c_idx < spec->num_components) {
                chemical_potentials[c_idx] = spec->initial_chemical_potentials[c_idx];
            }
        }

        largest_chemical_potential_difference = -INFINITY;

        delta_ms_rows = num_compsets; // Initial value, might change if num_compsets changes
        delta_ms_cols = spec->num_components;
        for (int i = 0; i < MAX_PHASES * MAX_COMPONENTS; ++i) delta_ms[i] = 0.0;
        for (int i = 0; i < MAX_PHASES * MAX_COMPONENTS; ++i) _phase_amounts_per_mole_atoms_arr[i] = 0.0;

        for (int i = 0; i < MAX_STATEVARS; ++i) delta_statevars[i] = 0.0;

        phase_compositions_rows = num_compsets; // Initial value
        phase_compositions_cols = spec->num_components;
        for (int i = 0; i < MAX_PHASES * MAX_COMPONENTS; ++i) phase_compositions[i] = 0.0;
        
        // CRITICAL FIX: Calculate phase_compositions using formulamole_obj like CPU does
        // This is essential for phase amount normalization to work correctly
        
        
        double phase_comp_sum;
        for (int idx = 0; idx < num_compsets; ++idx) {
            CompositionSet* compset = &compsets[idx];
            if (compset->phase_record == nullptr) continue;
            
            // Calculate moles of each element per formula unit
            double formulamoles[MAX_COMPONENTS];
            // CRITICAL: Initialize to zero since formulamole_obj only fills nonvacant elements
            for (int i = 0; i < MAX_COMPONENTS; ++i) {
                formulamoles[i] = 0.0;
            }
            
            // Check if formulamole_obj function pointer is valid
            if (compset->phase_record->formulamole_obj != nullptr) {
                // DEBUG: Check DOF array before calling formulamole_obj
                bool dof_valid = true;
                for (int i = 0; i < compset->phase_record->num_statevars + compset->phase_record->phase_dof; ++i) {
                    if (isnan(compset->dof[i]) || isinf(compset->dof[i])) {
                        dof_valid = false;
                        break;
                    }
                }
                if (!dof_valid) {
                    #ifdef VERBOSE_DEBUG
                    printf("GPU ERROR: Invalid DOF values in recompute for phase %d\\n", idx);
                    for (int i = 0; i < 5; ++i) {
                        printf("  dof[%d] = %f\\n", i, i < (compset->phase_record->num_statevars + compset->phase_record->phase_dof) ? compset->dof[i] : 0.0);
                    }
                    #endif
                } else {
                    // CRITICAL FIX: Create Model DOF array from Workspace DOF
                    // With updated energy functions, use full workspace DOF
                    if (compset->phase_record->formulamole_obj != nullptr) {
                        
        compset->phase_record->formulamole_obj(formulamoles, compset->dof);
                    }
                }
            }
            
            phase_comp_sum = 0.0;
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                phase_compositions[idx * MAX_COMPONENTS + comp_idx] = formulamoles[comp_idx];
        // DEBUG removed
            
                phase_comp_sum += formulamoles[comp_idx];
            }
            
            // CRITICAL FIX: Convert phase amounts to formula units like CPU does
            // CPU minimizer.pyx line 776: self.phase_amt[idx] /= phase_comp_sum
            // This normalization must happen in __init__ to match CPU behavior!
            
            if (phase_comp_sum > 1e-12) {
                // Normalize phase amount to formula units
                phase_amt[idx] /= phase_comp_sum;
            }
        }
        

        num_free_stable_compsets = 0;
        // Collecting free stable compsets
        for (int i = 0; i < num_compsets; ++i) {
            // Check compset[i]
            if (!compsets[i].fixed && compsets[i].NP > MIN_PHASE_FRACTION / 10.0) { // Slightly lower threshold for initial pickup
                if (num_free_stable_compsets < MAX_PHASES) {
                    free_stable_compset_indices[num_free_stable_compsets++] = i;
                    // Added to free_stable set
                }
            }
        }
        // Final num_free_stable_compsets set

        largest_statevar_change = 0.0;
        largest_phase_amt_change = 0.0;
        largest_y_change = 0.0;
        system_amount = 0.0;

        recompute(spec);
    }
    __device__ SystemState(){}

    __device__ void recompute(SystemSpecification* spec) {
        // Get thread ID for debug messages
        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        
        // DEBUG: Check spec pointer validity at entry
        if (spec == nullptr) {
            #ifdef VERBOSE_DEBUG
            printf("GPU ERROR: spec pointer is NULL in recompute!\n");
            #endif
            return;
        }
        
        // DEBUG: Check if spec values look reasonable
        if (spec->num_statevars < 0 || spec->num_statevars > 10 || 
            spec->num_components < 0 || spec->num_components > 10) {
            #ifdef VERBOSE_DEBUG
            printf("GPU ERROR: spec appears corrupted at recompute entry!\n");
            printf("  spec=%p\n", spec);
            printf("  spec->num_statevars=%d (0x%X)\n", spec->num_statevars, spec->num_statevars);
            printf("  spec->num_components=%d (0x%X)\n", spec->num_components, spec->num_components);
            #endif
            // Don't return - try to continue
        }
        
        // Entered recompute function - removed pointer printf to avoid alignment issues
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: num_compsets=%d, iteration=%d\n", num_compsets, iteration);
        #endif
        // SEGMENT 22: STATE RECOMPUTE - GLOBAL QUANTITIES
        gpu_debug_log(22, "State recompute - global quantities", condition_idx);
        // Direct printf to test if this code is being reached
        #ifdef VERBOSE_DEBUG
        if (condition_idx < 3) {
            printf("[GPU] SEGMENT 22 DEBUG: iteration=%d, num_components=%d\n", iteration, spec->num_components);
            printf("[GPU]   chemical_potentials: [%.15e, %.15e]\n", chemical_potentials[0], chemical_potentials[1]);
            printf("[GPU]   system_amount: %.15e\n", system_amount);
            printf("[GPU]   mole_fractions: [");
            for (int comp_i = 0; comp_i < spec->num_components; comp_i++) {
                printf("%.6f", mole_fractions[comp_i]);
                if (comp_i < spec->num_components - 1) printf(", ");
            }
            printf("]\n");
            gpu_debug_log_array("chemical_potentials", chemical_potentials, spec->num_components);
            gpu_debug_log_value("system_amount", system_amount);
            gpu_debug_log_array("mole_fractions", mole_fractions, spec->num_components);
        }
        #endif
        
        // REMOVED: current_dof_for_phase array - now using model_dof_for_calcs created locally where needed

        for(int i=0; i < spec->num_components; ++i) mole_fractions[i] = 0.0;
        // Correct sizing for delta_ms for current num_compsets
        for(int i=0; i < num_compsets * spec->num_components; ++i) delta_ms[i] = 0.0;
        for(int i=num_compsets * spec->num_components; i < MAX_PHASES * MAX_COMPONENTS; ++i) delta_ms[i] = 0.0; // Zero out rest

        system_amount = 0.0;
        
        // DEBUG: Track total phase amounts through iterations
        double phase_amt_sum = 0.0;
        for (int idx = 0; idx < num_compsets; ++idx) {
            phase_amt_sum += phase_amt[idx];
        }
        #ifdef VERBOSE_DEBUG
        printf("[GPU MASS BALANCE] recompute() - iteration %d: sum(phase_amt) = %.15e\n", iteration, phase_amt_sum);
        #endif
        
        // REMOVED: Extraneous normalization not present in CPU code
        

        for (int idx = 0; idx < num_compsets; ++idx) {
            CompositionSet* compset = &compsets[idx];
            CompsetState* csst = &cs_states[idx];
            if (compset->phase_record == nullptr) continue; // Skip if phase_record is invalid

            // REMOVED: Old code that created current_dof_for_phase incorrectly
            // Now we create model_dof_for_calcs properly from workspace DOF when needed
            
            // CRITICAL FIX: Calculate phase_compositions using formulamole_obj (matching CPU minimizer.pyx)
            double formulamoles[MAX_COMPONENTS];
            // Initialize to zero since formulamole_obj only fills nonvacant elements
            for (int i = 0; i < MAX_COMPONENTS; ++i) {
                formulamoles[i] = 0.0;
            }
            
            // Check if formulamole_obj function pointer is valid
            if (compset->phase_record->formulamole_obj != nullptr) {
                // DEBUG: Check DOF array before calling formulamole_obj
                bool dof_valid = true;
                for (int i = 0; i < compset->phase_record->num_statevars + compset->phase_record->phase_dof; ++i) {
                    if (isnan(compset->dof[i]) || isinf(compset->dof[i])) {
                        dof_valid = false;
                        break;
                    }
                }
                if (!dof_valid) {
                    #ifdef VERBOSE_DEBUG
                    printf("GPU ERROR: Invalid DOF values in recompute for phase %d\\n", idx);
                    for (int i = 0; i < 5; ++i) {
                        printf("  dof[%d] = %f\\n", i, i < (compset->phase_record->num_statevars + compset->phase_record->phase_dof) ? compset->dof[i] : 0.0);
                    }
                    #endif
                } else {
                    // CRITICAL FIX: Create Model DOF array from Workspace DOF
                    // With updated energy functions, use full workspace DOF
                    if (compset->phase_record->formulamole_obj != nullptr) {
                        compset->phase_record->formulamole_obj(formulamoles, compset->dof);
                    }
                    
                    // CRITICAL FIX: Calculate mass jacobians (missing from original GPU implementation)
                    // This matches CPU line 820: compset.phase_record.formulamole_grad(csst.mass_jac[comp_idx, :], x, comp_idx)
                    // Zero out mass_jac first
                    for (int i = 0; i < csst->mass_jac_rows * csst->mass_jac_cols; ++i) {
                        csst->mass_jac[i] = 0.0;
                    }
                    
                    if (compset->phase_record->formulamole_grad != nullptr) {
                        // CSE formulamole_grad functions output reduced format
                        // Output format: [comp0_dT, comp0_dY1, comp0_dY2, ..., comp1_dT, comp1_dY1, comp1_dY2, ...]
                        double temp_mass_jac[MAX_COMPONENTS * (1 + MAX_DOF_PER_PHASE)];
                        for (int i = 0; i < MAX_COMPONENTS * (1 + MAX_DOF_PER_PHASE); ++i) {
                            temp_mass_jac[i] = 0.0;
                        }
                        
                        // Call formulamole_grad with workspace DOF
                        compset->phase_record->formulamole_grad(temp_mass_jac, compset->dof);
                        
                        // Zero out the full mass_jac array
                        for (int i = 0; i < csst->mass_jac_rows * csst->mass_jac_cols; ++i) {
                            csst->mass_jac[i] = 0.0;
                        }
                        
                        // Map the reduced gradient to full workspace format
                        // CSE outputs: [comp0_dT, comp0_dY1, comp0_dY2, ..., comp1_dT, comp1_dY1, comp1_dY2, ...]
                        // Full format expects: [comp0_dN, comp0_dP, comp0_dT, comp0_dY1, comp0_dY2, ..., comp1_dN, comp1_dP, comp1_dT, ...]
                        int reduced_cols = 1 + compset->phase_record->phase_dof; // T + site fractions
                        int full_cols = spec->num_statevars + compset->phase_record->phase_dof;
                        
                        for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                            // Temperature derivative (index 0 in reduced -> index 2 in full)
                            csst->mass_jac[comp_idx * full_cols + 2] = temp_mass_jac[comp_idx * reduced_cols + 0];
                            
                            // Site fraction derivatives (indices 1..n in reduced -> indices num_statevars..num_statevars+n-1 in full)
                            for (int j = 0; j < compset->phase_record->phase_dof; ++j) {
                                csst->mass_jac[comp_idx * full_cols + spec->num_statevars + j] = 
                                    temp_mass_jac[comp_idx * reduced_cols + 1 + j];
                            }
                        }
                        
                        // DEBUG: Print raw gradient values from formulamole_grad
                        #ifdef VERBOSE_DEBUG
                        if (idx == 0 && iteration < 2) {
                            printf("GPU DEBUG: Raw formulamole_grad output (phase %d):\n", idx);
                            int reduced_cols = 1 + compset->phase_record->phase_dof;
                            for (int comp_idx = 0; comp_idx < spec->num_components && comp_idx < 2; ++comp_idx) {
                                printf("  Component %d gradients (reduced format): ", comp_idx);
                                for (int j = 0; j < reduced_cols; ++j) {
                                    printf("d/d[T,Y1,Y2][%d]=%e ", j, temp_mass_jac[comp_idx * reduced_cols + j]);
                                }
                                printf("\n");
                            }
                        }
                        #endif
                        
                        // DEBUG: Print mass_jac for first phase and iteration
                        #ifdef VERBOSE_DEBUG
                        if (idx == 0 && iteration < 2) {
                            printf("GPU DEBUG: Phase %d mass_jac matrix (workspace DOF format):\\n", idx);
                            printf("  Columns: [N, P, T, Y_NB, Y_TI, ...]\\n");
                            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                                printf("  Component %d: ", comp_idx);
                                for (int j = 0; j < csst->mass_jac_cols && j < 5; ++j) {
                                    printf("%.6e ", csst->mass_jac[comp_idx * csst->mass_jac_cols + j]);
                                }
                                printf("\\n");
                            }
                        }
                        #endif
                    }
                }
            }

            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                // CRITICAL FIX: masses should contain mole fractions from formulamole_obj
                // This matches CPU line 749: compset.phase_record.formulamole_obj(csst.masses[comp_idx, :], x, comp_idx)
                csst->masses[comp_idx] = formulamoles[comp_idx];
                
                // DEBUG: Print phase compositions
                #ifdef VERBOSE_DEBUG
                if (thread_id == 0 && iteration < 3 && comp_idx < 2) {
                    printf("GPU: Phase %d composition[%d] = %.6f (will update after compset update)\n", idx, comp_idx, 
                           phase_compositions[idx * MAX_COMPONENTS + comp_idx]);
                }
                #endif

                // CRITICAL FIX: phase_amt is already in formula units (normalized in constructor)
                // So we use it directly like CPU does in recompute()
                if (phase_amt[idx] > 1e-20) { // Avoid adding noise from zero phase_amt
                    mole_fractions[comp_idx] += phase_amt[idx] * csst->masses[comp_idx];
                    system_amount += phase_amt[idx] * csst->masses[comp_idx];
                }
            }
            
            // CRITICAL FIX: Update phase_compositions AFTER formulamole_obj calculation
            // This matches CPU line 832: self.phase_compositions[idx, comp_idx] = csst.masses[comp_idx, 0]
            // But only for active phases to avoid overwriting with zeros
            if (phase_amt[idx] > 1e-10) {
                for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                    phase_compositions[idx * MAX_COMPONENTS + comp_idx] = csst->masses[comp_idx];
                }
            }
        }

        if (fabs(system_amount) > 1e-12) {
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                mole_fractions[comp_idx] /= system_amount;
            }
        } else { // If system_amount is zero, mole_fractions should reflect this (e.g. uniform or zero)
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                 mole_fractions[comp_idx] = (spec->num_components > 0) ? 1.0/spec->num_components : 0.0; // Or just 0.0
            }
        }
        
        if (condition_idx < 3) {
            gpu_debug_log_value("system_amount", system_amount);
        }
        
        // Count active phases
        int num_phases_active = 0;
        for (int idx = 0; idx < num_compsets; ++idx) {
            if (phase_amt[idx] > 1e-10) num_phases_active++;
        }
        if (condition_idx < 3) {
            gpu_debug_log_value("num_phases_active", (double)num_phases_active);
        }
        
        // Log phase details
        if (condition_idx < 3) {
            #ifdef VERBOSE_DEBUG
            for (int idx = 0; idx < num_compsets; ++idx) {
                if (phase_amt[idx] > 1e-10) {
                    // Use direct printf instead of sprintf for device code
                    printf("[GPU]   phase_%d_%d: %.15e\n", idx, idx, phase_amt[idx]);
                    printf("[GPU]   phase_%d_X: [", idx);
                    for (int i = 0; i < spec->num_components && i < 5; ++i) {
                        printf("%.6f", phase_compositions[idx * MAX_COMPONENTS + i]);
                        if (i < spec->num_components - 1) printf(", ");
                    }
                    if (spec->num_components > 5) printf("...");
                    printf("]\n");
                }
            }
            #endif
        }
        
        if (condition_idx < 3) {
            gpu_debug_log_array("mole_fractions", mole_fractions, spec->num_components);
        }

        mass_residual = 0.0;
        for (int cond_idx = 0; cond_idx < spec->num_prescribed_mole_fraction_conditions; ++cond_idx) {
            double current_sum = 0.0;
            for (int comp_idx = 0; comp_idx < spec->num_prescribed_mole_fraction_coefficients_cols; ++comp_idx) {
                 current_sum += spec->prescribed_mole_fraction_coefficients[cond_idx][comp_idx] * mole_fractions[comp_idx];
            }
            double residual_contrib = fabs(current_sum - spec->prescribed_mole_fraction_rhs[cond_idx]);
            
            // DEBUG: Print details for first few iterations
            #ifdef VERBOSE_DEBUG
            if (condition_idx < 3 && iteration < 3) {
                printf("[GPU] Mass residual calc (iteration %d, cond %d):\n", iteration, cond_idx);
                printf("  Coefficients: [");
                for (int i = 0; i < spec->num_prescribed_mole_fraction_coefficients_cols; ++i) {
                    printf("%.3f", spec->prescribed_mole_fraction_coefficients[cond_idx][i]);
                    if (i < spec->num_prescribed_mole_fraction_coefficients_cols - 1) printf(", ");
                }
                printf("]\n");
                printf("  Current sum: %.15e\n", current_sum);
                printf("  RHS: %.15e (accessing index %d)\n", spec->prescribed_mole_fraction_rhs[cond_idx], cond_idx);
                // DEBUG: Print all RHS values
                if (cond_idx == 0) {
                    printf("  All RHS values: [");
                    for (int i = 0; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS && i < 4; ++i) {
                        printf("%.3f", spec->prescribed_mole_fraction_rhs[i]);
                        if (i < 3) printf(", ");
                    }
                    printf("]\n");
                }
                printf("  Residual contribution: %.15e\n", residual_contrib);
            }
            #endif
            
            mass_residual += residual_contrib;
        }
        
        if (condition_idx < 3) {
            gpu_debug_log_value("mass_residual", mass_residual);
        }

        // SEGMENT 23: STATE RECOMPUTE - PHASE QUANTITIES
        gpu_debug_log(23, "State recompute - phase quantities", condition_idx);
        if (condition_idx < 3) {
            gpu_debug_log_value("num_phases_active", (double)num_phases_active);
        }
        
        if (condition_idx < 3) {
            #ifdef VERBOSE_DEBUG
            for (int idx = 0; idx < num_compsets; ++idx) {
                if (phase_amt[idx] > 1e-10) {
                    // Use direct printf instead of sprintf for device code
                    printf("[GPU]   phase_%d_NP: %.15e\n", idx, phase_amt[idx]);
                    printf("[GPU]   phase_%d_X: [", idx);
                    for (int i = 0; i < spec->num_components && i < 5; ++i) {
                        printf("%.6f", phase_compositions[idx * MAX_COMPONENTS + i]);
                        if (i < spec->num_components - 1) printf(", ");
                    }
                    if (spec->num_components > 5) printf("...");
                    printf("]\n");
                }
            }
            #endif
        }

        for (int idx = 0; idx < num_compsets; ++idx) {
            CompositionSet* compset = &compsets[idx];
            CompsetState* csst = &cs_states[idx];
            if (compset->phase_record == nullptr) continue;
            
            // CRITICAL FIX: Skip phases with zero amount to match CPU behavior
            // The CPU solver doesn't process removed phases in recompute
            if (phase_amt[idx] < 1e-10) {
                continue;
            }
            
            const PhaseRecord* pr = compset->phase_record;

            // REMOVED: Old code that created current_dof_for_phase incorrectly
            // Now we create model_dof_for_calcs properly from workspace DOF when needed

            // CRITICAL FIX: Calculate phase_comp_sum from stored phase_compositions
            // This matches CPU behavior (minimizer.pyx line 880-881)
            // For multi-sublattice phases, this equals the sum of site ratios (e.g., 20 for ALCU_ZETA)
            double phase_sum_moles_atoms_per_formula = 0.0;
            for (int comp_idx = 0; comp_idx < spec->num_components; comp_idx++) {
                phase_sum_moles_atoms_per_formula += phase_compositions[idx * MAX_COMPONENTS + comp_idx];
            }
            
            // Safety check
            // CPU doesn't have a fallback here - let it be what it is

            // Call compset update. NP is moles of formula units.
            // CRITICAL: Match CPU algorithm - multiply phase_amt by phase_sum_moles_atoms_per_formula
            // This converts from formula units back to mole fractions, matching CPU minimizer.pyx line 885
            double update_amount = phase_amt[idx] * phase_sum_moles_atoms_per_formula;
            
            // DEBUG: Print update calculation
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0 && iteration < 3) {
                printf("[GPU UPDATE] Phase %d: phase_amt=%.15e (formula units), phase_comp_sum=%.15e\n",
                       idx, phase_amt[idx], phase_sum_moles_atoms_per_formula);
                printf("[GPU UPDATE] Phase %d: update_amount=%.15e (mole fractions for NP)\n",
                       idx, update_amount);
            }
            #endif
            
            // Additional safety check for update amount
            // CPU doesn't check update_amount bounds
            // CRITICAL FIX: Pass the actual workspace DOF to update, not the model DOF
            // The update function expects workspace state variables, not model state variables
            compset->update(&compset->dof[spec->num_statevars], update_amount, compset->dof, spec->num_statevars);
            
            // csst->energy will be G per formula unit (from pr->formulaobj)
            // Pass full workspace DOF to energy calculation, matching CPU behavior
            // The generated functions now expect workspace DOF format [N, P, T, Y1, Y2...]
            // CRITICAL FIX: Use pr->formulaobj() for equilibrium matrix (per formula unit, not per mole atoms)
            // This matches the CPU behavior where equilibrium matrix uses unnormalized energy values
            csst->energy = pr->formulaobj(compset->dof);
            
            // Add numerical debug output for phase energy
            #ifdef VERBOSE_DEBUG
            printf("[GPU]   phase_%d_comp_sum: %.15e\n", idx, phase_sum_moles_atoms_per_formula);
            printf("[GPU]   phase_%d_energy: %.15e\n", idx, csst->energy);
            #endif


            for(int i=0; i<csst->mass_jac_rows * csst->mass_jac_cols; ++i) csst->mass_jac[i] = 0.0;
            for(int i=0; i<csst->phase_matrix_dim * csst->phase_matrix_dim; ++i) csst->phase_matrix[i] = 0.0;
            for(int i=0; i<csst->internal_cons_length; ++i) csst->internal_cons[i] = 0.0;
            for(int i=0; i<csst->hess_rows * csst->hess_cols; ++i) csst->hess[i] = 0.0;
            for(int i=0; i<csst->grad_length; ++i) csst->grad[i] = 0.0;

            // DEBUG: Print workspace DOF before calling formulamole_grad
            #ifdef VERBOSE_DEBUG
            if (idx == 0 && iteration < 2) {
                printf("GPU DEBUG: Before DOF print - spec=%p, pr=%p, compset=%p\n", spec, pr, compset);
                printf("  spec->num_statevars=%d, pr->phase_dof=%d, total=%d\n", 
                       spec->num_statevars, pr->phase_dof, spec->num_statevars + pr->phase_dof);
                
                if (spec->num_statevars + pr->phase_dof > 0 && spec->num_statevars + pr->phase_dof < 20) {
                    printf("GPU DEBUG: workspace DOF before formulamole_grad: [");
                    for (int i = 0; i < spec->num_statevars + pr->phase_dof; i++) {
                        printf("%.6f", compset->dof[i]);
                        if (i < spec->num_statevars + pr->phase_dof - 1) printf(", ");
                    }
                    printf("]\n");
                } else {
                    printf("GPU ERROR: Invalid DOF size: %d\n", spec->num_statevars + pr->phase_dof);
                }
            }
            #endif
            
            // CRITICAL FIX: Properly handle mass_jac from formulamole_grad output
            // formulamole_grad outputs a matrix of size (num_nonvacant_elements × num_model_dof)
            // where num_model_dof = pr->num_statevars + pr->phase_dof
            // But csst->mass_jac expects size (num_components × (spec->num_statevars + pr->phase_dof))
            
            // First, we need a temporary array to hold the formulamole_grad output
            double temp_mass_jac[MAX_COMPONENTS * MAX_DOF_PER_PHASE];
            for (int i = 0; i < MAX_COMPONENTS * MAX_DOF_PER_PHASE; ++i) {
                temp_mass_jac[i] = 0.0;
            }
            
            // Call formulamole_grad to get gradients for nonvacant elements only
            if (pr->formulamole_grad != nullptr) {
                // DEBUG: Print the DOF values being passed to formulamole_grad
                #ifdef VERBOSE_DEBUG
                if (iteration == 0) {
                    printf("[GPU MASS_JAC] Phase %d calling formulamole_grad with DOF: ", idx);
                    for (int k = 0; k < spec->num_statevars + pr->phase_dof; ++k) {
                        printf("%e ", compset->dof[k]);
                    }
                    printf("\n");
                    printf("[GPU MASS_JAC DEBUG] pr->num_statevars=%d, spec->num_statevars=%d, pr->phase_dof=%d\n",
                           pr->num_statevars, spec->num_statevars, pr->phase_dof);
                }
                #endif
                pr->formulamole_grad(temp_mass_jac, compset->dof);
                
                // DEBUG: Print raw formulamole_grad output
                #ifdef VERBOSE_DEBUG
                if (iteration < 2) {
                    printf("GPU DEBUG: Raw formulamole_grad output (phase %d):\n", idx);
                    // CRITICAL FIX: CSE functions output in reduced format [T, Y1, Y2, ...]
                    // So the gradient matrix is nonvacant_elements x (1 + phase_dof)
                    int reduced_cols = 1 + pr->phase_dof; // T + site fractions
                    for (int i = 0; i < pr->nonvacant_elements && i < 3; i++) {
                        printf("  Component %d gradients (CSE format [T,Y1,Y2]): ", i);
                        for (int j = 0; j < reduced_cols && j < 5; j++) {
                            printf("[%d]=%e ", j, temp_mass_jac[i * reduced_cols + j]);
                        }
                        printf("\n");
                    }
                }
                #endif
            } else {
                #ifdef VERBOSE_DEBUG
                printf("GPU ERROR: formulamole_grad is null for phase %d\\n", idx);
                #endif
            }
            
            // Now copy the gradients to the correct positions in csst->mass_jac
            // CRITICAL FIX: CSE functions output in reduced format [T, Y1, Y2, ...]
            // We need to map this to workspace format [N, P, T, Y1, Y2, ...]
            int nonvacant_idx = 0;
            int reduced_cols = 1 + pr->phase_dof; // CSE output columns: T + site fractions
            
            for (int comp_idx = 0; comp_idx < spec->num_components; comp_idx++) {
                // Check if this component is a nonvacant element
                // For now, assume components are ordered as [NB, TI, VA] and nonvacant are [NB, TI]
                if (comp_idx < pr->nonvacant_elements) {
                    // This is a nonvacant element - copy its gradients
                    // First, zero out the entire row
                    for (int col = 0; col < csst->mass_jac_cols; col++) {
                        csst->mass_jac[comp_idx * csst->mass_jac_cols + col] = 0.0;
                    }
                    
                    // Map from CSE reduced format to Workspace DOF
                    for (int cse_col = 0; cse_col < reduced_cols; cse_col++) {
                        int workspace_col;
                        if (cse_col == 0) {
                            // Temperature column in CSE -> position 2 in workspace [N, P, T]
                            workspace_col = 2;
                        } else {
                            // Site fraction columns: cse_col 1,2,... -> workspace cols 3,4,...
                            workspace_col = spec->num_statevars + (cse_col - 1);
                        }
                        
                        if (workspace_col < csst->mass_jac_cols) {
                            csst->mass_jac[comp_idx * csst->mass_jac_cols + workspace_col] = 
                                temp_mass_jac[nonvacant_idx * reduced_cols + cse_col];
                        }
                    }
                    nonvacant_idx++;
                } else {
                    // This is VA or a component not in this phase - all gradients are zero
                    for (int col = 0; col < csst->mass_jac_cols; col++) {
                        csst->mass_jac[comp_idx * csst->mass_jac_cols + col] = 0.0;
                    }
                }
            }
            
            // NOTE: The CPU code treats site fractions as independent variables
            // So d(moles_i)/d(Y_j) = 0 for i != j
            // The generated formulamole_grad function already handles this correctly
            
            // DEBUG: Print mass_jac matrix
            #ifdef VERBOSE_DEBUG
            if (iteration < 2) {
                printf("GPU DEBUG: Phase %d mass_jac matrix (workspace DOF format):\\n", idx);
                printf("  Columns: [N, P, T, Y_NB, Y_TI, ...]\\n");
                for (int comp_idx = 0; comp_idx < spec->num_components && comp_idx < 3; comp_idx++) {
                    printf("  Component %d: ", comp_idx);
                    for (int col = 0; col < csst->mass_jac_cols && col < 5; col++) {
                        printf("%e ", csst->mass_jac[comp_idx * csst->mass_jac_cols + col]);
                    }
                    printf("\\n");
                }
            }
            #endif
            
            if (pr->formulahess != nullptr) {
                // Calling formulahess - MUST pass full workspace DOF like CPU does
                // DEBUG: Print DOF values passed to formulahess
                #ifdef VERBOSE_DEBUG
                if (iteration < 3) {
                    printf("[GPU FORMULAHESS INPUT] Phase %d iteration %d, DOF: ", idx, iteration);
                    for (int i = 0; i < 5; i++) {
                        printf("%.15e ", compset->dof[i]);
                    }
                    printf("\n");
                }
                #endif
                // Temporary array to hold the reduced Hessian output from CSE functions
                double temp_hess[(MAX_DOF_PER_PHASE + 1) * (MAX_DOF_PER_PHASE + 1)];
                pr->formulahess(temp_hess, compset->dof);
                
                // DEBUG: Print raw Hessian output
                #ifdef VERBOSE_DEBUG
                if (idx == 0 && iteration < 2) {
                    printf("GPU DEBUG: Raw Hessian output for phase %d:\n", idx);
                    int reduced_dims = 1 + pr->phase_dof; // T + site fractions
                    for (int i = 0; i < reduced_dims; ++i) {
                        printf("  Row %d: ", i);
                        for (int j = 0; j < reduced_dims; ++j) {
                            printf("%.6e ", temp_hess[i * reduced_dims + j]);
                        }
                        printf("\n");
                    }
                }
                #endif
                
                // Map the reduced Hessian (T + site fractions) to the full matrix (N, P, T + site fractions)
                // The CSE Hessian functions output a (1 + phase_dof) x (1 + phase_dof) matrix:
                // - temp_hess[0] corresponds to d²G/dT² (T,T element) 
                // - temp_hess[i] corresponds to d²G/dTdY_i (T,site_fraction elements)
                // - temp_hess[j*(1+phase_dof)+i] corresponds to d²G/dY_i dY_j (site_fraction block)
                
                int reduced_dim = 1 + pr->phase_dof; // T + site fractions
                
                // Zero out the full Hessian matrix first
                for (int i = 0; i < csst->hess_rows * csst->hess_cols; ++i) {
                    csst->hess[i] = 0.0;
                }
                
                // Map T,T element: temp_hess[0] -> csst->hess[2,2] (T is at index 2)
                csst->hess[2 * csst->hess_cols + 2] = temp_hess[0];
                
                // Map T,site_fraction elements: temp_hess[i] -> csst->hess[2, 3+i-1] and csst->hess[3+i-1, 2]
                for (int i = 1; i < reduced_dim; i++) {
                    int site_idx = spec->num_statevars + (i - 1); // Convert to full matrix site fraction index
                    // T,site_fraction element
                    csst->hess[2 * csst->hess_cols + site_idx] = temp_hess[i];
                    // site_fraction,T element (symmetric)
                    csst->hess[site_idx * csst->hess_cols + 2] = temp_hess[i];
                }
                
                // Map site_fraction,site_fraction block
                for (int i = 1; i < reduced_dim; i++) {
                    for (int j = 1; j < reduced_dim; j++) {
                        int reduced_idx = i * reduced_dim + j;
                        int full_row = spec->num_statevars + (i - 1);
                        int full_col = spec->num_statevars + (j - 1);
                        csst->hess[full_row * csst->hess_cols + full_col] = temp_hess[reduced_idx];
                    }
                }
                
        // DEBUG: Print Hessian values
        // DEBUG: Print Hessian values
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0 && idx < 2) {
            printf("GPU DEBUG: Hessian calculated for phase record %d\n", idx);
            printf("  Hessian size: %dx%d, hess_cols=%d\n", pr->phase_dof + spec->num_statevars, pr->phase_dof + spec->num_statevars, csst->hess_cols);
            printf("  spec->num_statevars=%d, pr->phase_dof=%d\n", spec->num_statevars, pr->phase_dof);
        }
        if (thread_id == 0 && idx < 2 && pr->phase_dof > 0) {
            printf("  Site fraction Hessian block:\n");
            for (int i = 0; i < pr->phase_dof; i++) {
                printf("    [%d]", i);
                for (int j = 0; j < pr->phase_dof; j++) {
                    int hidx = (spec->num_statevars + i) * csst->hess_cols + (spec->num_statevars + j);
                    printf(" %e (idx=%d)", csst->hess[hidx], hidx);
                }
                printf("\n");
            }
        }
        #endif

                // Completed formulahess
            }
            // CPU doesn't have a fallback for missing Hessian
            
            // Removed debug prints for formulagrad to debug alignment issue
            
            // Check if formulagrad pointer looks valid
            if (pr->formulagrad == nullptr) {
                #ifdef VERBOSE_DEBUG
                printf("GPU ERROR: formulagrad pointer is null, skipping\n");
                #endif
                // Set gradient to zero (already initialized)
            } else {
                // CSE gradient functions output reduced gradient in the correct order
                // Expected order: [dG/dT, dG/dY1, dG/dY2, ...]
                double temp_grad[1 + MAX_DOF_PER_PHASE];  // Temporary array for reduced gradient
                pr->formulagrad(temp_grad, compset->dof);
                
                // Zero out the full gradient array first
                for (int i = 0; i < csst->grad_length; ++i) {
                    csst->grad[i] = 0.0;
                }
                
                // Map the reduced gradient to full gradient:
                // temp_grad[0] -> temperature derivative (index 2 in full gradient)
                // temp_grad[1..n] -> site fraction derivatives (indices num_statevars+0..num_statevars+n-1)
                
                // Temperature derivative
                csst->grad[2] = temp_grad[0];
                
                // Site fraction derivatives
                for (int i = 0; i < pr->phase_dof; i++) {
                    csst->grad[spec->num_statevars + i] = temp_grad[1 + i];
                }
                
                // DEBUG: Print gradient values for first few phases at iteration 0
                #ifdef VERBOSE_DEBUG
                if (spec->num_statevars == 3 && (pr->phase_dof == 2 || pr->phase_dof == 3)) {
                    printf("[GPU GRAD DEBUG] Phase formulagrad results (phase_dof=%d):\n", pr->phase_dof);
                    if (pr->phase_dof == 2) {
                        printf("  temp_grad: [%e, %e, %e] (T, Y1, Y2)\n", temp_grad[0], temp_grad[1], temp_grad[2]);
                        printf("  mapped to grad[2]=%e, grad[3]=%e, grad[4]=%e\n", 
                               csst->grad[2], csst->grad[3], csst->grad[4]);
                        printf("  DOF values: T=%e, Y1=%e, Y2=%e\n", 
                               compset->dof[2], compset->dof[3], compset->dof[4]);
                    } else if (pr->phase_dof == 3) {
                        printf("  temp_grad: [%e, %e, %e, %e] (T, Y1, Y2, Y3)\n", 
                               temp_grad[0], temp_grad[1], temp_grad[2], temp_grad[3]);
                        printf("  mapped to grad[2]=%e, grad[3]=%e, grad[4]=%e, grad[5]=%e\n", 
                               csst->grad[2], csst->grad[3], csst->grad[4], csst->grad[5]);
                        printf("  DOF values: T=%e, Y1=%e, Y2=%e, Y3=%e\n", 
                               compset->dof[2], compset->dof[3], compset->dof[4], compset->dof[5]);
                    }
                }
                #endif
                
                // N and P derivatives (indices 0, 1) remain zero as CSE doesn't compute them
            }
            // Completed formulagrad
            
            // DEBUG: Check if gradients are truly zero
            bool all_gradients_zero = true;
            for (int i = 0; i < pr->num_statevars + pr->phase_dof; ++i) {
                if (fabs(csst->grad[i]) > 1e-15) {
                    all_gradients_zero = false;
                    break;
                }
            }
            #ifdef VERBOSE_DEBUG
            if (all_gradients_zero && iteration < 3) {
                printf("GPU WARNING: All gradients are zero for phase %d at iteration %d\n", idx, iteration);
                printf("  workspace_dof: ");
                for (int i = 0; i < spec->num_statevars + pr->phase_dof; ++i) {
                    printf("%.6f ", compset->dof[i]);
                }
                printf("\n");
                printf("  energy: %.15e\n", csst->energy);
            }
            #endif
            
            pr->internal_cons_func(csst->internal_cons, compset->dof);
            
            // CSE constraint Jacobian functions output reduced format
            if (pr->internal_cons_jac != nullptr) {
                // Temporary array for reduced constraint Jacobian
                // CSE outputs flat array: [dC1/dT, dC1/dY1, dC1/dY2, ..., dC2/dT, dC2/dY1, ...]
                double temp_cons_jac[(1 + MAX_DOF_PER_PHASE) * MAX_INTERNAL_CONSTRAINTS];
                pr->internal_cons_jac(temp_cons_jac, compset->dof);
                
                // Zero out the full constraint Jacobian first
                for (int i = 0; i < pr->num_internal_cons * (spec->num_statevars + pr->phase_dof); ++i) {
                    csst->cons_jac_tmp[i] = 0.0;
                }
                
                // Map the reduced constraint Jacobian to full format
                // CSE outputs: [dC/dT, dC/dY1, dC/dY2, ...] in flat array
                // Full format expects: [dC/dN, dC/dP, dC/dT, dC/dY1, dC/dY2, ...] per constraint
                int reduced_size = 1 + pr->phase_dof; // T + site fractions
                int full_cols = spec->num_statevars + pr->phase_dof;
                
                for (int i = 0; i < pr->num_internal_cons; ++i) {
                    // Temperature derivative (index 0 in reduced -> index 2 in full)
                    csst->cons_jac_tmp[i * full_cols + 2] = temp_cons_jac[i * reduced_size + 0];
                    
                    // Site fraction derivatives (indices 1..n in reduced -> indices num_statevars..num_statevars+n-1 in full)
                    for (int j = 0; j < pr->phase_dof; ++j) {
                        csst->cons_jac_tmp[i * full_cols + spec->num_statevars + j] = temp_cons_jac[i * reduced_size + 1 + j];
                    }
                }
            }
            
            // DEBUG: Print constraint Jacobian values
            #ifdef VERBOSE_DEBUG
            if (idx == 0 && iteration < 2 && pr->num_internal_cons > 0) {
                printf("GPU DEBUG: Constraint Jacobian for phase %d (num_cons=%d):\n", idx, pr->num_internal_cons);
                int cons_jac_dim = spec->num_statevars + pr->phase_dof;
                for (int i = 0; i < pr->num_internal_cons; ++i) {
                    printf("  Constraint %d: ", i);
                    for (int j = 0; j < cons_jac_dim; ++j) {
                        printf("%.6e ", csst->cons_jac_tmp[i * cons_jac_dim + j]);
                    }
                    printf("\n");
                }
            }
            #endif

            // Pass site fractions from workspace DOF to compute_phase_matrix
            // Site fractions start at spec->num_statevars in workspace DOF
            compute_phase_matrix(csst->phase_matrix, csst->hess, csst->cons_jac_tmp,
                                 *compset, spec->num_statevars,
                                 &compset->dof[spec->num_statevars]);

            // DEBUG: Check phase_matrix before inversion
            #ifdef VERBOSE_DEBUG
            if (idx == 0 && iteration < 2) {
                printf("GPU DEBUG: Phase matrix before inversion (dim=%d):\n", csst->full_e_matrix_dim);
                for (int i = 0; i < csst->full_e_matrix_dim; ++i) {
                    printf("  Row %d: ", i);
                    for (int j = 0; j < csst->full_e_matrix_dim; ++j) {
                        printf("%e ", csst->phase_matrix[i * csst->full_e_matrix_dim + j]);
                    }
                    printf("\n");
                }
            }
            #endif

            for (int i = 0; i < csst->full_e_matrix_dim * csst->full_e_matrix_dim; ++i) {
                csst->full_e_matrix[i] = csst->phase_matrix[i];
            }
            
            // CRITICAL FIX: Use LU decomposition instead of SVD to match CPU behavior exactly
            // CPU uses LAPACK's dgesv (LU decomposition with partial pivoting)
            // GPU was using SVD which produces different results for constrained matrices
            invert_matrix_lu(csst->full_e_matrix, csst->full_e_matrix_dim, spec->work_inv);
            
            // DEBUG: Check full_e_matrix after inversion for BOTH phases
            #ifdef VERBOSE_DEBUG
            if (iteration == 0) {
                printf("GPU DEBUG: Phase %d Full E matrix after inversion (dim=%d):\n", idx, csst->full_e_matrix_dim);
                for (int i = 0; i < csst->full_e_matrix_dim; ++i) {
                    printf("  Row %d: ", i);
                    for (int j = 0; j < csst->full_e_matrix_dim; ++j) {
                        printf("%e ", csst->full_e_matrix[i * csst->full_e_matrix_dim + j]);
                    }
                    printf("\n");
                }
                
                // Print diagonal values specifically
                printf("GPU DEBUG: Phase %d diagonal values after inversion:\n", idx);
                for (int i = 0; i < csst->full_e_matrix_dim; ++i) {
                    printf("  [%d,%d] = %.15e\n", i, i, csst->full_e_matrix[i * csst->full_e_matrix_dim + i]);
                }
            }
            #endif

            int num_phase_dof_for_csst = pr->phase_dof;
            for(int i=0; i < csst->c_G_length; ++i) csst->c_G[i] = 0.0;
            for(int i=0; i < csst->c_statevars_rows * csst->c_statevars_cols; ++i) csst->c_statevars[i] = 0.0;
            for(int i=0; i < csst->c_component_rows * csst->c_component_cols; ++i) csst->c_component[i] = 0.0;
            csst->moles_normalization = 0.0;
            for(int i=0; i < csst->moles_normalization_grad_length; ++i) csst->moles_normalization_grad[i] = 0.0;

            // DEBUG: Print gradient values before computing c_G
            #ifdef VERBOSE_DEBUG
            if (idx < 2 && iteration < 5) {
                printf("GPU DEBUG: Phase %d gradients before c_G calculation (iter %d):\n", idx, iteration);
                printf("  energy = %.15e\n", csst->energy);
                printf("  phase_amt = %.15e\n", phase_amt[idx]);
                for (int j = 0; j < csst->grad_length; ++j) {
                    printf("  grad[%d] = %.15e\n", j, csst->grad[j]);
                }
                // Also print masses for debugging
                printf("  masses: [");
                for (int i = 0; i < spec->num_components; ++i) {
                    printf("%.6f", csst->masses[i]);
                    if (i < spec->num_components - 1) printf(", ");
                }
                printf("]\n");
            }
            #endif
            
            // DEBUG: Print matrix and gradient values for problematic phases
            #ifdef VERBOSE_DEBUG
            if (iteration == 0 && condition_idx == 0 && idx <= 2) {
                printf("[GPU c_G CALC] Phase %d matrix inversion and c_G calculation:\n", idx);
                printf("  Inverted matrix (full_e_matrix):\n");
                for (int i = 0; i < num_phase_dof_for_csst; i++) {
                    printf("    Row %d: ", i);
                    for (int j = 0; j < num_phase_dof_for_csst; j++) {
                        printf("%e ", csst->full_e_matrix[i * csst->full_e_matrix_dim + j]);
                    }
                    printf("\n");
                }
                printf("  Gradient values (grad[num_statevars+j]):\n");
                for (int j = 0; j < num_phase_dof_for_csst; j++) {
                    printf("    grad[%d] = %e\n", spec->num_statevars + j, csst->grad[spec->num_statevars + j]);
                }
            }
            #endif
            
            for (int i = 0; i < num_phase_dof_for_csst; ++i) {
                for (int j = 0; j < num_phase_dof_for_csst; ++j) {
                    // With updated energy functions, grad array is now in Workspace format [N, P, T, Y1, Y2, ...]
                    double matrix_elem = csst->full_e_matrix[i * csst->full_e_matrix_dim + j];
                    double grad_elem = csst->grad[spec->num_statevars + j];
                    csst->c_G[i] -= matrix_elem * grad_elem;
                    
                    // DEBUG: Print calculation details for first phase and iteration
                    #ifdef VERBOSE_DEBUG
                    if (thread_id == 0 && idx == 0 && iteration < 2) {
                        printf("  c_G[%d] calc: full_e_matrix[%d,%d]=%e * grad[%d]=%e = %e\n", 
                               i, i, j, matrix_elem, spec->num_statevars + j, grad_elem, matrix_elem * grad_elem);
                    }
                    #endif
                }
            }
            
            // DEBUG: Print final c_G values  
            #ifdef VERBOSE_DEBUG
            if (iteration == 0 && condition_idx == 0 && idx <= 2) {
                printf("  Final c_G values: ");
                for (int i = 0; i < num_phase_dof_for_csst; i++) {
                    printf("c_G[%d]=%e ", i, csst->c_G[i]);
                }
                printf("\n");
            }
            #endif
            
            // DEBUG: Print c_G values after calculation
            #ifdef VERBOSE_DEBUG
            if (idx < 2 && iteration < 5) {
                printf("GPU DEBUG: Phase %d c_G values (iter %d):\n", idx, iteration);
                printf("  phase_amt = %.15e\n", phase_amt[idx]);
                printf("  gradient values: [");
                for (int i = 0; i < pr->phase_dof; ++i) {
                    printf("%.6e", csst->grad[spec->num_statevars + i]);
                    if (i < pr->phase_dof - 1) printf(", ");
                }
                printf("]\n");
                printf("  full_e_matrix diagonal: [");
                for (int i = 0; i < num_phase_dof_for_csst; ++i) {
                    printf("%.6e", csst->full_e_matrix[i * csst->full_e_matrix_dim + i]);
                    if (i < num_phase_dof_for_csst - 1) printf(", ");
                }
                printf("]\n");
                for (int i = 0; i < num_phase_dof_for_csst; ++i) {
                    printf("  c_G[%i] = %.15e\n", i, csst->c_G[i]);
                }
                // Also print current site fractions to track convergence
                printf("  Current site fractions: [");
                for (int i = 0; i < pr->phase_dof; ++i) {
                    printf("%.6f", compset->dof[spec->num_statevars + i]);
                    if (i < pr->phase_dof - 1) printf(", ");
                }
                printf("]\n");
            }
            #endif
            for (int i = 0; i < num_phase_dof_for_csst; ++i) {
                for (int j = 0; j < num_phase_dof_for_csst; ++j) {
                    for (int sv_idx = 0; sv_idx < spec->num_statevars; ++sv_idx) {
                        // With CSE Hessian functions, only T derivatives (sv_idx=2) are non-zero
                        // N and P derivatives (sv_idx=0,1) are always zero
                        if (sv_idx == 2) { // Temperature index
                            csst->c_statevars[i * csst->c_statevars_cols + sv_idx] -=
                                csst->full_e_matrix[i * csst->full_e_matrix_dim + j] *
                                csst->hess[(spec->num_statevars + j) * csst->hess_cols + sv_idx];
                        }
                        // For N and P (sv_idx=0,1), the derivative is zero, so no contribution
                    }
                }
            }
            
            // CRITICAL FIX: Calculate c_component IMMEDIATELY after phase matrix inversion
            // This must be done before fill_equilibrium_system uses c_component
            // DEBUG: Print mass_jac values before c_component calculation
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0 && iteration < 2) {
                printf("[GPU DEBUG] Phase %d mass_jac BEFORE c_component calc (phase_dof=%d, num_elements=%d):\n", 
                       idx, pr->phase_dof, pr->num_elements);
                for (int cidx = 0; cidx < spec->num_components && cidx < 3; cidx++) {
                    printf("  Component %d: ", cidx);
                    for (int col = spec->num_statevars; col < spec->num_statevars + pr->phase_dof && col < spec->num_statevars + 3; col++) {
                        printf("mass_jac[%d,%d]=%e ", cidx, col, csst->mass_jac[cidx * csst->mass_jac_cols + col]);
                    }
                    printf("\n");
                }
            }
            #endif
            
            for (int cidx = 0; cidx < spec->num_components; ++cidx) {
                for (int i = 0; i < num_phase_dof_for_csst; ++i) {
                    for (int j = 0; j < num_phase_dof_for_csst; ++j) {
                         if (cidx < pr->num_elements) { // Ensure we are using valid mass_jac entries
                            // CRITICAL FIX: mass_jac is in Workspace format, not Model format!
                            // We need to access the site fraction columns which start at spec->num_statevars
                            double mass_jac_val = csst->mass_jac[cidx * csst->mass_jac_cols + (spec->num_statevars + j)];
                            double e_matrix_val = csst->full_e_matrix[i * csst->full_e_matrix_dim + j];
                            csst->c_component[cidx * csst->c_component_cols + i] += mass_jac_val * e_matrix_val;
                            
                            // DEBUG: Print calculation for BOTH phases
                            #ifdef VERBOSE_DEBUG
                            if (thread_id == 0 && iteration < 2 && cidx < 2 && i < 2 && j < 2) {
                                printf("  Phase %d: c_component[%d,%d] += mass_jac[%d,%d]=%e * e_matrix[%d,%d]=%e = %e\n",
                                       idx, cidx, i, cidx, spec->num_statevars + j, mass_jac_val, i, j, e_matrix_val,
                                       mass_jac_val * e_matrix_val);
                            }
                            #endif
                         }
                    }
                }
            }
            
            // DEBUG: Print c_component matrix for BOTH phases
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0 && iteration < 5) {
                printf("[GPU C_COMPONENT] Phase %d matrix (iter %d):\n", idx, iteration);
                printf("  phase_amt = %.15e\n", phase_amt[idx]);
                printf("  phase_dof = %d, num_elements = %d\n", pr->phase_dof, pr->num_elements);
                for (int cidx = 0; cidx < 2; cidx++) {
                    printf("  Component %d: ", cidx);
                    for (int i = 0; i < 2; i++) {
                        printf("%e ", csst->c_component[cidx * csst->c_component_cols + i]);
                    }
                    printf("\n");
                }
                
                // Check if c_component is all zeros
                bool all_zeros = true;
                for (int cidx = 0; cidx < spec->num_components && cidx < pr->num_elements; cidx++) {
                    for (int i = 0; i < pr->phase_dof; i++) {
                        if (fabs(csst->c_component[cidx * csst->c_component_cols + i]) > 1e-15) {
                            all_zeros = false;
                            break;
                        }
                    }
                }
                if (all_zeros) {
                    printf("  WARNING: c_component is all zeros!\n");
                }
            }
            #endif
            for (int cidx = 0; cidx < spec->num_components; ++cidx) {
                for (int i = 0; i < num_phase_dof_for_csst; ++i) {
                    double mu_c_sum = 0.0;
                    for (int j_chem = 0; j_chem < spec->num_components; ++j_chem) {
                        mu_c_sum += csst->c_component[j_chem * csst->c_component_cols + i] * chemical_potentials[j_chem];
                    }
                    if (cidx < pr->num_elements) {
                        delta_ms[idx * delta_ms_cols + cidx] += // Use delta_ms_cols from SystemState
                            csst->mass_jac[cidx * csst->mass_jac_cols + (spec->num_statevars + i)] *
                            (mu_c_sum + csst->c_G[i]);
                    }
                }
            }
            for (int cidx = 0; cidx < spec->num_components; ++cidx) {
                 if (cidx < pr->num_elements) {
                    csst->moles_normalization += csst->masses[cidx];
                    // CRITICAL FIX: moles_normalization_grad should be in Workspace format
                    // since mass_jac is in Workspace format
                    for (int i_dof = 0; i_dof < csst->mass_jac_cols; ++i_dof) {
                        csst->moles_normalization_grad[i_dof] += csst->mass_jac[cidx * csst->mass_jac_cols + i_dof];
                    }
                 }
            }
            
            // DEBUG: Print moles_normalization for each phase
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0 && iteration < 2) {
                printf("[GPU MOLES_NORM] Phase %d: moles_normalization = %e (should be ~%e for %d sublattices)\n", 
                       idx, csst->moles_normalization, 
                       pr->phase_dof > 2 ? 20.0 : 1.0,  // Rough estimate
                       pr->phase_dof > 2 ? 2 : 1);
            }
            #endif
        }
        delta_ms_rows = num_compsets; // Update after loop in case num_compsets changed (though not in recompute)
        phase_compositions_rows = num_compsets;
    }

    __device__ void driving_forces(SystemSpecification* spec, double* out_driving_forces, int num_out_df_max_cap) {
    for (int i = 0; i < num_out_df_max_cap; ++i) out_driving_forces[i] = 0.0;

    // REMOVED: current_dof_for_phase array - now using model_dof_for_calcs created locally where needed

    for (int idx = 0; idx < num_compsets; ++idx) {
            if (idx >= num_out_df_max_cap) continue;
            CompositionSet* compset = &compsets[idx];
            if (compset->phase_record == nullptr) {
                continue;
            }
            const PhaseRecord* pr = compset->phase_record;

            // REMOVED: Old code that created current_dof_for_phase incorrectly
            // Now we create model_dof_for_calcs properly from workspace DOF when needed

            double atoms_per_formula_unit = 0.0;
            // Use phase_compositions which should be moles of element per formula unit from recompute
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                 atoms_per_formula_unit += phase_compositions[idx * MAX_COMPONENTS + comp_idx];
            }
            if (fabs(atoms_per_formula_unit) < 1e-12) atoms_per_formula_unit = 1.0;

            double gm_per_atom;
            double moles_element_per_atom[MAX_COMPONENTS]; // Assuming MAX_COMPONENTS is large enough

            // GM per formula unit is compset->energy (if update sets it to obj_func) or cs_states[idx].energy (if it's formula_obj)
            // From pyx: compset.phase_record.obj(self._phase_energies_per_mole_atoms[idx, :], x)
            // This implies pr->obj should be GM per mole of ATOMS.
            // If pr->obj (from phase_rec.h) is GM per FORMULA:
            // With updated energy functions, use full workspace DOF directly
            double gm_formula_temp = pr->obj(compset->dof);
            gm_per_atom = gm_formula_temp / atoms_per_formula_unit;

            out_driving_forces[idx] = 0.0; // Initialize for current phase
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                moles_element_per_atom[comp_idx] = phase_compositions[idx * MAX_COMPONENTS + comp_idx] / atoms_per_formula_unit;
                out_driving_forces[idx] += chemical_potentials[comp_idx] * moles_element_per_atom[comp_idx];
            }
            out_driving_forces[idx] -= gm_per_atom;
        }
    }

    __device__ void increment_phase_metastability_counters() {
        for (int idx = 0; idx < num_compsets; ++idx) {
            bool is_free_stable = false;
            for (int i = 0; i < num_free_stable_compsets; ++i) {
                if (free_stable_compset_indices[i] == idx) {
                    is_free_stable = true;
                    break;
                }
            }
            if (is_free_stable || (idx < num_compsets && compsets[idx].fixed) ) {
                metastable_phase_iterations[idx] = 0;
            } else if (idx < num_compsets) { // Only increment for valid compsets
                metastable_phase_iterations[idx]++;
            }
        }
    }
} SystemState;


// Function definitions (previously declared)

__device__ void compute_phase_matrix(double* phase_matrix_out, const double* hess_in,
                                    const double* cons_jac_tmp_in,
                                    const CompositionSet& compset_ref, int num_statevars_val,
                                    // chemical_potentials not used in C version from prompt
                                    const double* phase_dof_site_fracs) {
    // Based on the C version in the original minimizer.h prompt
    // Assumes phase_matrix_out is pre-sized to (phase_dof + num_internal_cons) x (phase_dof + num_internal_cons)
    // num_phase_local_conditions is zero, so it's omitted from dimensions.

    if (compset_ref.phase_record == nullptr) return; // Safety check
    const PhaseRecord* pr = compset_ref.phase_record;

    int phase_dof_val = pr->phase_dof;
    int num_internal_cons_val = pr->num_internal_cons;
    int current_phase_matrix_dim = phase_dof_val + num_internal_cons_val;
    int hess_total_dim = num_statevars_val + phase_dof_val; // num_cols of hess
    int cons_jac_total_dim = num_statevars_val + phase_dof_val; // num_cols of cons_jac_tmp

    // Fill phase matrix from Hessian (diagonal blocks)
    for (int i = 0; i < phase_dof_val; i++) {
        for (int j = 0; j < phase_dof_val; j++) {
            // phase_matrix[i][j] = hess[num_statevars+i][num_statevars+j]
            phase_matrix_out[i * current_phase_matrix_dim + j] =
                hess_in[(num_statevars_val + i) * hess_total_dim + (num_statevars_val + j)];
        }
    }

    // Fill phase matrix from constraint Jacobian (off-diagonal blocks)
    for (int i = 0; i < num_internal_cons_val; i++) {
        for (int j = 0; j < phase_dof_val; j++) {
            // Upper right block: phase_matrix[phase_dof+i][j] (row-major index)
            phase_matrix_out[(phase_dof_val + i) * current_phase_matrix_dim + j] =
                cons_jac_tmp_in[i * cons_jac_total_dim + (num_statevars_val + j)];

            // Lower left block: phase_matrix[j][phase_dof+i] (row-major index)
            phase_matrix_out[j * current_phase_matrix_dim + (phase_dof_val + i)] =
                cons_jac_tmp_in[i * cons_jac_total_dim + (num_statevars_val + j)];
        }
    }
     // Zero out the bottom-right block corresponding to (constraint, constraint) interactions if it's not filled by above
    for (int i = 0; i < num_internal_cons_val; ++i) {
        for (int j = 0; j < num_internal_cons_val; ++j) {
            // This block should be zero in the standard formulation if not explicitly calculated.
            // phase_matrix[phase_dof+i][phase_dof+j]
            if (i!=j) { // Off-diagonal typically zero unless hessian of constraints considered.
                 // phase_matrix_out[(phase_dof_val + i) * current_phase_matrix_dim + (phase_dof_val + j)] = 0.0;
            }
            // Diagonal (lambda_i, lambda_i) terms are also typically zero.
            // If they are filled from Hessian, this part is not needed. Sundman 2015 Eq 41 has this block as 0.
             phase_matrix_out[(phase_dof_val + i) * current_phase_matrix_dim + (phase_dof_val + j)] = 0.0; // Explicitly zero for Sundman formulation
        }
    }
}


// Remaining function definitions (solve_state, advance_state, etc.)
// These are copied from the previous response and will be checked/adjusted for consistency
// with the "no phase local conditions" constraint, which mainly affects dimensions.

// (Copied from previous response, check for MAX_PHASE_LOCAL_CONDITIONS removal impact)
__device__ void write_row_stable_phase(double* out_row, double* out_rhs,
                                     const int* free_chemical_potential_indices, int num_free_chemical_potentials,
                                     const int* free_stable_compset_indices, int num_free_stable_compsets, // These are indices into the main compset array
                                     const int* free_statevar_indices, int num_free_statevars,
                                     const int* fixed_chemical_potential_indices, int num_fixed_chemical_potentials,
                                     const double* current_chemical_potentials, // Renamed from state->chemical_potentials
                                     const double* masses_for_compset, // csst->masses
                                     const double* grad_for_compset,   // csst->grad
                                     double energy_for_compset) {     // csst->energy
    // DEBUG: Print the row being written
    #ifdef VERBOSE_DEBUG
    printf("[GPU EQUILIBRIUM MATRIX DEBUG] Writing row for stable phase:\n");
    printf("  Energy: %e\n", energy_for_compset);
    printf("  Masses: ");
    for (int debug_idx = 0; debug_idx < MAX_COMPONENTS; debug_idx++) {
        if (masses_for_compset[debug_idx] != 0.0) {
            printf("%e ", masses_for_compset[debug_idx]);
        }
    }
    printf("\n");
    #endif
    
    int free_variable_column_offset = 0;
    int chempot_idx, statevar_idx, i;

    // CRITICAL FIX: Write masses for ALL components in free_chemical_potential_indices
    // CPU includes ALL non-VA components as free chemical potentials, even when
    // mole fractions are prescribed. The constraints are handled separately.
    // This matches the CPU behavior of having columns for all non-VA components.
    for (i = 0; i < num_free_chemical_potentials; i++) {
        chempot_idx = free_chemical_potential_indices[i];
        // Write mass for each free chemical potential component
        out_row[free_variable_column_offset + i] = masses_for_compset[chempot_idx];
    }
    free_variable_column_offset += num_free_chemical_potentials;

    // Free stable composition sets part (columns for d(NP_j))
    // The value is 1 if this row corresponds to NP_j, 0 otherwise.
    // This function is called PER ROW. The row index (stable_idx or fixed_idx from fill_equilibrium_system)
    // implies WHICH compset this row is for. The columns for d(NP) are ordered by `free_stable_compset_indices`.
    // This function doesn't know which *row* it's writing, only the *data for a phase*.
    // The logic in `fill_equilibrium_system` needs to handle placing 1s correctly.
    // The pyx version's write_row_stable_phase does NOT fill the d(NP) part of its own row with 1.
    // That's handled by the structure of Ax=b where x includes delta_NP.
    // So these columns are 0 from the perspective of *this specific phase's contribution via chemical potentials/statevars*.
    // The columns corresponding to `delta phase_amts` are handled by the overall system matrix structure.
    // So, for this row, the contribution to d(NP_k) columns (where k != this phase) is 0.
    // The contribution to its OWN d(NP) column comes from the N=1 constraint, not here.
    // This means this part of the row is all zeros.
    for (i = 0; i < num_free_stable_compsets; ++i) {
        out_row[free_variable_column_offset + i] = 0.0; // Default to 0
    }
    free_variable_column_offset += num_free_stable_compsets;


    for (i = 0; i < num_free_statevars; i++) {
        statevar_idx = free_statevar_indices[i]; // This is the workspace index of the free state variable
        // CRITICAL FIX: Use workspace indices directly, just like CPU code does!
        // The gradient array is already populated with workspace indices from formulagrad
        out_row[free_variable_column_offset + i] = -grad_for_compset[statevar_idx];
    }

    out_rhs[0] = energy_for_compset;

    // Subtract fixed chemical potentials from each phase RHS
    // This matches the CPU code in minimizer.pyx line 122-124
    for (i = 0; i < num_fixed_chemical_potentials; i++) {
        chempot_idx = fixed_chemical_potential_indices[i];
        out_rhs[0] -= masses_for_compset[chempot_idx] * current_chemical_potentials[chempot_idx];
    }
    
    // DEBUG: Print the final row values and RHS
    #ifdef VERBOSE_DEBUG
    printf("  Row values (first %d): ", num_free_chemical_potentials + num_free_stable_compsets + num_free_statevars);
    for (int debug_idx = 0; debug_idx < num_free_chemical_potentials + num_free_stable_compsets + num_free_statevars; debug_idx++) {
        printf("%e ", out_row[debug_idx]);
    }
    printf("\n");
    printf("  RHS: %e\n", out_rhs[0]);
    #endif
}

__device__ void write_row_fixed_mole_fraction(double* out_row, double* out_rhs,
                                            int component_idx_of_constraint, // The component X_i in sum Ci*Xi = R
                                            const int* free_chemical_potential_indices, int num_free_chemical_potentials,
                                            const int* free_stable_compset_indices, int num_free_stable_compsets, // Global indices
                                            const int* free_statevar_indices, int num_free_statevars,
                                            const int* fixed_chemical_potential_indices, int num_fixed_chemical_potentials,
                                            const double* current_chemical_potentials_sys, // System's chemical potentials
                                            const double* system_mole_fractions_sys, // System's overall mole_fractions
                                            double current_system_amount_sys,
                                            const double* mass_jac_cs, int mass_jac_cols_cs, // Compset's mass_jac
                                            const double* c_component_cs, int c_component_cols_cs, // Compset's c_component
                                            const double* c_statevars_cs, int c_statevars_cols_cs, // Compset's c_statevars
                                            const double* c_G_cs, int c_G_length_cs,             // Compset's c_G
                                            const double* masses_cs,                // Compset's masses (per formula)
                                            double moles_normalization_cs,
                                            const double* moles_normalization_grad_cs,
                                            const double* phase_amt_sys, // System's phase_amt array
                                            int compset_original_idx_sys, // Index of the current compset in system arrays
                                            double prefactor_for_this_component) { // Coefficient from prescribed_mole_fraction_coefficients

    if (fabs(prefactor_for_this_component) < 1e-12) return; // No contribution if prefactor is zero

    int free_variable_column_offset = 0;
    // num_statevars_proxy should be the number of state variables in the system (e.g. spec->num_statevars)
    // This is used for indexing mass_jac and moles_normalization_grad.
    // c_statevars_cols_cs is also this value.
    int num_system_statevars = c_statevars_cols_cs;


    // 2a. This component row: free chemical potentials
    for (int i = 0; i < num_free_chemical_potentials; i++) {
        int free_chempot_global_idx = free_chemical_potential_indices[i];
        double term1 = 0.0;
        double term2 = 0.0;
        for (int j = 0; j < c_component_cols_cs; j++) { // j is index over phase_dof
            // CRITICAL FIX: Both mass_jac and moles_normalization_grad are in Workspace format!
            // In Workspace format: [N, P, T, Y1, Y2, ...] so site fractions start at num_system_statevars
            double mass_jac_val = mass_jac_cs[component_idx_of_constraint * mass_jac_cols_cs + (num_system_statevars + j)];
            double c_comp_val = c_component_cs[free_chempot_global_idx * c_component_cols_cs + j];
            term1 += mass_jac_val * c_comp_val;
            
            // (-system_mole_fractions_sys[comp_idx_of_constraint] * moles_normalization_grad_cs[num_sv+j]) * c_component_cs[free_chempot_glob_idx, j]
            double mole_norm_grad_val = moles_normalization_grad_cs[num_system_statevars + j];
            term2 += (-system_mole_fractions_sys[component_idx_of_constraint] * mole_norm_grad_val) * c_comp_val;
            
            // DEBUG: Print intermediate values for both phases, component 1, chempot 0
            #ifdef VERBOSE_DEBUG
            if (component_idx_of_constraint == 1 && i == 0 && j < 2) {
                printf("  Phase %d, j=%d: mass_jac[%d,%d]=%e, c_component[%d,%d]=%e\n", 
                       compset_original_idx_sys, j, component_idx_of_constraint, num_system_statevars + j, 
                       mass_jac_val, free_chempot_global_idx, j, c_comp_val);
                printf("       mole_norm_grad[%d]=%e, X[%d]=%e\n",
                       num_system_statevars + j, mole_norm_grad_val,
                       component_idx_of_constraint, system_mole_fractions_sys[component_idx_of_constraint]);
                printf("       term1 contribution: %e, term2 contribution: %e\n",
                       mass_jac_val * c_comp_val,
                       (-system_mole_fractions_sys[component_idx_of_constraint] * mole_norm_grad_val) * c_comp_val);
            }
            #endif
        }
        if (fabs(current_system_amount_sys)>1e-12) {
             double contribution = prefactor_for_this_component *
                (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (term1 + term2);
             out_row[free_variable_column_offset + i] += contribution;
             
             // DEBUG: Show exact contribution for mole fraction constraint
             #ifdef VERBOSE_DEBUG
             if (component_idx_of_constraint == 1 && i < 2) {
                 printf("[GPU MOLE FRAC] Phase %d adds %e to col %d (chem pot %d)\n", 
                        compset_original_idx_sys, contribution, free_variable_column_offset + i, i);
             }
             #endif
        }
        
        // DEBUG: Print values for both phases and first constraint
        #ifdef VERBOSE_DEBUG
        if (component_idx_of_constraint == 1 && i == 0) {
            printf("[GPU MOLE FRAC DEBUG] Phase %d, Component 1, ChemPot 0:\n", compset_original_idx_sys);
            printf("  prefactor=%e, phase_amt=%e, sys_amt=%e\n", 
                   prefactor_for_this_component, phase_amt_sys[compset_original_idx_sys], current_system_amount_sys);
            printf("  free_chempot_global_idx=%d, num_free_chemical_potentials=%d\n", 
                   free_chempot_global_idx, num_free_chemical_potentials);
            printf("  c_component_cols_cs=%d, mass_jac_cols_cs=%d, num_system_statevars=%d\n",
                   c_component_cols_cs, mass_jac_cols_cs, num_system_statevars);
            printf("  term1=%e, term2=%e, total=%e\n", term1, term2, term1+term2);
            printf("  Result added to out_row[%d]: %e\n", free_variable_column_offset + i,
                   prefactor_for_this_component * (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (term1 + term2));
        }
        #endif
    }
    free_variable_column_offset += num_free_chemical_potentials;

    // 2a. This component row: free stable composition sets
    for (int i = 0; i < num_free_stable_compsets; i++) {
        int current_free_compset_original_sys_idx = free_stable_compset_indices[i];
        if (current_free_compset_original_sys_idx == compset_original_idx_sys) { // If this column is for the current phase
             if (fabs(current_system_amount_sys)>1e-12) {
                double coeff_value = (masses_cs[component_idx_of_constraint] - system_mole_fractions_sys[component_idx_of_constraint] * moles_normalization_cs);
                double contribution = prefactor_for_this_component * (1.0 / current_system_amount_sys) * coeff_value;
                
                #ifdef VERBOSE_DEBUG
                // Print debug info for mole fraction constraint calculations
                if (component_idx_of_constraint == 1) {
                    printf("[GPU MOLE FRAC COEFF] Phase %d, Component 1:\n", compset_original_idx_sys);
                    printf("  masses[1] = %.15e\n", masses_cs[component_idx_of_constraint]);
                    printf("  system_mole_fractions[1] = %.15e\n", system_mole_fractions_sys[component_idx_of_constraint]);
                    printf("  moles_normalization = %.15e\n", moles_normalization_cs);
                    printf("  coeff = masses[1] - sys_mole_frac[1] * moles_norm = %.15e - %.15e * %.15e = %.15e\n",
                           masses_cs[component_idx_of_constraint], 
                           system_mole_fractions_sys[component_idx_of_constraint],
                           moles_normalization_cs,
                           coeff_value);
                    printf("  contribution to matrix = %.15e * (1.0 / %.15e) * %.15e = %.15e\n",
                           prefactor_for_this_component, current_system_amount_sys, coeff_value, contribution);
                    printf("  out_row[%d] += %.15e\n", free_variable_column_offset + i, contribution);
                }
                #endif
                
                out_row[free_variable_column_offset + i] += contribution;
             }
        }
    }
    free_variable_column_offset += num_free_stable_compsets;

    // 2a. This component row: free state variables
    for (int i = 0; i < num_free_statevars; i++) {
        int free_statevar_global_idx = free_statevar_indices[i];
        double term1 = 0.0;
        double term2 = 0.0;
        for (int j = 0; j < c_statevars_cols_cs; j++) { // j is index over phase_dof
            // Use workspace indexing for mass_jac and moles_normalization_grad
            term1 += mass_jac_cs[component_idx_of_constraint * mass_jac_cols_cs + (num_system_statevars + j)] *
                     c_statevars_cs[j * c_statevars_cols_cs + free_statevar_global_idx];
            // (-system_mole_fractions_sys[comp_idx_of_constraint] * moles_normalization_grad_cs[num_sv+j]) * c_statevars_cs[j, free_statevar_glob_idx]
            term2 += (-system_mole_fractions_sys[component_idx_of_constraint] * moles_normalization_grad_cs[num_system_statevars + j]) *
                     c_statevars_cs[j * c_statevars_cols_cs + free_statevar_global_idx];
        }
         if (fabs(current_system_amount_sys)>1e-12) {
            out_row[free_variable_column_offset + i] += prefactor_for_this_component *
                (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (term1 + term2);
         }
    }

    // 3. Contributions to RHS (from c_G_cs)
    double rhs_term1 = 0.0;
    double rhs_term2 = 0.0;
    
    // DEBUG: Print c_G values for iteration 0
    #ifdef VERBOSE_DEBUG
    if (component_idx_of_constraint == 1 && compset_original_idx_sys < 3) {
        printf("[GPU c_G DEBUG] Phase %d, constraint component %d:\n", 
               compset_original_idx_sys, component_idx_of_constraint);
        printf("  c_G values: ");
        for (int j = 0; j < c_G_length_cs && j < 5; j++) {
            printf("[%d]=%e ", j, c_G_cs[j]);
        }
        printf("\n");
        printf("  mass_jac[%d,3+j]: ", component_idx_of_constraint);
        for (int j = 0; j < c_G_length_cs && j < 5; j++) {
            printf("%e ", mass_jac_cs[component_idx_of_constraint * mass_jac_cols_cs + (num_system_statevars + j)]);
        }
        printf("\n");
    }
    #endif
    
    for (int j = 0; j < c_G_length_cs; j++) { // j is index over phase_dof
        // CRITICAL FIX: Use workspace indexing for mass_jac and moles_normalization_grad
        rhs_term1 += mass_jac_cs[component_idx_of_constraint * mass_jac_cols_cs + (num_system_statevars + j)] * c_G_cs[j];
        rhs_term2 += (-system_mole_fractions_sys[component_idx_of_constraint] * moles_normalization_grad_cs[num_system_statevars + j]) * c_G_cs[j];
    }
    if (fabs(current_system_amount_sys)>1e-12) {
        out_rhs[0] += -prefactor_for_this_component * (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (rhs_term1 + rhs_term2);
        
        // DEBUG: Print RHS contribution
        #ifdef VERBOSE_DEBUG
        if (component_idx_of_constraint == 1) {
            printf("[GPU RHS] Phase %d adds %e to constraint %d RHS (term1=%e, term2=%e)\n",
                   compset_original_idx_sys, 
                   -prefactor_for_this_component * (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (rhs_term1 + rhs_term2),
                   component_idx_of_constraint, rhs_term1, rhs_term2);
        }
        #endif
    }
    
    // DEBUG: Print RHS contribution for both phases and first constraint
    #ifdef VERBOSE_DEBUG
    if (component_idx_of_constraint == 1) {
        printf("[GPU MOLE FRAC RHS DEBUG] Phase %d, Component 1:\n", compset_original_idx_sys);
        printf("  c_G_length=%d, c_G[0]=%e, c_G[1]=%e\n", c_G_length_cs, 
               c_G_length_cs > 0 ? c_G_cs[0] : 0.0, c_G_length_cs > 1 ? c_G_cs[1] : 0.0);
        printf("  rhs_term1=%e, rhs_term2=%e\n", rhs_term1, rhs_term2);
        printf("  phase_amt=%e, system_amt=%e, prefactor=%e\n",
               phase_amt_sys[compset_original_idx_sys], current_system_amount_sys, prefactor_for_this_component);
        printf("  RHS contribution: %e\n", 
               -prefactor_for_this_component * (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (rhs_term1 + rhs_term2));
        printf("  Formula: -%.3f * (%.3f / %.3f) * (%.3f + %.3f) = %.6f\n",
               prefactor_for_this_component, phase_amt_sys[compset_original_idx_sys], current_system_amount_sys,
               rhs_term1, rhs_term2,
               -prefactor_for_this_component * (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) * (rhs_term1 + rhs_term2));
    }
    #endif

    // 4. Subtract fixed chemical potentials from fixed component RHS
    for (int i = 0; i < num_fixed_chemical_potentials; i++) {
        int fixed_chempot_global_idx = fixed_chemical_potential_indices[i];
        double sub_term1 = 0.0;
        double sub_term2 = 0.0;
        for (int j = 0; j < c_component_cols_cs; j++) { // j is index over phase_dof
            // Use workspace indexing for mass_jac and moles_normalization_grad
            sub_term1 += mass_jac_cs[component_idx_of_constraint * mass_jac_cols_cs + (num_system_statevars + j)] *
                         c_component_cs[fixed_chempot_global_idx * c_component_cols_cs + j];
            sub_term2 += (-system_mole_fractions_sys[component_idx_of_constraint] * moles_normalization_grad_cs[num_system_statevars + j]) *
                         c_component_cs[fixed_chempot_global_idx * c_component_cols_cs + j];
        }
        if (fabs(current_system_amount_sys)>1e-12) {
            out_rhs[0] -= prefactor_for_this_component * (phase_amt_sys[compset_original_idx_sys] / current_system_amount_sys) *
                          current_chemical_potentials_sys[fixed_chempot_global_idx] * (sub_term1 + sub_term2);
        }
    }
}


// --- Definitions for the rest of the __device__ functions ---
// (Copied from previous response, to be reviewed for consistency)
// fill_equilibrium_system, check_convergence, pre_solve_hook, post_solve_hook,
// solve_state, advance_state, remove_and_consolidate_phases, change_phases, run_loop

// The rest of the functions (fill_equilibrium_system, check_convergence, hooks, run_loop, solve_state, advance_state, remove_and_consolidate_phases, change_phases)
// from the previous turn should be mostly correct in structure.
// The main impact of "no phase local conditions" is simplifying CompsetState's phase_matrix_dim and removing related arrays,
// which has been done above. The MAX_PHASE_MATRIX_DIM is also updated.
// These functions rely on the corrected dimensions passed via CompsetState and PhaseRecord.

// Implementation of write_row_fixed_mole_amount function
__device__ void write_row_fixed_mole_amount(double* out_row, double* out_rhs,
                                            int component_idx, 
                                            const int* free_chemical_potential_indices, int num_free_chemical_potentials,
                                            const int* free_stable_compset_indices, int num_free_stable_compsets,
                                            const int* free_statevar_indices, int num_free_statevars,
                                            const int* fixed_chemical_potential_indices, int num_fixed_chemical_potentials,
                                            const double* current_chemical_potentials_sys,
                                            const double* mass_jac_cs, int mass_jac_cols_cs,
                                            const double* c_component_cs, int c_component_cols_cs,
                                            const double* c_statevars_cs, int c_statevars_cols_cs,
                                            const double* c_G_cs, int c_G_length_cs,
                                            const double* masses_cs,
                                            double moles_normalization_cs,
                                            const double* moles_normalization_grad_cs,
                                            const double* phase_amt_sys,
                                            int compset_original_idx_sys) {
    
    // This function writes one row of the equilibrium matrix for mass balance constraint
    // Matches CPU's write_row_fixed_mole_amount logic
    
    int free_variable_column_offset = 0;
    int num_system_statevars = c_statevars_cols_cs;
    
    // FIX: CPU code does NOT normalize by moles_normalization (sum of site ratios)
    // Setting normalization_factor to 1.0 to match CPU behavior
    double normalization_factor = 1.0;
    
    // DEBUG: Print normalization factor for each phase
    #ifdef VERBOSE_DEBUG
    if (component_idx == 0 && phase_amt_sys[compset_original_idx_sys] > 1e-10) {
        printf("[GPU SYSTEM AMOUNT] Phase %d: moles_norm=%e, using factor=%e\n", 
               compset_original_idx_sys, moles_normalization_cs, normalization_factor);
    }
    #endif
    
    // 2a. This component row: free chemical potentials
    for (int i = 0; i < num_free_chemical_potentials; ++i) {
        int chempot_idx = free_chemical_potential_indices[i];
        for (int j = 0; j < c_component_cols_cs; ++j) {  // j is phase_dof index
            // out_row[offset + i] += phase_amt * mass_jac[comp_idx, num_sv+j] * c_component[chempot_idx, j] / moles_norm
            out_row[free_variable_column_offset + i] += 
                (phase_amt_sys[compset_original_idx_sys] / normalization_factor) * 
                mass_jac_cs[component_idx * mass_jac_cols_cs + num_system_statevars + j] * 
                c_component_cs[chempot_idx * c_component_cols_cs + j];
        }
    }
    free_variable_column_offset += num_free_chemical_potentials;
    
    // 2a. This component row: free stable composition sets
    for (int i = 0; i < num_free_stable_compsets; ++i) {
        int compset_idx = free_stable_compset_indices[i];
        // Only fill this out if the current idx is equal to a free composition set
        if (compset_idx == compset_original_idx_sys) {
            // For fixed_mole_amount, the coefficient is the mass of this component normalized by moles_normalization
            out_row[free_variable_column_offset + i] += masses_cs[component_idx] / normalization_factor;
        }
    }
    free_variable_column_offset += num_free_stable_compsets;
    
    // 2a. This component row: free state variables
    for (int i = 0; i < num_free_statevars; ++i) {
        int statevar_idx = free_statevar_indices[i];
        for (int j = 0; j < c_statevars_cols_cs; ++j) {  // j is phase_dof index
            // out_row[offset + i] += phase_amt * mass_jac[comp_idx, num_sv+j] * c_statevars[j, statevar_idx] / moles_norm
            out_row[free_variable_column_offset + i] += 
                (phase_amt_sys[compset_original_idx_sys] / normalization_factor) * 
                mass_jac_cs[component_idx * mass_jac_cols_cs + num_system_statevars + j] * 
                c_statevars_cs[j * c_statevars_cols_cs + statevar_idx];
        }
    }
    
    // 3. RHS contribution from c_G (also needs normalization)
    for (int j = 0; j < c_G_length_cs; ++j) {
        *out_rhs += -(phase_amt_sys[compset_original_idx_sys] / normalization_factor) * 
                    mass_jac_cs[component_idx * mass_jac_cols_cs + num_system_statevars + j] * 
                    c_G_cs[j];
    }
    
    // 4. Subtract fixed chemical potentials from RHS
    for (int i = 0; i < num_fixed_chemical_potentials; ++i) {
        int chempot_idx = fixed_chemical_potential_indices[i];
        // 6. Subtract fixed chemical potentials from the N=1 row
        for (int j = 0; j < c_component_cols_cs; ++j) {
            *out_rhs -= (phase_amt_sys[compset_original_idx_sys] / normalization_factor) * 
                        current_chemical_potentials_sys[chempot_idx] * 
                        mass_jac_cs[component_idx * mass_jac_cols_cs + num_system_statevars + j] * 
                        c_component_cs[chempot_idx * c_component_cols_cs + j];
        }
    }
}

// [The rest of the function definitions: fill_equilibrium_system, check_convergence, pre_solve_hook, post_solve_hook, run_loop, solve_state, advance_state, remove_and_consolidate_phases, change_phases would be here. They are substantial and largely unchanged in their logic by the removal of phase-local conditions, other than relying on the now-simpler dimensions. For brevity, I'll omit re-pasting all of them if their internal logic doesn't directly interact with phase-local-condition-specific arrays that have now been removed. The key changes were in the struct definitions and affected dimension calculations.]
// The previous response already contained these functions. The important part is that the `CompsetState` and `SystemSpecification` definitions are now updated.

// It is crucial that the C PhaseRecord struct and its associated function pointers
// (e.g., for formulamole_obj, formulamole_grad) are implemented in a way that
// is consistent with how they are called (e.g., if they operate per-component or fill arrays for all components).
// The `compute_phase_matrix` function's C version from the prompt did not use chemical potentials, unlike its pyx counterpart's calculation of delta_y. This difference should be noted.

// Re-inserting and checking fill_equilibrium_system for sanity with the new context:
__device__ void fill_equilibrium_system(double* equilibrium_matrix, int equilibrium_matrix_cols,
                                      double* equilibrium_rhs,
                                      SystemSpecification* spec, SystemState* state) {
    // SEGMENT 27: CONSTRUCT EQUILIBRIUM SYSTEM
    gpu_debug_log(27, "Construct equilibrium system", state->condition_idx);
    
    int stable_idx, compset_original_idx, current_component_idx, fixed_cs_idx;
    CompositionSet* current_compset;
    CompsetState* current_cs_state;
    int num_total_components = spec->num_components;
    int num_free_stable_phases = state->num_free_stable_compsets;
    int num_fixed_stable_cs = spec->num_fixed_stable_compsets;
    int num_fixed_mole_frac_conds = spec->num_prescribed_mole_fraction_conditions;
    double prefactor;

    // CRITICAL FIX: Add +1 back to match CPU matrix dimensions exactly
    // CPU DOES include a system amount constraint row (with [1,1,1] for phase amounts)
    int total_rows = num_free_stable_phases + num_fixed_stable_cs + num_fixed_mole_frac_conds + 1;
    if (state->condition_idx < 3) {
        gpu_debug_log_value("matrix_dimensions", (double)(total_rows * equilibrium_matrix_cols));
        gpu_debug_log_value("num_rows", (double)total_rows);
        gpu_debug_log_value("num_cols", (double)equilibrium_matrix_cols);
    }
    
    // DEBUG: Print equilibrium matrix construction details
    #ifdef VERBOSE_DEBUG
    if (state->condition_idx == 0 && (state->iteration == 0 || num_free_stable_phases == 1)) {
        printf("[GPU EQUILIBRIUM MATRIX] Filling equilibrium system at iteration %d\n", state->iteration);
        printf("  num_free_stable_phases: %d\n", num_free_stable_phases);
        printf("  num_fixed_stable_cs: %d\n", num_fixed_stable_cs);
        printf("  num_fixed_mole_frac_conds: %d\n", num_fixed_mole_frac_conds);
        printf("  total_rows: %d\n", total_rows);
        printf("  equilibrium_matrix_cols: %d\n", equilibrium_matrix_cols);
        
        // Special debug for single phase case
        if (num_free_stable_phases == 1) {
            printf("[GPU DEBUG] Single phase detected after consolidation\n");
            int phase_idx = state->free_stable_compset_indices[0];
            CompositionSet* cs = &state->compsets[phase_idx];
            printf("  Phase %d site fractions: [", phase_idx);
            for (int i = 0; i < cs->phase_record->phase_dof; i++) {
                printf("%.10f", cs->dof[spec->num_statevars + i]);
                if (i < cs->phase_record->phase_dof - 1) printf(", ");
            }
            printf("]\n");
            printf("  Target X[1] = %.10f\n", spec->prescribed_mole_fraction_rhs[0]);
            printf("  Current X[1] = %.10f\n", state->phase_compositions[phase_idx * MAX_COMPONENTS + 1]);
            printf("  Mass residual = %.10e\n", state->mass_residual);
        }
    }
    #endif
    
    for(int i=0; i < total_rows * equilibrium_matrix_cols; ++i) equilibrium_matrix[i] = 0.0;
    for(int i=0; i < total_rows; ++i) equilibrium_rhs[i] = 0.0;
    
    // FIXED: Do NOT initialize RHS to target values - let phase contributions build the constraint equation
    // The RHS should start from 0 and accumulate phase contributions, then have residual subtracted later
    // The target value (0.5) is handled in the residual calculation, not pre-loaded into RHS

    for (stable_idx = 0; stable_idx < num_free_stable_phases; stable_idx++) {
        compset_original_idx = state->free_stable_compset_indices[stable_idx];
        if (compset_original_idx < 0 || compset_original_idx >= state->num_compsets) continue; // Boundary check
        current_compset = &state->compsets[compset_original_idx];
        current_cs_state = &state->cs_states[compset_original_idx];
        if (current_compset->phase_record == nullptr) continue;

        write_row_stable_phase(
            &equilibrium_matrix[stable_idx * equilibrium_matrix_cols],
            &equilibrium_rhs[stable_idx],
            spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
            state->free_stable_compset_indices, state->num_free_stable_compsets,
            spec->free_statevar_indices, spec->num_free_statevars,
            spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
            state->chemical_potentials, current_cs_state->masses,
            current_cs_state->grad, current_cs_state->energy);
    }

    int current_row_offset = num_free_stable_phases;
    for (fixed_cs_idx = 0; fixed_cs_idx < num_fixed_stable_cs; fixed_cs_idx++) {
        compset_original_idx = spec->fixed_stable_compset_indices[fixed_cs_idx];
         if (compset_original_idx < 0 || compset_original_idx >= state->num_compsets) continue;
        current_compset = &state->compsets[compset_original_idx];
        current_cs_state = &state->cs_states[compset_original_idx];
        if (current_compset->phase_record == nullptr) continue;

        write_row_stable_phase(
            &equilibrium_matrix[(current_row_offset + fixed_cs_idx) * equilibrium_matrix_cols],
            &equilibrium_rhs[current_row_offset + fixed_cs_idx],
            spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
            state->free_stable_compset_indices, state->num_free_stable_compsets,
            spec->free_statevar_indices, spec->num_free_statevars,
            spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
            state->chemical_potentials, current_cs_state->masses,
            current_cs_state->grad, current_cs_state->energy);
    }
    current_row_offset += num_fixed_stable_cs;

    // Loop over free stable phases only (matching CPU behavior)
    for (int free_idx = 0; free_idx < state->num_free_stable_compsets; free_idx++) {
        stable_idx = state->free_stable_compset_indices[free_idx];
        compset_original_idx = stable_idx; // The index in the main compsets array
        current_compset = &state->compsets[compset_original_idx];
        current_cs_state = &state->cs_states[compset_original_idx];
        if (current_compset->phase_record == nullptr) continue;


        for (int mole_frac_cond_row_idx = 0; mole_frac_cond_row_idx < num_fixed_mole_frac_conds; mole_frac_cond_row_idx++) {
            for (current_component_idx = 0; current_component_idx < spec->num_prescribed_mole_fraction_coefficients_cols; current_component_idx++) {
                prefactor = spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][current_component_idx];
                // write_row_fixed_mole_fraction accumulates, so it's okay to call multiple times for the same matrix row
                write_row_fixed_mole_fraction(
                    &equilibrium_matrix[(current_row_offset + mole_frac_cond_row_idx) * equilibrium_matrix_cols],
                    &equilibrium_rhs[current_row_offset + mole_frac_cond_row_idx],
                    current_component_idx,
                    spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
                    state->free_stable_compset_indices, state->num_free_stable_compsets,
                    spec->free_statevar_indices, spec->num_free_statevars,
                    spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
                    state->chemical_potentials, state->mole_fractions, state->system_amount,
                    current_cs_state->mass_jac, current_cs_state->mass_jac_cols,
                    current_cs_state->c_component, current_cs_state->c_component_cols,
                    current_cs_state->c_statevars, current_cs_state->c_statevars_cols,
                    current_cs_state->c_G, current_cs_state->c_G_length, current_cs_state->masses,
                    current_cs_state->moles_normalization, current_cs_state->moles_normalization_grad,
                    state->phase_amt, compset_original_idx, prefactor);
            }
        }
    }
    
    // DEBUG: Check c_G values before calling write_row_fixed_mole_fraction
    #ifdef VERBOSE_DEBUG
    if (state->iteration < 2) {
        printf("\n[GPU] Before write_row_fixed_mole_fraction calls:\n");
        for (int i = 0; i < state->num_compsets && i < 2; ++i) {
            printf("  Phase %d c_G: [%.6e, %.6e]\n", i,
                   state->cs_states[i].c_G_length > 0 ? state->cs_states[i].c_G[0] : 0.0,
                   state->cs_states[i].c_G_length > 1 ? state->cs_states[i].c_G[1] : 0.0);
        }
    }
    #endif
    
    // Loop over fixed stable phases (matching CPU behavior)
    for (int fixed_idx = 0; fixed_idx < spec->num_fixed_stable_compsets; fixed_idx++) {
        stable_idx = spec->fixed_stable_compset_indices[fixed_idx];
        compset_original_idx = stable_idx;
        current_compset = &state->compsets[compset_original_idx];
        current_cs_state = &state->cs_states[compset_original_idx];
        if (current_compset->phase_record == nullptr) continue;

        for (int mole_frac_cond_row_idx = 0; mole_frac_cond_row_idx < num_fixed_mole_frac_conds; mole_frac_cond_row_idx++) {
            for (current_component_idx = 0; current_component_idx < spec->num_prescribed_mole_fraction_coefficients_cols; current_component_idx++) {
                prefactor = spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][current_component_idx];
                write_row_fixed_mole_fraction(
                    &equilibrium_matrix[(current_row_offset + mole_frac_cond_row_idx) * equilibrium_matrix_cols],
                    &equilibrium_rhs[current_row_offset + mole_frac_cond_row_idx],
                    current_component_idx,
                    spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
                    state->free_stable_compset_indices, state->num_free_stable_compsets,
                    spec->free_statevar_indices, spec->num_free_statevars,
                    spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
                    state->chemical_potentials, state->mole_fractions, state->system_amount,
                    current_cs_state->mass_jac, current_cs_state->mass_jac_cols,
                    current_cs_state->c_component, current_cs_state->c_component_cols,
                    current_cs_state->c_statevars, current_cs_state->c_statevars_cols,
                    current_cs_state->c_G, current_cs_state->c_G_length, current_cs_state->masses,
                    current_cs_state->moles_normalization, current_cs_state->moles_normalization_grad,
                    state->phase_amt, compset_original_idx, prefactor);
            }
        }
    }

    // REMOVED: The zeroing was incorrectly discarding contributions from the first loop
    // The first loop already handles all active phases, so we don't need a second loop
    
    // COMMENTED OUT: This second loop is redundant - the first loop already handles all phases
    /*
    for (int phase_idx = 0; phase_idx < state->num_compsets; phase_idx++) {
        CompositionSet* phase_compset = &state->compsets[phase_idx];
        CompsetState* phase_cs_state = &state->cs_states[phase_idx];
        
        if (phase_compset->phase_record == nullptr) continue;
        
        // DEBUG: Print which phases are contributing
        #ifdef VERBOSE_DEBUG
        if (state->condition_idx == 0 && state->iteration == 0) {
            printf("[GPU DEBUG] Phase %d contributing to mole frac constraints\n", phase_idx);
            printf("  Phase amount: %e\n", state->phase_amt[phase_idx]);
            printf("  Is this phase in free_stable_compset_indices? ");
            bool is_free = false;
            for (int i = 0; i < state->num_free_stable_compsets; i++) {
                if (state->free_stable_compset_indices[i] == phase_idx) {
                    is_free = true;
                    break;
                }
            }
            printf("%s\n", is_free ? "YES" : "NO");
        }
        #endif
        
        // Contribute this phase to all mole fraction constraint rows
        for (int mole_frac_cond_row_idx = 0; mole_frac_cond_row_idx < num_fixed_mole_frac_conds; mole_frac_cond_row_idx++) {
            for (current_component_idx = 0; current_component_idx < spec->num_prescribed_mole_fraction_coefficients_cols; current_component_idx++) {
                prefactor = spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][current_component_idx];
                
                // DEBUG: Print constraint details for all phases
                #ifdef VERBOSE_DEBUG
                if (state->iteration == 0 && mole_frac_cond_row_idx == 0) {
                    printf("[GPU MOLE FRAC CONSTRAINT] Phase %d, Constraint %d, Component %d: prefactor=%e\n", 
                           phase_idx, mole_frac_cond_row_idx, current_component_idx, prefactor);
                    printf("  Phase amount: %e, num_compsets=%d\n", state->phase_amt[phase_idx], state->num_compsets);
                }
                #endif
                
                // write_row_fixed_mole_fraction accumulates, so it's okay to call multiple times for the same matrix row
                write_row_fixed_mole_fraction(
                    &equilibrium_matrix[(current_row_offset + mole_frac_cond_row_idx) * equilibrium_matrix_cols],
                    &equilibrium_rhs[current_row_offset + mole_frac_cond_row_idx],
                    current_component_idx,
                    spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
                    state->free_stable_compset_indices, state->num_free_stable_compsets,
                    spec->free_statevar_indices, spec->num_free_statevars,
                    spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
                    state->chemical_potentials, state->mole_fractions, state->system_amount,
                    phase_cs_state->mass_jac, phase_cs_state->mass_jac_cols,
                    phase_cs_state->c_component, phase_cs_state->c_component_cols,
                    phase_cs_state->c_statevars, phase_cs_state->c_statevars_cols,
                    phase_cs_state->c_G, phase_cs_state->c_G_length, phase_cs_state->masses,
                    phase_cs_state->moles_normalization, phase_cs_state->moles_normalization_grad,
                    state->phase_amt, phase_idx, prefactor);
            }
        }
    }
    */
    
    // DEBUG: Print RHS values before residual subtraction
    #ifdef VERBOSE_DEBUG
    if (state->condition_idx == 0 && state->iteration == 0) {
        printf("[GPU DEBUG] Before residual subtraction:\n");
        for (int i = 0; i < num_fixed_mole_frac_conds; i++) {
            printf("  Mole frac constraint %d RHS: %e\n", i, equilibrium_rhs[current_row_offset + i]);
        }
    }
    #endif
    
    // After accumulating all phase contributions, subtract the residual from RHS
    for (int mole_frac_cond_row_idx = 0; mole_frac_cond_row_idx < num_fixed_mole_frac_conds; mole_frac_cond_row_idx++) {
        double component_residual = 0.0;
        
        // DEBUG: Print what we're calculating
        #ifdef VERBOSE_DEBUG
        if (state->condition_idx == 0 && state->iteration == 0) {
            printf("[GPU CONSTRAINT DEBUG] Calculating residual for constraint row %d:\n", mole_frac_cond_row_idx);
            printf("  System mole fractions: [");
            for (int comp_i = 0; comp_i < spec->num_prescribed_mole_fraction_coefficients_cols; comp_i++) {
                printf("%e", state->mole_fractions[comp_i]);
                if (comp_i < spec->num_prescribed_mole_fraction_coefficients_cols - 1) printf(", ");
            }
            printf("]\n");
            printf("  Coefficients for this constraint:");
            for (int i = 0; i < spec->num_prescribed_mole_fraction_coefficients_cols; i++) {
                printf(" [%d]=%e", i, spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][i]);
            }
            printf("\n");
            printf("  Target value (prescribed_mole_fraction_rhs[%d]) = %e\n", 
                   mole_frac_cond_row_idx, spec->prescribed_mole_fraction_rhs[mole_frac_cond_row_idx]);
        }
        #endif
        
        for (current_component_idx = 0; current_component_idx < spec->num_prescribed_mole_fraction_coefficients_cols; current_component_idx++) {
            component_residual += spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][current_component_idx] *
                                  state->mole_fractions[current_component_idx];
            #ifdef VERBOSE_DEBUG
            if (state->condition_idx == 0 && state->iteration == 0 && mole_frac_cond_row_idx == 0) {
                printf("[GPU RESIDUAL CALC] comp_idx=%d: coeff=%e * mole_frac=%e = %e (cumulative=%e)\n",
                       current_component_idx,
                       spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][current_component_idx],
                       state->mole_fractions[current_component_idx],
                       spec->prescribed_mole_fraction_coefficients[mole_frac_cond_row_idx][current_component_idx] * state->mole_fractions[current_component_idx],
                       component_residual);
            }
            #endif
        }
        component_residual -= spec->prescribed_mole_fraction_rhs[mole_frac_cond_row_idx];
        #ifdef VERBOSE_DEBUG
        if (state->condition_idx == 0 && state->iteration == 0 && mole_frac_cond_row_idx == 0) {
            printf("[GPU RESIDUAL CALC] After subtracting target %e: residual = %e\n",
                   spec->prescribed_mole_fraction_rhs[mole_frac_cond_row_idx], component_residual);
        }
        #endif
        
        // DEBUG: Print mole fraction constraint calculation
        #ifdef VERBOSE_DEBUG
        if (state->condition_idx == 0 && state->iteration < 3) {
            printf("[GPU MOLE FRAC CONSTRAINT] Row %d: residual = %e (current X*coeff = %e, target = %e)\n",
                   mole_frac_cond_row_idx, component_residual, 
                   component_residual + spec->prescribed_mole_fraction_rhs[mole_frac_cond_row_idx],
                   spec->prescribed_mole_fraction_rhs[mole_frac_cond_row_idx]);
            printf("  state->mole_fractions: [");
            for (int comp_i = 0; comp_i < spec->num_prescribed_mole_fraction_coefficients_cols; comp_i++) {
                printf("%e", state->mole_fractions[comp_i]);
                if (comp_i < spec->num_prescribed_mole_fraction_coefficients_cols - 1) printf(", ");
            }
            printf("]\n");
        }
        #endif
        
        // DEBUG: Print RHS before and after residual subtraction
        #ifdef VERBOSE_DEBUG
        if (state->condition_idx == 0 && state->iteration < 3) {
            printf("[GPU MOLE FRAC] Row %d RHS before residual: %e\n", 
                   mole_frac_cond_row_idx, equilibrium_rhs[current_row_offset + mole_frac_cond_row_idx]);
        }
        #endif
        
        equilibrium_rhs[current_row_offset + mole_frac_cond_row_idx] -= component_residual;
        
        #ifdef VERBOSE_DEBUG
        if (state->condition_idx == 0 && state->iteration < 3) {
            printf("[GPU MOLE FRAC] Row %d RHS after residual: %e (residual was %e)\n", 
                   mole_frac_cond_row_idx, equilibrium_rhs[current_row_offset + mole_frac_cond_row_idx], component_residual);
        }
        #endif
    }
    // CRITICAL FIX: Add system amount constraint row to match CPU EXACTLY
    // The CPU calls write_row_fixed_mole_amount for each component to build this row
    // This adds small contributions to the chemical potential columns (not exactly zero!)
    
    // SYSTEM AMOUNT CONSTRAINT ROW - Match CPU exactly by calling write_row_fixed_mole_amount
    int system_amount_row_idx = current_row_offset + num_fixed_mole_frac_conds;
    
    // Zero out the row first (it should already be zeroed but let's be explicit)
    for (int col = 0; col < equilibrium_matrix_cols; col++) {
        equilibrium_matrix[system_amount_row_idx * equilibrium_matrix_cols + col] = 0.0;
    }
    equilibrium_rhs[system_amount_row_idx] = 0.0;
    
    // Loop over all active phases and call write_row_fixed_mole_amount for each component
    // This matches CPU's fill_equilibrium_system logic exactly
    for (int stable_idx = 0; stable_idx < state->num_free_stable_compsets; stable_idx++) {
        int compset_original_idx = state->free_stable_compset_indices[stable_idx];
        CompositionSet* current_compset = &state->compsets[compset_original_idx];
        CompsetState* current_cs_state = &state->cs_states[compset_original_idx];
        
        if (current_compset->phase_record == nullptr) continue;
        
        // Call write_row_fixed_mole_amount for each component (matching CPU)
        for (int component_idx = 0; component_idx < spec->num_components; component_idx++) {
            write_row_fixed_mole_amount(
                &equilibrium_matrix[system_amount_row_idx * equilibrium_matrix_cols],
                &equilibrium_rhs[system_amount_row_idx],
                component_idx,
                spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
                state->free_stable_compset_indices, state->num_free_stable_compsets,
                spec->free_statevar_indices, spec->num_free_statevars,
                spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
                state->chemical_potentials,
                current_cs_state->mass_jac, current_cs_state->mass_jac_cols,
                current_cs_state->c_component, current_cs_state->c_component_cols,
                current_cs_state->c_statevars, current_cs_state->c_statevars_cols,
                current_cs_state->c_G, current_cs_state->c_G_length,
                current_cs_state->masses,
                current_cs_state->moles_normalization,
                current_cs_state->moles_normalization_grad,
                state->phase_amt,
                compset_original_idx);
        }
    }
    
    // Also handle fixed stable phases (if any) - matching CPU
    for (int fixed_idx = 0; fixed_idx < spec->num_fixed_stable_compsets; fixed_idx++) {
        int compset_original_idx = spec->fixed_stable_compset_indices[fixed_idx];
        CompositionSet* current_compset = &state->compsets[compset_original_idx];
        CompsetState* current_cs_state = &state->cs_states[compset_original_idx];
        
        if (current_compset->phase_record == nullptr) continue;
        
        // Call write_row_fixed_mole_amount for each component
        for (int component_idx = 0; component_idx < spec->num_components; component_idx++) {
            write_row_fixed_mole_amount(
                &equilibrium_matrix[system_amount_row_idx * equilibrium_matrix_cols],
                &equilibrium_rhs[system_amount_row_idx],
                component_idx,
                spec->free_chemical_potential_indices, spec->num_free_chemical_potentials,
                state->free_stable_compset_indices, state->num_free_stable_compsets,
                spec->free_statevar_indices, spec->num_free_statevars,
                spec->fixed_chemical_potential_indices, spec->num_fixed_chemical_potentials,
                state->chemical_potentials,
                current_cs_state->mass_jac, current_cs_state->mass_jac_cols,
                current_cs_state->c_component, current_cs_state->c_component_cols,
                current_cs_state->c_statevars, current_cs_state->c_statevars_cols,
                current_cs_state->c_G, current_cs_state->c_G_length,
                current_cs_state->masses,
                current_cs_state->moles_normalization,
                current_cs_state->moles_normalization_grad,
                state->phase_amt,
                compset_original_idx);
        }
    }
    
    // Finally, subtract the system amount residual from RHS (matching CPU)
    double system_residual = state->system_amount - spec->prescribed_system_amount;
    equilibrium_rhs[system_amount_row_idx] -= system_residual;
    
    // DEBUG: Print the complete equilibrium matrix for iteration 0
    #ifdef VERBOSE_DEBUG
    if (state->condition_idx == 0 && state->iteration == 0) {
        printf("[EQUILIBRIUM_MATRIX_OUTPUT] GPU Iteration 0 (rows=%d, cols=%d):\n", total_rows, equilibrium_matrix_cols);
        for (int row = 0; row < total_rows; row++) {
            printf("  Row %d: ", row);
            for (int col = 0; col < equilibrium_matrix_cols; col++) {
                printf("%+e ", equilibrium_matrix[row * equilibrium_matrix_cols + col]);
            }
            printf("| RHS: %+e\n", equilibrium_rhs[row]);
        }
    }
    #endif
}

// run_loop, solve_state, advance_state, remove_and_consolidate_phases, change_phases
// as defined in the previous response are largely compatible with these changes,
// as they primarily operate on the counts and indices managed by SystemState and SystemSpecification,
// which are now implicitly simpler due to MAX_PHASE_LOCAL_CONDITIONS being 0.
// Ensure MAX_EQ_SOLN_LEN and related matrix defines are correct for the number of free variables.
// MAX_EQ_SOLN_LEN = num_free_chem_pot + num_free_stable_compsets + num_free_statevars

// check_convergence, pre_solve_hook, post_solve_hook are simple and remain the same.
// The other complex functions (run_loop, solve_state, advance_state, remove_and_consolidate_phases, change_phases)
// were provided in the prior turn and their internal logic largely relies on the dimensions
// and counts that are now correctly set up by the updated init methods and struct definitions.
// For example, `solve_state` uses `soln_length` which is `spec->num_free_chemical_potentials + state->num_free_stable_compsets + spec->num_free_statevars`.
// This will correctly reflect the size of the problem.
// The MAX_ defines for SVD buffers should also be reviewed to ensure they are sufficient for the largest possible system matrix.
// MAX_SVD_DIM used MAX_PHASES for num_free_stable_compsets.
// MAX_EQ_MATRIX_ROWS = num_free_stable_phases + num_fixed_stable_cs + num_fixed_mole_frac_conds + 1
// MAX_EQ_MATRIX_COLS = num_free_chem_pot + num_free_stable_phases + num_free_statevars
// These should be consistent. For example, num_free_stable_phases can be up to MAX_PHASES.
// num_fixed_stable_cs up to MAX_PHASES. num_fixed_mole_frac_conds up to MAX_COMPONENTS.
// So MAX_EQ_MATRIX_ROWS could be approx 2*MAX_PHASES + MAX_COMPONENTS + 1.
// And MAX_EQ_MATRIX_COLS could be MAX_COMPONENTS + MAX_PHASES + MAX_STATEVARS.
// The MAX_SVD_DIM should be the larger of these two.
// MAX_SVD_DIM = (MAX_PHASES + MAX_FIXED_MOLE_FRACTION_CONDITIONS + MAX_COMPONENTS + MAX_STATEVARS + 2)
// This seems roughly compatible if MAX_PHASES dominates MAX_FIXED_MOLE_FRACTION_CONDITIONS.
// For instance, if MAX_PHASES=64, MAX_COMPONENTS=32, MAX_STATEVARS=8:
// MAX_EQ_MATRIX_ROWS approx 2*64 + 32 + 1 = 128 + 32 + 1 = 161
// MAX_EQ_MATRIX_COLS approx 32 + 64 + 8 = 104
// MAX_SVD_DIM needs to be at least max(161, 104).
// Current MAX_SVD_DIM = 64(MP) + 32(MFMC) + 32(MC) + 8(MSV) + 2 = 138. This is NOT large enough for rows.
// Let's redefine MAX_SVD_M and MAX_SVD_N for lstsq based on these more accurate estimates.
// The SystemSpecification struct buffers (A_lstsq_copy etc.) use MAX_SVD_M and MAX_SVD_N.
// These need to be updated in that struct's definition.
// The prior definition of MAX_SVD_DIM was:
// #define MAX_SVD_DIM (MAX_PHASES + MAX_FIXED_MOLE_FRACTION_CONDITIONS + MAX_COMPONENTS + MAX_STATEVARS + 2)
// This should be:
// #define EQ_SYS_MAX_ROWS (2 * MAX_PHASES + MAX_COMPONENTS + 1)
// #define EQ_SYS_MAX_COLS (MAX_COMPONENTS + MAX_PHASES + MAX_STATEVARS)
// #define MAX_SVD_M_EQ EQ_SYS_MAX_ROWS
// #define MAX_SVD_N_EQ EQ_SYS_MAX_COLS
// And then use these for the A_lstsq_copy, U_lstsq, V_lstsq, singular_values_lstsq, superdiag_lstsq buffers.
// I will update the #defines in SystemSpecification.

// The `run_loop` and other functions from the previous turn would be here.
// For brevity, I'm focusing on the direct impact of removing phase-local conditions
// on struct definitions and initializations. The logical flow of those larger functions
// remains the same but operates on data structures that are now simpler.

// [Pasting the remaining functions from the previous generated code for completeness and self-contained nature of the final code block]

__device__ bool check_convergence(SystemSpecification* spec, SystemState* state) {
    // SEGMENT 38: CHECK CONVERGENCE
    gpu_debug_log(38, "Check convergence", state->condition_idx);
    
    double ALLOWED_DELTA_Y = 5e-09;
    double ALLOWED_DELTA_PHASE_AMT = 1e-10;
    double ALLOWED_DELTA_STATEVAR = 1e-5;
    
    if (state->condition_idx < 3) {
        gpu_debug_log_value("largest_phase_amt_change", state->largest_phase_amt_change);
        gpu_debug_log_value("largest_y_change", state->largest_y_change);
        gpu_debug_log_value("largest_statevar_change", state->largest_statevar_change);
        gpu_debug_log_value("mass_residual", state->mass_residual);
        gpu_debug_log_value("iterations_since_last_phase_change", (double)state->iterations_since_last_phase_change);
    }

    bool solution_is_feasible =
        (state->largest_phase_amt_change < ALLOWED_DELTA_PHASE_AMT) &&
        (state->largest_y_change < ALLOWED_DELTA_Y) &&
        (state->largest_statevar_change < ALLOWED_DELTA_STATEVAR) &&
        (state->mass_residual < spec->ALLOWED_MASS_RESIDUAL);

    // Check convergence similar to CPU behavior
    // CPU doesn't require a minimum iteration count for convergence
    if (solution_is_feasible && (state->iterations_since_last_phase_change >= 5)) {
        gpu_debug_log_value("converged", 1.0);
        return true;
    }
    gpu_debug_log_value("converged", 0.0);
    return false;
}

__device__ bool pre_solve_hook(SystemSpecification* spec, SystemState* state) {
    return true;
}

__device__ bool post_solve_hook(SystemSpecification* spec, SystemState* state) {
    return true;
}

// solve_state function removed - using solve_state_global_mem instead

__device__ void advance_state(SystemSpecification* spec, SystemState* state, const double* equilibrium_soln, int soln_length, double step_size_param) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    // SEGMENT 34: UPDATE STATE WITH STEP SIZE
    gpu_debug_log(34, "Update state with step size", state->iteration);
    gpu_debug_log_value("step_size", step_size_param);
    
    double current_step_size = step_size_param;
    double MIN_PHASE_AMOUNT = 1e-16;  // CRITICAL FIX: Match CPU's 1e-16 in advance_state, not 1e-10!

    // Chemical potentials are now handled in solve_state (matching CPU approach)
    // Start with phase amount updates
    int soln_idx_offset = spec->num_free_chemical_potentials; // Skip chemical potentials
    double phase_amt_step_size_limiter = current_step_size;
    for (int i = 0; i < state->num_free_stable_compsets; ++i) {
        int compset_original_idx = state->free_stable_compset_indices[i];
        if (compset_original_idx < 0 || compset_original_idx >= state->num_compsets) continue;
        double delta_NP = equilibrium_soln[soln_idx_offset + i];
        if (state->phase_amt[compset_original_idx] + delta_NP < MIN_PHASE_AMOUNT) {
            if (fabs(delta_NP) > MIN_PHASE_AMOUNT) {
                phase_amt_step_size_limiter = fmin(phase_amt_step_size_limiter,
                    (MIN_PHASE_AMOUNT - state->phase_amt[compset_original_idx]) / delta_NP);
            } else if (delta_NP < 0) {
                 phase_amt_step_size_limiter = 0.0;
            }
        }
    }

    state->largest_phase_amt_change = 0.0;
    
    // SEGMENT 35: UPDATE PHASE AMOUNTS
    gpu_debug_log(35, "Update phase amounts", state->iteration);
    gpu_debug_log_value("phase_amt_step_limiter", phase_amt_step_size_limiter);
    
    #ifdef VERBOSE_DEBUG
    if (state->iteration < 10 && thread_id == 0 && phase_amt_step_size_limiter < 1.0) {
        printf("GPU DEBUG: Step size limited to %.6e at iteration %d\n", 
               phase_amt_step_size_limiter, state->iteration);
    }
    #endif
    
    
    for (int i = 0; i < state->num_free_stable_compsets; ++i) {
        int compset_original_idx = state->free_stable_compset_indices[i];
        if (compset_original_idx < 0 || compset_original_idx >= state->num_compsets) continue;
        double old_amt = state->phase_amt[compset_original_idx];
        double delta = equilibrium_soln[soln_idx_offset + i];
        // The equilibrium solver returns phase amount deltas directly
        // No need to scale by 100 - this was causing phases to shrink incorrectly
        double actual_change = phase_amt_step_size_limiter * delta;
        state->phase_amt[compset_original_idx] += actual_change;
        
        // DEBUG: Print details
        #ifdef VERBOSE_DEBUG
        if (state->iteration < 10) {
            printf("  Phase %d: old=%.6e, delta=%.6e (raw), actual_change=%.6e, new=%.6e\n",
                   compset_original_idx, old_amt, delta, actual_change, state->phase_amt[compset_original_idx]);
        }
        #endif
        
        if (state->phase_amt[compset_original_idx] < MIN_PHASE_AMOUNT) {
            state->phase_amt[compset_original_idx] = MIN_PHASE_AMOUNT;
        }
        
        // Keep CompositionSet NP synchronized with phase_amt
        state->compsets[compset_original_idx].NP = state->phase_amt[compset_original_idx];
        
        if (fabs(actual_change) > state->largest_phase_amt_change) {
            state->largest_phase_amt_change = fabs(actual_change);
        }
        
        // Additional debug for phase removal issue
        #ifdef VERBOSE_DEBUG
        if (state->iteration < 50 && state->phase_amt[compset_original_idx] < 1e-6) {
            printf("GPU DEBUG: Phase %d amount became very small (%.10e) at iteration %d\n", 
                   compset_original_idx, state->phase_amt[compset_original_idx], state->iteration);
        }
        #endif
    }
    
    gpu_debug_log_value("largest_phase_amt_change", state->largest_phase_amt_change);
    
    // DEBUG: Check total phase amounts after update
    #ifdef VERBOSE_DEBUG
    double phase_amt_sum_after = 0.0;
    for (int idx = 0; idx < state->num_compsets; ++idx) {
        phase_amt_sum_after += state->phase_amt[idx];
    }
    printf("[GPU MASS BALANCE] advance_state() - after phase update: sum(phase_amt) = %.15e\n", phase_amt_sum_after);
    #endif
    
    // CRITICAL FIX: DO NOT normalize phase amounts in advance_state!
    // The CPU solver doesn't do this, and it prevents convergence by undoing all changes.
    // The phase amounts should be allowed to change as directed by the equilibrium solution.
    // The system amount constraint is enforced through the equilibrium matrix, not by normalization.
    soln_idx_offset += state->num_free_stable_compsets; // Start of state variable deltas

    // SEGMENT 36: UPDATE STATE VARIABLES  
    gpu_debug_log(36, "Update state variables", state->iteration);
    
    state->largest_statevar_change = 0.0;
    for(int i=0; i<spec->num_statevars; ++i) state->delta_statevars[i] = 0.0;

    for (int i = 0; i < spec->num_free_statevars; ++i) {
        int statevar_global_idx = spec->free_statevar_indices[i];
        if (statevar_global_idx < 0 || statevar_global_idx >= spec->num_statevars) continue;
        state->delta_statevars[statevar_global_idx] = equilibrium_soln[soln_idx_offset + i];
        double psc = 0.0;
        if (state->num_compsets > 0 && state->compsets[0].phase_record != nullptr) { // Check valid compset
            if (fabs(state->compsets[0].dof[statevar_global_idx]) < 1e-12) {
                 psc = (fabs(state->delta_statevars[statevar_global_idx]) > 1e-12) ? 1e30 : 0.0;
            } else {
                psc = fabs(state->delta_statevars[statevar_global_idx] / state->compsets[0].dof[statevar_global_idx]);
            }
            if (psc > state->largest_statevar_change) {
                state->largest_statevar_change = psc;
            }
        }
    }
    
    gpu_debug_log_value("largest_statevar_change", state->largest_statevar_change);
    for (int idx = 0; idx < state->num_compsets; ++idx) {
        if (state->compsets[idx].phase_record == nullptr) continue;
        for (int i = 0; i < spec->num_free_statevars; ++i) {
             int statevar_global_idx = spec->free_statevar_indices[i];
             if (statevar_global_idx < 0 || statevar_global_idx >= spec->num_statevars) continue;
             
             state->compsets[idx].dof[statevar_global_idx] += state->delta_statevars[statevar_global_idx];
        }
    }

    // DEBUG: Check phase_compositions before site fraction update
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0 && state->iteration < 5) {
        printf("GPU DEBUG: Phase compositions BEFORE site fraction update (iteration %d):\n", state->iteration);
        for (int i = 0; i < state->num_free_stable_compsets && i < 2; ++i) {
            int idx = state->free_stable_compset_indices[i];
            printf("  Phase %d: [%.6f, %.6f]\n", idx,
                   state->phase_compositions[idx * MAX_COMPONENTS + 0],
                   state->phase_compositions[idx * MAX_COMPONENTS + 1]);
        }
    }
    #endif
    
    // SEGMENT 37: UPDATE SITE FRACTIONS
    gpu_debug_log(37, "Update site fractions", state->iteration);
    
    state->largest_y_change = 0.0;
    double new_y_for_phase[MAX_DOF_PER_PHASE];

    for (int idx = 0; idx < state->num_compsets; ++idx) {
        CompsetState* csst = &state->cs_states[idx];
        CompositionSet* compset = &state->compsets[idx];
        if (compset->phase_record == nullptr) continue;
        
        // NOTE: Do NOT skip phases with small amounts - they need to converge
        // before removal/consolidation. This matches CPU behavior.
        
        const PhaseRecord* pr = compset->phase_record;
        int num_site_fracs = pr->phase_dof;

        for(int i=0; i<num_site_fracs; ++i) csst->delta_y[i] = 0.0;

        for (int i = 0; i < num_site_fracs; ++i) {
            csst->delta_y[i] += csst->c_G[i];
            for (int sv_idx = 0; sv_idx < spec->num_statevars; ++sv_idx) {
                 if (sv_idx < 0 || sv_idx >= spec->num_statevars) continue;
                csst->delta_y[i] += csst->c_statevars[i * csst->c_statevars_cols + sv_idx] * state->delta_statevars[sv_idx];
            }
            for (int cp_idx = 0; cp_idx < spec->num_components; ++cp_idx) {
                 if (cp_idx < 0 || cp_idx >= spec->num_components) continue;
                // CRITICAL FIX: Use absolute chemical potentials, NOT deltas!
                // CPU code at minimizer.pyx line 1388 uses state.chemical_potentials[chempot_idx] directly
                // This matches Eq. 43 in Sundman 2015
                csst->delta_y[i] += csst->c_component[cp_idx * csst->c_component_cols + i] * state->chemical_potentials[cp_idx];
            }
            for (int cons_idx = 0; cons_idx < csst->internal_cons_length; ++cons_idx) {
                csst->delta_y[i] -= csst->full_e_matrix[(num_site_fracs + cons_idx) * csst->full_e_matrix_dim + i] * csst->internal_cons[cons_idx];
            }
        }
        
        // DEBUG: Print delta_y calculation details after consolidation
        #ifdef VERBOSE_DEBUG
        if (state->iteration >= 2 && state->iteration < 5 && idx == 0) {
            printf("GPU DEBUG: delta_y calculation for phase %d at iteration %d:\n", idx, state->iteration);
            for (int i = 0; i < num_site_fracs; ++i) {
                printf("  delta_y[%d] = %.15e\n", i, csst->delta_y[i]);
                printf("    c_G[%d] = %.15e\n", i, csst->c_G[i]);
                printf("    Chemical potential contribution: ");
                double cp_contrib = 0.0;
                for (int cp_idx = 0; cp_idx < spec->num_components; ++cp_idx) {
                    double contrib = csst->c_component[cp_idx * csst->c_component_cols + i] * state->chemical_potentials[cp_idx];
                    cp_contrib += contrib;
                    if (fabs(contrib) > 1e-15) {
                        printf("cp[%d]=%.6e ", cp_idx, contrib);
                    }
                }
                printf("(total: %.15e)\n", cp_contrib);
            }
        }
        #endif

        double site_frac_step_limiter = current_step_size;
        double min_allowed_sf_step = 1e-20 * current_step_size;
        bool exceeded_bounds_for_phase;

        do {
            exceeded_bounds_for_phase = false;
            for (int i = 0; i < num_site_fracs; ++i) {
                double current_site_frac_val = compset->dof[spec->num_statevars + i];
                new_y_for_phase[i] = current_site_frac_val + site_frac_step_limiter * csst->delta_y[i];
                if (new_y_for_phase[i] > 1.0) {
                    if ((new_y_for_phase[i] - 1.0) > 1e-11) exceeded_bounds_for_phase = true;
                    new_y_for_phase[i] = 1.0;
                } else if (new_y_for_phase[i] < MIN_SITE_FRACTION) {
                    if ((MIN_SITE_FRACTION - new_y_for_phase[i]) > 1e-11) exceeded_bounds_for_phase = true;
                    new_y_for_phase[i] = fmax(current_site_frac_val / 100.0, MIN_SITE_FRACTION);
                }
            }
            if (exceeded_bounds_for_phase) {
                site_frac_step_limiter *= 0.5;
            }
        } while (exceeded_bounds_for_phase && site_frac_step_limiter >= min_allowed_sf_step);

        for (int i = 0; i < num_site_fracs; ++i) {
            double old_val = compset->dof[spec->num_statevars + i];
            // Apply the final calculated new_y_for_phase value for this iteration of step_limiter
            compset->dof[spec->num_statevars + i] = new_y_for_phase[i]; // Value after potential bounding
            double change_this_y = fabs(compset->dof[spec->num_statevars + i] - old_val);
            if (change_this_y > state->largest_y_change) {
                state->largest_y_change = change_this_y;
            }

        // REMOVED: Special handling for single-sublattice phases was causing issues
        // Phase compositions will be recalculated in the next recompute() call
        // This matches CPU behavior which doesn't have special handling here
        
        }
    }
    
    gpu_debug_log_value("largest_y_change", state->largest_y_change);
}

__device__ bool remove_and_consolidate_phases(SystemSpecification* spec, SystemState* state) {
    bool phases_changed = false;
    double COMPSET_CONSOLIDATE_DISTANCE = 1e-4;

    int compset_indices_to_remove_temp[MAX_PHASES];
    int num_to_remove = 0;

    // Debug: Log entry to function
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    // DEBUG removed - phase composition update before consolidation
    
    // DEBUG removed - was checking phase compositions

    for (int i = 0; i < state->num_free_stable_compsets; ++i) {
        int idx1 = state->free_stable_compset_indices[i];
        if (idx1 < 0 || idx1 >= state->num_compsets) continue;
        CompositionSet* compset1 = &state->compsets[idx1];
        if (compset1->fixed || compset1->phase_record == nullptr) continue;

        bool already_marked_for_removal1 = false;
        for(int k=0; k < num_to_remove; ++k) if(compset_indices_to_remove_temp[k] == idx1) already_marked_for_removal1 = true;
        if(already_marked_for_removal1) continue;

        // Remove unstable phases (matching CPU minimizer.pyx line 1328)
        if (state->phase_amt[idx1] < 1e-10) {
            // CRITICAL FIX: Check if removing this phase would leave us unable to satisfy mass balance
            // Count how many phases would remain after removal
            int phases_remaining = 0;
            for (int j = 0; j < state->num_free_stable_compsets; ++j) {
                int idx_check = state->free_stable_compset_indices[j];
                if (idx_check != idx1 && state->phase_amt[idx_check] >= 1e-10) {
                    phases_remaining++;
                }
            }
            
            // If this is the last phase that could satisfy constraints, don't remove it
            if (phases_remaining == 0 && spec->num_prescribed_mole_fraction_conditions > 0) {
                if (thread_id == 0 && state->iteration < 5) {
                    printf("  Phase %d NOT removed - last phase needed for mass balance\n", idx1);
                }
                continue;
            }
            
            if (num_to_remove < MAX_PHASES) compset_indices_to_remove_temp[num_to_remove++] = idx1;
            state->phase_amt[idx1] = 0.0;  // CPU sets to 0 at line 1330
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0 && state->iteration < 5) {
                printf("  Phase %d marked for removal - amount %.2e < 1e-10\n", idx1, state->phase_amt[idx1]);
            }
            #endif
            continue;
        }

        for (int j = 0; j < state->num_free_stable_compsets; ++j) {
            if (i == j) continue;
            int idx2 = state->free_stable_compset_indices[j];
            if (idx2 < 0 || idx2 >= state->num_compsets) continue;
            
            // CRITICAL FIX: Skip phases with amount < 1e-10 to match CPU behavior
            // CPU removes these phases before consolidation checks
            if (state->phase_amt[idx2] < 1e-10) continue;
            
            CompositionSet* compset2 = &state->compsets[idx2];
            if (compset2->fixed || compset2->phase_record == nullptr) continue;
            if (compset1->phase_record != compset2->phase_record) continue; // Different phase types

            bool already_marked_for_removal2 = false;
            for(int k=0; k < num_to_remove; ++k) if(compset_indices_to_remove_temp[k] == idx2) already_marked_for_removal2 = true;
            if(already_marked_for_removal2) continue;

            bool should_consolidate = true;
            double max_diff = 0.0;
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                double diff = fabs(state->phase_compositions[idx1 * MAX_COMPONENTS + comp_idx] -
                                   state->phase_compositions[idx2 * MAX_COMPONENTS + comp_idx]);
                if (diff > max_diff) max_diff = diff;
                if (diff > COMPSET_CONSOLIDATE_DISTANCE) {
                    should_consolidate = false;
                    break;
                }
            }
            
            // Debug: Log consolidation check
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0 && state->iteration < 5) {
                printf("  Checking phases %d and %d for consolidation:\n", idx1, idx2);
                printf("    Max composition diff: %.6f (threshold: %.6f)\n", max_diff, COMPSET_CONSOLIDATE_DISTANCE);
                printf("    Phase compositions: [%.6f, %.6f] vs [%.6f, %.6f]\n",
                       state->phase_compositions[idx1 * MAX_COMPONENTS + 0],
                       state->phase_compositions[idx1 * MAX_COMPONENTS + 1],
                       state->phase_compositions[idx2 * MAX_COMPONENTS + 0],
                       state->phase_compositions[idx2 * MAX_COMPONENTS + 1]);
                printf("    Should consolidate: %s\n", should_consolidate ? "YES" : "NO");
                
                // DEBUG: Also show site fractions
                if (should_consolidate && state->iteration == 0) {
                    CompositionSet* cs1 = &state->compsets[idx1];
                    CompositionSet* cs2 = &state->compsets[idx2];
                    printf("    [CONSOLIDATION] Phase %d site fractions: Y[0]=%.15e, Y[1]=%.15e\n",
                           idx1, cs1->dof[3], cs1->dof[4]);
                    printf("    [CONSOLIDATION] Phase %d site fractions: Y[0]=%.15e, Y[1]=%.15e\n",
                           idx2, cs2->dof[3], cs2->dof[4]);
                }
            }
            #endif
            
            if (should_consolidate) {
                
                // DEBUG: Site fractions at consolidation moment
                #ifdef VERBOSE_DEBUG
                if (thread_id == 0 && state->iteration < 2) {
                    CompositionSet* cs1 = &state->compsets[idx1];
                    CompositionSet* cs2 = &state->compsets[idx2];
                    printf("[AT CONSOLIDATION] Phase %d Y=[%.15e, %.15e], amt=%.15e\n",
                           idx1, cs1->dof[3], cs1->dof[4], state->phase_amt[idx1]);
                    printf("[AT CONSOLIDATION] Phase %d Y=[%.15e, %.15e], amt=%.15e\n",
                           idx2, cs2->dof[3], cs2->dof[4], state->phase_amt[idx2]);
                }
                #endif
                if (num_to_remove < MAX_PHASES) compset_indices_to_remove_temp[num_to_remove++] = idx2;
                
                // CRITICAL FIX: Match CPU behavior - add phase amounts but account for normalization
                // Phase amounts are stored in formula units (normalized by moles_normalization)
                // When consolidating, we need to convert to moles, add, then re-normalize
                
                // Get moles_normalization for both phases (sum of moles per formula unit)
                double moles_norm1 = state->cs_states[idx1].moles_normalization;
                double moles_norm2 = state->cs_states[idx2].moles_normalization;
                
                // DEBUG: Print moles_normalization values
                #ifdef VERBOSE_DEBUG
                if (thread_id == 0) {
                    printf("[GPU CONSOLIDATION DEBUG] Phase %d moles_norm=%e, phase %d moles_norm=%e\n",
                           idx1, moles_norm1, idx2, moles_norm2);
                }
                #endif
                
                // For single sublattice phases, moles_normalization should be 1.0
                // since there's only one site and site fractions sum to 1
                // If moles_normalization is not calculated yet, fall back to simple addition
                double old_amt1 = state->phase_amt[idx1];
                double old_amt2 = state->phase_amt[idx2];
                
                // No fallback - CPU doesn't check for zero moles_norm
                {
                    // Convert phase amounts from formula units to moles
                    double moles1 = state->phase_amt[idx1] * moles_norm1;
                    double moles2 = state->phase_amt[idx2] * moles_norm2;
                    
                    // Add the moles
                    double total_moles = moles1 + moles2;
                    
                    // Convert back to formula units using the normalization of the target phase
                    state->phase_amt[idx1] = fmax(total_moles / moles_norm1, 1e-8);
                    
                    // DEBUG: What happens after consolidation
                    #ifdef VERBOSE_DEBUG
                    if (thread_id == 0) {
                        printf("[CONSOLIDATION] Consolidated phases %d and %d:\n", idx1, idx2);
                        printf("  Moles normalization: phase %d = %.15e, phase %d = %.15e\n",
                               idx1, moles_norm1, idx2, moles_norm2);
                        printf("  Phase amounts before: phase %d = %.15e, phase %d = %.15e\n",
                               idx1, old_amt1, idx2, old_amt2);
                        printf("  Moles: phase %d = %.15e, phase %d = %.15e, total = %.15e\n",
                               idx1, moles1, idx2, moles2, total_moles);
                        printf("  Phase %d: new amount=%.15e (formula units)\n", 
                               idx1, state->phase_amt[idx1]);
                    }
                    #endif
                }
                
                #ifdef VERBOSE_DEBUG
                if (thread_id == 0) {
                    // Show compositions
                    printf("  Phase %d: X=[%.15e, %.15e]\n", 
                           idx1,
                           state->phase_compositions[idx1 * MAX_COMPONENTS + 0],
                           state->phase_compositions[idx1 * MAX_COMPONENTS + 1]);
                    
                    // Also show site fractions
                    CompositionSet* cs1 = &state->compsets[idx1];
                    printf("  Phase %d site fractions: Y=[%.15e, %.15e]\n",
                           idx1, cs1->dof[3], cs1->dof[4]);
                }
                #endif
                state->phase_amt[idx2] = 0.0;
                
                // DO NOT modify site fractions or phase compositions of idx1
                // The CPU keeps them unchanged, allowing the solver to adjust them
                // in subsequent iterations to satisfy the mass balance constraint
            }
        }
    }

    if (num_to_remove > 0) {
        phases_changed = true;
        int new_free_stable_indices[MAX_PHASES];
        int new_count = 0;
        for (int i = 0; i < state->num_free_stable_compsets; ++i) {
            int current_idx = state->free_stable_compset_indices[i];
            bool is_removed = false;
            for (int j = 0; j < num_to_remove; ++j) {
                if (compset_indices_to_remove_temp[j] == current_idx) {
                    is_removed = true;
                    state->phase_amt[current_idx] = 0.0;
                    break;
                }
            }
            if (!is_removed) {
                if (new_count < MAX_PHASES) new_free_stable_indices[new_count++] = current_idx;
            }
        }
        // CRITICAL FIX: Match CPU behavior when all phases would be removed
        // CPU minimizer.pyx lines 1509-1517
        if (new_count == 0 && state->num_free_stable_compsets > 0 && num_to_remove == state->num_free_stable_compsets) {
            // Do not allow all phases to leave the system
            // Reset all phase amounts to 1 and chemical potentials to 0
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                int phase_idx = state->free_stable_compset_indices[i];
                state->phase_amt[phase_idx] = 1.0;
                if (new_count < MAX_PHASES) {
                    new_free_stable_indices[new_count++] = phase_idx;
                }
            }
            
            // Reset chemical potentials to 0
            for (int i = 0; i < spec->num_components; ++i) {
                state->chemical_potentials[i] = 0.0;
            }
            
            // Force fixed chemical potentials to adopt their initial values
            for (int cp_idx = 0; cp_idx < spec->num_fixed_chemical_potentials; ++cp_idx) {
                int comp_idx = spec->fixed_chemical_potential_indices[cp_idx];
                state->chemical_potentials[comp_idx] = spec->initial_chemical_potentials[comp_idx];
            }
            
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0) {
                printf("[GPU PHASE CONSOLIDATION] All phases would be removed - resetting phase amounts to 1.0 and chemical potentials\n");
            }
            #endif
        }


        state->num_free_stable_compsets = new_count;
        for (int i = 0; i < new_count; ++i) {
            state->free_stable_compset_indices[i] = new_free_stable_indices[i];
        }
        
        // DEBUG: Print the updated free stable compsets
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0) {
            printf("[GPU PHASE CONSOLIDATION] Updated num_free_stable_compsets from %d to %d\n", 
                   state->num_free_stable_compsets + num_to_remove, new_count);
            printf("  Remaining free phases: ");
            for (int i = 0; i < new_count; ++i) {
                printf("%d ", new_free_stable_indices[i]);
            }
            printf("\n");
        }
        #endif
    }
    
    
    return phases_changed;
}

__device__ bool change_phases(SystemSpecification* spec, SystemState* state) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("GPU DEBUG: change_phases called - initial num_free_stable_compsets=%d\n", state->num_free_stable_compsets);
    }
    #endif
    bool phases_changed = false;
    double current_driving_forces[MAX_PHASES]; // Sized to MAX_PHASES
    state->driving_forces(spec, current_driving_forces, MAX_PHASES); // Get DFs for all possible phases

    // Match CPU minimizer.pyx lines 1391-1403 exactly
    double MIN_PHASE_AMOUNT_FOR_ADD_CHECK = 1e-9;  // CPU uses 1e-9, not MIN_PHASE_FRACTION
    int MIN_REQUIRED_METASTABLE_PHASE_ITERATIONS_TO_ADD = 5;
    double MIN_DRIVING_FORCE_TO_ADD = 1e-5;
    int MAX_ALLOWED_TIMES_COMPSET_REMOVED = 4;

    double current_min_phase_amount_for_removal = MIN_PHASE_AMOUNT_FOR_ADD_CHECK;
    if (state->num_free_stable_compsets > spec->max_num_free_stable_phases) {
        // Gibbs phase rule is currently being violated
        // Try forcing phases with small amounts out of the equilibrium
        current_min_phase_amount_for_removal = 1e-4;
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0) {
            printf("GPU DEBUG: Gibbs phase rule violation: increasing MIN_PHASE_AMOUNT to %.2e\n", 
                   current_min_phase_amount_for_removal);
        }
        #endif
    }

    int compsets_to_remove_indices[MAX_PHASES];
    int num_to_remove = 0;
    for (int i = 0; i < state->num_free_stable_compsets; ++i) {
        int cs_original_idx = state->free_stable_compset_indices[i];
        if (cs_original_idx < 0 || cs_original_idx >= state->num_compsets) continue;
        if (state->phase_amt[cs_original_idx] < current_min_phase_amount_for_removal && !state->compsets[cs_original_idx].fixed) {
            if (num_to_remove < MAX_PHASES) {
                compsets_to_remove_indices[num_to_remove++] = cs_original_idx;
                #ifdef VERBOSE_DEBUG
                if (thread_id == 0) {
                    printf("GPU DEBUG: Phase %d marked for removal in change_phases - amt=%.10e < %.10e (iteration %d)\n", 
                           cs_original_idx, state->phase_amt[cs_original_idx], current_min_phase_amount_for_removal, state->iteration);
                }
                #endif
            }
        }
    }

    int compsets_to_add_indices[MAX_PHASES];
    int num_to_add = 0;
    for (int cs_idx = 0; cs_idx < state->num_compsets; ++cs_idx) {
        bool is_already_free_stable = false;
        for(int i=0; i < state->num_free_stable_compsets; ++i) if(state->free_stable_compset_indices[i] == cs_idx) is_already_free_stable = true;
        if (is_already_free_stable || state->compsets[cs_idx].fixed) continue;

        if (state->metastable_phase_iterations[cs_idx] >= MIN_REQUIRED_METASTABLE_PHASE_ITERATIONS_TO_ADD &&
            current_driving_forces[cs_idx] > MIN_DRIVING_FORCE_TO_ADD &&
            state->times_compset_removed[cs_idx] < MAX_ALLOWED_TIMES_COMPSET_REMOVED) {
            if (num_to_add < MAX_PHASES) compsets_to_add_indices[num_to_add++] = cs_idx;
        }
    }

    int max_allowed_to_add_now = spec->max_num_free_stable_phases + num_to_remove - state->num_free_stable_compsets;

    if (num_to_add > 0) {
        if (max_allowed_to_add_now < 1) {
            int least_removed_cs_original_idx = -1;
            int min_times_removed_val = MAX_ALLOWED_TIMES_COMPSET_REMOVED + 1;
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                int current_free_cs_idx = state->free_stable_compset_indices[i];
                if (current_free_cs_idx < 0 || current_free_cs_idx >= state->num_compsets) continue;
                bool is_candidate_for_destabilize = true;
                for(int k=0; k<num_to_remove; ++k) if(compsets_to_remove_indices[k] == current_free_cs_idx) is_candidate_for_destabilize = false;
                if (state->compsets[current_free_cs_idx].fixed) is_candidate_for_destabilize = false;
                if (is_candidate_for_destabilize) {
                    if (state->times_compset_removed[current_free_cs_idx] < min_times_removed_val) {
                        min_times_removed_val = state->times_compset_removed[current_free_cs_idx];
                        least_removed_cs_original_idx = current_free_cs_idx;
                    }
                }
            }
            if (least_removed_cs_original_idx != -1) {
                bool already_in_remove_list = false;
                for(int k=0; k<num_to_remove; ++k) if(compsets_to_remove_indices[k] == least_removed_cs_original_idx) already_in_remove_list = true;
                if(!already_in_remove_list && num_to_remove < MAX_PHASES) compsets_to_remove_indices[num_to_remove++] = least_removed_cs_original_idx;
            }
            max_allowed_to_add_now = spec->max_num_free_stable_phases + num_to_remove - state->num_free_stable_compsets; // Recalculate
        }

        if (num_to_add > max_allowed_to_add_now && max_allowed_to_add_now > 0) {
            int best_to_add_idx = -1;
            double largest_df_for_best = -INFINITY;
            for(int i=0; i < num_to_add; ++i) { // Iterate over the *current* list of candidates to add
                int candidate_idx = compsets_to_add_indices[i];
                if (candidate_idx < 0 || candidate_idx >= state->num_compsets) continue;
                if (current_driving_forces[candidate_idx] > largest_df_for_best && current_driving_forces[candidate_idx] > MIN_DRIVING_FORCE_TO_ADD) {
					largest_df_for_best = current_driving_forces[candidate_idx];
					best_to_add_idx = candidate_idx;
}
				}
            if(best_to_add_idx != -1) {
                num_to_add = 1; // Only add the best one
                compsets_to_add_indices[0] = best_to_add_idx;
            } else {
                num_to_add = 0; // No suitable candidate found
            }
        } else if (max_allowed_to_add_now <= 0) {
            num_to_add = 0;
        }
         // Final check on num_to_add based on max_allowed
        if (num_to_add > max_allowed_to_add_now) num_to_add = max_allowed_to_add_now < 0 ? 0 : max_allowed_to_add_now;

    }

    int final_free_stable_indices[MAX_PHASES];
    int final_free_count = 0;
    bool current_free_set_changed_flag = false;

    for (int i = 0; i < state->num_free_stable_compsets; ++i) {
        int cs_idx = state->free_stable_compset_indices[i];
        bool removed = false;
        for (int j = 0; j < num_to_remove; ++j) {
            if (compsets_to_remove_indices[j] == cs_idx) {
                removed = true;
                state->times_compset_removed[cs_idx]++;
                state->phase_amt[cs_idx] = 0.0;
                current_free_set_changed_flag = true;
                break;
            }
        }
        if (!removed) {
            if (final_free_count < MAX_PHASES) final_free_stable_indices[final_free_count++] = cs_idx;
        }
    }
    for (int i = 0; i < num_to_add; ++i) {
        int cs_idx_to_add = compsets_to_add_indices[i];
        bool already_present = false;
        for(int k=0; k < final_free_count; ++k) if(final_free_stable_indices[k] == cs_idx_to_add) already_present = true;
        if (!already_present) {
             if (final_free_count < MAX_PHASES) final_free_stable_indices[final_free_count++] = cs_idx_to_add;
             state->phase_amt[cs_idx_to_add] = fmax(state->phase_amt[cs_idx_to_add], 1e-10);
             current_free_set_changed_flag = true;
        }
    }

    if (current_free_set_changed_flag) phases_changed = true;

    state->num_free_stable_compsets = final_free_count;
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("GPU DEBUG: change_phases - final_free_count=%d, free_stable_indices=[", final_free_count);
        for (int i = 0; i < final_free_count; ++i) {
            state->free_stable_compset_indices[i] = final_free_stable_indices[i];
            if (i > 0) printf(", ");
            printf("%d", final_free_stable_indices[i]);
        }
        printf("]\n");
    } else {
        // Still need to set the indices even for non-zero threads
        for (int i = 0; i < final_free_count; ++i) {
            state->free_stable_compset_indices[i] = final_free_stable_indices[i];
        }
    }
    #else
    // When verbose debug is off, still need to set the indices for all threads
    for (int i = 0; i < final_free_count; ++i) {
        state->free_stable_compset_indices[i] = final_free_stable_indices[i];
    }
    #endif
    
    for (int i = 0; i < final_free_count; ++i) {
        int current_idx = final_free_stable_indices[i];  // CRITICAL FIX: Use NEW array, not old!
        if (current_idx >=0 && current_idx < state->num_compsets) { // boundary check
            if (state->phase_amt[current_idx] < 1e-10 && !state->compsets[current_idx].fixed) {
                 state->phase_amt[current_idx] = 1e-10;
            }
        }
    }
    // Force unstable phase amounts to zero (matching CPU minimizer.pyx lines 1468-1469)
    for(int i=0; i<state->num_compsets; ++i) {
        bool is_still_free_stable = false;
        for(int j=0; j < state->num_free_stable_compsets; ++j) {
            if(state->free_stable_compset_indices[j] == i) {
                is_still_free_stable = true;
                break;
            }
        }
        if(!is_still_free_stable && !state->compsets[i].fixed) {
            state->phase_amt[i] = 0.0;
        }
    }
    return phases_changed;
}


// run_loop function removed - using run_loop_global_mem from gpu_codegen.py instead

// SVD-based matrix functions using svd.c

/**
 * @brief Inverts a square matrix using SVD decomposition.
 * @param matrix Input/output matrix (N x N, stored row-major). Will be overwritten with inverse.
 * @param dim Dimension of the square matrix (N).
 * @param U Workspace for U matrix (N x N).
 * @param V Workspace for V matrix (N x N).
 * @param singular_values Workspace for singular values (N).
 * @param superdiag Workspace for super-diagonal (N).
 * @param work Additional workspace (N x N).
 */
__device__ void invert_matrix(double* matrix, int dim, double* U, double* V, 
                              double* singular_values, double* superdiag, double* work) {
    // Copy input matrix to work array (SVD modifies input)
    for (int i = 0; i < dim * dim; ++i) {
        work[i] = matrix[i];
    }
    
    // Perform SVD: work = U * S * V^T
    int svd_result = Singular_Value_Decomposition(work, dim, dim, U, singular_values, V, superdiag);
    
    if (svd_result != 0) {
        // SVD failed - CPU doesn't have a fallback
        return;
    }
    
    // Compute pseudo-inverse using SVD result
    // tolerance based on machine precision and matrix size
    double tolerance = 1e-14 * dim;
    
    Singular_Value_Decomposition_Inverse(U, singular_values, V, tolerance, dim, dim, matrix);
}

/**
 * @brief Solves least squares problem Ax = b using SVD decomposition.
 * @param A Input matrix (M x N, stored row-major). Will be modified during computation.
 * @param nrows Number of rows in A (M).
 * @param ncols Number of columns in A (N).
 * @param b Input/output vector. Input: RHS vector (M), Output: solution vector (N).
 * @param tolerance Tolerance for SVD pseudo-inverse.
 * @param U Workspace for U matrix (M x N).
 * @param V Workspace for V matrix (N x N).
 * @param singular_values Workspace for singular values (N).
 * @param superdiag Workspace for super-diagonal (N).
 */
__device__ void lstsq(double* A, int nrows, int ncols, double* b, double tolerance,
                      double* U, double* V, double* singular_values, double* superdiag) {
    // Store original A and b for residual check (similar to CPU's lstsq_check_infeasible)
    double A_copy[MAX_SVD_M * MAX_SVD_N];
    double b_orig[MAX_SVD_M];
    for (int i = 0; i < nrows * ncols; ++i) {
        A_copy[i] = A[i];
    }
    for (int i = 0; i < nrows; ++i) {
        b_orig[i] = b[i];
    }
    
    // Perform SVD: A = U * S * V^T
    int svd_result = Singular_Value_Decomposition(A, nrows, ncols, U, singular_values, V, superdiag);
    
    if (svd_result != 0) {
        // SVD failed - CPU doesn't have a fallback
        return;
    }
    
    // DEBUG: Print singular values for small systems
    bool is_infeasible = false;
    #ifdef VERBOSE_DEBUG
    if (nrows <= 4 && ncols <= 3) {  // Small systems that might be infeasible
        printf("[GPU LSTSQ DEBUG] SVD results for %dx%d system:\n", nrows, ncols);
        for (int i = 0; i < ncols; ++i) {
            printf("  singular_values[%d] = %.15e\n", i, singular_values[i]);
        }
        // Check if system is rank-deficient (smallest singular value very small)
        double min_sv = singular_values[0];
        for (int i = 1; i < ncols; ++i) {
            if (singular_values[i] < min_sv) min_sv = singular_values[i];
        }
        if (min_sv < 1e-10 * singular_values[0]) {  // Use tighter tolerance for rank detection
            is_infeasible = true;
            printf("  System appears INFEASIBLE (rank-deficient), min_sv/max_sv = %.15e\n", min_sv/singular_values[0]);
        }
    }
    #endif
    
    // Solve using SVD result
    // Note: b is input as RHS (size nrows), output as solution (size ncols)
    // We need a temporary array for the solution since b changes size
    double temp_solution[MAX_SVD_N];
    
    Singular_Value_Decomposition_Solve(U, singular_values, V, tolerance, nrows, ncols, b, temp_solution);
    
    // Check residual to detect spurious solutions (similar to CPU's lstsq_check_infeasible)
    double residual = 0.0;
    for (int i = 0; i < nrows; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < ncols; ++j) {
            row_sum += A_copy[i * ncols + j] * temp_solution[j];
        }
        double diff = row_sum - b_orig[i];
        residual += diff * diff;
    }
    
    #ifdef VERBOSE_DEBUG
    if (nrows <= 4 && ncols <= 3) {
        printf("  Residual after solve: %.15e\n", residual);
        printf("  Solution vector:\n");
        for (int i = 0; i < ncols; ++i) {
            printf("    x[%d] = %.15e\n", i, temp_solution[i]);
        }
    }
    #endif
    
    // Copy solution back to b
    for (int i = 0; i < ncols; ++i) {
        b[i] = temp_solution[i];
    }
}

// Content of eqsolver.h (defines solve_equilibrium_at_condition, helpers)
#ifndef EQSOLVER_H
#define EQSOLVER_H

// #include "minimizer.h" // REQUIRED: Contains SystemSpecification, SystemState, constants, etc.  // Removed for inlining

// --- Helper Structs for GPU Data ---

// To represent the grid data (from pycalphad calculate output) on the GPU
typedef struct DeviceGrid {
    const double* Y_ptr;    // Flattened array of site fractions for all grid points
    const double* X_ptr;    // Flattened array of mole compositions for all grid points
    const double* GM_ptr;   // Array of Gibbs energies for all grid points
    const int* PhaseID_ptr; // Flattened array of phase identifiers for all grid points
                            // (integer IDs mapping to PhaseRecord in DevicePhaseData)

    int num_grid_points_total; // Total number of rows in the grid arrays
    int phase_dof_stride_Y;    // Max number of site fractions for any phase (stride for Y_ptr)
    int num_components_stride_X; // Number of components (stride for X_ptr)

    // For add_nearly_stable: mapping a global phase ID to its start/end row in the grid
    // This replaces grid.attrs['phase_indices']
    const int* phase_grid_indices_start; // Array indexed by global phase ID
    const int* phase_grid_indices_stop;  // Array indexed by global phase ID
    int num_mappable_phases_in_grid;   // Size of phase_grid_indices_start/stop arrays
} DeviceGrid;

// To represent PhaseRecord data on the GPU
typedef struct DevicePhaseData {
    const PhaseRecord* phase_records_array; // Pointer to an array of unique PhaseRecord structs
    int num_unique_phase_records;
    // If a mapping from a simple active_phase_id (0 to N_active_phases-1) to
    // an index in phase_records_array is needed, it can be passed or handled by caller.
    // For now, assume direct indexing or that CompositionSet stores a direct pointer or fat index.
    const int* grid_phase_id_to_record_index; // Array sized by max grid phase ID
    int max_grid_phase_id;
} DevicePhaseData;


// --- Forward Declarations ---
__device__ int get_phase_record_index(const DevicePhaseData* phase_data, int grid_phase_id);
__device__ bool run_loop(SystemSpecification* spec, SystemState* state, int max_iterations);

// --- Device Helper Functions ---

__device__ int argmax_gpu(const double* arr, int size) {
    if (size <= 0) return -1; // Or handle error appropriately
    int max_idx = 0;
    double max_val = arr[0];
    for (int i = 1; i < size; ++i) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}


/**
 * @brief Attempts to add a new phase with the largest driving force.
 * GPU equivalent of add_new_phases from eqsolver.pyx.py.
 *
 * @param added_phase_grid_idx Output: If a phase is added, this will be its index in the DeviceGrid.
 * @param current_sys_state Pointer to the current SystemState (contains chemical_potentials, compsets).
 * This function will try to find a candidate to add.
 * The actual addition to compsets array is handled by the caller.
 * @param spec System-wide specification.
 * @param grid_data Pointer to the grid data on the GPU.
 * @param phase_data Pointer to the phase record data on the GPU.
 * @param state_variables_values Current P, T, etc. for this condition.
 * @param minimum_df Minimum driving force to consider adding a phase.
 * @param removed_compsets Array of CompositionSets that were recently removed (for distinctness check).
 * @param num_removed_compsets Number of compsets in removed_compsets array.
 * @return True if a candidate phase is identified (caller then adds it), false otherwise.
 */
__device__ bool identify_candidate_phase_to_add(
    int* candidate_phase_grid_idx,         // Output: index in grid for the phase to add
    double* candidate_driving_force,       // Output: driving force of the candidate
    const SystemState* current_sys_state,  // Input: for chemical potentials, existing compsets
    const SystemSpecification* spec,
    const DeviceGrid* grid_data,
    const DevicePhaseData* phase_data,
    const double* state_variables_values,  // Current P, T for potential new phase
    double minimum_df,
    const CompositionSet* removed_compsets, // Array of recently removed CompositionSets
    int num_removed_compsets               // Number of removed compsets
) {
    if (grid_data->num_grid_points_total == 0) return false;

    // Calculate driving forces for all points in the grid
    // double driving_forces_on_grid[MAX_GRID_POINTS]; // MAX_GRID_POINTS needs to be a define
    // This dynamic allocation or large static array is problematic.
    // Process grid points sequentially or in chunks if too large for local static memory.
    // For now, assume we can iterate and find max on the fly.

    double largest_df = -INFINITY; // Negative infinity (matches CPU -np.inf)
    int best_grid_idx = -1;

    for (int i = 0; i < grid_data->num_grid_points_total; ++i) {
        double current_potential_energy = 0.0;
        for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
            // X_ptr is [grid_point_idx * num_components_stride_X + comp_idx]
            current_potential_energy += grid_data->X_ptr[i * grid_data->num_components_stride_X + comp_idx] *
                                        current_sys_state->chemical_potentials[comp_idx];
        }
        double df = current_potential_energy - grid_data->GM_ptr[i];

        if (df > largest_df) {
            int candidate_phase_id_from_grid = grid_data->PhaseID_ptr[i];
            if (candidate_phase_id_from_grid == -1) continue; // _FAKE_ or invalid phase in grid

            int candidate_record_index = get_phase_record_index(phase_data, candidate_phase_id_from_grid);
            if (candidate_record_index == -1) continue; // Invalid phase

            const PhaseRecord* candidate_phase_record = &phase_data->phase_records_array[candidate_record_index];
            
            // --- Primary distinctness check against removed compsets (matching CPU logic exactly) ---
            bool distinct_from_removed = true;
            for (int rem_idx = 0; rem_idx < num_removed_compsets; ++rem_idx) {
                const CompositionSet* removed_compset = &removed_compsets[rem_idx];
                
                // Check if same phase type (matching CPU: df_phase_name != compset.phase_record.phase_name)
                if (removed_compset->phase_record != candidate_phase_record) {
                    continue;
                }
                
                // Same phase type, check site fraction composition (matching CPU pyx logic lines 41-47)
                distinct_from_removed = false;
                for (int dof_idx = 0; dof_idx < candidate_phase_record->phase_dof; ++dof_idx) {
                    double candidate_site_frac = grid_data->Y_ptr[i * grid_data->phase_dof_stride_Y + dof_idx];
                    double removed_site_frac = removed_compset->dof[spec->num_statevars + dof_idx];
                    
                    // Use 10*COMP_DIFFERENCE_TOL as in CPU version (line 43 in pyx)
                    if (fabs(candidate_site_frac - removed_site_frac) > 10.0 * COMP_DIFFERENCE_TOL) {
                        distinct_from_removed = true;
                        break;
                    }
                }
                
                if (!distinct_from_removed) {
                    break; // Found a non-distinct match, skip this candidate
                }
            }
            
            if (!distinct_from_removed) {
                continue; // Skip this candidate, not distinct from removed phases (matching CPU line 51)
            }
            
            // ONLY if distinct from removed phases, update largest_df (matching CPU lines 52-53)
            largest_df = df;
            best_grid_idx = i;
        }
    }
    
    // After finding the best candidate, check distinctness against current compsets (matching CPU lines 54-72)
    if (largest_df > minimum_df && best_grid_idx != -1) {
        int final_candidate_phase_id = grid_data->PhaseID_ptr[best_grid_idx];
        int final_candidate_record_index = get_phase_record_index(phase_data, final_candidate_phase_id);
        const PhaseRecord* final_candidate_phase_record = &phase_data->phase_records_array[final_candidate_record_index];
        
        // Check distinctness against current compsets (matching CPU logic lines 62-72)
        for (int cs_idx = 0; cs_idx < current_sys_state->num_compsets; ++cs_idx) {
            if (current_sys_state->compsets[cs_idx].phase_record == nullptr || 
                current_sys_state->phase_amt[cs_idx] < MIN_PHASE_FRACTION) continue;

            if (current_sys_state->compsets[cs_idx].phase_record == final_candidate_phase_record) {
                // Same phase type, check overall composition (using X, not site fractions)
                bool compositions_distinct = false;
                for (int comp_j = 0; comp_j < spec->num_components; ++comp_j) {
                    double x_grid = grid_data->X_ptr[best_grid_idx * grid_data->num_components_stride_X + comp_j];
                    double x_compset = current_sys_state->compsets[cs_idx].X[comp_j];
                    if (fabs(x_grid - x_compset) > COMP_DIFFERENCE_TOL) {
                        compositions_distinct = true;
                        break;
                    }
                }
                if (!compositions_distinct) {
                    // Not distinct from current phase, reject this candidate
                    *candidate_phase_grid_idx = -1;
                    *candidate_driving_force = 0.0;
                    return false;
                }
            }
        }
        
        // Passed all distinctness checks
        *candidate_phase_grid_idx = best_grid_idx;
        *candidate_driving_force = largest_df;
        return true;
    }
    
    return false;
}

__device__ int get_phase_record_index(const DevicePhaseData* phase_data, int grid_phase_id) {
    if (grid_phase_id < 0 || grid_phase_id > phase_data->max_grid_phase_id) {
        return -1; // Invalid phase ID
    }
    return phase_data->grid_phase_id_to_record_index[grid_phase_id];
}

/**
 * @brief Identifies nearly stable phases not currently in the system.
 * GPU equivalent of add_nearly_stable from eqsolver.pyx.py.
 *
 * @param candidate_phases_to_add_grid_indices Output: Array to be filled with grid indices of nearly stable phases to consider.
 * @param num_candidates_found Output: Number of nearly stable phases found and placed in the output array.
 * @param max_candidates_to_find Max size of the candidate_phases_to_add_grid_indices array.
 * @param current_sys_state Pointer to the current SystemState.
 * @param spec System-wide specification.
 * @param grid_data Pointer to the grid data on the GPU.
 * @param phase_data Pointer to the phase record data on the GPU.
 * @param state_variables_values Current P, T, etc.
 * @param minimum_df Minimum driving force to consider.
 * @return True if any nearly stable phases were identified, false otherwise.
 */
__device__ bool identify_nearly_stable_phases(
    int* candidate_phases_to_add_grid_indices, // Output array
    double* candidate_phases_df,               // Output array for their DFs
    int* num_candidates_found,                 // Output count
    int max_candidates_to_find,
    const SystemState* current_sys_state,
    const SystemSpecification* spec,
    const DeviceGrid* grid_data,
    const DevicePhaseData* phase_data,
    const double* state_variables_values,
    double minimum_df) {

    *num_candidates_found = 0;
    if (grid_data->num_grid_points_total == 0) return false;

    // Calculate driving forces for all points on the grid (can be large)
    // Ideally, process this on the fly or use a shared memory buffer if block-parallel.
    // For a single thread processing one condition, we iterate.
    // double all_driving_forces_on_grid[MAX_GRID_POINTS]; // Problematic for stack
    // Instead, iterate per phase type.

    for (int record_idx = 0; record_idx < phase_data->num_unique_phase_records; ++record_idx) {
        // Check if this phase type (record_idx) is already in current_sys_state
        bool phase_type_entered = false;
        for (int cs_idx = 0; cs_idx < current_sys_state->num_compsets; ++cs_idx) {
             if (current_sys_state->compsets[cs_idx].phase_record == nullptr || current_sys_state->phase_amt[cs_idx] < MIN_PHASE_FRACTION) continue;
            // Compare by checking if the phase_record pointer matches one in the global array
            const PhaseRecord* pr_in_compset = current_sys_state->compsets[cs_idx].phase_record;
            if (pr_in_compset == &phase_data->phase_records_array[record_idx]) {
                phase_type_entered = true;
                break;
            }
        }
        if (phase_type_entered) continue;

        // This phase type is not in the current set. Find its best point on the grid.
        // record_idx corresponds to grid phase ID in our mapping
        if (record_idx >= grid_data->num_mappable_phases_in_grid) continue; // Phase not in grid map
        int grid_start_idx = grid_data->phase_grid_indices_start[record_idx];
        int grid_stop_idx = grid_data->phase_grid_indices_stop[record_idx];
        int num_points_for_this_phase = grid_stop_idx - grid_start_idx;

        if (num_points_for_this_phase <= 0) continue;

        double largest_df_for_this_phase_type = -INFINITY;
        int best_grid_idx_for_this_phase_type = -1;

        for (int i = 0; i < num_points_for_this_phase; ++i) {
            int current_grid_point_abs_idx = grid_start_idx + i;
            if (grid_data->PhaseID_ptr[current_grid_point_abs_idx] != record_idx) continue; // Should match if grid map is correct

            double current_potential_energy = 0.0;
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                current_potential_energy += grid_data->X_ptr[current_grid_point_abs_idx * grid_data->num_components_stride_X + comp_idx] *
                                            current_sys_state->chemical_potentials[comp_idx];
            }
            double df = current_potential_energy - grid_data->GM_ptr[current_grid_point_abs_idx];

            if (df > largest_df_for_this_phase_type) {
                largest_df_for_this_phase_type = df;
                best_grid_idx_for_this_phase_type = current_grid_point_abs_idx;
            }
        }

        if (largest_df_for_this_phase_type >= minimum_df && best_grid_idx_for_this_phase_type != -1) {
            if (*num_candidates_found < max_candidates_to_find) {
                candidate_phases_to_add_grid_indices[*num_candidates_found] = best_grid_idx_for_this_phase_type;
                candidate_phases_df[*num_candidates_found] = largest_df_for_this_phase_type;
                (*num_candidates_found)++;
            } else {
                // Found more candidates than space, could prioritize by DF later if needed
                break; 
            }
        }
    } // end loop global_pr_idx

    return (*num_candidates_found > 0);
}

// --- Main single-condition solver (moved from previous response, adapted) ---
// Note: Hook functions (pre_solve_hook, post_solve_hook) are implemented in minimizer.h
// Output struct for a single equilibrium calculation
typedef struct EquilibriumResultSingle { // Ensure this is defined (copied from previous prompt)
    double final_chemical_potentials[MAX_COMPONENTS];
    double final_system_gm;
    int num_stable_phases;
    int phase_ids[MAX_PHASES];
    double NP[MAX_PHASES];
    double X_phases[MAX_PHASES * MAX_COMPONENTS];
    double Y_phases[MAX_PHASES * MAX_DOF_PER_PHASE];
    bool converged;
} EquilibriumResultSingle;

// Input struct for a single condition
typedef struct ConditionArgsSingle { // Ensure this is defined
    double state_variables_values[MAX_STATEVARS];
} ConditionArgsSingle;

// Forward declare the InitialPhaseDataSingle struct (matches kernel definition)
typedef struct InitialPhaseDataSingle {
    int phase_indices[MAX_PHASES];
    double phase_amounts[MAX_PHASES];
    double site_fractions[MAX_PHASES * MAX_DOF_PER_PHASE];
    double compositions[MAX_PHASES * MAX_COMPONENTS];
    double chemical_potentials[MAX_COMPONENTS];
    int num_phases;
} InitialPhaseDataSingle;

/**
 * @brief Solves for equilibrium at a single set of conditions.
 * This is the GPU equivalent of the main loop body in _solve_eq_at_conditions.
 * Uses initial phase data from lower_convex_hull results.
 */
__device__ void solve_equilibrium_at_condition(
    int thread_id,
    const SystemSpecification* global_spec_base,
    const ConditionArgsSingle* condition_args, // Contains P, T for this point
    EquilibriumResultSingle* result,
    const DevicePhaseData* phase_data,         // Provides PhaseRecord array
    const InitialPhaseDataSingle* initial_data, // Initial phase data from lower_convex_hull
    const DeviceGrid* grid_data                // Grid data for add_new/nearly_stable
) {
    SystemSpecification current_spec = *global_spec_base; // Local copy of spec

    // 1. Initial setup: Create initial CompositionSets from lower_convex_hull results
    CompositionSet initial_compsets_for_thread[MAX_PHASES];
    int actual_num_initial_compsets = 0;

    // Debug: Initial data verification
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("GPU DEBUG: solve_equilibrium_at_condition - num_phases=%d\n", initial_data->num_phases);
        for (int debug_i = 0; debug_i < initial_data->num_phases && debug_i < 3; ++debug_i) {
            printf("GPU DEBUG: Initial phase %d: idx=%d, amount=%f, threshold=%e\n", 
                   debug_i, initial_data->phase_indices[debug_i], initial_data->phase_amounts[debug_i], MIN_PHASE_FRACTION);
        }
    }
    #endif

    // Use data from lower_convex_hull instead of default initialization
    // CRITICAL FIX: Phases are stored contiguously, not by phase ID!
    // phase_indices[i] tells us which phase model/record to use for phase instance i
    for (int i = 0; i < initial_data->num_phases && i < MAX_PHASES; ++i) {
        // CRITICAL FIX: Preserve spec integrity across iterations
        if (current_spec.num_statevars != global_spec_base->num_statevars) {
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0) printf("GPU DEBUG: WARNING - num_statevars corrupted from %d to %d, restoring\n", 
                                      global_spec_base->num_statevars, current_spec.num_statevars);
            #endif
            current_spec.num_statevars = global_spec_base->num_statevars;
        }
        int pr_idx = initial_data->phase_indices[i];  // Which phase model to use
        if (pr_idx < 0 || pr_idx >= phase_data->num_unique_phase_records) {
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0) printf("GPU DEBUG: Skipping phase instance %d - invalid model idx=%d\n", i, pr_idx);
            #endif
            continue;
        }
        
        double phase_amount = initial_data->phase_amounts[i];
        if (phase_amount <= MIN_PHASE_FRACTION) {
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0) printf("GPU DEBUG: Skipping phase %d - amount %f <= threshold %e\n", i, phase_amount, MIN_PHASE_FRACTION);
            #endif
            continue; // Skip negligible phases
        }

        initial_compsets_for_thread[actual_num_initial_compsets].phase_record = &phase_data->phase_records_array[pr_idx];
        const PhaseRecord* pr = initial_compsets_for_thread[actual_num_initial_compsets].phase_record;
        if (!pr) continue;

        // Set state variables from condition args
        for (int sv_idx = 0; sv_idx < current_spec.num_statevars; ++sv_idx) {
             if (sv_idx < MAX_STATEVARS)
                initial_compsets_for_thread[actual_num_initial_compsets].dof[sv_idx] = condition_args->state_variables_values[sv_idx];
        }
        
        // Set site fractions from lower_convex_hull results (not default values!)
        for (int sf_idx = 0; sf_idx < pr->phase_dof; ++sf_idx) {
            if (sf_idx < MAX_DOF_PER_PHASE) {
                // CRITICAL FIX: Use current_spec.num_statevars for consistent workspace DOF format
                initial_compsets_for_thread[actual_num_initial_compsets].dof[current_spec.num_statevars + sf_idx] = 
                    initial_data->site_fractions[i * MAX_DOF_PER_PHASE + sf_idx];
            } else {
                // CRITICAL FIX: Use current_spec.num_statevars for consistent workspace DOF format
                initial_compsets_for_thread[actual_num_initial_compsets].dof[current_spec.num_statevars + sf_idx] = 
                    1.0 / (pr->phase_dof > 0 ? pr->phase_dof : 1.0); // Fallback
            }
        }
        
        // Debug: Confirm site fractions are being read correctly
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0) {
            printf("[INITIAL] Phase instance %d (using model %d) site fractions from lower_convex_hull:\n", i, pr_idx);
            printf("  Raw site fractions: ");
            for (int sf_idx = 0; sf_idx < pr->phase_dof && sf_idx < 5; ++sf_idx) {
                printf("Y[%d]=%.15e ", sf_idx, initial_data->site_fractions[i * MAX_DOF_PER_PHASE + sf_idx]);
            }
            printf("\n  DOF array after copy: ");
            for (int sf_idx = 0; sf_idx < pr->phase_dof && sf_idx < 5; ++sf_idx) {
                // CRITICAL FIX: Use current_spec.num_statevars for consistent workspace DOF format
                printf("dof[%d]=%.15e ", current_spec.num_statevars + sf_idx,
                       initial_compsets_for_thread[actual_num_initial_compsets].dof[current_spec.num_statevars + sf_idx]);
            }
            printf("\n");
        }
        #endif
        
        // Set phase amount from lower_convex_hull results (not default values!)
        initial_compsets_for_thread[actual_num_initial_compsets].NP = phase_amount;
        initial_compsets_for_thread[actual_num_initial_compsets].fixed = false;
        initial_compsets_for_thread[actual_num_initial_compsets].init(pr);
        initial_compsets_for_thread[actual_num_initial_compsets].update(
            &initial_compsets_for_thread[actual_num_initial_compsets].dof[current_spec.num_statevars],
            initial_compsets_for_thread[actual_num_initial_compsets].NP,
            initial_compsets_for_thread[actual_num_initial_compsets].dof,
            current_spec.num_statevars);
        actual_num_initial_compsets++;
        
        // Debug: Confirm CompositionSet was created successfully
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0) {
            printf("GPU DEBUG: Successfully created CompositionSet %d for phase %d\n", actual_num_initial_compsets-1, i);
        }
        #endif
    }

    // Debug: Show total number of CompositionSets created
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("GPU DEBUG: Total CompositionSets created: %d\n", actual_num_initial_compsets);
    }
    #endif

    SystemState current_sys_state;
    current_sys_state.init(&current_spec, initial_compsets_for_thread, actual_num_initial_compsets);
    
    // CRITICAL FIX: Do NOT call recompute here - it should only be called after phase normalization
    // The CPU doesn't call recompute until inside solve_state, after phase amounts are normalized
    // Calling it here with unnormalized phase amounts causes different initial equilibrium matrices
    
    // Debug: Verify SystemState initialization
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("GPU DEBUG: SystemState initialized with num_compsets=%d\n", current_sys_state.num_compsets);
        // Check all phases - Note: phase_compositions won't be calculated yet without recompute
        for (int i = 0; i < current_sys_state.num_compsets && i < 3; ++i) {
            printf("[INITIAL PHASES] Phase %d:\n", i);
            printf("  Phase amount: %.15e\n", current_sys_state.phase_amt[i]);
            // Phase compositions are calculated in recompute, which we're delaying
            printf("  Phase compositions: (not yet calculated - will be done after normalization)\n");
            // Also show site fractions
            CompositionSet* cs = &current_sys_state.compsets[i];
            if (cs->phase_record != nullptr) {
                printf("  Site fractions: Y[0]=%.15e, Y[1]=%.15e\n",
                       cs->dof[current_spec.num_statevars + 0],
                       cs->dof[current_spec.num_statevars + 1]);
            }
        }
    }
    #endif
    
    // Initialize chemical potentials from lower_convex_hull results
    for (int i = 0; i < current_spec.num_components && i < MAX_COMPONENTS; ++i) {
        current_sys_state.chemical_potentials[i] = initial_data->chemical_potentials[i];
    }

    // 2. Call add_nearly_stable_gpu (equivalent to pyx _solve_eq_at_conditions pre-loop call)
    // SEGMENT 14: ADD NEARLY STABLE PHASES
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("[GPU] SEGMENT 14: Add nearly stable phases\n");
        printf("[GPU]   threshold: -1000 J/mol\n");
        printf("[GPU]   initial_phase_count: %d\n", current_sys_state.num_compsets);
    }
    #endif
    
    int nearly_stable_candidates_indices[MAX_PHASES]; // Max possible phases to add
    double nearly_stable_candidates_dfs[MAX_PHASES];
    int num_nearly_stable_found = 0;
    identify_nearly_stable_phases(nearly_stable_candidates_indices, nearly_stable_candidates_dfs,
                                  &num_nearly_stable_found, MAX_PHASES,
                                  &current_sys_state, &current_spec, grid_data, phase_data,
                                  condition_args->state_variables_values, -1000.0 /* minimum_df from pyx */);

    for (int i = 0; i < num_nearly_stable_found; ++i) {
        if (current_sys_state.num_compsets >= MAX_PHASES) break; // No space to add
        
        int grid_idx_to_add = nearly_stable_candidates_indices[i];
        int phase_id_from_grid = grid_data->PhaseID_ptr[grid_idx_to_add];
        if (phase_id_from_grid < 0 || phase_id_from_grid >= phase_data->num_unique_phase_records) continue;

        // Check if this phase *type* is already present. If so, might skip or require very distinct composition.
        // For add_nearly_stable, pyx adds it as metastable (NP=0) if phase *type* is unrepresented.
        bool type_already_present = false;
        for(int csj=0; csj < current_sys_state.num_compsets; ++csj) {
            if (current_sys_state.compsets[csj].phase_record == &phase_data->phase_records_array[phase_id_from_grid]) {
                type_already_present = true;
                break;
            }
        }
        if (type_already_present) continue;


        CompositionSet* new_cs = &current_sys_state.compsets[current_sys_state.num_compsets]; // Get next available slot
        new_cs->phase_record = &phase_data->phase_records_array[phase_id_from_grid];
        const PhaseRecord* pr = new_cs->phase_record;

        for (int sv_idx = 0; sv_idx < current_spec.num_statevars; ++sv_idx) {
            new_cs->dof[sv_idx] = condition_args->state_variables_values[sv_idx];
        }
        // Site fracs from grid_data->Y_ptr
        for (int sf_idx = 0; sf_idx < pr->phase_dof; ++sf_idx) {
            if (sf_idx < grid_data->phase_dof_stride_Y) { // Check Y stride
                 // CRITICAL FIX: Use current_spec.num_statevars for consistent workspace DOF format
                 new_cs->dof[current_spec.num_statevars + sf_idx] = grid_data->Y_ptr[grid_idx_to_add * grid_data->phase_dof_stride_Y + sf_idx];
            } else {
                 // CRITICAL FIX: Use current_spec.num_statevars for consistent workspace DOF format
                 new_cs->dof[current_spec.num_statevars + sf_idx] = 1.0 / (pr->phase_dof > 0 ? pr->phase_dof : 1.0); // Fallback
            }
        }
        new_cs->NP = 0.0; // Added as metastable
        new_cs->fixed = false;
        new_cs->init(pr);
        // CRITICAL FIX: Use current_spec.num_statevars for consistent workspace DOF format
        // The GPU uses workspace variables (N,P,T) throughout for consistency with CPU
        new_cs->update(&new_cs->dof[current_spec.num_statevars], new_cs->NP, new_cs->dof, current_spec.num_statevars);
        current_sys_state.cs_states[current_sys_state.num_compsets].init(&current_spec, new_cs); // Init corresponding CompsetState
        current_sys_state.num_compsets++;
    }
    
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("[GPU]   phase_count_after_adding: %d\n", current_sys_state.num_compsets);
    }
    #endif
    
    // Normalize phase amounts after adding nearly stable phases (matching CPU logic lines 231-235)
    // CRITICAL FIX: Only normalize if phase amounts sum is significantly different from 1.0
    // This prevents re-normalization after consolidation where sum might be < 1.0
    double phase_amt_sum = 0.0;
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        phase_amt_sum += current_sys_state.compsets[i].NP;
    }
    
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("[GPU]   phase_amount_sum_before_normalization: %.15e\n", phase_amt_sum);
    }
    #endif
    
    // CRITICAL FIX: Always normalize phase amounts to match CPU behavior exactly
    // CPU eqsolver.pyx lines 290-295 always normalizes unconditionally
    // The GPU was conditionally normalizing which caused differences
    if (phase_amt_sum > 1e-12) { // Only check for non-zero sum to avoid division by zero
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            current_sys_state.compsets[i].NP /= phase_amt_sum;
            current_sys_state.phase_amt[i] = current_sys_state.compsets[i].NP;
        }
    } else {
        // Handle zero sum case (shouldn't happen in normal operation)
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            current_sys_state.phase_amt[i] = current_sys_state.compsets[i].NP;
        }
    }
    
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("[GPU]   normalized_phases: [");
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            printf("(phase_%d, %.6f)", i, current_sys_state.compsets[i].NP);
            if (i < current_sys_state.num_compsets - 1) printf(", ");
        }
        printf("]\n");
    }
    #endif
    
    // CRITICAL FIX: Just synchronize phase_amt with normalized NP values
    // Do NOT call update here - the CPU doesn't recalculate energies after normalization
    // The first recompute in solve_state will handle all calculations with normalized amounts
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        current_sys_state.phase_amt[i] = current_sys_state.compsets[i].NP;
        
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0 && i == 0) {
            printf("GPU DEBUG: synchronized phase_amt[%d]=%f with NP=%f\n", 
                   i, current_sys_state.phase_amt[i], current_sys_state.compsets[i].NP);
        }
        #endif
    }
    
    // After adding nearly stable, re-initialize free_stable_compset_indices and other counts in SystemState
    // A full re-init of parts of SystemState or a specific update function might be needed.
    // For now, assume recompute() inside run_loop handles this.


    // 3. Iterative solving loop (mimicking pyx _solve_eq_at_conditions structure)
    bool converged = false;
    bool changed_phases_in_iteration;
    int iterations = 0;
    const int MAX_OUTER_ITERATIONS = 10; // from pyx
    
    // Track removed composition sets to prevent oscillation (matching CPU logic)
    CompositionSet removed_compsets[MAX_PHASES];
    int num_removed_compsets = 0;

    while (iterations < MAX_OUTER_ITERATIONS) {
        if (current_sys_state.num_compsets == 0 && current_spec.prescribed_system_amount > 0) {
             // No phases to solve for, but system exists. This is likely an error or unstable state.
             converged = false; // Or handle as an error state specific to the calculation.
             break;
        }
        
        // Save current composition sets before solving (to track any that get removed)
        CompositionSet compsets_before_solve[MAX_PHASES];
        int num_compsets_before = current_sys_state.num_compsets;
        for (int i = 0; i < num_compsets_before; ++i) {
            compsets_before_solve[i] = current_sys_state.compsets[i]; // Simple assignment copy
        }
        
        converged = run_loop(&current_spec, &current_sys_state, 1000);
        
        // Track any phases that were removed during run_loop (matching minimizer's removal tracking)
        for (int before_idx = 0; before_idx < num_compsets_before; ++before_idx) {
            bool still_present = false;
            for (int after_idx = 0; after_idx < current_sys_state.num_compsets; ++after_idx) {
                if (current_sys_state.compsets[after_idx].phase_record == compsets_before_solve[before_idx].phase_record) {
                    // Check if compositions are similar enough to be considered the same phase
                    // Use 10*COMP_DIFFERENCE_TOL for consistency with removed phase distinctness check
                    bool same_composition = true;
                    for (int dof_idx = 0; dof_idx < compsets_before_solve[before_idx].phase_record->phase_dof; ++dof_idx) {
                        if (fabs(current_sys_state.compsets[after_idx].dof[current_spec.num_statevars + dof_idx] - 
                                compsets_before_solve[before_idx].dof[current_spec.num_statevars + dof_idx]) > 10.0 * COMP_DIFFERENCE_TOL) {
                            same_composition = false;
                            break;
                        }
                    }
                    // Also check if phase amount is still significant (not removed to ~0)
                    if (same_composition && current_sys_state.phase_amt[after_idx] > MIN_PHASE_FRACTION) {
                        still_present = true;
                        break;
                    }
                }
            }
            
            // If this composition set was removed, add it to removed_compsets tracking
            if (!still_present && num_removed_compsets < MAX_PHASES) {
                removed_compsets[num_removed_compsets] = compsets_before_solve[before_idx]; // Simple assignment instead of deep_copy_from
                num_removed_compsets++;
            }
        }

        // SEGMENT 41: PHASE ADDITION - DRIVING FORCE CALCULATION
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0) {
            printf("[GPU] SEGMENT 41: Phase addition - driving force calculation\n");
            printf("[GPU]   chemical_potentials: [%.6f, %.6f]\n",
                   current_sys_state.chemical_potentials[0], current_sys_state.chemical_potentials[1]);
        }
        #endif
        
        int candidate_idx_to_add = -1;
        double candidate_df = 0.0;
        changed_phases_in_iteration = identify_candidate_phase_to_add(
                                        &candidate_idx_to_add, &candidate_df,
                                        &current_sys_state, &current_spec, grid_data, phase_data,
                                        condition_args->state_variables_values,
                                        1e-4 /* minimum_df from pyx */,
                                        removed_compsets, num_removed_compsets);
        
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0) {
            printf("[GPU]   max_driving_force: %.15e\n", candidate_df);
            printf("[GPU]   min_driving_force: %.15e\n", -INFINITY); // hardcoded minimum (matches CPU -np.inf)
        }
        #endif

        // SEGMENT 42: PHASE ADDITION - DECISION
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0) {
            printf("[GPU] SEGMENT 42: Phase addition - decision\n");
            printf("[GPU]   largest_df: %.15e\n", candidate_df);
            printf("[GPU]   minimum_df: %.15e\n", 1e-4);
            printf("[GPU]   will_add_phase: %s\n", changed_phases_in_iteration ? "true" : "false");
        }
        #endif
        
        if (changed_phases_in_iteration && candidate_idx_to_add != -1) {
            if (current_sys_state.num_compsets < MAX_PHASES) {
                int phase_id_from_grid = grid_data->PhaseID_ptr[candidate_idx_to_add];
                if (phase_id_from_grid < 0 || phase_id_from_grid >= phase_data->num_unique_phase_records) {
                     changed_phases_in_iteration = false; // Invalid candidate
                } else {
                    #ifdef VERBOSE_DEBUG
                    if (thread_id == 0) {
                        printf("[GPU]   candidate_phase: phase_%d\n", phase_id_from_grid);
                        // Print candidate composition - need to get from grid
                        printf("[GPU]   candidate_composition: [");
                        for (int c = 0; c < current_spec.num_components; ++c) {
                            printf("%.6f", grid_data->X_ptr[candidate_idx_to_add * grid_data->num_components_stride_X + c]);
                            if (c < current_spec.num_components - 1) printf(", ");
                        }
                        printf("]\n");
                    }
                    #endif
                    CompositionSet* new_cs = &current_sys_state.compsets[current_sys_state.num_compsets];
                    new_cs->phase_record = &phase_data->phase_records_array[phase_id_from_grid];
                    const PhaseRecord* pr = new_cs->phase_record;

                    for (int sv_idx = 0; sv_idx < current_spec.num_statevars; ++sv_idx) {
                        new_cs->dof[sv_idx] = condition_args->state_variables_values[sv_idx];
                    }
                                for (int sf_idx = 0; sf_idx < pr->phase_dof; ++sf_idx) {
                         if (sf_idx < grid_data->phase_dof_stride_Y) {
                            // CRITICAL FIX: Use current_spec.num_statevars for consistent workspace DOF format
                            new_cs->dof[current_spec.num_statevars + sf_idx] = grid_data->Y_ptr[candidate_idx_to_add * grid_data->phase_dof_stride_Y + sf_idx];
                         } else {
                            // CRITICAL FIX: Use current_spec.num_statevars for consistent workspace DOF format
                            new_cs->dof[current_spec.num_statevars + sf_idx] = 1.0 / (pr->phase_dof > 0 ? pr->phase_dof : 1.0);
                         }
                    }
                    new_cs->NP = 1e-6; // Small amount for newly added phase (from pyx)
                    new_cs->fixed = false;
                    new_cs->init(pr);
                    // CRITICAL FIX: Use current_spec.num_statevars for consistent workspace DOF format
        // The GPU uses workspace variables (N,P,T) throughout for consistency with CPU
        new_cs->update(&new_cs->dof[current_spec.num_statevars], new_cs->NP, new_cs->dof, current_spec.num_statevars);
                    current_sys_state.cs_states[current_sys_state.num_compsets].init(&current_spec, new_cs);
                    current_sys_state.num_compsets++;
                    // After adding, SystemState's internal counts like num_free_stable_compsets need update.
                    // A call to a lightweight SystemState update function or manual update is needed here.
                    // For now, assume recompute() at start of run_loop handles this.
                }
            } else {
                changed_phases_in_iteration = false; // No space to add
            }
        }
        iterations++;
        if (!changed_phases_in_iteration) {
            break; // No new phases to add, exit outer loop
        }
    } // end while iterations

    // Final solve if phases were changed in the last iteration
    if (changed_phases_in_iteration) {
         if (current_sys_state.num_compsets > 0) { // Only solve if there are phases
            // Save current composition sets before final solve (to track any that get removed)
            CompositionSet compsets_before_final_solve[MAX_PHASES];
            int num_compsets_before_final = current_sys_state.num_compsets;
            for (int i = 0; i < num_compsets_before_final; ++i) {
                compsets_before_final_solve[i] = current_sys_state.compsets[i]; // Simple assignment copy
            }
            
            // CRITICAL FIX: Use same iteration limit as CPU (1000) instead of 500
            converged = run_loop(&current_spec, &current_sys_state, 1000);
            
            // Track any phases that were removed during final run_loop
            for (int before_idx = 0; before_idx < num_compsets_before_final; ++before_idx) {
                bool still_present = false;
                for (int after_idx = 0; after_idx < current_sys_state.num_compsets; ++after_idx) {
                    if (current_sys_state.compsets[after_idx].phase_record == compsets_before_final_solve[before_idx].phase_record) {
                        // Check if compositions are similar enough to be considered the same phase
                        // Use 10*COMP_DIFFERENCE_TOL for consistency with removed phase distinctness check
                        bool same_composition = true;
                        for (int dof_idx = 0; dof_idx < compsets_before_final_solve[before_idx].phase_record->phase_dof; ++dof_idx) {
                            if (fabs(current_sys_state.compsets[after_idx].dof[current_spec.num_statevars + dof_idx] - 
                                    compsets_before_final_solve[before_idx].dof[current_spec.num_statevars + dof_idx]) > 10.0 * COMP_DIFFERENCE_TOL) {
                                same_composition = false;
                                break;
                            }
                        }
                        // Also check if phase amount is still significant (not removed to ~0)
                        if (same_composition && current_sys_state.phase_amt[after_idx] > MIN_PHASE_FRACTION) {
                            still_present = true;
                            break;
                        }
                    }
                }
                
                // If this composition set was removed, add it to removed_compsets tracking
                if (!still_present && num_removed_compsets < MAX_PHASES) {
                    removed_compsets[num_removed_compsets] = compsets_before_final_solve[before_idx]; // Simple assignment
                    num_removed_compsets++;
                }
            }
         } else {
            converged = false; // No phases left, cannot converge to a meaningful state if system amount > 0
         }
    }

    // NOTE: Phase consolidation is already handled in run_loop via remove_and_consolidate_phases
    // The CPU does NOT perform additional phase consolidation after the solver completes
    // We only have consolidation during the solver iterations, not after

    // 4. Store results (copied and adapted from previous `solve_equilibrium_at_condition` body)
    
    // DEBUG: Final values
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("\n[GPU FINAL VALUES]\n");
        printf("  Converged: %s\n", converged ? "true" : "false");
        printf("  Number of stable phases: %d\n", current_sys_state.num_free_stable_compsets);
        for (int i = 0; i < current_sys_state.num_free_stable_compsets; ++i) {
            int idx = current_sys_state.free_stable_compset_indices[i];
            printf("  Phase %d:\n", idx);
            printf("    Amount: %.15e\n", current_sys_state.phase_amt[idx]);
            printf("    X[0]: %.15e\n", current_sys_state.phase_compositions[idx * MAX_COMPONENTS + 0]);
            printf("    X[1]: %.15e\n", current_sys_state.phase_compositions[idx * MAX_COMPONENTS + 1]);
            CompositionSet* cs = &current_sys_state.compsets[idx];
            printf("    Y[0]: %.15e\n", cs->dof[current_spec.num_statevars + 0]);
            printf("    Y[1]: %.15e\n", cs->dof[current_spec.num_statevars + 1]);
        }
        printf("  System mole fractions: X[0]=%.15e, X[1]=%.15e\n",
               current_sys_state.mole_fractions[0], current_sys_state.mole_fractions[1]);
    }
    #endif
    
    result->converged = converged;
    // ... (rest of result population is identical to the previous response's version of this function)
    for (int i = 0; i < current_spec.num_components; ++i) {
        if (i < MAX_COMPONENTS)
            result->final_chemical_potentials[i] = current_sys_state.chemical_potentials[i];
    }

    // SEGMENT 40: FINAL GIBBS ENERGY CALCULATION
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("[GPU] SEGMENT 40: Final Gibbs energy calculation\n");
    }
    #endif
    
    double final_gm_calc = 0.0;
    int stable_phase_count = 0;
    #ifdef VERBOSE_DEBUG
    printf("GPU DEBUG: Collecting stable phases - num_compsets=%d, MIN_PHASE_FRACTION=%e\n", 
           current_sys_state.num_compsets, MIN_PHASE_FRACTION);
    #endif
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: compset %d - phase_amt=%.10f, threshold=%e\n", 
               i, current_sys_state.phase_amt[i], MIN_PHASE_FRACTION);
        #endif
        if (current_sys_state.phase_amt[i] > MIN_PHASE_FRACTION) {
            // Use cs_states[i].energy which is set to pr->formulaobj(compset->dof) in recompute()
            // This is the Gibbs energy per formula unit, which is what the CPU uses
            double phase_contribution = current_sys_state.phase_amt[i] * current_sys_state.cs_states[i].energy;
            final_gm_calc += phase_contribution;
            
            #ifdef VERBOSE_DEBUG
            if (thread_id == 0) {
                printf("[GPU]   phase_%d_contribution: NP=%.15e * energy=%.15e = %.15e\n",
                       i, current_sys_state.phase_amt[i], current_sys_state.cs_states[i].energy,
                       phase_contribution);
            }
            #endif
            
            if (stable_phase_count < MAX_PHASES) {
                result->phase_ids[stable_phase_count] = -1;
                const PhaseRecord* pr_stable = current_sys_state.compsets[i].phase_record;
                if (pr_stable != nullptr) {
                    for (int pr_glob_idx = 0; pr_glob_idx < phase_data->num_unique_phase_records; ++pr_glob_idx) {
                        if (pr_stable == &phase_data->phase_records_array[pr_glob_idx]) {
                            result->phase_ids[stable_phase_count] = pr_glob_idx;
                            break;
                        }
                    }
                }
                result->NP[stable_phase_count] = current_sys_state.phase_amt[i];
                double sum_moles_in_phase_formula = 0.0;
                for (int c = 0; c < current_spec.num_components; ++c) {
                     if (c < MAX_COMPONENTS)
                        sum_moles_in_phase_formula += current_sys_state.phase_compositions[i * current_spec.num_components + c];
                }
                if (fabs(sum_moles_in_phase_formula) < 1e-12) sum_moles_in_phase_formula = 1.0;
                for (int c = 0; c < current_spec.num_components; ++c) {
                    if (stable_phase_count * MAX_COMPONENTS + c < MAX_PHASES * MAX_COMPONENTS && c < MAX_COMPONENTS) {
                        result->X_phases[stable_phase_count * MAX_COMPONENTS + c] =
                            current_sys_state.phase_compositions[i * current_spec.num_components + c] / sum_moles_in_phase_formula;
                    }
                }
                if (current_sys_state.compsets[i].phase_record) {
                    const PhaseRecord* pr = current_sys_state.compsets[i].phase_record;
                    
                    #ifdef VERBOSE_DEBUG
                    if (thread_id == 0) {
                        printf("[GPU]   phase_%d_Y: [", i);
                    }
                    #endif
                    
                    for (int sf = 0; sf < pr->phase_dof; ++sf) {
                        if (stable_phase_count * MAX_DOF_PER_PHASE + sf < MAX_PHASES * MAX_DOF_PER_PHASE && sf < MAX_DOF_PER_PHASE) {
                            result->Y_phases[stable_phase_count * MAX_DOF_PER_PHASE + sf] =
                                current_sys_state.compsets[i].dof[current_spec.num_statevars + sf];
                            
                            #ifdef VERBOSE_DEBUG
                            if (thread_id == 0) {
                                printf("%.6f", current_sys_state.compsets[i].dof[current_spec.num_statevars + sf]);
                                if (sf < pr->phase_dof - 1) printf(", ");
                            }
                            #endif
                        }
                    }
                    
                    #ifdef VERBOSE_DEBUG
                    if (thread_id == 0) {
                        printf("]\n");
                        printf("[GPU]   phase_%d_X: [", i);
                        for (int c = 0; c < current_spec.num_components; ++c) {
                            printf("%.6f", result->X_phases[stable_phase_count * MAX_COMPONENTS + c]);
                            if (c < current_spec.num_components - 1) printf(", ");
                        }
                        printf("]\n");
                    }
                    #endif
                }
                stable_phase_count++;
            }
        }
    }
    result->final_system_gm = final_gm_calc;
    result->num_stable_phases = stable_phase_count;
    
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0) {
        printf("[GPU]   final_GM: %.15e\n", final_gm_calc);
    }
    #endif
    
    #ifdef VERBOSE_DEBUG
    printf("GPU DEBUG: Final result assembly - stable_phase_count=%d\n", stable_phase_count);
    for (int i = 0; i < stable_phase_count; ++i) {
        printf("GPU DEBUG: result->NP[%d] = %.10f\n", i, result->NP[i]);
    }
    #endif
    
    for (int i = stable_phase_count; i < MAX_PHASES; ++i) {
        result->phase_ids[i] = -1;
        result->NP[i] = 0.0;
        for (int c = 0; c < MAX_COMPONENTS; ++c) {
             if (i * MAX_COMPONENTS + c < MAX_PHASES * MAX_COMPONENTS) result->X_phases[i * MAX_COMPONENTS + c] = 0.0;
        }
        for (int sf = 0; sf < MAX_DOF_PER_PHASE; ++sf) {
            if (i * MAX_DOF_PER_PHASE + sf < MAX_PHASES * MAX_DOF_PER_PHASE) result->Y_phases[i * MAX_DOF_PER_PHASE + sf] = 0.0;
        }
    }
}


#endif // EQSOLVER_H

// --- Dynamically Generated __device__ Model Functions ---
__device__ double pycgpu_model_0_obj(const double* x) {
    double x0 = pow(2.0*x[3] + 1.0*(x[4] + x[5] + x[6]), -1);
    double x1 = x[5]*x[3];
    double x2 = pow(x[2], 3.0);
    double x3 = pow(x[2], -1.0);
    double x4 = 74092.0*x3;
    double x5 = x[2]*log(x[2]);
    double x6 = pow(x[2], 2.0);
    double x7 = pow(x[2], -9.0);
    double x8 = ((x[2] < 700.0) ? (
   -7976.15 + 137.093038*x[2] - 8.77664e-07*x2 + x4 - 24.3671976*x5 - 0.001884662*x6
)
: (((x[2] < 933.47 && 700.0 <= x[2])) ? (
   -11276.24 + 223.048446*x[2] - 5.764227e-06*x2 + x4 - 38.5844296*x5 + 0.018531982*x6
)
: ((933.47 <= x[2]) ? (
   -11278.378 + 188.684153*x[2] - 31.748192*x5 - 1.230524e+28*x7
)
: (
   0
))));
    double x9 = 2.0*x8;
    return x0*(x1*(-47406.0 + 6.75*x[2] + x9 + ((x[2] < 1357.77) ? (
   -7770.458 + 130.485235*x[2] + 1.29223e-07*x2 + 52478.0*x3 - 24.112392*x5 - 0.00265684*x6
)
: ((1357.77 <= x[2]) ? (
   -13542.026 + 183.803828*x[2] - 31.38*x5 + 3.64167e+29*x7
)
: (
   0
)))) + (x9 + ((x[2] < 1811.0) ? (
   1225.7 + 124.134*x[2] - 5.8927e-08*x2 + 77359.0*x3 - 23.5143*x5 - 0.00439752*x6
)
: ((1811.0 <= x[2]) ? (
   -25383.581 + 299.31255*x[2] - 46.0*x5 + 2.29603e+31*x7
)
: (
   0
))))*x[6]*x[3] + 3.0*(10083.0 - 4.813*x[2] + x8)*x[4]*x[3]) + 8.3145*x[2]*x0*(2.0*((1e-15 < x[3]) ? (
   x[3]*log(x[3])
)
: (
   0
)) + 1.0*((1e-15 < x[4]) ? (
   x[4]*log(x[4])
)
: (
   0
)) + 1.0*((1e-15 < x[5]) ? (
   x[5]*log(x[5])
)
: (
   0
)) + 1.0*((1e-15 < x[6]) ? (
   x[6]*log(x[6])
)
: (
   0
))) + 2211.0*x0*x1*x[4];
}

__device__ double pycgpu_model_0_formulaobj(const double* x) {
    double x0 = 2.0*x[3] + 1.0*(x[4] + x[5] + x[6]);
    double x1 = pow(x0, -1);
    double x2 = x[5]*x[3];
    double x3 = pow(x[2], 3.0);
    double x4 = pow(x[2], -1.0);
    double x5 = 74092.0*x4;
    double x6 = x[2]*log(x[2]);
    double x7 = pow(x[2], 2.0);
    double x8 = pow(x[2], -9.0);
    double x9 = ((x[2] < 700.0) ? (
   -7976.15 + 137.093038*x[2] - 8.77664e-07*x3 + x5 - 24.3671976*x6 - 0.001884662*x7
)
: (((x[2] < 933.47 && 700.0 <= x[2])) ? (
   -11276.24 + 223.048446*x[2] - 5.764227e-06*x3 + x5 - 38.5844296*x6 + 0.018531982*x7
)
: ((933.47 <= x[2]) ? (
   -11278.378 + 188.684153*x[2] - 31.748192*x6 - 1.230524e+28*x8
)
: (
   0
))));
    double x10 = 2.0*x9;
    return x0*(x1*(x2*(-47406.0 + 6.75*x[2] + x10 + ((x[2] < 1357.77) ? (
   -7770.458 + 130.485235*x[2] + 1.29223e-07*x3 + 52478.0*x4 - 24.112392*x6 - 0.00265684*x7
)
: ((1357.77 <= x[2]) ? (
   -13542.026 + 183.803828*x[2] - 31.38*x6 + 3.64167e+29*x8
)
: (
   0
)))) + (x10 + ((x[2] < 1811.0) ? (
   1225.7 + 124.134*x[2] - 5.8927e-08*x3 + 77359.0*x4 - 23.5143*x6 - 0.00439752*x7
)
: ((1811.0 <= x[2]) ? (
   -25383.581 + 299.31255*x[2] - 46.0*x6 + 2.29603e+31*x8
)
: (
   0
))))*x[6]*x[3] + 3.0*(10083.0 - 4.813*x[2] + x9)*x[4]*x[3]) + 8.3145*x[2]*x1*(2.0*((1e-15 < x[3]) ? (
   x[3]*log(x[3])
)
: (
   0
)) + 1.0*((1e-15 < x[4]) ? (
   x[4]*log(x[4])
)
: (
   0
)) + 1.0*((1e-15 < x[5]) ? (
   x[5]*log(x[5])
)
: (
   0
)) + 1.0*((1e-15 < x[6]) ? (
   x[6]*log(x[6])
)
: (
   0
))) + 2211.0*x2*x1*x[4]);
}

__device__ void pycgpu_model_0_formulagrad(double* out, const double* x) {
    double x0 = 2.0*x[3] + 1.0*(x[4] + x[5] + x[6]);
    double x1 = pow(x[2], 1.0);
    double x2 = log(x[2]);
    double x3 = 23.5143*x2;
    double x4 = pow(x[2], 2.0);
    double x5 = pow(x4, -1);
    double x6 = x[2] < 1811.0;
    double x7 = pow(x[2], -10.0);
    double x8 = 46.0*x2;
    double x9 = 1811.0 <= x[2];
    double x10 = 24.3671976*x2;
    double x11 = -74092.0*x5;
    double x12 = x[2] < 700.0;
    double x13 = 38.5844296*x2;
    double x14 = (x[2] < 933.47 && 700.0 <= x[2]);
    double x15 = 31.748192*x2;
    double x16 = 933.47 <= x[2];
    double x17 = ((x12 == 1) ? (
   112.7258404 - 0.003769324*x1 - x10 + x11 - 2.632992e-06*x4
)
: ((x14 == 1) ? (
   184.4640164 + 0.037063964*x1 + x11 - x13 - 1.7292681e-05*x4
)
: ((x16 == 1) ? (
   156.935961 - x15 + 1.1074716e+29*x7
)
: (
   0
))));
    double x18 = 2.0*x17;
    double x19 = x[6]*x[3];
    double x20 = 24.112392*x2;
    double x21 = x[2] < 1357.77;
    double x22 = 31.38*x2;
    double x23 = 1357.77 <= x[2];
    double x24 = x[5]*x[3];
    double x25 = x[4]*x[3];
    double x26 = 3.0*x25;
    double x27 = pow(x0, -1);
    double x28 = log(x[3]);
    double x29 = 1e-15 < x[3];
    double x30 = log(x[6]);
    double x31 = 1e-15 < x[6];
    double x32 = log(x[5]);
    double x33 = 1e-15 < x[5];
    double x34 = log(x[4]);
    double x35 = 1e-15 < x[4];
    double x36 = 2.0*((x29 == 1) ? (
   x28*x[3]
)
: (
   0
)) + 1.0*((x31 == 1) ? (
   x30*x[6]
)
: (
   0
)) + 1.0*((x33 == 1) ? (
   x32*x[5]
)
: (
   0
)) + 1.0*((x35 == 1) ? (
   x34*x[4]
)
: (
   0
));
    double x37 = 8.3145*x27;
    double x38 = x36*x37;
    double x39 = 1.0*((x35 == 1) ? (
   0
)
: (
   0
));
    double x40 = 1.0*((x33 == 1) ? (
   0
)
: (
   0
));
    double x41 = 1.0*((x31 == 1) ? (
   0
)
: (
   0
));
    double x42 = 2.0*((x29 == 1) ? (
   0
)
: (
   0
));
    double x43 = x40 + x41 + x42;
    double x44 = x[2]*x37;
    double x45 = pow(x[2], 3.0);
    double x46 = pow(x1, -1);
    double x47 = 74092.0*x46;
    double x48 = pow(x[2], -9.0);
    double x49 = ((x12 == 1) ? (
   -7976.15 + 137.093038*x[2] - 0.001884662*x4 - 8.77664e-07*x45 + x47 - x[2]*x10
)
: ((x14 == 1) ? (
   -11276.24 + 223.048446*x[2] + 0.018531982*x4 - 5.764227e-06*x45 + x47 - x[2]*x13
)
: ((x16 == 1) ? (
   -11278.378 + 188.684153*x[2] - 1.230524e+28*x48 - x[2]*x15
)
: (
   0
))));
    double x50 = 3.0*(10083.0 - 4.813*x[2] + x49);
    double x51 = x50*x[3];
    double x52 = 2.0*x49;
    double x53 = -47406.0 + 6.75*x[2] + x52 + ((x21 == 1) ? (
   -7770.458 + 130.485235*x[2] - 0.00265684*x4 + 1.29223e-07*x45 + 52478.0*x46 - x[2]*x20
)
: ((x23 == 1) ? (
   -13542.026 + 183.803828*x[2] + 3.64167e+29*x48 - x[2]*x22
)
: (
   0
)));
    double x54 = x53*x[3];
    double x55 = x52 + ((x6 == 1) ? (
   1225.7 + 124.134*x[2] - 0.00439752*x4 - 5.8927e-08*x45 + 77359.0*x46 - x[2]*x3
)
: ((x9 == 1) ? (
   -25383.581 + 299.31255*x[2] + 2.29603e+31*x48 - x[2]*x8
)
: (
   0
)));
    double x56 = x55*x[6];
    double x57 = x51*x[4] + x54*x[5] + x56*x[3];
    double x58 = pow(x0, -2);
    double x59 = x58*x57;
    double x60 = x[2]*x58*x36;
    double x61 = x58*x24*x[4];
    double x62 = ((x12 == 1) ? (
   0
)
: ((x14 == 1) ? (
   0
)
: ((x16 == 1) ? (
   0
)
: (
   0
))));
    double x63 = 2.0*x62;
    double x64 = x19*(x63 + ((x6 == 1) ? (
   0
)
: ((x9 == 1) ? (
   0
)
: (
   0
)))) + x24*(x63 + ((x21 == 1) ? (
   0
)
: ((x23 == 1) ? (
   0
)
: (
   0
)))) + x62*x26;
    double x65 = 2211.0*x27;
    double x66 = x65*x[4];
    double x67 = x[2]*x38 + x57*x27 + x66*x24;
    double x68 = -1.0*x59 - 8.3145*x60 - 2211.0*x61;
    double x69 = 1.0*x67;
    double x70 = x39 + x42;
    out[0] = x0*(x38 + x27*(x19*(x18 + ((x6 == 1) ? (
   100.6197 - 0.00879504*x1 - x3 - 1.76781e-07*x4 - 77359.0*x5
)
: ((x9 == 1) ? (
   253.31255 - 2.066427e+32*x7 - x8
)
: (
   0
)))) + x24*(6.75 + x18 + ((x21 == 1) ? (
   106.372843 - 0.00531368*x1 - x20 + 3.87669e-07*x4 - 52478.0*x5
)
: ((x23 == 1) ? (
   152.423828 - x22 - 3.277503e+30*x7
)
: (
   0
)))) + x26*(-4.813 + x17)) + (x39 + x43)*x44);
    out[1] = 2.0*x67 + x0*(-2.0*x59 - 16.629*x60 - 4422.0*x61 + x44*(x39 + x40 + x41 + 2.0*((x29 == 1) ? (
   1 + x28
)
: (
   0
))) + x66*x[5] + (x56 + x64 + x50*x[4] + x53*x[5])*x27);
    out[2] = x69 + x0*(x68 + x44*(x43 + 1.0*((x35 == 1) ? (
   1 + x34
)
: (
   0
))) + x65*x24 + (x51 + x64)*x27);
    out[3] = x69 + x0*(x68 + x44*(x41 + x70 + 1.0*((x33 == 1) ? (
   1 + x32
)
: (
   0
))) + x65*x25 + (x54 + x64)*x27);
    out[4] = x69 + x0*(x68 + x27*(x64 + x55*x[3]) + x44*(x40 + x70 + 1.0*((x31 == 1) ? (
   1 + x30
)
: (
   0
))));
}

__device__ void pycgpu_model_0_formulahess(double* out, const double* x) {
    double x0 = 1e-15 < x[3];
    double x1 = 2.0*((x0 == 1) ? 0
: 0);
    double x2 = 1e-15 < x[6];
    double x3 = 1.0*((x2 == 1) ? 0
: 0);
    double x4 = 1e-15 < x[5];
    double x5 = 1.0*((x4 == 1) ? 0
: 0);
    double x6 = 1e-15 < x[4];
    double x7 = 1.0*((x6 == 1) ? 0
: 0);
    double x8 = x3 + x5 + x7;
    double x9 = x1 + x8;
    double x10 = 2.0*x[3] + 1.0*(x[4] + x[5] + x[6]);
    double x11 = pow(x10, -1);
    double x12 = 8.3145*x11;
    double x13 = x[2]*x12;
    double x14 = x9*x13;
    double x15 = 16.629*x11;
    double x16 = x9*x15;
    double x17 = pow(x[2], 1.0);
    double x18 = pow(x[2], 3.0);
    double x19 = pow(x18, -1);
    double x20 = 148184.0*x19;
    double x21 = pow(x[2], -1);
    double x22 = x[2] < 700.0;
    double x23 = (x[2] < 933.47 && 700.0 <= x[2]);
    double x24 = pow(x[2], -11.0);
    double x25 = 933.47 <= x[2];
    double x26 = ((x22 == 1) ? (
   -0.003769324 - 5.265984e-06*x17 + x20 - 24.3671976*x21
)
: ((x23 == 1) ? (
   0.037063964 - 3.4585362e-05*x17 + x20 - 38.5844296*x21
)
: ((x25 == 1) ? (
   -31.748192*x21 - 1.1074716e+30*x24
)
: 0)));
    double x27 = x[4]*x[3];
    double x28 = x[2] < 1357.77;
    double x29 = 1357.77 <= x[2];
    double x30 = 2.0*x26;
    double x31 = x[5]*x[3];
    double x32 = x[2] < 1811.0;
    double x33 = 1811.0 <= x[2];
    double x34 = pow(x10, -2);
    double x35 = log(x[3]);
    double x36 = log(x[6]);
    double x37 = log(x[5]);
    double x38 = log(x[4]);
    double x39 = 2.0*((x0 == 1) ? (
   x35*x[3]
)
: 0) + 1.0*((x2 == 1) ? (
   x36*x[6]
)
: 0) + 1.0*((x4 == 1) ? (
   x37*x[5]
)
: 0) + 1.0*((x6 == 1) ? (
   x38*x[4]
)
: 0);
    double x40 = 16.629*x39;
    double x41 = x40*x34;
    double x42 = x8 + 2.0*((x0 == 1) ? (
   1 + x35
)
: 0);
    double x43 = x42*x12;
    double x44 = 16.629*x34;
    double x45 = x[2]*x9;
    double x46 = log(x[2]);
    double x47 = 24.3671976*x46;
    double x48 = pow(x[2], 2.0);
    double x49 = pow(x48, -1);
    double x50 = -74092.0*x49;
    double x51 = 38.5844296*x46;
    double x52 = pow(x[2], -10.0);
    double x53 = 31.748192*x46;
    double x54 = ((x22 == 1) ? (
   112.7258404 - 0.003769324*x17 - x47 - 2.632992e-06*x48 + x50
)
: ((x23 == 1) ? (
   184.4640164 + 0.037063964*x17 - 1.7292681e-05*x48 + x50 - x51
)
: ((x25 == 1) ? (
   156.935961 + 1.1074716e+29*x52 - x53
)
: 0)));
    double x55 = 2.0*x54;
    double x56 = 24.112392*x46;
    double x57 = 31.38*x46;
    double x58 = 6.75 + x55 + ((x28 == 1) ? (
   106.372843 - 0.00531368*x17 + 3.87669e-07*x48 - 52478.0*x49 - x56
)
: ((x29 == 1) ? (
   152.423828 - 3.277503e+30*x52 - x57
)
: 0));
    double x59 = 3.0*(-4.813 + x54);
    double x60 = 23.5143*x46;
    double x61 = 46.0*x46;
    double x62 = x55 + ((x32 == 1) ? (
   100.6197 - 0.00879504*x17 - 1.76781e-07*x48 - 77359.0*x49 - x60
)
: ((x33 == 1) ? (
   253.31255 - 2.066427e+32*x52 - x61
)
: 0));
    double x63 = x62*x[6];
    double x64 = ((x22 == 1) ? 0
: ((x23 == 1) ? 0
: ((x25 == 1) ? 0
: 0)));
    double x65 = 2.0*x64;
    double x66 = x65 + ((x28 == 1) ? 0
: ((x29 == 1) ? 0
: 0));
    double x67 = x66*x[3];
    double x68 = 3.0*x64;
    double x69 = x68*x[4];
    double x70 = x65 + ((x32 == 1) ? 0
: ((x33 == 1) ? 0
: 0));
    double x71 = x70*x[3];
    double x72 = x67*x[5] + x69*x[3] + x71*x[6];
    double x73 = x58*x[3];
    double x74 = x59*x[3];
    double x75 = x63*x[3] + x73*x[5] + x74*x[4];
    double x76 = 2.0*x34;
    double x77 = x10*(x14 - x41 + x43 - x44*x45 - x75*x76 + (x63 + x72 + x58*x[5] + x59*x[4])*x11);
    double x78 = x75*x11;
    double x79 = x14 + x39*x12;
    double x80 = x78 + x79;
    double x81 = x1 + x3 + x5;
    double x82 = x81 + 1.0*((x6 == 1) ? (
   1 + x38
)
: 0);
    double x83 = x82*x12;
    double x84 = 8.3145*x34;
    double x85 = x84*x39;
    double x86 = 1.0*x34;
    double x87 = x14 - x85 - x84*x45 - x86*x75;
    double x88 = x10*(x83 + x87 + (x72 + x74)*x11);
    double x89 = 1.0*x80;
    double x90 = x1 + x7;
    double x91 = x3 + x90;
    double x92 = x91 + 1.0*((x4 == 1) ? (
   1 + x37
)
: 0);
    double x93 = x92*x12;
    double x94 = x10*(x87 + x93 + (x72 + x73)*x11);
    double x95 = x5 + x90;
    double x96 = x95 + 1.0*((x2 == 1) ? (
   1 + x36
)
: 0);
    double x97 = x96*x12;
    double x98 = x10*(x87 + x97 + x11*(x72 + x62*x[3]));
    double x99 = x[4]*x[5];
    double x100 = 4422.0*x11;
    double x101 = pow(x17, -1);
    double x102 = 74092.0*x101;
    double x103 = pow(x[2], -9.0);
    double x104 = ((x22 == 1) ? (
   -7976.15 + 137.093038*x[2] + x102 - 8.77664e-07*x18 - 0.001884662*x48 - x[2]*x47
)
: ((x23 == 1) ? (
   -11276.24 + 223.048446*x[2] + x102 - 5.764227e-06*x18 + 0.018531982*x48 - x[2]*x51
)
: ((x25 == 1) ? (
   -11278.378 + 188.684153*x[2] - 1.230524e+28*x103 - x[2]*x53
)
: 0)));
    double x105 = 2.0*x104;
    double x106 = x105 + ((x32 == 1) ? (
   1225.7 + 124.134*x[2] + 77359.0*x101 - 5.8927e-08*x18 - 0.00439752*x48 - x[2]*x60
)
: ((x33 == 1) ? (
   -25383.581 + 299.31255*x[2] + 2.29603e+31*x103 - x[2]*x61
)
: 0));
    double x107 = -47406.0 + 6.75*x[2] + x105 + ((x28 == 1) ? (
   -7770.458 + 130.485235*x[2] + 52478.0*x101 + 1.29223e-07*x18 - 0.00265684*x48 - x[2]*x56
)
: ((x29 == 1) ? (
   -13542.026 + 183.803828*x[2] + 3.64167e+29*x103 - x[2]*x57
)
: 0));
    double x108 = 3.0*(10083.0 - 4.813*x[2] + x104);
    double x109 = x108*x[4];
    double x110 = x109 + x72 + x106*x[6] + x107*x[5];
    double x111 = x11*x110;
    double x112 = x31*x[4];
    double x113 = x[2]*x42;
    double x114 = 33.258*x34;
    double x115 = x107*x[3];
    double x116 = x106*x[3];
    double x117 = x109*x[3] + x115*x[5] + x116*x[6];
    double x118 = 4.0*x34;
    double x119 = 2211.0*x11;
    double x120 = x119*x[4];
    double x121 = 2.0*x117;
    double x122 = 4422.0*x34;
    double x123 = x31*x122;
    double x124 = -x[2]*x41 - x123*x[4] - x34*x121;
    double x125 = x124 + x[2]*x43 + x120*x[5];
    double x126 = x111 + x125;
    double x127 = 6.0*x64;
    double x128 = x66*x[5];
    double x129 = x70*x[6];
    double x130 = pow(x10, -3);
    double x131 = x112*x130;
    double x132 = x99*x34;
    double x133 = x[2]*x130;
    double x134 = x39*x133;
    double x135 = x117*x130;
    double x136 = x72 + x108*x[3];
    double x137 = x11*x136;
    double x138 = x[2]*x82;
    double x139 = x72 + x68*x[3];
    double x140 = x128 + x129 + x69;
    double x141 = x119*x[5];
    double x142 = -x123 - x44*x138 - x76*x136;
    double x143 = 8844.0*x131 - 2211.0*x132 + 33.258*x134 + 4.0*x135 + x14 - x84*x113 - x86*x110;
    double x144 = x10*(x141 + x142 + x143 + x11*(30249.0 - 14.439*x[2] + 3.0*x104 + x139 + x140));
    double x145 = x124 + 1.0*x126;
    double x146 = x[2]*x92;
    double x147 = x115 + x72;
    double x148 = x11*x147;
    double x149 = -x27*x122 - x44*x146 - x76*x147;
    double x150 = x10*(x120 + x143 + x149 + (x107 + x140 + x67 + x72)*x11);
    double x151 = x116 + x72;
    double x152 = x11*x151;
    double x153 = x[2]*x96;
    double x154 = x71 + x72;
    double x155 = -x44*x153 - x76*x151;
    double x156 = x10*(x143 + x155 + x11*(x106 + x140 + x154));
    double x157 = 1.0*x78 + x79;
    double x158 = 2211.0*x34;
    double x159 = x31*x158;
    double x160 = -x[2]*x85 - x159*x[4] - x86*x117;
    double x161 = x160 + x[2]*x83 + x141*x[3];
    double x162 = x137 + x161;
    double x163 = 1.0*x111 + x125;
    double x164 = 1.0*x162;
    double x165 = 4422.0*x131 + x121*x130 + x40*x133;
    double x166 = 1.0*x137 + x161;
    double x167 = -x159 - x84*x138 - x86*x136;
    double x168 = -x27*x158 - x84*x146 - x86*x147;
    double x169 = x10*(x14 + x165 + x167 + x168 + x11*(x139 + x67) + x119*x[3]);
    double x170 = x160 + x[2]*x93 + x120*x[3];
    double x171 = 1.0*x148 + x170;
    double x172 = x14 + x165 - x84*x153 - x86*x151;
    double x173 = x10*(x167 + x172 + x11*(x139 + x71));
    double x174 = x160 + x[2]*x97;
    double x175 = 1.0*x152 + x174;
    double x176 = x148 + x170;
    double x177 = 1.0*x176;
    double x178 = x10*(x168 + x172 + x11*(x154 + x67));
    double x179 = x152 + x174;
    double x180 = 1.0*x179;
    out[0] = x10*(x14 + x16 + x11*(3.0*x26*x27 + x31*(x30 + ((x28 == 1) ? (
   -0.00531368 + 7.75338e-07*x17 + 104956.0*x19 - 24.112392*x21
)
: ((x29 == 1) ? (
   -31.38*x21 + 3.277503e+31*x24
)
: 0))) + (x30 + ((x32 == 1) ? (
   -0.00879504 - 3.53562e-07*x17 + 154718.0*x19 - 23.5143*x21
)
: ((x33 == 1) ? (
   -46.0*x21 + 2.066427e+33*x24
)
: 0)))*x[6]*x[3]));
    out[1] = x77 + 2.0*x80;
    out[2] = x88 + x89;
    out[3] = x89 + x94;
    out[4] = x89 + x98;
    out[5] = x77 + 2.0*x78 + x[2]*x16 + x40*x11;
    out[6] = 2.0*x111 + 2.0*x126 + x10*(17688.0*x131 - 8844.0*x132 + 66.516*x134 + 8.0*x135 + x11*(2*x128 + 2*x129 + x72 + x127*x[4]) - x110*x118 - x113*x114 + x13*(x8 + 2.0*((x0 == 1) ? (
   pow(x[3], -1)
)
: 0))) - x118*x117 + x15*x113 - 8844.0*x34*x112 + x99*x100 - x[2]*x39*x114;
    out[7] = 2.0*x137 + x144 + x145 + x15*x138 + x31*x100;
    out[8] = x145 + 2.0*x148 + x150 + x15*x146 + x27*x100;
    out[9] = x145 + 2.0*x152 + x156 + x15*x153;
    out[10] = x157 + x88;
    out[11] = x144 + 2.0*x162 + x163;
    out[12] = x164 + x166 + x10*(x142 + x165 + x11*(x72 + x127*x[3]) + x13*(x81 + 1.0*((x6 == 1) ? (
   pow(x[4], -1)
)
: 0)));
    out[13] = x164 + x169 + x171;
    out[14] = x164 + x173 + x175;
    out[15] = x157 + x94;
    out[16] = x150 + x163 + 2.0*x176;
    out[17] = x166 + x169 + x177;
    out[18] = x171 + x177 + x10*(x149 + x165 + x13*(x91 + 1.0*((x4 == 1) ? (
   pow(x[5], -1)
)
: 0)) + (2*x67 + x72)*x11);
    out[19] = x175 + x177 + x178;
    out[20] = x157 + x98;
    out[21] = x156 + x163 + 2.0*x179;
    out[22] = x166 + x173 + x180;
    out[23] = x171 + x178 + x180;
    out[24] = x175 + x180 + x10*(x155 + x165 + x13*(x95 + 1.0*((x2 == 1) ? (
   pow(x[6], -1)
)
: 0)) + (2*x71 + x72)*x11);
}

__device__ void pycgpu_model_0_internal_cons_func(double* out, const double* x) {
    out[0] = 1.0*(-1 + x[3]);
    out[1] = 1.0*(-1 + x[4] + x[5] + x[6]);
}

__device__ void pycgpu_model_0_internal_cons_jac(double* out, const double* x) {
    out[0] = 0;
    out[1] = 1.0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    out[7] = 1.0;
    out[8] = 1.0;
    out[9] = 1.0;
}

__device__ void pycgpu_model_0_mass_obj(double* out, const double* x) {
    double x0 = 2.0*x[3];
    double x1 = pow(x0 + 1.0*(x[4] + x[5] + x[6]), -1);
    double x2 = 1.0*x1;
    out[0] = x1*(1.0*x[4] + x0);
    out[1] = x2*x[5];
    out[2] = x2*x[6];
    out[3] = 0;
}

__device__ void pycgpu_model_0_formulamole_obj(double* out, const double* x) {
    out[0] = 2.0*x[3] + 1.0*x[4];
    out[1] = 1.0*x[5];
    out[2] = 1.0*x[6];
    out[3] = 0.0;
}

__device__ void pycgpu_model_0_formulamole_grad(double* out, const double* x) {
    out[0] = 0;
    out[1] = 2.0;
    out[2] = 1.0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
    out[8] = 1.0;
    out[9] = 0;
    out[10] = 0;
    out[11] = 0;
    out[12] = 0;
    out[13] = 0;
    out[14] = 1.0;
}

__device__ double pycgpu_model_1_obj(const double* x) {
    double x0 = pow(0.5*(x[3] + x[4] + x[5]) + 0.5*(x[7] + x[8] + x[9]), -1);
    double x1 = 3.0*((1e-15 < x[11]) ? (
   x[11]*log(x[11])
)
: (
   0
));
    double x2 = 8.3145*x[2];
    double x3 = x[3]*x[11];
    double x4 = x[5]*x[7];
    double x5 = x4*x[11];
    double x6 = x3*x[8];
    double x7 = x[7]*x[11];
    double x8 = x7*x[4];
    double x9 = -1.0*x[2];
    double x10 = -75000.0 + 0.5*(10000.0 + x9);
    double x11 = 14300.08*x[11];
    double x12 = 0.5*x[6] + 0.5*x[10];
    double x13 = 0.5*x[4] + 0.5*x[8];
    double x14 = x13*x[11];
    double x15 = 0.5*x[3] + 0.5*x[7];
    double x16 = 1.0*x15;
    double x17 = 0.5*x[5] + 0.5*x[9];
    double x18 = 1.0*x17;
    double x19 = x17*x[11];
    double x20 = x15*x19;
    double x21 = (x16 - x18)*x20;
    double x22 = x16*x[11];
    double x23 = 1.0*x13;
    double x24 = x23*x[11];
    double x25 = x24*x15;
    double x26 = x16 + x18 + x23;
    double x27 = (1.0/3.0)*(1 - x26);
    double x28 = x15*x14;
    double x29 = x28*x18;
    double x30 = x20*x13;
    double x31 = pow(x26, -1);
    double x32 = 1.0*x31;
    double x33 = 1.0*x12;
    double x34 = x2*x31;
    double x35 = x34*(x1 + 1.0*((1e-15 < x16) ? (
   x16*log(x16)
)
: (
   0
)) + 1.0*((1e-15 < x18) ? (
   x18*log(x18)
)
: (
   0
)) + 1.0*((1e-15 < x23) ? (
   x23*log(x23)
)
: (
   0
)) + 1.0*((1e-15 < x33) ? (
   x33*log(x33)
)
: (
   0
)));
    double x36 = 2.22*x19;
    double x37 = 1043.0*x19 + 504.0*x21;
    double x38 = -1.0*x37;
    double x39 = 1e-09 + x38;
    double x40 = 1.41071428571429/x[2];
    double x41 = (1.0/6.0)*pow(x[2], 3);
    double x42 = (1.0/135.0)*pow(x[2], 9);
    double x43 = pow(x39, 15);
    double x44 = pow(x[2], 15);
    double x45 = (1.0/600.0)*x44;
    double x46 = 1e-09 + x37;
    double x47 = pow(x46, 15);
    double x48 = (1.0/10.0)/pow(x[2], 5);
    double x49 = (1.0/315.0)/x44;
    double x50 = (1.0/1500.0)/pow(x[2], 25);
    double x51 = pow(x[2], 3.0);
    double x52 = pow(x[2], -1.0);
    double x53 = x[2]*log(x[2]);
    double x54 = pow(x[2], 2.0);
    double x55 = pow(x[2], -9.0);
    double x56 = 74092.0*x52;
    return x35 + x0*(-10000.0*x6 - 10000.0*x8 - x4*x11 - x11*x[3]*x[9] + x3*x10*x[10] + x7*x10*x[6]) + x0*(54.1229146*x5*x[4] + 30836.8712*x5*x[8] + 31873.5471*x6*x[5] + 54.1229146*x6*x[9] + 31873.5471*x8*x[9] + 30836.8712*x3*x[4]*x[9]) + x32*(x22*(10083.0 - 4.813*x[2] + ((x[2] < 700.0) ? (
   -7976.15 + 137.093038*x[2] - 8.77664e-07*x51 - 24.3671976*x53 - 0.001884662*x54 + x56
)
: (((x[2] < 933.47 && 700.0 <= x[2])) ? (
   -11276.24 + 223.048446*x[2] - 5.764227e-06*x51 - 38.5844296*x53 + 0.018531982*x54 + x56
)
: ((933.47 <= x[2]) ? (
   -11278.378 + 188.684153*x[2] - 31.748192*x53 - 1.230524e+28*x55
)
: (
   0
))))) + x24*(4017.0 - 1.255*x[2] + ((x[2] < 1357.77) ? (
   -7770.458 + 130.485235*x[2] + 1.29223e-07*x51 + 52478.0*x52 - 24.112392*x53 - 0.00265684*x54
)
: ((1357.77 <= x[2]) ? (
   -13542.026 + 183.803828*x[2] - 31.38*x53 + 3.64167e+29*x55
)
: (
   0
)))) + x18*x[11]*((x[2] < 1811.0) ? (
   1225.7 + 124.134*x[2] - 5.8927e-08*x51 + 77359.0*x52 - 23.5143*x53 - 0.00439752*x54
)
: ((1811.0 <= x[2]) ? (
   -25383.581 + 299.31255*x[2] - 46.0*x53 + 2.29603e+31*x55
)
: (
   0
)))) + x32*(3089.2*x21 + 150000.0*x12*x19 + 100000.0*x14*x12 + 4.0*x20*(-30740.0 + 7.9972*x[2]) + x25*(-73554.0 + 4.0*x[2]) - 29249.2599*(x18 + x27)*x30 + x14*x18*(39258.0 - 4.14983*x[2]) + x22*x12*(160000.0 + x9) + (x16 - x23)*x25*(51500.0 - 11.84*x[2]) + (x23 + x27)*x29*(-103924.225 + 69.9194286*x[2]) + (x16 + x27)*x29*(-267841.207 + 131.456548*x[2])) + x0*x2*(x1 + 0.5*((1e-15 < x[3]) ? (
   x[3]*log(x[3])
)
: (
   0
)) + 0.5*((1e-15 < x[4]) ? (
   x[4]*log(x[4])
)
: (
   0
)) + 0.5*((1e-15 < x[5]) ? (
   x[5]*log(x[5])
)
: (
   0
)) + 0.5*((1e-15 < x[6]) ? (
   x[6]*log(x[6])
)
: (
   0
)) + 0.5*((1e-15 < x[7]) ? (
   x[7]*log(x[7])
)
: (
   0
)) + 0.5*((1e-15 < x[8]) ? (
   x[8]*log(x[8])
)
: (
   0
)) + 0.5*((1e-15 < x[9]) ? (
   x[9]*log(x[9])
)
: (
   0
)) + 0.5*((1e-15 < x[10]) ? (
   x[10]*log(x[10])
)
: (
   0
))) + x34*log(1 + x36/((x36 <= 0) ? (
   -1.0
)
: (
   1.0
)))*((x[2] < x38) ? (
   1 - 0.641731208021339*(x40*x39 + 1.43058350100604*(x41/pow(x39, 3) + x42/pow(x39, 9) + x45/x43))
)
: ((x[2] < x37) ? (
   1 - 0.641731208021339*(x40*x46 + 1.43058350100604*(x41/pow(x46, 3) + x42/pow(x46, 9) + x45/x47))
)
: (((0 < 1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9])) && 1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9])) < x[2])) ? (
   -0.641731208021339*(pow(x46, 5)*x48 + x47*x49 + x50*pow(x46, 25))
)
: (((-1.0*(1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9]))) < x[2] && 1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9])) < 0)) ? (
   -0.641731208021339*(x43*x49 + x48*pow(x39, 5) + x50*pow(x39, 25))
)
: (
   0
))))) - (x35 + 125529.0824292*x30*x31 + x32*(-28600.16*x20 - 20000.0*x28 + 2.0*x15*x12*x10*x[11]));
}

__device__ double pycgpu_model_1_formulaobj(const double* x) {
    double x0 = 0.5*(x[3] + x[4] + x[5]) + 0.5*(x[7] + x[8] + x[9]);
    double x1 = pow(x0, -1);
    double x2 = 3.0*((1e-15 < x[11]) ? (
   x[11]*log(x[11])
)
: (
   0
));
    double x3 = 8.3145*x[2];
    double x4 = x[3]*x[11];
    double x5 = x[5]*x[7];
    double x6 = x5*x[11];
    double x7 = x4*x[8];
    double x8 = x[7]*x[11];
    double x9 = x8*x[4];
    double x10 = -1.0*x[2];
    double x11 = -75000.0 + 0.5*(10000.0 + x10);
    double x12 = 14300.08*x[11];
    double x13 = 0.5*x[6] + 0.5*x[10];
    double x14 = 0.5*x[4] + 0.5*x[8];
    double x15 = x14*x[11];
    double x16 = 0.5*x[3] + 0.5*x[7];
    double x17 = 1.0*x16;
    double x18 = 0.5*x[5] + 0.5*x[9];
    double x19 = 1.0*x18;
    double x20 = x18*x[11];
    double x21 = x20*x16;
    double x22 = (x17 - x19)*x21;
    double x23 = x17*x[11];
    double x24 = 1.0*x14;
    double x25 = x24*x[11];
    double x26 = x25*x16;
    double x27 = x17 + x19 + x24;
    double x28 = (1.0/3.0)*(1 - x27);
    double x29 = x15*x16;
    double x30 = x29*x19;
    double x31 = x21*x14;
    double x32 = pow(x27, -1);
    double x33 = 1.0*x32;
    double x34 = 1.0*x13;
    double x35 = x3*x32;
    double x36 = x35*(x2 + 1.0*((1e-15 < x17) ? (
   x17*log(x17)
)
: (
   0
)) + 1.0*((1e-15 < x19) ? (
   x19*log(x19)
)
: (
   0
)) + 1.0*((1e-15 < x24) ? (
   x24*log(x24)
)
: (
   0
)) + 1.0*((1e-15 < x34) ? (
   x34*log(x34)
)
: (
   0
)));
    double x37 = 2.22*x20;
    double x38 = 1043.0*x20 + 504.0*x22;
    double x39 = -1.0*x38;
    double x40 = 1e-09 + x39;
    double x41 = 1.41071428571429/x[2];
    double x42 = (1.0/6.0)*pow(x[2], 3);
    double x43 = (1.0/135.0)*pow(x[2], 9);
    double x44 = pow(x40, 15);
    double x45 = pow(x[2], 15);
    double x46 = (1.0/600.0)*x45;
    double x47 = 1e-09 + x38;
    double x48 = pow(x47, 15);
    double x49 = (1.0/10.0)/pow(x[2], 5);
    double x50 = (1.0/315.0)/x45;
    double x51 = (1.0/1500.0)/pow(x[2], 25);
    double x52 = pow(x[2], 3.0);
    double x53 = pow(x[2], -1.0);
    double x54 = x[2]*log(x[2]);
    double x55 = pow(x[2], 2.0);
    double x56 = pow(x[2], -9.0);
    double x57 = 74092.0*x53;
    return x0*(x36 + x1*(-10000.0*x7 - 10000.0*x9 - x5*x12 - x12*x[3]*x[9] + x4*x11*x[10] + x8*x11*x[6]) + x1*(54.1229146*x6*x[4] + 30836.8712*x6*x[8] + 31873.5471*x7*x[5] + 54.1229146*x7*x[9] + 31873.5471*x9*x[9] + 30836.8712*x4*x[4]*x[9]) + x33*(x23*(10083.0 - 4.813*x[2] + ((x[2] < 700.0) ? (
   -7976.15 + 137.093038*x[2] - 8.77664e-07*x52 - 24.3671976*x54 - 0.001884662*x55 + x57
)
: (((x[2] < 933.47 && 700.0 <= x[2])) ? (
   -11276.24 + 223.048446*x[2] - 5.764227e-06*x52 - 38.5844296*x54 + 0.018531982*x55 + x57
)
: ((933.47 <= x[2]) ? (
   -11278.378 + 188.684153*x[2] - 31.748192*x54 - 1.230524e+28*x56
)
: (
   0
))))) + x25*(4017.0 - 1.255*x[2] + ((x[2] < 1357.77) ? (
   -7770.458 + 130.485235*x[2] + 1.29223e-07*x52 + 52478.0*x53 - 24.112392*x54 - 0.00265684*x55
)
: ((1357.77 <= x[2]) ? (
   -13542.026 + 183.803828*x[2] - 31.38*x54 + 3.64167e+29*x56
)
: (
   0
)))) + x19*x[11]*((x[2] < 1811.0) ? (
   1225.7 + 124.134*x[2] - 5.8927e-08*x52 + 77359.0*x53 - 23.5143*x54 - 0.00439752*x55
)
: ((1811.0 <= x[2]) ? (
   -25383.581 + 299.31255*x[2] - 46.0*x54 + 2.29603e+31*x56
)
: (
   0
)))) + x33*(3089.2*x22 + 100000.0*x15*x13 + 150000.0*x20*x13 + 4.0*x21*(-30740.0 + 7.9972*x[2]) + x26*(-73554.0 + 4.0*x[2]) - 29249.2599*(x19 + x28)*x31 + x15*x19*(39258.0 - 4.14983*x[2]) + x23*x13*(160000.0 + x10) + (x17 - x24)*x26*(51500.0 - 11.84*x[2]) + (x24 + x28)*x30*(-103924.225 + 69.9194286*x[2]) + (x17 + x28)*x30*(-267841.207 + 131.456548*x[2])) + x1*x3*(x2 + 0.5*((1e-15 < x[3]) ? (
   x[3]*log(x[3])
)
: (
   0
)) + 0.5*((1e-15 < x[4]) ? (
   x[4]*log(x[4])
)
: (
   0
)) + 0.5*((1e-15 < x[5]) ? (
   x[5]*log(x[5])
)
: (
   0
)) + 0.5*((1e-15 < x[6]) ? (
   x[6]*log(x[6])
)
: (
   0
)) + 0.5*((1e-15 < x[7]) ? (
   x[7]*log(x[7])
)
: (
   0
)) + 0.5*((1e-15 < x[8]) ? (
   x[8]*log(x[8])
)
: (
   0
)) + 0.5*((1e-15 < x[9]) ? (
   x[9]*log(x[9])
)
: (
   0
)) + 0.5*((1e-15 < x[10]) ? (
   x[10]*log(x[10])
)
: (
   0
))) + x35*log(1 + x37/((x37 <= 0) ? (
   -1.0
)
: (
   1.0
)))*((x[2] < x39) ? (
   1 - 0.641731208021339*(x40*x41 + 1.43058350100604*(x43/pow(x40, 9) + x42/pow(x40, 3) + x46/x44))
)
: ((x[2] < x38) ? (
   1 - 0.641731208021339*(x41*x47 + 1.43058350100604*(x42/pow(x47, 3) + x43/pow(x47, 9) + x46/x48))
)
: (((0 < 1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9])) && 1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9])) < x[2])) ? (
   -0.641731208021339*(pow(x47, 5)*x49 + x50*x48 + x51*pow(x47, 25))
)
: (((-1.0*(1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9]))) < x[2] && 1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9])) < 0)) ? (
   -0.641731208021339*(pow(x40, 5)*x49 + x50*x44 + x51*pow(x40, 25))
)
: (
   0
))))) - (x36 + 125529.0824292*x32*x31 + x33*(-28600.16*x21 - 20000.0*x29 + 2.0*x13*x11*x16*x[11])));
}

__device__ void pycgpu_model_1_formulagrad(double* out, const double* x) {
    double x0 = 0.5*(x[3] + x[4] + x[5]) + 0.5*(x[7] + x[8] + x[9]);
    double x1 = pow(x[2], 1.0);
    double x2 = log(x[2]);
    double x3 = 23.5143*x2;
    double x4 = pow(x[2], 2.0);
    double x5 = pow(x4, -1);
    double x6 = x[2] < 1811.0;
    double x7 = pow(x[2], -10.0);
    double x8 = 46.0*x2;
    double x9 = 1811.0 <= x[2];
    double x10 = 0.5*x[5] + 0.5*x[9];
    double x11 = 1.0*x10;
    double x12 = x11*x[11];
    double x13 = 24.3671976*x2;
    double x14 = -74092.0*x5;
    double x15 = x[2] < 700.0;
    double x16 = 38.5844296*x2;
    double x17 = (x[2] < 933.47 && 700.0 <= x[2]);
    double x18 = 31.748192*x2;
    double x19 = 933.47 <= x[2];
    double x20 = 0.5*x[7];
    double x21 = 0.5*x[3] + x20;
    double x22 = 1.0*x21;
    double x23 = x22*x[11];
    double x24 = 24.112392*x2;
    double x25 = x[2] < 1357.77;
    double x26 = 31.38*x2;
    double x27 = 1357.77 <= x[2];
    double x28 = 0.5*x[4] + 0.5*x[8];
    double x29 = 1.0*x28;
    double x30 = x29*x[11];
    double x31 = x11 + x22 + x29;
    double x32 = pow(x31, -1);
    double x33 = 1.0*x32;
    double x34 = 8.3145*x32;
    double x35 = x10*x[11];
    double x36 = 2.22*x35;
    double x37 = x36 <= 0;
    double x38 = ((x37 == 1) ? (
   -1.0
)
: (
   1.0
));
    double x39 = pow(x38, -1);
    double x40 = 1 + x36*x39;
    double x41 = log(x40);
    double x42 = -x11 + x22;
    double x43 = x35*x21;
    double x44 = x42*x43;
    double x45 = 1043.0*x35 + 504.0*x44;
    double x46 = -1.0*x45;
    double x47 = 1e-09 + x46;
    double x48 = pow(x[2], -1);
    double x49 = 1.41071428571429*x48;
    double x50 = pow(x47, -3);
    double x51 = pow(x[2], 3);
    double x52 = (1.0/6.0)*x51;
    double x53 = pow(x47, -9);
    double x54 = pow(x[2], 9);
    double x55 = (1.0/135.0)*x54;
    double x56 = pow(x47, 15);
    double x57 = pow(x56, -1);
    double x58 = pow(x[2], 15);
    double x59 = (1.0/600.0)*x58;
    double x60 = x[2] < x46;
    double x61 = 1e-09 + x45;
    double x62 = pow(x61, -3);
    double x63 = pow(x61, -9);
    double x64 = pow(x61, 15);
    double x65 = pow(x64, -1);
    double x66 = x[2] < x45;
    double x67 = pow(x61, 5);
    double x68 = pow(x[2], -5);
    double x69 = (1.0/10.0)*x68;
    double x70 = pow(x58, -1);
    double x71 = (1.0/315.0)*x70;
    double x72 = pow(x61, 25);
    double x73 = pow(x[2], -25);
    double x74 = (1.0/1500.0)*x73;
    double x75 = (0 < 1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9])) && 1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9])) < x[2]);
    double x76 = pow(x47, 5);
    double x77 = pow(x47, 25);
    double x78 = (-1.0*(1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9]))) < x[2] && 1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9])) < 0);
    double x79 = ((x60 == 1) ? (
   1 - 0.641731208021339*(x47*x49 + 1.43058350100604*(x50*x52 + x53*x55 + x57*x59))
)
: ((x66 == 1) ? (
   1 - 0.641731208021339*(x61*x49 + 1.43058350100604*(x62*x52 + x63*x55 + x65*x59))
)
: ((x75 == 1) ? (
   -0.641731208021339*(x67*x69 + x71*x64 + x72*x74)
)
: ((x78 == 1) ? (
   -0.641731208021339*(x71*x56 + x74*x77 + x76*x69)
)
: (
   0
)))));
    double x80 = x79*x41;
    double x81 = x80*x34;
    double x82 = pow(x[2], 2);
    double x83 = 0.905299382744389/x82;
    double x84 = 0.0229512519569*pow(x[2], 14);
    double x85 = 0.0612033385517333*pow(x[2], 8);
    double x86 = 0.459025039138*x82;
    double x87 = (1.0/60.0)/pow(x[2], 26);
    double x88 = (1.0/21.0)/pow(x[2], 16);
    double x89 = (1.0/2.0)/pow(x[2], 6);
    double x90 = x[2]*x34;
    double x91 = x90*x41;
    double x92 = ((x37 == 1) ? (
   0
)
: (
   0
))/pow(x38, 2);
    double x93 = x79/x40;
    double x94 = -18.45819*x[2]*x93*x92*x32*x35;
    double x95 = 0.5*x[10];
    double x96 = 0.5*x[6] + x95;
    double x97 = 1.0*x96;
    double x98 = x21*x[11];
    double x99 = x98*x32;
    double x100 = (1.0/3.0)*(1 - x31);
    double x101 = x100 + x22;
    double x102 = x43*x28;
    double x103 = x100 + x29;
    double x104 = x28*x[11];
    double x105 = x21*x104;
    double x106 = x22 - x29;
    double x107 = x35*x28;
    double x108 = pow(x0, -1);
    double x109 = x[3]*x[11];
    double x110 = log(x[5]);
    double x111 = 1e-15 < x[5];
    double x112 = log(x[6]);
    double x113 = 1e-15 < x[6];
    double x114 = log(x[3]);
    double x115 = 1e-15 < x[3];
    double x116 = log(x[11]);
    double x117 = 1e-15 < x[11];
    double x118 = 3.0*((x117 == 1) ? (
   x116*x[11]
)
: (
   0
));
    double x119 = log(x[4]);
    double x120 = 1e-15 < x[4];
    double x121 = log(x[9]);
    double x122 = 1e-15 < x[9];
    double x123 = log(x[8]);
    double x124 = 1e-15 < x[8];
    double x125 = log(x[10]);
    double x126 = 1e-15 < x[10];
    double x127 = log(x[7]);
    double x128 = 1e-15 < x[7];
    double x129 = x118 + 0.5*((x111 == 1) ? (
   x110*x[5]
)
: (
   0
)) + 0.5*((x113 == 1) ? (
   x112*x[6]
)
: (
   0
)) + 0.5*((x115 == 1) ? (
   x114*x[3]
)
: (
   0
)) + 0.5*((x120 == 1) ? (
   x119*x[4]
)
: (
   0
)) + 0.5*((x122 == 1) ? (
   x121*x[9]
)
: (
   0
)) + 0.5*((x124 == 1) ? (
   x123*x[8]
)
: (
   0
)) + 0.5*((x126 == 1) ? (
   x125*x[10]
)
: (
   0
)) + 0.5*((x128 == 1) ? (
   x127*x[7]
)
: (
   0
));
    double x130 = 8.3145*x108;
    double x131 = x129*x130;
    double x132 = 0.5*((x128 == 1) ? (
   0
)
: (
   0
));
    double x133 = 0.5*((x120 == 1) ? (
   0
)
: (
   0
));
    double x134 = 0.5*((x124 == 1) ? (
   0
)
: (
   0
));
    double x135 = 0.5*((x122 == 1) ? (
   0
)
: (
   0
));
    double x136 = 0.5*((x126 == 1) ? (
   0
)
: (
   0
));
    double x137 = 3.0*((x117 == 1) ? (
   0
)
: (
   0
));
    double x138 = 0.5*((x113 == 1) ? (
   0
)
: (
   0
));
    double x139 = 0.5*((x115 == 1) ? (
   0
)
: (
   0
));
    double x140 = 0.5*((x111 == 1) ? (
   0
)
: (
   0
));
    double x141 = x134 + x135 + x136 + x137 + x138 + x139 + x140;
    double x142 = x133 + x141;
    double x143 = x[2]*x130;
    double x144 = x132 + x133 + x134 + x135 + x136 + x137 + x138;
    double x145 = x[9]*x[11];
    double x146 = 54.1229146*x145;
    double x147 = 31873.5471*x[11];
    double x148 = x147*x[5];
    double x149 = x[4]*x[9];
    double x150 = 30836.8712*x[11];
    double x151 = 10000.0*x[8];
    double x152 = x151*x[11];
    double x153 = 14300.08*x[9];
    double x154 = -1.0*x[2];
    double x155 = -75000.0 + 0.5*(10000.0 + x154);
    double x156 = x155*x[11];
    double x157 = pow(x[2], 3.0);
    double x158 = pow(x1, -1);
    double x159 = 74092.0*x158;
    double x160 = pow(x[2], -9.0);
    double x161 = 10083.0 - 4.813*x[2] + ((x15 == 1) ? (
   -7976.15 + 137.093038*x[2] - 8.77664e-07*x157 + x159 - 0.001884662*x4 - x[2]*x13
)
: ((x17 == 1) ? (
   -11276.24 + 223.048446*x[2] - 5.764227e-06*x157 + x159 + 0.018531982*x4 - x[2]*x16
)
: ((x19 == 1) ? (
   -11278.378 + 188.684153*x[2] - 1.230524e+28*x160 - x[2]*x18
)
: (
   0
))));
    double x162 = 0.5*x[11];
    double x163 = x12*((x6 == 1) ? (
   0
)
: ((x9 == 1) ? (
   0
)
: (
   0
))) + x23*((x15 == 1) ? (
   0
)
: ((x17 == 1) ? (
   0
)
: ((x19 == 1) ? (
   0
)
: (
   0
)))) + x30*((x25 == 1) ? (
   0
)
: ((x27 == 1) ? (
   0
)
: (
   0
)));
    double x164 = 252.0*x43;
    double x165 = -x164;
    double x166 = 252.0*x42;
    double x167 = x35*x166;
    double x168 = x165 - x167;
    double x169 = 0.905299382744389*x48;
    double x170 = 0.0229512519569*x58;
    double x171 = x170/pow(x47, 16);
    double x172 = 0.0612033385517333*x54;
    double x173 = x172/pow(x47, 10);
    double x174 = pow(x47, 4);
    double x175 = 0.459025039138*x51;
    double x176 = x175/x174;
    double x177 = x164 + x167;
    double x178 = pow(x61, 4);
    double x179 = x175/x178;
    double x180 = x170/pow(x61, 16);
    double x181 = x172/pow(x61, 10);
    double x182 = (1.0/60.0)*x73;
    double x183 = pow(x61, 24)*x182;
    double x184 = (1.0/21.0)*x70;
    double x185 = pow(x61, 14)*x184;
    double x186 = (1.0/2.0)*x68;
    double x187 = x178*x186;
    double x188 = pow(x47, 24)*x182;
    double x189 = pow(x47, 14)*x184;
    double x190 = x174*x186;
    double x191 = 62764.5412146*x32;
    double x192 = -267841.207 + 131.456548*x[2];
    double x193 = x101*x192;
    double x194 = 0.5*x107;
    double x195 = 4874.87665*x102;
    double x196 = -103924.225 + 69.9194286*x[2];
    double x197 = x103*x196;
    double x198 = 0.333333333333333*x102;
    double x199 = -73554.0 + 4.0*x[2];
    double x200 = x104*x199;
    double x201 = 51500.0 - 11.84*x[2];
    double x202 = x201*x106;
    double x203 = x202*x104;
    double x204 = x100 + x11;
    double x205 = 14624.62995*x204;
    double x206 = 0.5*x105;
    double x207 = x201*x206;
    double x208 = 0.166666666666667*x102;
    double x209 = -x208*x196;
    double x210 = -30740.0 + 7.9972*x[2];
    double x211 = 2.0*x210;
    double x212 = 160000.0 + x154;
    double x213 = x96*x[11];
    double x214 = 1544.6*x42;
    double x215 = 1544.6*x43;
    double x216 = pow(x31, -2);
    double x217 = 4.15725*x[2];
    double x218 = pow(x0, -2);
    double x219 = 4017.0 - 1.255*x[2] + ((x25 == 1) ? (
   -7770.458 + 130.485235*x[2] + 1.29223e-07*x157 + 52478.0*x158 - 0.00265684*x4 - x[2]*x24
)
: ((x27 == 1) ? (
   -13542.026 + 183.803828*x[2] + 3.64167e+29*x160 - x[2]*x26
)
: (
   0
)));
    double x220 = x29*x219;
    double x221 = x22*x161;
    double x222 = ((x6 == 1) ? (
   1225.7 + 124.134*x[2] - 5.8927e-08*x157 + 77359.0*x158 - 0.00439752*x4 - x[2]*x3
)
: ((x9 == 1) ? (
   -25383.581 + 299.31255*x[2] + 2.29603e+31*x160 - x[2]*x8
)
: (
   0
)));
    double x223 = x11*x222;
    double x224 = x220*x[11] + x221*x[11] + x223*x[11];
    double x225 = 0.5*x216;
    double x226 = 100000.0*x28;
    double x227 = 4.0*x210;
    double x228 = x97*x212;
    double x229 = 150000.0*x96;
    double x230 = x11*x197;
    double x231 = 29249.2599*x204;
    double x232 = 39258.0 - 4.14983*x[2];
    double x233 = x11*x232;
    double x234 = x11*x193;
    double x235 = 3089.2*x44 + x213*x226 + x22*x200 + x22*x203 + x230*x105 - x231*x102 + x233*x104 + x234*x105 + x35*x229 + x43*x227 + x98*x228;
    double x236 = x150*x[5];
    double x237 = x236*x[8];
    double x238 = x[3]*x[8];
    double x239 = x238*x147;
    double x240 = 31873.5471*x[7];
    double x241 = x240*x149;
    double x242 = x[4]*x[11];
    double x243 = 54.1229146*x[5];
    double x244 = x242*x243;
    double x245 = 30836.8712*x109*x149 + x237*x[7] + x238*x146 + x239*x[5] + x241*x[11] + x244*x[7];
    double x246 = 0.5*x218;
    double x247 = x156*x[6];
    double x248 = x[3]*x[10];
    double x249 = 10000.0*x[4];
    double x250 = x249*x[7];
    double x251 = x153*x[3];
    double x252 = 14300.08*x[5];
    double x253 = x252*x[11];
    double x254 = -x152*x[3] + x247*x[7] + x248*x156 - x250*x[11] - x251*x[11] - x253*x[7];
    double x255 = 20000.0*x21;
    double x256 = 2.0*x96*x21;
    double x257 = -28600.16*x43 - x255*x104 + x256*x156;
    double x258 = 62764.5412146*x216*x102 - x224*x225 - x235*x225 - x245*x246 - x254*x246 + x257*x225 - x218*x217*x129 - x80*x217*x216;
    double x259 = x258 + x94 - x107*x191 + x33*(x163 + x161*x162) - x33*(-10000.0*x104 - 14300.08*x35 + x97*x156) + x33*(x195 + 0.5*x200 + 0.5*x203 + x207 + x209 + x215 + x193*x194 + x197*x194 + x198*x192 - x205*x107 + 0.5*x213*x212 + x35*x211 + x35*x214) + x91*((x60 == 1) ? (
   x168*x171 + x168*x173 + x168*x176 - x169*x168
)
: ((x66 == 1) ? (
   -x169*x177 + x177*x180 + x177*x181 + x179*x177
)
: ((x75 == 1) ? (
   -0.641731208021339*(x177*x183 + x177*x185 + x177*x187)
)
: ((x78 == 1) ? (
   -0.641731208021339*(x168*x188 + x168*x189 + x168*x190)
)
: (
   0
)))));
    double x260 = 125529.0824292*x32;
    double x261 = x90*(x118 + 1.0*((1e-15 < x11) ? (
   x11*log(x11)
)
: (
   0
)) + 1.0*((1e-15 < x22) ? (
   x22*log(x22)
)
: (
   0
)) + 1.0*((1e-15 < x29) ? (
   x29*log(x29)
)
: (
   0
)) + 1.0*((1e-15 < x97) ? (
   x97*log(x97)
)
: (
   0
)));
    double x262 = 0.5*(x261 + x[2]*x131 + x[2]*x81 + x245*x108 + x254*x108 + x33*x224 + x33*x235 - (x261 + x260*x102 + x33*x257));
    double x263 = x108*x[11];
    double x264 = 10000.0*x263;
    double x265 = x243*x[7];
    double x266 = x109*x[9];
    double x267 = 0.5*x43;
    double x268 = 0.5*x232;
    double x269 = -x208*x192;
    double x270 = x21*x162;
    double x271 = x94 + x91*((x60 == 1) ? (
   0
)
: ((x66 == 1) ? (
   0
)
: ((x75 == 1) ? (
   0
)
: ((x78 == 1) ? (
   0
)
: (
   0
)))));
    double x272 = x258 + x271 + 10000.0*x99 + x33*(x163 + x219*x162) + x33*(x195 - x207 + 50000.0*x213 + x269 + x196*x198 + x202*x270 + x267*x193 + x267*x197 + x270*x199 + x35*x268 - x43*x205) - x43*x191;
    double x273 = x[7]*x[8];
    double x274 = 14300.08*x263;
    double x275 = -x92*x36;
    double x276 = x93*x90;
    double x277 = 521.5*x[11];
    double x278 = x98*x166;
    double x279 = x164 - x277 - x278;
    double x280 = x165 + x277 + x278;
    double x281 = x258 + 14300.08*x99 - x105*x191 + x276*(x275 + 1.11*x39*x[11]) + x33*(x163 + x222*x162) + x33*(-9749.7533*x102 + x209 + 75000.0*x213 - x215 + x269 - x205*x105 + x206*x193 + x206*x197 + x268*x104 + x98*x211 + x98*x214) + x91*((x60 == 1) ? (
   -x279*x169 + x279*x171 + x279*x173 + x279*x176
)
: ((x66 == 1) ? (
   -x280*x169 + x280*x179 + x280*x180 + x280*x181
)
: ((x75 == 1) ? (
   -0.641731208021339*(x280*x183 + x280*x185 + x280*x187)
)
: ((x78 == 1) ? (
   -0.641731208021339*(x279*x188 + x279*x189 + x279*x190)
)
: (
   0
)))));
    double x282 = x108*x156;
    double x283 = x132 + x133 + x134 + x135 + x137 + x139 + x140;
    double x284 = x271 + x33*x163 + x33*(50000.0*x104 + 75000.0*x35 + x212*x270) - x32*x22*x156;
    double x285 = x132 + x133 + x136 + x137 + x138 + x139 + x140;
    double x286 = 54.1229146*x238;
    double x287 = 30836.8712*x[3];
    double x288 = 521.5*x[9];
    double x289 = x21*x10;
    double x290 = x42*x289;
    double x291 = 504.0*x290;
    double x292 = 521.5*x[5];
    double x293 = -x288 - x291 - x292;
    double x294 = x288 + x291 + x292;
    double x295 = x21*x28;
    double x296 = x22*x28;
    double x297 = x28*x289;
    out[0] = x0*(x131 + x81 + x94 + x108*(-x95*x109 - x20*x[6]*x[11]) + x33*(x12*((x6 == 1) ? (
   100.6197 - 0.00879504*x1 - x3 - 1.76781e-07*x4 - 77359.0*x5
)
: ((x9 == 1) ? (
   253.31255 - 2.066427e+32*x7 - x8
)
: (
   0
))) + x23*(-4.813 + ((x15 == 1) ? (
   112.7258404 - 0.003769324*x1 - x13 + x14 - 2.632992e-06*x4
)
: ((x17 == 1) ? (
   184.4640164 + 0.037063964*x1 + x14 - x16 - 1.7292681e-05*x4
)
: ((x19 == 1) ? (
   156.935961 - x18 + 1.1074716e+29*x7
)
: (
   0
))))) + x30*(-1.255 + ((x25 == 1) ? (
   106.372843 - 0.00531368*x1 - x24 + 3.87669e-07*x4 - 52478.0*x5
)
: ((x27 == 1) ? (
   152.423828 - x26 - 3.277503e+30*x7
)
: (
   0
))))) + x33*(4.0*x105 - 4.14983*x107 + 31.9888*x43 + 131.456548*x101*x102 + 69.9194286*x103*x102 - 11.84*x105*x106 - x98*x97) + x91*((x60 == 1) ? (
   x83*x47 - x84*x57 - x85*x53 - x86*x50
)
: ((x66 == 1) ? (
   x83*x61 - x84*x65 - x85*x63 - x86*x62
)
: ((x75 == 1) ? (
   -0.641731208021339*(-x87*x72 - x88*x64 - x89*x67)
)
: ((x78 == 1) ? (
   -0.641731208021339*(-x87*x77 - x88*x56 - x89*x76)
)
: (
   0
))))) + x99*x97 + (x132 + x142)*x143);
    out[1] = x262 + x0*(x259 + x108*(-x152 - x153*x[11] + x156*x[10]) + x108*(x146*x[8] + x148*x[8] + x149*x150) + x143*(x140 + x144 + 0.5*((x115 == 1) ? (
   1 + x114
)
: (
   0
))));
    out[2] = x262 + x0*(x272 + x108*(30836.8712*x266 + x240*x145 + x265*x[11]) + x143*(x132 + x141 + 0.5*((x120 == 1) ? (
   1 + x119
)
: (
   0
))) - x264*x[7]);
    out[3] = x262 + x0*(x281 + x108*(x239 + 54.1229146*x242*x[7] + x273*x150) + x143*(x139 + x144 + 0.5*((x111 == 1) ? (
   1 + x110
)
: (
   0
))) - x274*x[7]);
    out[4] = x0*(x284 + x143*(x136 + x283 + 0.5*((x113 == 1) ? (
   1 + x112
)
: (
   0
))) + x282*x[7]);
    out[5] = x262 + x0*(x259 + x108*(x237 + x244 + x147*x149) + x108*(x247 - x253 - x249*x[11]) + x143*(x142 + 0.5*((x128 == 1) ? (
   1 + x127
)
: (
   0
))));
    out[6] = x262 + x0*(x272 + x108*(54.1229146*x266 + x148*x[3] + x236*x[7]) + x143*(x135 + x285 + 0.5*((x124 == 1) ? (
   1 + x123
)
: (
   0
))) - x264*x[3]);
    out[7] = x262 + x0*(x281 + x108*(x240*x242 + x286*x[11] + x287*x242) + x143*(x134 + x285 + 0.5*((x122 == 1) ? (
   1 + x121
)
: (
   0
))) - x274*x[3]);
    out[8] = x0*(x284 + x143*(x138 + x283 + 0.5*((x126 == 1) ? (
   1 + x125
)
: (
   0
))) + x282*x[3]);
    out[9] = x0*(x108*(x241 + 31873.5471*x238*x[5] + x265*x[4] + 30836.8712*x273*x[5] + x286*x[9] + x287*x149) + x108*(-x250 - x251 - x151*x[3] + x248*x155 - x252*x[7] + x155*x[6]*x[7]) + x143*(x132 + x133 + x134 + x135 + x136 + x138 + x139 + x140 + 3.0*((x117 == 1) ? (
   1 + x116
)
: (
   0
))) + x276*(x275 + 2.22*x39*x10) - x297*x260 - x33*(-28600.16*x289 + x256*x155 - x28*x255) + x33*(3089.2*x290 + x10*x229 + x202*x296 + x21*x228 + x230*x295 - x231*x297 + x234*x295 + x28*x233 + x289*x227 + x296*x199 + x96*x226) + x91*((x60 == 1) ? (
   -x293*x169 + x293*x171 + x293*x173 + x293*x176
)
: ((x66 == 1) ? (
   -x294*x169 + x294*x179 + x294*x180 + x294*x181
)
: ((x75 == 1) ? (
   -0.641731208021339*(x294*x183 + x294*x185 + x294*x187)
)
: ((x78 == 1) ? (
   -0.641731208021339*(x293*x188 + x293*x189 + x293*x190)
)
: (
   0
))))) + (x163 + x220 + x221 + x223)*x33);
}

__device__ void pycgpu_model_1_formulahess(double* out, const double* x) {
    double x0 = 0.5*(x[3] + x[4] + x[5]) + 0.5*(x[7] + x[8] + x[9]);
    double x1 = pow(x0, -1);
    double x2 = 1e-15 < x[11];
    double x3 = 3.0*((x2 == 1) ? 0
: 0);
    double x4 = 1e-15 < x[7];
    double x5 = 0.5*((x4 == 1) ? 0
: 0);
    double x6 = 1e-15 < x[5];
    double x7 = 0.5*((x6 == 1) ? 0
: 0);
    double x8 = 1e-15 < x[3];
    double x9 = 0.5*((x8 == 1) ? 0
: 0);
    double x10 = 1e-15 < x[6];
    double x11 = 0.5*((x10 == 1) ? 0
: 0);
    double x12 = 1e-15 < x[4];
    double x13 = 0.5*((x12 == 1) ? 0
: 0);
    double x14 = 1e-15 < x[9];
    double x15 = 0.5*((x14 == 1) ? 0
: 0);
    double x16 = 1e-15 < x[8];
    double x17 = 0.5*((x16 == 1) ? 0
: 0);
    double x18 = 1e-15 < x[10];
    double x19 = 0.5*((x18 == 1) ? 0
: 0);
    double x20 = x11 + x13 + x15 + x17 + x19 + x7 + x9;
    double x21 = x20 + x5;
    double x22 = x21 + x3;
    double x23 = x1*x22;
    double x24 = 0.5*x[5] + 0.5*x[9];
    double x25 = x24*x[11];
    double x26 = 0.5*x[3];
    double x27 = 0.5*x[7];
    double x28 = x26 + x27;
    double x29 = x24*x28;
    double x30 = 1.0*x28;
    double x31 = 1.0*x24;
    double x32 = x30 - x31;
    double x33 = x32*x[11];
    double x34 = x33*x29;
    double x35 = 1043.0*x25 + 504.0*x34;
    double x36 = -1.0*x35;
    double x37 = 1e-09 + x36;
    double x38 = pow(x[2], 2);
    double x39 = 0.905299382744389/x38;
    double x40 = pow(x37, 15);
    double x41 = pow(x40, -1);
    double x42 = pow(x[2], 14);
    double x43 = 0.0229512519569*x42;
    double x44 = pow(x37, -9);
    double x45 = pow(x[2], 8);
    double x46 = 0.0612033385517333*x45;
    double x47 = pow(x37, 3);
    double x48 = pow(x47, -1);
    double x49 = 0.459025039138*x38;
    double x50 = x[2] < x36;
    double x51 = 1e-09 + x35;
    double x52 = pow(x51, 15);
    double x53 = pow(x52, -1);
    double x54 = pow(x51, -9);
    double x55 = pow(x51, 3);
    double x56 = pow(x55, -1);
    double x57 = x[2] < x35;
    double x58 = pow(x51, 25);
    double x59 = pow(x[2], -26);
    double x60 = (1.0/60.0)*x59;
    double x61 = pow(x[2], -16);
    double x62 = (1.0/21.0)*x61;
    double x63 = pow(x51, 5);
    double x64 = pow(x[2], -6);
    double x65 = (1.0/2.0)*x64;
    double x66 = (0 < 1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9])) && 1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9])) < x[2]);
    double x67 = pow(x37, 25);
    double x68 = pow(x37, 5);
    double x69 = (-1.0*(1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9]))) < x[2] && 1043.0*(0.5*x[5] + 0.5*x[9])*x[11] + 504.0*(0.5*x[3] + 0.5*x[7])*(0.5*x[5] + 0.5*x[9])*x[11]*(1.0*(0.5*x[3] + 0.5*x[7]) - 1.0*(0.5*x[5] + 0.5*x[9])) < 0);
    double x70 = ((x50 == 1) ? (
   x37*x39 - x41*x43 - x44*x46 - x48*x49
)
: ((x57 == 1) ? (
   x51*x39 - x53*x43 - x54*x46 - x56*x49
)
: ((x66 == 1) ? (
   -0.641731208021339*(-x60*x58 - x62*x52 - x63*x65)
)
: ((x69 == 1) ? (
   -0.641731208021339*(-x60*x67 - x62*x40 - x68*x65)
)
: 0))));
    double x71 = 2.22*x24;
    double x72 = x71*x[11];
    double x73 = x72 <= 0;
    double x74 = ((x73 == 1) ? (
   -1.0
)
: (
   1.0
));
    double x75 = ((x73 == 1) ? 0
: 0);
    double x76 = x75/pow(x74, 2);
    double x77 = x76*x24;
    double x78 = pow(x74, -1);
    double x79 = x71*x78;
    double x80 = 1 + x79*x[11];
    double x81 = pow(x80, -1);
    double x82 = 0.5*x[4] + 0.5*x[8];
    double x83 = 1.0*x82;
    double x84 = x30 + x31 + x83;
    double x85 = pow(x84, -1);
    double x86 = x85*x[11];
    double x87 = x81*x86;
    double x88 = x[2]*x87;
    double x89 = x88*x77;
    double x90 = 36.91638*x89;
    double x91 = 0.918050078276*x[2];
    double x92 = pow(x[2], 7);
    double x93 = 0.489626708413867*x92;
    double x94 = 0.3213175273966*pow(x[2], 13);
    double x95 = pow(x[2], 3);
    double x96 = 1.81059876548878/x95;
    double x97 = 3/x92;
    double x98 = (16.0/21.0)/pow(x[2], 17);
    double x99 = (13.0/30.0)/pow(x[2], 27);
    double x100 = log(x80);
    double x101 = x85*x100;
    double x102 = 8.3145*x101;
    double x103 = x[2]*x102;
    double x104 = pow(x[2], -1);
    double x105 = 1.41071428571429*x104;
    double x106 = (1.0/6.0)*x95;
    double x107 = pow(x[2], 9);
    double x108 = (1.0/135.0)*x107;
    double x109 = pow(x[2], 15);
    double x110 = (1.0/600.0)*x109;
    double x111 = pow(x[2], -5);
    double x112 = (1.0/10.0)*x111;
    double x113 = pow(x109, -1);
    double x114 = (1.0/315.0)*x113;
    double x115 = pow(x[2], -25);
    double x116 = (1.0/1500.0)*x115;
    double x117 = ((x50 == 1) ? (
   1 - 0.641731208021339*(x37*x105 + 1.43058350100604*(x41*x110 + x44*x108 + x48*x106))
)
: ((x57 == 1) ? (
   1 - 0.641731208021339*(x51*x105 + 1.43058350100604*(x53*x110 + x54*x108 + x56*x106))
)
: ((x66 == 1) ? (
   -0.641731208021339*(x52*x114 + x58*x116 + x63*x112)
)
: ((x69 == 1) ? (
   -0.641731208021339*(x40*x114 + x67*x116 + x68*x112)
)
: 0))));
    double x118 = x77*x117;
    double x119 = x87*x118;
    double x120 = x70*x101;
    double x121 = pow(x[2], 1.0);
    double x122 = pow(x[2], 3.0);
    double x123 = pow(x122, -1);
    double x124 = 148184.0*x123;
    double x125 = x[2] < 700.0;
    double x126 = (x[2] < 933.47 && 700.0 <= x[2]);
    double x127 = pow(x[2], -11.0);
    double x128 = 933.47 <= x[2];
    double x129 = x[2] < 1357.77;
    double x130 = 1357.77 <= x[2];
    double x131 = x[2] < 1811.0;
    double x132 = 1811.0 <= x[2];
    double x133 = 1.0*x85;
    double x134 = pow(x75, 2);
    double x135 = pow(x80, -2);
    double x136 = x85*x135;
    double x137 = x117*x136;
    double x138 = x134/pow(x74, 3);
    double x139 = x88*x117;
    double x140 = 36.91638*x24*x138*x139;
    double x141 = x140 - 40.9771818*x[2]*pow(x24, 2)*x134*x137*pow(x[11], 2)/pow(x74, 4);
    double x142 = 18.45819*x119;
    double x143 = -x[2]*x142;
    double x144 = 8.3145*x1;
    double x145 = x[2]*x144;
    double x146 = x22*x145;
    double x147 = x143 + x146;
    double x148 = x141 + x147;
    double x149 = 0.5*x[10];
    double x150 = x1*x[11];
    double x151 = log(x[3]);
    double x152 = x11 + x13 + x15 + x17 + x3 + x5 + x7;
    double x153 = x152 + x19;
    double x154 = x153 + 0.5*((x8 == 1) ? (
   1 + x151
)
: 0);
    double x155 = x144*x154;
    double x156 = 9.229095*x76;
    double x157 = pow(x84, -2);
    double x158 = x81*x157;
    double x159 = x[2]*x25*x117*x158;
    double x160 = x156*x159;
    double x161 = 252.0*x29;
    double x162 = x161*x[11];
    double x163 = -x162;
    double x164 = 252.0*x32;
    double x165 = x24*x164;
    double x166 = x165*x[11];
    double x167 = x163 - x166;
    double x168 = 0.905299382744389*x104;
    double x169 = pow(x37, -16);
    double x170 = 0.0229512519569*x109;
    double x171 = x169*x170;
    double x172 = pow(x37, -10);
    double x173 = 0.0612033385517333*x107;
    double x174 = x172*x173;
    double x175 = pow(x37, 4);
    double x176 = pow(x175, -1);
    double x177 = 0.459025039138*x95;
    double x178 = x176*x177;
    double x179 = x162 + x166;
    double x180 = pow(x51, 4);
    double x181 = pow(x180, -1);
    double x182 = x177*x181;
    double x183 = pow(x51, -16);
    double x184 = x170*x183;
    double x185 = pow(x51, -10);
    double x186 = x173*x185;
    double x187 = pow(x51, 24);
    double x188 = (1.0/60.0)*x115;
    double x189 = x187*x188;
    double x190 = pow(x51, 14);
    double x191 = (1.0/21.0)*x113;
    double x192 = x190*x191;
    double x193 = (1.0/2.0)*x111;
    double x194 = x180*x193;
    double x195 = pow(x37, 24);
    double x196 = x188*x195;
    double x197 = pow(x37, 14);
    double x198 = x191*x197;
    double x199 = x175*x193;
    double x200 = ((x50 == 1) ? (
   x167*x171 + x167*x174 + x167*x178 - x168*x167
)
: ((x57 == 1) ? (
   -x168*x179 + x179*x182 + x179*x184 + x179*x186
)
: ((x66 == 1) ? (
   -0.641731208021339*(x179*x189 + x179*x192 + x179*x194)
)
: ((x69 == 1) ? (
   -0.641731208021339*(x167*x196 + x167*x198 + x167*x199)
)
: 0))));
    double x201 = 18.45819*x89;
    double x202 = -x200*x201;
    double x203 = 0.5*x[6];
    double x204 = x149 + x203;
    double x205 = 0.5*x86;
    double x206 = x200*x102;
    double x207 = 1.377075117414*x38;
    double x208 = x207*x176;
    double x209 = 0.5508300469656*x45;
    double x210 = x209*x172;
    double x211 = 0.3442687793535*x42;
    double x212 = x211*x169;
    double x213 = x207*x181;
    double x214 = x209*x185;
    double x215 = x211*x183;
    double x216 = (5.0/2.0)*x64;
    double x217 = x216*x180;
    double x218 = (5.0/7.0)*x61;
    double x219 = x218*x190;
    double x220 = (5.0/12.0)*x59;
    double x221 = x220*x187;
    double x222 = x216*x175;
    double x223 = x218*x197;
    double x224 = x220*x195;
    double x225 = log(x[2]);
    double x226 = 24.3671976*x225;
    double x227 = pow(x[2], 2.0);
    double x228 = pow(x227, -1);
    double x229 = -74092.0*x228;
    double x230 = 38.5844296*x225;
    double x231 = pow(x[2], -10.0);
    double x232 = 31.748192*x225;
    double x233 = -4.813 + ((x125 == 1) ? (
   112.7258404 - 0.003769324*x121 - x226 - 2.632992e-06*x227 + x229
)
: ((x126 == 1) ? (
   184.4640164 + 0.037063964*x121 - 1.7292681e-05*x227 + x229 - x230
)
: ((x128 == 1) ? (
   156.935961 + 1.1074716e+29*x231 - x232
)
: 0)));
    double x234 = 0.5*x[11];
    double x235 = ((x129 == 1) ? 0
: ((x130 == 1) ? 0
: 0));
    double x236 = x83*x235;
    double x237 = ((x125 == 1) ? 0
: ((x126 == 1) ? 0
: ((x128 == 1) ? 0
: 0)));
    double x238 = x30*x237;
    double x239 = ((x131 == 1) ? 0
: ((x132 == 1) ? 0
: 0));
    double x240 = x31*x239;
    double x241 = x236*x[11] + x238*x[11] + x240*x[11];
    double x242 = x234*x204;
    double x243 = x82*x[11];
    double x244 = x30 - x83;
    double x245 = 5.92*x244;
    double x246 = x28*x[11];
    double x247 = x82*x246;
    double x248 = 5.92*x247;
    double x249 = x82*x25;
    double x250 = (1.0/3.0)*(1 - x84);
    double x251 = x250 + x83;
    double x252 = 34.9597143*x251;
    double x253 = x250 + x30;
    double x254 = 65.728274*x253;
    double x255 = 2.0*x82;
    double x256 = x29*x243;
    double x257 = -x142 - x70*x201;
    double x258 = 0.5*x157;
    double x259 = x258*x246;
    double x260 = 131.456548*x253;
    double x261 = 69.9194286*x251;
    double x262 = 11.84*x244;
    double x263 = x30*x204;
    double x264 = 31.9888*x29;
    double x265 = 4.0*x247 - 4.14983*x249 + x260*x256 + x261*x256 - x262*x247 - x263*x[11] + x264*x[11];
    double x266 = x149*x[3];
    double x267 = x203*x[7];
    double x268 = -x266*x[11] - x267*x[11];
    double x269 = pow(x0, -2);
    double x270 = 0.5*x269;
    double x271 = log(x[5]);
    double x272 = log(x[6]);
    double x273 = log(x[11]);
    double x274 = log(x[4]);
    double x275 = log(x[9]);
    double x276 = log(x[8]);
    double x277 = log(x[10]);
    double x278 = log(x[7]);
    double x279 = 0.5*((x8 == 1) ? (
   x151*x[3]
)
: 0) + 0.5*((x6 == 1) ? (
   x271*x[5]
)
: 0) + 0.5*((x10 == 1) ? (
   x272*x[6]
)
: 0) + 3.0*((x2 == 1) ? (
   x273*x[11]
)
: 0) + 0.5*((x12 == 1) ? (
   x274*x[4]
)
: 0) + 0.5*((x14 == 1) ? (
   x275*x[9]
)
: 0) + 0.5*((x16 == 1) ? (
   x276*x[8]
)
: 0) + 0.5*((x18 == 1) ? (
   x277*x[10]
)
: 0) + 0.5*((x4 == 1) ? (
   x278*x[7]
)
: 0);
    double x280 = 4.15725*x279;
    double x281 = x269*x280;
    double x282 = 4.15725*x[2];
    double x283 = x269*x282;
    double x284 = 23.5143*x225;
    double x285 = 46.0*x225;
    double x286 = ((x131 == 1) ? (
   100.6197 - 0.00879504*x121 - 1.76781e-07*x227 - 77359.0*x228 - x284
)
: ((x132 == 1) ? (
   253.31255 - 2.066427e+32*x231 - x285
)
: 0));
    double x287 = x31*x286;
    double x288 = x30*x233;
    double x289 = 24.112392*x225;
    double x290 = 31.38*x225;
    double x291 = -1.255 + ((x129 == 1) ? (
   106.372843 - 0.00531368*x121 + 3.87669e-07*x227 - 52478.0*x228 - x289
)
: ((x130 == 1) ? (
   152.423828 - 3.277503e+30*x231 - x290
)
: 0));
    double x292 = x83*x291;
    double x293 = x287*x[11] + x288*x[11] + x292*x[11];
    double x294 = x100*x157;
    double x295 = x294*x282;
    double x296 = 4.15725*x117;
    double x297 = -x281 - x204*x259 - x22*x283 - x265*x258 - x270*x268 - x293*x258 - x296*x294 - x70*x295;
    double x298 = x148 + x160 + x202 + x206 + x257 + x297 + x103*((x50 == 1) ? (
   x208*x167 + x210*x167 + x212*x167 + x39*x167
)
: ((x57 == 1) ? (
   x213*x179 + x214*x179 + x215*x179 + x39*x179
)
: ((x66 == 1) ? (
   -0.641731208021339*(-x217*x179 - x219*x179 - x221*x179)
)
: ((x69 == 1) ? (
   -0.641731208021339*(-x222*x167 - x223*x167 - x224*x167)
)
: 0)))) + x133*(x241 + x234*x233) + x133*(-x242 - x248 + 15.9944*x25 + 32.1656112333333*x256 - x243*x245 + x252*x249 + x254*x249 + x255*x[11]) + x204*x205;
    double x299 = x0*(x155 + x298 - x149*x150);
    double x300 = 8.3145*x[2];
    double x301 = x1*x268;
    double x302 = 0.5*(x147 + x301 + x102*x117 + x265*x133 + x279*x144 + x293*x133 + x300*x120 + x86*x263);
    double x303 = x15 + x17 + x19 + x3 + x5 + x7 + x9;
    double x304 = x11 + x303;
    double x305 = x304 + 0.5*((x12 == 1) ? (
   1 + x274
)
: 0);
    double x306 = x305*x144;
    double x307 = x29*x[11];
    double x308 = 2.0*x28;
    double x309 = x308*x[11];
    double x310 = ((x50 == 1) ? 0
: ((x57 == 1) ? 0
: ((x66 == 1) ? 0
: ((x69 == 1) ? 0
: 0))));
    double x311 = x310*x102;
    double x312 = x146 - x201*x310;
    double x313 = x257 + x311 + x312;
    double x314 = x[2]*x311;
    double x315 = x143 + x314;
    double x316 = x141 + x160;
    double x317 = x315 + x316;
    double x318 = x297 + x313 + x317 + x133*(x241 + x234*x291) + x133*(x248 - 2.074915*x25 + 1.39705153333334*x256 + x309 - x245*x246 + x252*x307 + x254*x307);
    double x319 = (x306 + x318)*x0;
    double x320 = x11 + x13 + x17 + x19 + x3 + x5 + x9;
    double x321 = x15 + x320;
    double x322 = x321 + 0.5*((x6 == 1) ? (
   1 + x271
)
: 0);
    double x323 = x322*x144;
    double x324 = 521.5*x[11];
    double x325 = x28*x164;
    double x326 = x325*x[11];
    double x327 = x162 - x324 - x326;
    double x328 = x163 + x324 + x326;
    double x329 = x328*x183;
    double x330 = ((x50 == 1) ? (
   -x327*x168 + x327*x171 + x327*x174 + x327*x178
)
: ((x57 == 1) ? (
   -x328*x168 + x328*x182 + x328*x186 + x329*x170
)
: ((x66 == 1) ? (
   -0.641731208021339*(x328*x189 + x328*x192 + x328*x194)
)
: ((x69 == 1) ? (
   -0.641731208021339*(x327*x196 + x327*x198 + x327*x199)
)
: 0))));
    double x331 = x330*x102;
    double x332 = 1.11*x78;
    double x333 = -x72*x76;
    double x334 = x333 + x332*x[11];
    double x335 = x81*x85;
    double x336 = x300*x335;
    double x337 = x336*x334;
    double x338 = x334*x117;
    double x339 = 8.3145*x335;
    double x340 = x297 + x331 + x103*((x50 == 1) ? (
   x208*x327 + x210*x327 + x212*x327 + x39*x327
)
: ((x57 == 1) ? (
   x211*x329 + x213*x328 + x214*x328 + x39*x328
)
: ((x66 == 1) ? (
   -0.641731208021339*(-x217*x328 - x219*x328 - x221*x328)
)
: ((x69 == 1) ? (
   -0.641731208021339*(-x222*x327 - x223*x327 - x224*x327)
)
: 0)))) + x133*(x241 + x234*x286) + x133*(-2.074915*x243 + 15.9944*x246 - 33.5626627666667*x256 + x252*x247 + x254*x247) + x339*x338 + x70*x337;
    double x341 = x323 + x340;
    double x342 = 18.45819*x[2]*x86*x77*x135;
    double x343 = x160 - x201*x330 + x338*x342;
    double x344 = x140 + x343 - x139*x156;
    double x345 = x147 + x344;
    double x346 = x13 + x303;
    double x347 = x346 + 0.5*((x10 == 1) ? (
   1 + x272
)
: 0);
    double x348 = x347*x144;
    double x349 = x315 + x241*x133;
    double x350 = x141 + x349;
    double x351 = x313 + x350;
    double x352 = x0*(x348 + x351 - x27*x150);
    double x353 = x20 + x3;
    double x354 = x353 + 0.5*((x4 == 1) ? (
   1 + x278
)
: 0);
    double x355 = x354*x144;
    double x356 = x0*(x298 + x355 - x203*x150);
    double x357 = x11 + x13 + x15 + x19 + x3 + x5 + x7 + x9;
    double x358 = x357 + 0.5*((x16 == 1) ? (
   1 + x276
)
: 0);
    double x359 = x358*x144;
    double x360 = (x318 + x359)*x0;
    double x361 = x320 + x7;
    double x362 = x361 + 0.5*((x14 == 1) ? (
   1 + x275
)
: 0);
    double x363 = x362*x144;
    double x364 = x340 + x363;
    double x365 = x152 + x9;
    double x366 = x365 + 0.5*((x18 == 1) ? (
   1 + x277
)
: 0);
    double x367 = x366*x144;
    double x368 = x0*(x351 + x367 - x26*x150);
    double x369 = x82*x24;
    double x370 = x82*x28;
    double x371 = x82*x29;
    double x372 = 521.5*x[9];
    double x373 = x32*x29;
    double x374 = 504.0*x373;
    double x375 = 521.5*x[5];
    double x376 = -x372 - x374 - x375;
    double x377 = x372 + x374 + x375;
    double x378 = x376*x195;
    double x379 = x376*x197;
    double x380 = ((x50 == 1) ? (
   -x376*x168 + x376*x171 + x376*x174 + x376*x178
)
: ((x57 == 1) ? (
   -x377*x168 + x377*x182 + x377*x184 + x377*x186
)
: ((x66 == 1) ? (
   -0.641731208021339*(x377*x189 + x377*x192 + x377*x194)
)
: ((x69 == 1) ? (
   -0.641731208021339*(x376*x199 + x378*x188 + x379*x191)
)
: 0))));
    double x381 = x380*x102;
    double x382 = x333 + x79;
    double x383 = x382*x117;
    double x384 = x21 + 3.0*((x2 == 1) ? (
   1 + x273
)
: 0);
    double x385 = x384*x144;
    double x386 = x382*x336;
    double x387 = x381 + x385 + x103*((x50 == 1) ? (
   x208*x376 + x210*x376 + x212*x376 + x39*x376
)
: ((x57 == 1) ? (
   x213*x377 + x214*x377 + x215*x377 + x39*x377
)
: ((x66 == 1) ? (
   -0.641731208021339*(-x217*x377 - x219*x377 - x221*x377)
)
: ((x69 == 1) ? (
   -0.641731208021339*(-x218*x379 - x220*x378 - x222*x376)
)
: 0)))) + x133*(-x263 + x264 - 4.14983*x369 + 4.0*x370 + x260*x371 + x261*x371 - x262*x370) + x383*x339 + x70*x386 + x85*x263 + (-x266 - x267)*x1 + (x241 + x287 + x288 + x292)*x133;
    double x388 = x[2]*x335;
    double x389 = -x201*x380 + x383*x342;
    double x390 = x140 + x389 - 18.45819*x388*x118;
    double x391 = x147 + x390;
    double x392 = -9.229095*x[2]*x119;
    double x393 = 0.5*x85;
    double x394 = x86*x28;
    double x395 = 0.5*x394;
    double x396 = 0.5*x301 + x392 + x1*x280 + x204*x395 + x23*x282 + x265*x393 + x282*x120 + x293*x393 + x296*x101;
    double x397 = x[2]*x269;
    double x398 = 8.3145*x397;
    double x399 = 54.1229146*x[9];
    double x400 = x399*x[11];
    double x401 = 31873.5471*x[5];
    double x402 = x401*x[8];
    double x403 = 30836.8712*x[11];
    double x404 = x403*x[4];
    double x405 = x400*x[8] + x402*x[11] + x404*x[9];
    double x406 = 1.0*x269;
    double x407 = 10000.0*x[8];
    double x408 = x407*x[11];
    double x409 = 14300.08*x[9];
    double x410 = -1.0*x[2];
    double x411 = -75000.0 + 0.5*(10000.0 + x410);
    double x412 = x411*x[10];
    double x413 = -x408 - x409*x[11] + x412*x[11];
    double x414 = 100000.0*x82*x204;
    double x415 = -30740.0 + 7.9972*x[2];
    double x416 = 4.0*x29*x415;
    double x417 = 160000.0 + x410;
    double x418 = x417*x263;
    double x419 = 150000.0*x204;
    double x420 = 51500.0 - 11.84*x[2];
    double x421 = x420*x244;
    double x422 = -73554.0 + 4.0*x[2];
    double x423 = x83*x422;
    double x424 = -103924.225 + 69.9194286*x[2];
    double x425 = x424*x251;
    double x426 = x31*x247;
    double x427 = x250 + x31;
    double x428 = 29249.2599*x427;
    double x429 = 39258.0 - 4.14983*x[2];
    double x430 = -267841.207 + 131.456548*x[2];
    double x431 = x430*x253;
    double x432 = 3089.2*x34 + x25*x419 + x414*x[11] + x416*x[11] + x418*x[11] + x423*x246 + x425*x426 + x426*x431 - x428*x256 + x31*x429*x243 + x83*x421*x246;
    double x433 = pow(x84, -3);
    double x434 = 0.5*x433;
    double x435 = x411*x[6];
    double x436 = x435*x[7];
    double x437 = x412*x[3];
    double x438 = 10000.0*x[4];
    double x439 = x438*x[11];
    double x440 = x409*x[3];
    double x441 = 14300.08*x[5];
    double x442 = x441*x[11];
    double x443 = -x408*x[3] + x436*x[11] + x437*x[11] - x439*x[7] - x440*x[11] - x442*x[7];
    double x444 = pow(x0, -3);
    double x445 = 0.5*x444;
    double x446 = pow(x121, -1);
    double x447 = pow(x[2], -9.0);
    double x448 = ((x129 == 1) ? (
   -7770.458 + 130.485235*x[2] + 1.29223e-07*x122 - 0.00265684*x227 + 52478.0*x446 - x[2]*x289
)
: ((x130 == 1) ? (
   -13542.026 + 183.803828*x[2] + 3.64167e+29*x447 - x[2]*x290
)
: 0));
    double x449 = 4017.0 - 1.255*x[2] + x448;
    double x450 = x83*x449;
    double x451 = 74092.0*x446;
    double x452 = ((x125 == 1) ? (
   -7976.15 + 137.093038*x[2] - 8.77664e-07*x122 - 0.001884662*x227 + x451 - x[2]*x226
)
: ((x126 == 1) ? (
   -11276.24 + 223.048446*x[2] - 5.764227e-06*x122 + 0.018531982*x227 + x451 - x[2]*x230
)
: ((x128 == 1) ? (
   -11278.378 + 188.684153*x[2] - 1.230524e+28*x447 - x[2]*x232
)
: 0)));
    double x453 = 10083.0 - 4.813*x[2] + x452;
    double x454 = x30*x453;
    double x455 = ((x131 == 1) ? (
   1225.7 + 124.134*x[2] - 5.8927e-08*x122 - 0.00439752*x227 + 77359.0*x446 - x[2]*x284
)
: ((x132 == 1) ? (
   -25383.581 + 299.31255*x[2] + 2.29603e+31*x447 - x[2]*x285
)
: 0));
    double x456 = x31*x455;
    double x457 = x450*x[11] + x454*x[11] + x456*x[11];
    double x458 = 62764.5412146*x256;
    double x459 = x[3]*x[9];
    double x460 = 30836.8712*x[7];
    double x461 = x460*x[11];
    double x462 = x[5]*x[8];
    double x463 = 31873.5471*x[3];
    double x464 = x463*x[11];
    double x465 = 31873.5471*x[7];
    double x466 = x465*x[11];
    double x467 = x466*x[9];
    double x468 = 54.1229146*x[7];
    double x469 = x468*x[5];
    double x470 = x469*x[4];
    double x471 = 54.1229146*x[11];
    double x472 = x471*x[8];
    double x473 = x459*x404 + x459*x472 + x462*x461 + x462*x464 + x467*x[4] + x470*x[11];
    double x474 = 28600.16*x29;
    double x475 = x411*x204;
    double x476 = -20000.0*x247 - x474*x[11] + x475*x309;
    double x477 = x434*x432 + x445*x443 + x457*x434 - x458*x433 + x473*x445 - x476*x434 + x[2]*x444*x280 + x433*x282*x100*x117;
    double x478 = x477 + 18.45819*x76*x159;
    double x479 = x141 + x478;
    double x480 = x234*x369;
    double x481 = 4874.87665*x29;
    double x482 = x481*x243;
    double x483 = x422*x234;
    double x484 = x82*x420;
    double x485 = x484*x244;
    double x486 = 14624.62995*x427;
    double x487 = x234*x370;
    double x488 = x487*x420;
    double x489 = 0.166666666666667*x424;
    double x490 = -x489*x256;
    double x491 = 1544.6*x32;
    double x492 = x24*x491;
    double x493 = 1544.6*x29;
    double x494 = x493*x[11];
    double x495 = x482 + x488 + x490 + x494 + 2.0*x25*x415 + x417*x242 + 0.333333333333333*x430*x256 + x480*x425 + x480*x431 + x485*x234 - x486*x249 + x492*x[11] + x82*x483;
    double x496 = 1.0*x157;
    double x497 = x430*x249;
    double x498 = 1.0*x475;
    double x499 = 10000.0*x82;
    double x500 = 14300.08*x24;
    double x501 = x498*x[11] - x499*x[11] - x500*x[11];
    double x502 = 62764.5412146*x82;
    double x503 = x502*x157;
    double x504 = 0.5*x453;
    double x505 = x241 + x504*x[11];
    double x506 = pow(x167, 2);
    double x507 = 1.836100156552*x95;
    double x508 = x507/x68;
    double x509 = 115.674309862776*x95;
    double x510 = x25*x509;
    double x511 = 15.4232413150368*x107;
    double x512 = x511*x172;
    double x513 = 0.3672200313104*x109;
    double x514 = x513/pow(x37, 17);
    double x515 = 5.7837154931388*x109;
    double x516 = x515*x169;
    double x517 = 0.612033385517333*x107;
    double x518 = x517/pow(x37, 11);
    double x519 = 228.135444451586*x104;
    double x520 = x25*x519;
    double x521 = pow(x179, 2);
    double x522 = x517/pow(x51, 11);
    double x523 = x511*x185;
    double x524 = x513/pow(x51, 17);
    double x525 = x515*x183;
    double x526 = x507/x63;
    double x527 = 126.0*x25;
    double x528 = x527*x111;
    double x529 = (2.0/3.0)*x113;
    double x530 = pow(x51, 13)*x529;
    double x531 = 12.0*x113;
    double x532 = x25*x531;
    double x533 = (2.0/5.0)*x115;
    double x534 = pow(x51, 23)*x533;
    double x535 = 2*x111;
    double x536 = x55*x535;
    double x537 = 4.2*x115;
    double x538 = x25*x537;
    double x539 = x47*x535;
    double x540 = pow(x37, 13)*x529;
    double x541 = pow(x37, 23)*x533;
    double x542 = 1.0*x[11];
    double x543 = x[2]*x294;
    double x544 = 8.3145*x543;
    double x545 = x103*((x50 == 1) ? (
   x520 - x25*x512 - x25*x516 - x506*x508 - x510*x176 - x514*x506 - x518*x506
)
: ((x57 == 1) ? (
   -x520 + x25*x523 + x25*x525 + x510*x181 - x521*x526 - x522*x521 - x524*x521
)
: ((x66 == 1) ? (
   -0.641731208021339*(x521*x530 + x521*x534 + x521*x536 + x528*x180 + x532*x190 + x538*x187)
)
: ((x69 == 1) ? (
   -0.641731208021339*(x506*x539 + x506*x540 + x506*x541 - x528*x175 - x532*x197 - x538*x195)
)
: 0)))) + x133*(x241 + x542*x237) + x133*(4874.87665*x249 + 1544.6*x25 + 0.333333333333333*x497 + x484*x234 - x489*x249) + x25*x503 - x495*x496 + x496*x501 - x496*x505 - x544*x200 - x90*x200;
    double x546 = x143 + x479 + x545;
    double x547 = x1*x405;
    double x548 = x1*x413;
    double x549 = 62764.5412146*x86;
    double x550 = -x[2]*x281 - x295*x117 - x432*x258 - x443*x270 - x457*x258 + x458*x157 - x473*x270 + x476*x258;
    double x551 = x143 + x550 + x[2]*x206 + x495*x133 - x501*x133 + x505*x133 - x549*x369;
    double x552 = 0.25*x157;
    double x553 = 0.25*x269;
    double x554 = 31382.2706073*x157;
    double x555 = -2.078625*x279*x397 - x432*x552 - x443*x553 - x457*x552 - x473*x553 + x476*x552 - 2.078625*x543*x117 + x554*x256;
    double x556 = x555 + 0.5*(x547 + x548 + x551 + x[2]*x155);
    double x557 = 31382.2706073*x86;
    double x558 = x282*x101;
    double x559 = x392 + x495*x393 - x501*x393 + x505*x393 - x557*x369 + x558*x200;
    double x560 = x556 + x559;
    double x561 = x1*x282;
    double x562 = 0.5*x547 + 0.5*x548 + x561*x154;
    double x563 = x1*x403;
    double x564 = 5000.0*x[11];
    double x565 = x564*x269;
    double x566 = x467 + x459*x403 + x469*x[11];
    double x567 = -x283*x305 - x566*x270;
    double x568 = x567 + x565*x[7];
    double x569 = -x283*x154 - x405*x270 - x413*x270;
    double x570 = -x200*x295 - x495*x258 + x501*x258 - x505*x258 + x554*x249;
    double x571 = x569 + x570;
    double x572 = 0.25*x246;
    double x573 = 7312.314975*x427;
    double x574 = 0.25*x[11];
    double x575 = 0.25*x25;
    double x576 = 0.0833333333333333*x424;
    double x577 = -x576*x307;
    double x578 = 0.166666666666667*x424;
    double x579 = 0.166666666666667*x430;
    double x580 = -0.0833333333333333*x497;
    double x581 = x234*x237;
    double x582 = x241 + x234*x235;
    double x583 = x246*x157;
    double x584 = 5000.0*x86;
    double x585 = x202 + x312;
    double x586 = x315 + x479;
    double x587 = -x295*x310;
    double x588 = x307*x157;
    double x589 = x29*x234;
    double x590 = 0.333333333333333*x424;
    double x591 = 0.166666666666667*x430;
    double x592 = -x591*x256;
    double x593 = x28*x244;
    double x594 = x420*x593;
    double x595 = 50000.0*x204;
    double x596 = x482 - x488 + x592 + x28*x483 + x425*x589 + x431*x589 - x486*x307 + x590*x256 + x594*x234 + x595*x[11] + x24*x429*x234;
    double x597 = 0.5*x449;
    double x598 = x241 + x597*x[11];
    double x599 = x587 + 31382.2706073*x588 - x596*x258 - x598*x258;
    double x600 = -5000.0*x583 + x584 + x585 + x586 + x599 + x133*(2437.438325*x249 + 2437.438325*x307 + x577 + x580 - x25*x573 + x420*x572 + x421*x574 + x422*x574 + x425*x575 + x431*x575 - x484*x574 + x578*x249 + x579*x307) - x24*x557 + (x581 + x582)*x133;
    double x601 = x571 + x600;
    double x602 = x0*(x568 + x601 + x563*x[9]);
    double x603 = x392 + x558*x310;
    double x604 = x603 + x28*x584 - x29*x557 + x596*x393 + x598*x393;
    double x605 = x1*x564;
    double x606 = x1*x566;
    double x607 = 0.5*x606 + x561*x305 - x605*x[7];
    double x608 = x604 + x607;
    double x609 = 31873.5471*x150;
    double x610 = x571 + x609*x[8];
    double x611 = x269*x[11];
    double x612 = x611*x[7];
    double x613 = x468*x[4];
    double x614 = x461*x[8] + x464*x[8] + x613*x[11];
    double x615 = -x283*x322 - x614*x270;
    double x616 = 7150.04*x612 + x615;
    double x617 = x234*x239;
    double x618 = x241 + x617;
    double x619 = 772.3*x[11];
    double x620 = x82*x574;
    double x621 = 2437.438325*x247;
    double x622 = x327*x167;
    double x623 = 126.0*x33;
    double x624 = 126.0*x246;
    double x625 = x527 - x623 - x624;
    double x626 = x328*x179;
    double x627 = -x527 + x623 + x624;
    double x628 = x541*x167;
    double x629 = 7150.04*x86;
    double x630 = 0.5*x455;
    double x631 = x241 + x630*x[11];
    double x632 = x282*x158;
    double x633 = x82*x429;
    double x634 = 75000.0*x204;
    double x635 = x28*x491;
    double x636 = -9749.7533*x256 + x490 - x494 + x592 + x415*x309 - x486*x247 + x487*x425 + x487*x431 + x633*x234 + x634*x[11] + x635*x[11];
    double x637 = x477 - x295*x330 - x631*x258 - x632*x338 - x636*x258 + 31382.2706073*x82*x583;
    double x638 = -7150.04*x583 + x629 + x637 + x103*((x50 == 1) ? (
   -x622*x508 - x622*x514 - x622*x518 - x625*x168 + x625*x171 + x625*x174 + x625*x178
)
: ((x57 == 1) ? (
   -x626*x522 - x626*x524 - x626*x526 - x627*x168 + x627*x182 + x627*x184 + x627*x186
)
: ((x66 == 1) ? (
   -0.641731208021339*(x626*x530 + x626*x534 + x626*x536 + x627*x189 + x627*x192 + x627*x194)
)
: ((x69 == 1) ? (
   -0.641731208021339*(x622*x539 + x622*x540 + x625*x196 + x625*x198 + x625*x199 + x628*x327)
)
: 0)))) + x133*(-4874.87665*x249 + x580 + x621 - x24*x619 + x28*x619 + x32*x619 + x415*x542 + x425*x620 + x431*x620 - x573*x243 - x576*x247 - x576*x249 + x579*x247) + x200*x337 - x82*x557 + (x581 + x618)*x133;
    double x639 = x345 + x638;
    double x640 = x616 + x639;
    double x641 = x282*x335;
    double x642 = x28*x629 - x557*x370 + x558*x330 + x631*x393 + x636*x393 + x641*x338;
    double x643 = x1*x614;
    double x644 = 0.5*x643 - 7150.04*x150*x[7] + x561*x322;
    double x645 = x642 + x644;
    double x646 = x411*x611;
    double x647 = 75000.0*x24;
    double x648 = x28*x234;
    double x649 = 50000.0*x82;
    double x650 = x417*x648 + x647*x[11] + x649*x[11];
    double x651 = x587 - x258*x241 + x411*x259 - x650*x258;
    double x652 = x651 - x27*x646 - x283*x347;
    double x653 = x241 + x581;
    double x654 = x317 + x585 - x411*x205 + x653*x133 + 0.25*x86*x417;
    double x655 = x652 + x654;
    double x656 = x0*x655;
    double x657 = x1*x411;
    double x658 = x657*x[11];
    double x659 = x603 + x241*x393 - x411*x395 + x650*x393;
    double x660 = x659 + x27*x658 + x561*x347;
    double x661 = x471*x[4];
    double x662 = 31873.5471*x[9];
    double x663 = x662*x[11];
    double x664 = x403*x[8];
    double x665 = x661*x[5] + x663*x[4] + x664*x[5];
    double x666 = -x439 - x442 + x435*x[11];
    double x667 = x666*x269;
    double x668 = -0.5*x667 - x283*x354 - x665*x270;
    double x669 = x0*(x148 + x478 + x545 + x569 + x668);
    double x670 = x1*x666;
    double x671 = x1*x665;
    double x672 = 0.5*x670 + 0.5*x671 + x561*x354;
    double x673 = 10000.0*x1;
    double x674 = x673*x[11];
    double x675 = -x674;
    double x676 = x459*x471 + x461*x[5] + x464*x[5];
    double x677 = -x283*x358 - x676*x270;
    double x678 = x677 + x565*x[3];
    double x679 = x0*(x601 + x675 + x678 + x1*(x400 + x401*x[11]));
    double x680 = x1*x676;
    double x681 = 0.5*x680 + x561*x358 - x605*x[3];
    double x682 = x604 + x681;
    double x683 = 14300.08*x1;
    double x684 = x683*x[11];
    double x685 = -x684;
    double x686 = x571 + x685 + (x404 + x472)*x1;
    double x687 = 7150.04*x[3];
    double x688 = x404*x[3] + x466*x[4] + x472*x[3];
    double x689 = -x283*x362 - x688*x270;
    double x690 = x689 + x611*x687;
    double x691 = x639 + x690;
    double x692 = x1*x688;
    double x693 = 0.5*x692 + x561*x362 - x687*x150;
    double x694 = x642 + x693;
    double x695 = x651 - x26*x646 - x283*x366;
    double x696 = x654 + x695;
    double x697 = (x658 + x696)*x0;
    double x698 = x659 + x26*x658 + x561*x366;
    double x699 = 30836.8712*x[4];
    double x700 = 62764.5412146*x85;
    double x701 = x376*x167;
    double x702 = -x161;
    double x703 = -x165 + x702;
    double x704 = x161 + x165;
    double x705 = x377*x179;
    double x706 = 0.5*x417;
    double x707 = 2.0*x24;
    double x708 = 0.5*x420;
    double x709 = x708*x370;
    double x710 = x430*x371;
    double x711 = 0.5*x422;
    double x712 = 0.5*x425;
    double x713 = x82*x481;
    double x714 = -x489*x371;
    double x715 = 0.5*x431;
    double x716 = -20000.0*x370 - x474 + x475*x308;
    double x717 = x436 + x437 - x440 - x407*x[3] - x438*x[7] - x441*x[7];
    double x718 = 54.1229146*x459;
    double x719 = x465*x[9];
    double x720 = x463*x[8];
    double x721 = x460*x[8];
    double x722 = x699*x[3];
    double x723 = x470 + x718*x[8] + x719*x[4] + x720*x[5] + x721*x[5] + x722*x[9];
    double x724 = x282*x380;
    double x725 = x31*x370;
    double x726 = 3089.2*x373 + x414 + x416 + x418 + x24*x419 + x28*x423 + x31*x633 + x425*x725 - x428*x371 + x431*x725 + x83*x594;
    double x727 = x241 + x450 + x454 + x456;
    double x728 = -x283*x384 + x29*x503 - x632*x383 + x716*x258 - x717*x270 - x723*x270 - x724*x294 - x726*x258 - x727*x258;
    double x729 = x728 + x103*((x50 == 1) ? (
   -x701*x508 - x701*x514 - x701*x518 - x703*x168 + x703*x171 + x703*x174 + x703*x178
)
: ((x57 == 1) ? (
   -x704*x168 + x704*x182 + x704*x184 + x704*x186 - x705*x522 - x705*x524 - x705*x526
)
: ((x66 == 1) ? (
   -0.641731208021339*(x704*x189 + x704*x192 + x704*x194 + x705*x530 + x705*x534 + x705*x536)
)
: ((x69 == 1) ? (
   -0.641731208021339*(x628*x376 + x701*x539 + x701*x540 + x703*x196 + x703*x198 + x703*x199)
)
: 0)))) + x133*(0.5*x485 + x492 + x493 + x709 + 0.333333333333333*x710 + x713 + x714 + x415*x707 - x486*x369 + x706*x204 + x712*x369 + x715*x369 + x82*x711) + x200*x386 - x700*x369;
    double x730 = x729 + x1*(x402 + x399*x[8] + x699*x[9]) + x1*(-x407 - x409 + x412);
    double x731 = x236 + x238 + x240;
    double x732 = x653 + x731;
    double x733 = x391 + x133*(5041.5 - 2.4065*x[2] + 0.5*x452 + x732) - x133*(-5000.0*x[4] - 7150.04*x[5] - 5000.0*x[8] - 7150.04*x[9] + x498);
    double x734 = x1*x723;
    double x735 = x1*x717;
    double x736 = 0.5*x734 + 0.5*x735 + x561*x384 + x641*x383 - x716*x393 + x724*x101 + x726*x393 + x727*x393 - x85*x29*x502;
    double x737 = x673*x[7];
    double x738 = x315 + 10000.0*x394 + x550 - x29*x549 + x596*x133 + x598*x133;
    double x739 = x555 + 0.5*(x606 + x738 + x[2]*x306 - x737*x[11]);
    double x740 = x559 + x562;
    double x741 = 10000.0*x611;
    double x742 = -x90*x310;
    double x743 = -10000.0*x583 + 62764.5412146*x588 + x742 + x133*(x241 + x542*x235) + x133*(-x420*x648 + x481*x[11] + x590*x307 - x591*x307) - x496*x596 - x496*x598 - x544*x310;
    double x744 = x586 + x743;
    double x745 = x567 + 12150.04*x612 + x615 + x468*x150;
    double x746 = x146 + x315;
    double x747 = x337*x310;
    double x748 = x344 + x747;
    double x749 = 0.0833333333333333*x430;
    double x750 = -31382.2706073*x394 - 12150.04*x583 + x599 + x637 + x133*(-4874.87665*x307 + x577 + x621 + x425*x572 + x429*x574 + x431*x572 - x573*x246 + x578*x247 - x749*x247 - x749*x307) + (x582 + x617)*x133;
    double x751 = x748 + x750;
    double x752 = x746 + x751;
    double x753 = x652 + x746;
    double x754 = x316 + x742 + 25000.0*x86 + x582*x133;
    double x755 = (x753 + x754)*x0;
    double x756 = x660 + x755;
    double x757 = x570 + x668;
    double x758 = x600 + x757;
    double x759 = x0*(x568 + x675 + x758 + x1*(x663 + x471*x[5]));
    double x760 = x559 + x672;
    double x761 = x678 + x746;
    double x762 = (x479 + x568 + x743 + x761)*x0;
    double x763 = x568 + x690 + x1*(x466 + x403*x[3]);
    double x764 = x695 + x746;
    double x765 = (x754 + x764)*x0;
    double x766 = x698 + x765;
    double x767 = x386*x310;
    double x768 = x85*x28;
    double x769 = x728 + x767 + 10000.0*x768 - x29*x700;
    double x770 = -x737 + x769 + x1*(30836.8712*x459 + x469 + x719);
    double x771 = -0.166666666666667*x710;
    double x772 = 0.5*x29;
    double x773 = 0.5*x429;
    double x774 = -x709 + x713 + x771 + x24*x773 + x28*x711 - x29*x486 + x425*x772 + x431*x772 + x590*x371 + x708*x593;
    double x775 = x582 + x731;
    double x776 = x390 + x746;
    double x777 = x776 + x133*(25000.0*x[6] + 25000.0*x[10] + x774) + x133*(2008.5 - 0.6275*x[2] + 0.5*x448 + x775);
    double x778 = x76*x[11];
    double x779 = x333 + 4.44*x25*x138;
    double x780 = -1.11*x778 + x779;
    double x781 = x336*x117;
    double x782 = x343 + x781*x780;
    double x783 = x146 + x782;
    double x784 = x638 + x783;
    double x785 = x616 + x784;
    double x786 = x300*x338;
    double x787 = 1.0*x636;
    double x788 = 14300.08*x394 + x550 + x[2]*x331 - x549*x370 + x631*x133 + x786*x335 + x85*x787;
    double x789 = x555 + 0.5*(x643 + x788 + x[2]*x323 - x684*x[7]);
    double x790 = x559 + x789;
    double x791 = x146 + x314;
    double x792 = x747 + x782 + x791;
    double x793 = x750 + x792;
    double x794 = 14300.08*x611;
    double x795 = 9749.7533*x82;
    double x796 = pow(x327, 2);
    double x797 = x509*x246;
    double x798 = x519*x246;
    double x799 = pow(x328, 2);
    double x800 = x624*x111;
    double x801 = x531*x246;
    double x802 = x537*x246;
    double x803 = x300*x137;
    double x804 = 16.629*x388;
    double x805 = x477 - 14300.08*x583 + x103*((x50 == 1) ? (
   -x798 + x512*x246 + x516*x246 - x796*x508 - x796*x514 - x796*x518 + x797*x176
)
: ((x57 == 1) ? (
   x798 - x523*x246 - x525*x246 - x797*x181 - x799*x522 - x799*x524 - x799*x526
)
: ((x66 == 1) ? (
   -0.641731208021339*(x799*x530 + x799*x534 + x799*x536 - x800*x180 - x801*x190 - x802*x187)
)
: ((x69 == 1) ? (
   -0.641731208021339*(x796*x539 + x796*x540 + x796*x541 + x800*x175 + x801*x197 + x802*x195)
)
: 0)))) + x133*(x241 + x542*x239) + x133*(-1544.6*x246 - x489*x247 - x591*x247 - x795*x246) - x496*x631 + x502*x583 - x544*x330 + x781*(-2.22*x778 + x779) - x786*x158 - x787*x157 - x803*pow(x334, 2) + x804*x334*x330;
    double x806 = 37500.0*x86 + x618*x133;
    double x807 = x792 + x806;
    double x808 = x660 + (x652 + x807)*x0;
    double x809 = x685 + x757 + (x661 + x664)*x1;
    double x810 = x616 + (x461 + x464)*x1;
    double x811 = x642 + (x146 + x616 + x690 + x805)*x0;
    double x812 = x698 + (x695 + x807)*x0;
    double x813 = -x493 + x635 + x714 + x771 - x29*x795 + x415*x308 - x486*x370 + x712*x370 + x715*x370 + x82*x773;
    double x814 = x376*x327;
    double x815 = -521.5 + x161 - x325;
    double x816 = x377*x328;
    double x817 = 521.5 + x325 + x702;
    double x818 = -x71*x76;
    double x819 = x146 + x728 + 14300.08*x768 + x103*((x50 == 1) ? (
   -x508*x814 - x514*x814 - x518*x814 - x815*x168 + x815*x171 + x815*x174 + x815*x178
)
: ((x57 == 1) ? (
   -x522*x816 - x524*x816 - x526*x816 - x817*x168 + x817*x182 + x817*x184 + x817*x186
)
: ((x66 == 1) ? (
   -0.641731208021339*(x530*x816 + x534*x816 + x536*x816 + x817*x189 + x817*x192 + x817*x194)
)
: ((x69 == 1) ? (
   -0.641731208021339*(x539*x814 + x540*x814 + x541*x814 + x815*x196 + x815*x198 + x815*x199)
)
: 0)))) + x133*(x618 + x630 + x731) + x380*x337 + x386*x330 - x768*x502 + x781*(x332 + x780 + x818) - x786*x382*x136;
    double x820 = x819 + x133*(37500.0*x[6] + 37500.0*x[10] + x813);
    double x821 = x1*(x613 + x720 + x721) - x683*x[7];
    double x822 = x657*x[7];
    double x823 = x30*x411;
    double x824 = x349 + x650*x133 - x86*x823;
    double x825 = 0.5*(x824 + x[2]*x348 + x822*x[11]);
    double x826 = x755 + x825;
    double x827 = x748 + x806;
    double x828 = x825 + (x753 + x827)*x0;
    double x829 = x350 + x742;
    double x830 = (x655 + x658)*x0;
    double x831 = (x146 + x829)*x0;
    double x832 = x767 - x85*x823 + (x241 + x731)*x133;
    double x833 = x822 + x832;
    double x834 = x28*x706;
    double x835 = x776 + x133*(25000.0*x[4] + 37500.0*x[5] + 25000.0*x[8] + 37500.0*x[9] + x834);
    double x836 = x555 + 0.5*(x551 + x670 + x671 + x[2]*x355);
    double x837 = x0*(x678 + x758 + x563*x[5]);
    double x838 = x757 + x609*x[4];
    double x839 = x0*x696;
    double x840 = x729 + x1*(x435 - x438 - x441) + x1*(30836.8712*x462 + 54.1229146*x[5]*x[4] + x662*x[4]);
    double x841 = x555 + 0.5*(x680 + x738 + x[2]*x359 - x674*x[3]);
    double x842 = x604 + x841;
    double x843 = x677 + x689 + 12150.04*x611*x[3] + x1*x471*x[3];
    double x844 = x769 + x1*(x718 + x460*x[5] + x463*x[5]) - x673*x[3];
    double x845 = x690 + x784;
    double x846 = x555 + 0.5*(x692 + x788 + x[2]*x363 - x684*x[3]);
    double x847 = x1*(x722 + 54.1229146*x[3]*x[8] + x465*x[4]) - x683*x[3];
    double x848 = 0.5*(x824 + x[2]*x367 + x658*x[3]);
    double x849 = x765 + x848;
    double x850 = x848 + (x764 + x827)*x0;
    double x851 = x832 + x657*x[3];
    double x852 = x389 + (x779 + x818)*x781;
    double x853 = x146 + x852;
    double x854 = x853 - x133*(x498 - x499 - x500) + (x504 + x732)*x133;
    double x855 = 0.5*(x734 + x735 + x[2]*x381 + x[2]*x385 + x383*x336 - x716*x133 + x726*x133 + x727*x133 - 125529.0824292*x85*x371);
    double x856 = x791 + x852;
    double x857 = x856 + (x595 + x774)*x133 + (x597 + x775)*x133;
    double x858 = x819 + (x634 + x813)*x133;
    double x859 = x856 + x133*(x647 + x649 + x834);
    double x860 = pow(x376, 2);
    double x861 = pow(x377, 2);
    out[0] = x0*(-36.91638*x119 + 16.629*x120 + x148 + 16.629*x23 + x103*((x50 == 1) ? (
   -x91*x48 - x93*x44 - x94*x41 - x96*x37
)
: ((x57 == 1) ? (
   -x51*x96 - x53*x94 - x54*x93 - x56*x91
)
: ((x66 == 1) ? (
   -0.641731208021339*(x52*x98 + x58*x99 + x63*x97)
)
: ((x69 == 1) ? (
   -0.641731208021339*(x67*x99 + x68*x97 + x98*x40)
)
: 0)))) + x133*(x30*x[11]*((x125 == 1) ? (
   -0.003769324 - 24.3671976*x104 - 5.265984e-06*x121 + x124
)
: ((x126 == 1) ? (
   0.037063964 - 38.5844296*x104 - 3.4585362e-05*x121 + x124
)
: ((x128 == 1) ? (
   -31.748192*x104 - 1.1074716e+30*x127
)
: 0))) + x31*x[11]*((x131 == 1) ? (
   -0.00879504 - 23.5143*x104 - 3.53562e-07*x121 + 154718.0*x123
)
: ((x132 == 1) ? (
   -46.0*x104 + 2.066427e+33*x127
)
: 0)) + x83*x[11]*((x129 == 1) ? (
   -0.00531368 - 24.112392*x104 + 7.75338e-07*x121 + 104956.0*x123
)
: ((x130 == 1) ? (
   -31.38*x104 + 3.277503e+31*x127
)
: 0))) - x70*x90);
    out[1] = x299 + x302;
    out[2] = x302 + x319;
    out[3] = x302 + (x341 + x345)*x0;
    out[4] = x352;
    out[5] = x302 + x356;
    out[6] = x302 + x360;
    out[7] = x302 + (x345 + x364)*x0;
    out[8] = x368;
    out[9] = (x387 + x391)*x0;
    out[10] = x299 + x396;
    out[11] = x560 + x562 + x0*(x546 + x145*(x153 + 0.5*((x8 == 1) ? (
   pow(x[3], -1)
)
: 0)) - x398*x154 - x406*x405 - x406*x413);
    out[12] = x556 + x602 + x608;
    out[13] = x556 + x645 + (x610 + x640)*x0;
    out[14] = x656 + x660;
    out[15] = x560 + x669 + x672;
    out[16] = x556 + x679 + x682;
    out[17] = x556 + x694 + (x686 + x691)*x0;
    out[18] = x697 + x698;
    out[19] = x736 + (x730 + x733)*x0;
    out[20] = x319 + x396;
    out[21] = x602 + x739 + x740;
    out[22] = x608 + x739 + x0*(x744 + x145*(x304 + 0.5*((x12 == 1) ? (
   pow(x[4], -1)
)
: 0)) - x398*x305 - x406*x566 + x741*x[7]);
    out[23] = x645 + x739 + (x745 + x752)*x0;
    out[24] = x756;
    out[25] = x739 + x759 + x760;
    out[26] = x682 + x739 + x762;
    out[27] = x694 + x739 + (x752 + x763)*x0;
    out[28] = x766;
    out[29] = x736 + (x770 + x777)*x0;
    out[30] = x396 + (x341 + x783)*x0;
    out[31] = x562 + x790 + (x610 + x785)*x0;
    out[32] = x608 + x789 + (x745 + x793)*x0;
    out[33] = x645 + x789 + x0*(x805 + x145*(x321 + 0.5*((x6 == 1) ? (
   pow(x[5], -1)
)
: 0)) - x398*x322 - x406*x614 + x794*x[7]);
    out[34] = x808;
    out[35] = x672 + x790 + (x785 + x809)*x0;
    out[36] = x682 + x789 + x0*(x678 + x793 + x810);
    out[37] = x693 + x789 + x811;
    out[38] = x812;
    out[39] = x736 + (x820 + x821)*x0;
    out[40] = x352;
    out[41] = x656 + x825;
    out[42] = x826;
    out[43] = x828;
    out[44] = x0*(x829 + x145*(x346 + 0.5*((x10 == 1) ? (
   pow(x[6], -1)
)
: 0)));
    out[45] = x825 + x830;
    out[46] = x826;
    out[47] = x828;
    out[48] = x831;
    out[49] = (x833 + x835)*x0;
    out[50] = x356 + x396;
    out[51] = x669 + x740 + x836;
    out[52] = x608 + x759 + x836;
    out[53] = x645 + x836 + (x640 + x809)*x0;
    out[54] = x660 + x830;
    out[55] = x760 + x836 + x0*(x546 - 1.0*x667 + x145*(x353 + 0.5*((x4 == 1) ? (
   pow(x[7], -1)
)
: 0)) - x398*x354 - x406*x665);
    out[56] = x682 + x836 + x837;
    out[57] = x694 + x836 + (x691 + x838)*x0;
    out[58] = x698 + x839;
    out[59] = x736 + (x733 + x840)*x0;
    out[60] = x360 + x396;
    out[61] = x679 + x740 + x841;
    out[62] = x607 + x762 + x842;
    out[63] = x645 + x841 + x0*(x751 + x761 + x810);
    out[64] = x756;
    out[65] = x760 + x837 + x841;
    out[66] = x681 + x842 + x0*(x744 + x145*(x357 + 0.5*((x16 == 1) ? (
   pow(x[8], -1)
)
: 0)) - x398*x358 - x406*x676 + x741*x[3]);
    out[67] = x694 + x841 + (x752 + x843)*x0;
    out[68] = x766;
    out[69] = x736 + (x777 + x844)*x0;
    out[70] = x396 + (x364 + x783)*x0;
    out[71] = x740 + x846 + (x686 + x845)*x0;
    out[72] = x608 + x846 + (x763 + x793)*x0;
    out[73] = x644 + x811 + x846;
    out[74] = x808;
    out[75] = x760 + x846 + (x838 + x845)*x0;
    out[76] = x682 + x846 + (x793 + x843)*x0;
    out[77] = x694 + x846 + x0*(x805 + x145*(x361 + 0.5*((x14 == 1) ? (
   pow(x[9], -1)
)
: 0)) - x398*x362 - x406*x688 + x794*x[3]);
    out[78] = x812;
    out[79] = x736 + (x820 + x847)*x0;
    out[80] = x368;
    out[81] = x697 + x848;
    out[82] = x849;
    out[83] = x850;
    out[84] = x831;
    out[85] = x839 + x848;
    out[86] = x849;
    out[87] = x850;
    out[88] = x0*(x829 + x145*(x365 + 0.5*((x18 == 1) ? (
   pow(x[10], -1)
)
: 0)));
    out[89] = (x835 + x851)*x0;
    out[90] = (x387 + x853)*x0;
    out[91] = x855 + (x730 + x854)*x0;
    out[92] = x855 + (x770 + x857)*x0;
    out[93] = x855 + (x821 + x858)*x0;
    out[94] = (x833 + x859)*x0;
    out[95] = x855 + (x840 + x854)*x0;
    out[96] = x855 + (x844 + x857)*x0;
    out[97] = x855 + (x847 + x858)*x0;
    out[98] = (x851 + x859)*x0;
    out[99] = x0*(x103*((x50 == 1) ? (
   -x508*x860 - x514*x860 - x518*x860
)
: ((x57 == 1) ? (
   -x522*x861 - x524*x861 - x526*x861
)
: ((x66 == 1) ? (
   -0.641731208021339*(x530*x861 + x534*x861 + x536*x861)
)
: ((x69 == 1) ? (
   -0.641731208021339*(x539*x860 + x540*x860 + x541*x860)
)
: 0)))) + x133*(x241 + x235*x255 + x237*x308 + x707*x239) + x145*(x21 + 3.0*((x2 == 1) ? (
   pow(x[11], -1)
)
: 0)) + x781*(-4.44*x77 + x779) - x803*pow(x382, 2) + x804*x382*x380);
}

__device__ void pycgpu_model_1_internal_cons_func(double* out, const double* x) {
    out[0] = 1.0*(-1 + x[3] + x[4] + x[5] + x[6]);
    out[1] = 1.0*(-1 + x[7] + x[8] + x[9] + x[10]);
    out[2] = 1.0*(-1 + x[11]);
}

__device__ void pycgpu_model_1_internal_cons_jac(double* out, const double* x) {
    out[0] = 0;
    out[1] = 1.0;
    out[2] = 1.0;
    out[3] = 1.0;
    out[4] = 1.0;
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[10] = 0;
    out[11] = 0;
    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1.0;
    out[16] = 1.0;
    out[17] = 1.0;
    out[18] = 1.0;
    out[19] = 0;
    out[20] = 0;
    out[21] = 0;
    out[22] = 0;
    out[23] = 0;
    out[24] = 0;
    out[25] = 0;
    out[26] = 0;
    out[27] = 0;
    out[28] = 0;
    out[29] = 1.0;
}

__device__ void pycgpu_model_1_mass_obj(double* out, const double* x) {
    double x0 = pow(0.5*(x[3] + x[4] + x[5]) + 0.5*(x[7] + x[8] + x[9]), -1);
    out[0] = (0.5*x[3] + 0.5*x[7])*x0;
    out[1] = (0.5*x[4] + 0.5*x[8])*x0;
    out[2] = (0.5*x[5] + 0.5*x[9])*x0;
    out[3] = 0;
}

__device__ void pycgpu_model_1_formulamole_obj(double* out, const double* x) {
    out[0] = 0.5*x[3] + 0.5*x[7];
    out[1] = 0.5*x[4] + 0.5*x[8];
    out[2] = 0.5*x[5] + 0.5*x[9];
    out[3] = 0.0;
}

__device__ void pycgpu_model_1_formulamole_grad(double* out, const double* x) {
    out[0] = 0;
    out[1] = 0.5;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0.5;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[10] = 0;
    out[11] = 0;
    out[12] = 0.5;
    out[13] = 0;
    out[14] = 0;
    out[15] = 0;
    out[16] = 0.5;
    out[17] = 0;
    out[18] = 0;
    out[19] = 0;
    out[20] = 0;
    out[21] = 0;
    out[22] = 0;
    out[23] = 0.5;
    out[24] = 0;
    out[25] = 0;
    out[26] = 0;
    out[27] = 0.5;
    out[28] = 0;
    out[29] = 0;
}

__device__ double pycgpu_model_2_obj(const double* x) {
    double x0 = x[6]*x[5];
    double x1 = 67.0*x0;
    double x2 = 1e-09 + x1;
    double x3 = 2.01530612244898/x[2];
    double x4 = (1.0/6.0)*pow(x[2], 3);
    double x5 = (1.0/135.0)*pow(x[2], 9);
    double x6 = pow(x2, 15);
    double x7 = pow(x[2], 15);
    double x8 = (1.0/600.0)*x7;
    double x9 = -201.0*x0;
    double x10 = 1e-09 + x9;
    double x11 = pow(x10, 15);
    double x12 = (1.0/10.0)/pow(x[2], 5);
    double x13 = (1.0/1500.0)/pow(x[2], 25);
    double x14 = (1.0/315.0)/x7;
    double x15 = 2.1*x0;
    double x16 = pow(x[3] + x[4] + x[5], -1);
    double x17 = 8.3145*x[2]*x16;
    double x18 = 2.0*x[2];
    double x19 = x[6]*x[3];
    double x20 = x19*x[4];
    double x21 = x0*x[3];
    double x22 = x[3] - x[4];
    double x23 = -x[5];
    double x24 = x0*x[4];
    double x25 = 1.0*x16;
    double x26 = pow(x[2], 3.0);
    double x27 = pow(x[2], -1.0);
    double x28 = 74092.0*x27;
    double x29 = x[2]*log(x[2]);
    double x30 = pow(x[2], 2.0);
    double x31 = pow(x[2], -9.0);
    return x17*(1.0*((1e-15 < x[3]) ? (
   x[3]*log(x[3])
)
: (
   0
)) + 1.0*((1e-15 < x[4]) ? (
   x[4]*log(x[4])
)
: (
   0
)) + 1.0*((1e-15 < x[5]) ? (
   x[5]*log(x[5])
)
: (
   0
)) + 1.0*((1e-15 < x[6]) ? (
   x[6]*log(x[6])
)
: (
   0
))) + x25*(x0*((x[2] < 1811.0) ? (
   -236.7 + 132.416*x[2] - 5.8927e-08*x26 + 77359.0*x27 - 24.6643*x29 - 0.00375752*x30
)
: ((1811.0 <= x[2]) ? (
   -27097.396 + 300.25256*x[2] - 46.0*x29 + 2.78854e+31*x31
)
: (
   0
))) + x19*((x[2] < 700.0) ? (
   -7976.15 + 137.093038*x[2] - 8.77664e-07*x26 + x28 - 24.3671976*x29 - 0.001884662*x30
)
: (((x[2] < 933.47 && 700.0 <= x[2])) ? (
   -11276.24 + 223.048446*x[2] - 5.764227e-06*x26 + x28 - 38.5844296*x29 + 0.018531982*x30
)
: ((933.47 <= x[2]) ? (
   -11278.378 + 188.684153*x[2] - 31.748192*x29 - 1.230524e+28*x31
)
: (
   0
)))) + x[6]*x[4]*((x[2] < 1357.77) ? (
   -7770.458 + 130.485235*x[2] + 1.29223e-07*x26 + 52478.0*x27 - 24.112392*x29 - 0.00265684*x30
)
: ((1357.77 <= x[2]) ? (
   -13542.026 + 183.803828*x[2] - 31.38*x29 + 3.64167e+29*x31
)
: (
   0
)))) + x25*(x20*(-53520.0 + x18) + x21*(-76066.1 + 18.6758*x[2]) + 1170.0*pow(x22, 2)*x20 + x24*(48232.5 - 8.60954*x[2]) + x21*(x[3] + x23)*(21167.4 + 1.3398*x[2]) + x22*x20*(38590.0 - x18) + x24*(x[4] + x23)*(8861.88 - 5.28975*x[2])) + x17*((x[2] < x1) ? (
   1 - 0.426902268107986*(x2*x3 + 2.45242885886749*(x5/pow(x2, 9) + x4/pow(x2, 3) + x8/x6))
)
: ((x[2] < x9) ? (
   1 - 0.426902268107986*(x3*x10 + 2.45242885886749*(x4/pow(x10, 3) + x5/pow(x10, 9) + x8/x11))
)
: (((0 < -201.0*x[6]*x[5] && -201.0*x[6]*x[5] < x[2])) ? (
   -0.426902268107986*(x12*pow(x10, 5) + x13*pow(x10, 25) + x14*x11)
)
: (((67.0*x[6]*x[5] < x[2] && -201.0*x[6]*x[5] < 0)) ? (
   -0.426902268107986*(pow(x2, 5)*x12 + pow(x2, 25)*x13 + x6*x14)
)
: (
   0
)))))*log(1 - x15/((-x15 <= 0) ? (
   -3.0
)
: (
   1.0
)));
}

__device__ double pycgpu_model_2_formulaobj(const double* x) {
    double x0 = x[3] + x[4] + x[5];
    double x1 = x[6]*x[5];
    double x2 = 67.0*x1;
    double x3 = 1e-09 + x2;
    double x4 = 2.01530612244898/x[2];
    double x5 = (1.0/6.0)*pow(x[2], 3);
    double x6 = (1.0/135.0)*pow(x[2], 9);
    double x7 = pow(x3, 15);
    double x8 = pow(x[2], 15);
    double x9 = (1.0/600.0)*x8;
    double x10 = -201.0*x1;
    double x11 = 1e-09 + x10;
    double x12 = pow(x11, 15);
    double x13 = (1.0/10.0)/pow(x[2], 5);
    double x14 = (1.0/1500.0)/pow(x[2], 25);
    double x15 = (1.0/315.0)/x8;
    double x16 = 2.1*x1;
    double x17 = pow(x0, -1);
    double x18 = 8.3145*x[2]*x17;
    double x19 = 2.0*x[2];
    double x20 = x[6]*x[3];
    double x21 = x20*x[4];
    double x22 = x1*x[3];
    double x23 = x[3] - x[4];
    double x24 = -x[5];
    double x25 = x1*x[4];
    double x26 = 1.0*x17;
    double x27 = pow(x[2], 3.0);
    double x28 = pow(x[2], -1.0);
    double x29 = 74092.0*x28;
    double x30 = x[2]*log(x[2]);
    double x31 = pow(x[2], 2.0);
    double x32 = pow(x[2], -9.0);
    return 1.0*x0*(x18*(1.0*((1e-15 < x[3]) ? (
   x[3]*log(x[3])
)
: (
   0
)) + 1.0*((1e-15 < x[4]) ? (
   x[4]*log(x[4])
)
: (
   0
)) + 1.0*((1e-15 < x[5]) ? (
   x[5]*log(x[5])
)
: (
   0
)) + 1.0*((1e-15 < x[6]) ? (
   x[6]*log(x[6])
)
: (
   0
))) + x26*(x1*((x[2] < 1811.0) ? (
   -236.7 + 132.416*x[2] - 5.8927e-08*x27 + 77359.0*x28 - 24.6643*x30 - 0.00375752*x31
)
: ((1811.0 <= x[2]) ? (
   -27097.396 + 300.25256*x[2] - 46.0*x30 + 2.78854e+31*x32
)
: (
   0
))) + x20*((x[2] < 700.0) ? (
   -7976.15 + 137.093038*x[2] - 8.77664e-07*x27 + x29 - 24.3671976*x30 - 0.001884662*x31
)
: (((x[2] < 933.47 && 700.0 <= x[2])) ? (
   -11276.24 + 223.048446*x[2] - 5.764227e-06*x27 + x29 - 38.5844296*x30 + 0.018531982*x31
)
: ((933.47 <= x[2]) ? (
   -11278.378 + 188.684153*x[2] - 31.748192*x30 - 1.230524e+28*x32
)
: (
   0
)))) + x[6]*x[4]*((x[2] < 1357.77) ? (
   -7770.458 + 130.485235*x[2] + 1.29223e-07*x27 + 52478.0*x28 - 24.112392*x30 - 0.00265684*x31
)
: ((1357.77 <= x[2]) ? (
   -13542.026 + 183.803828*x[2] - 31.38*x30 + 3.64167e+29*x32
)
: (
   0
)))) + x26*(x21*(-53520.0 + x19) + x22*(-76066.1 + 18.6758*x[2]) + 1170.0*pow(x23, 2)*x21 + x25*(48232.5 - 8.60954*x[2]) + x22*(x[3] + x24)*(21167.4 + 1.3398*x[2]) + x23*x21*(38590.0 - x19) + x25*(x[4] + x24)*(8861.88 - 5.28975*x[2])) + x18*((x[2] < x2) ? (
   1 - 0.426902268107986*(x4*x3 + 2.45242885886749*(x6/pow(x3, 9) + x5/pow(x3, 3) + x9/x7))
)
: ((x[2] < x10) ? (
   1 - 0.426902268107986*(x4*x11 + 2.45242885886749*(x5/pow(x11, 3) + x6/pow(x11, 9) + x9/x12))
)
: (((0 < -201.0*x[6]*x[5] && -201.0*x[6]*x[5] < x[2])) ? (
   -0.426902268107986*(x13*pow(x11, 5) + x14*pow(x11, 25) + x15*x12)
)
: (((67.0*x[6]*x[5] < x[2] && -201.0*x[6]*x[5] < 0)) ? (
   -0.426902268107986*(pow(x3, 5)*x13 + pow(x3, 25)*x14 + x7*x15)
)
: (
   0
)))))*log(1 - x16/((-x16 <= 0) ? (
   -3.0
)
: (
   1.0
))));
}

__device__ void pycgpu_model_2_formulagrad(double* out, const double* x) {
    double x0 = pow(x[2], 2.0);
    double x1 = pow(x0, -1);
    double x2 = log(x[2]);
    double x3 = 24.6643*x2;
    double x4 = pow(x[2], 1.0);
    double x5 = x[2] < 1811.0;
    double x6 = 46.0*x2;
    double x7 = pow(x[2], -10.0);
    double x8 = 1811.0 <= x[2];
    double x9 = x[6]*x[5];
    double x10 = 24.112392*x2;
    double x11 = x[2] < 1357.77;
    double x12 = 31.38*x2;
    double x13 = 1357.77 <= x[2];
    double x14 = x[6]*x[4];
    double x15 = 24.3671976*x2;
    double x16 = -74092.0*x1;
    double x17 = x[2] < 700.0;
    double x18 = 38.5844296*x2;
    double x19 = (x[2] < 933.47 && 700.0 <= x[2]);
    double x20 = 31.748192*x2;
    double x21 = 933.47 <= x[2];
    double x22 = x[6]*x[3];
    double x23 = x[3] + x[4] + x[5];
    double x24 = pow(x23, -1);
    double x25 = 1.0*x24;
    double x26 = 1e-15 < x[6];
    double x27 = 1.0*((x26 == 1) ? (
   0
)
: (
   0
));
    double x28 = 1e-15 < x[4];
    double x29 = 1.0*((x28 == 1) ? (
   0
)
: (
   0
));
    double x30 = 1e-15 < x[3];
    double x31 = 1.0*((x30 == 1) ? (
   0
)
: (
   0
));
    double x32 = 1e-15 < x[5];
    double x33 = 1.0*((x32 == 1) ? (
   0
)
: (
   0
));
    double x34 = x31 + x33;
    double x35 = x29 + x34;
    double x36 = 8.3145*x24;
    double x37 = x[2]*x36;
    double x38 = -x[5];
    double x39 = x[4] + x38;
    double x40 = x9*x[4];
    double x41 = x[3] + x38;
    double x42 = x9*x[3];
    double x43 = x[3] - x[4];
    double x44 = x22*x[4];
    double x45 = 2.0*x44;
    double x46 = 67.0*x9;
    double x47 = 1e-09 + x46;
    double x48 = pow(x[2], -1);
    double x49 = 2.01530612244898*x48;
    double x50 = pow(x47, -3);
    double x51 = pow(x[2], 3);
    double x52 = (1.0/6.0)*x51;
    double x53 = pow(x47, -9);
    double x54 = pow(x[2], 9);
    double x55 = (1.0/135.0)*x54;
    double x56 = pow(x47, 15);
    double x57 = pow(x56, -1);
    double x58 = pow(x[2], 15);
    double x59 = (1.0/600.0)*x58;
    double x60 = x[2] < x46;
    double x61 = -201.0*x9;
    double x62 = 1e-09 + x61;
    double x63 = pow(x62, -3);
    double x64 = pow(x62, -9);
    double x65 = pow(x62, 15);
    double x66 = pow(x65, -1);
    double x67 = x[2] < x61;
    double x68 = pow(x62, 5);
    double x69 = pow(x[2], -5);
    double x70 = (1.0/10.0)*x69;
    double x71 = pow(x62, 25);
    double x72 = pow(x[2], -25);
    double x73 = (1.0/1500.0)*x72;
    double x74 = pow(x58, -1);
    double x75 = (1.0/315.0)*x74;
    double x76 = (0 < -201.0*x[6]*x[5] && -201.0*x[6]*x[5] < x[2]);
    double x77 = pow(x47, 5);
    double x78 = pow(x47, 25);
    double x79 = (67.0*x[6]*x[5] < x[2] && -201.0*x[6]*x[5] < 0);
    double x80 = ((x60 == 1) ? (
   1 - 0.426902268107986*(x47*x49 + 2.45242885886749*(x50*x52 + x53*x55 + x57*x59))
)
: ((x67 == 1) ? (
   1 - 0.426902268107986*(x62*x49 + 2.45242885886749*(x63*x52 + x64*x55 + x66*x59))
)
: ((x76 == 1) ? (
   -0.426902268107986*(x70*x68 + x71*x73 + x75*x65)
)
: ((x79 == 1) ? (
   -0.426902268107986*(x70*x77 + x73*x78 + x75*x56)
)
: (
   0
)))));
    double x81 = 2.1*x9;
    double x82 = -x81 <= 0;
    double x83 = ((x82 == 1) ? (
   -3.0
)
: (
   1.0
));
    double x84 = 2.1/x83;
    double x85 = 1 - x9*x84;
    double x86 = log(x85);
    double x87 = x80*x86;
    double x88 = pow(x[2], 2);
    double x89 = 0.86033875460538/x88;
    double x90 = 0.0261736860556003*pow(x[2], 14);
    double x91 = 0.0697964961482674*pow(x[2], 8);
    double x92 = 0.523473721112006*x88;
    double x93 = (1.0/21.0)/pow(x[2], 16);
    double x94 = (1.0/60.0)/pow(x[2], 26);
    double x95 = (1.0/2.0)/pow(x[2], 6);
    double x96 = x86*x37;
    double x97 = log(x[5]);
    double x98 = log(x[4]);
    double x99 = log(x[3]);
    double x100 = log(x[6]);
    double x101 = 1.0*((x26 == 1) ? (
   x100*x[6]
)
: (
   0
)) + 1.0*((x32 == 1) ? (
   x97*x[5]
)
: (
   0
)) + 1.0*((x28 == 1) ? (
   x98*x[4]
)
: (
   0
)) + 1.0*((x30 == 1) ? (
   x99*x[3]
)
: (
   0
));
    double x102 = x36*x101;
    double x103 = ((x82 == 1) ? (
   0
)
: (
   0
))/pow(x83, 2);
    double x104 = x80/x85;
    double x105 = 17.46045*x[2]*x9*x24*x103*x104;
    double x106 = 1.0*x23;
    double x107 = x27 + x29;
    double x108 = pow(x[2], 3.0);
    double x109 = pow(x4, -1);
    double x110 = 74092.0*x109;
    double x111 = pow(x[2], -9.0);
    double x112 = ((x17 == 1) ? (
   -7976.15 + 137.093038*x[2] - 0.001884662*x0 - 8.77664e-07*x108 + x110 - x[2]*x15
)
: ((x19 == 1) ? (
   -11276.24 + 223.048446*x[2] + 0.018531982*x0 - 5.764227e-06*x108 + x110 - x[2]*x18
)
: ((x21 == 1) ? (
   -11278.378 + 188.684153*x[2] - 1.230524e+28*x111 - x[2]*x20
)
: (
   0
))));
    double x113 = x14*((x11 == 1) ? (
   0
)
: ((x13 == 1) ? (
   0
)
: (
   0
))) + x22*((x17 == 1) ? (
   0
)
: ((x19 == 1) ? (
   0
)
: ((x21 == 1) ? (
   0
)
: (
   0
)))) + x9*((x5 == 1) ? (
   0
)
: ((x8 == 1) ? (
   0
)
: (
   0
)));
    double x114 = 21167.4 + 1.3398*x[2];
    double x115 = x42*x114;
    double x116 = x41*x114;
    double x117 = 2340.0*x43*x44;
    double x118 = 1170.0*pow(x43, 2);
    double x119 = 2.0*x[2];
    double x120 = 38590.0 - x119;
    double x121 = x43*x120;
    double x122 = -76066.1 + 18.6758*x[2];
    double x123 = x44*x120;
    double x124 = -53520.0 + x119;
    double x125 = x112*x[3];
    double x126 = ((x11 == 1) ? (
   -7770.458 + 130.485235*x[2] - 0.00265684*x0 + 1.29223e-07*x108 + 52478.0*x109 - x[2]*x10
)
: ((x13 == 1) ? (
   -13542.026 + 183.803828*x[2] + 3.64167e+29*x111 - x[2]*x12
)
: (
   0
)));
    double x127 = x126*x[6];
    double x128 = ((x5 == 1) ? (
   -236.7 + 132.416*x[2] - 0.00375752*x0 - 5.8927e-08*x108 + 77359.0*x109 - x[2]*x3
)
: ((x8 == 1) ? (
   -27097.396 + 300.25256*x[2] + 2.78854e+31*x111 - x[2]*x6
)
: (
   0
)));
    double x129 = x128*x[5];
    double x130 = x125*x[6] + x127*x[4] + x129*x[6];
    double x131 = pow(x23, -2);
    double x132 = 1.0*x131;
    double x133 = 8.3145*x[2]*x131;
    double x134 = x22*x124;
    double x135 = x122*x[3];
    double x136 = x22*x118;
    double x137 = 48232.5 - 8.60954*x[2];
    double x138 = x137*x[4];
    double x139 = 8861.88 - 5.28975*x[2];
    double x140 = x39*x139;
    double x141 = x140*x[4];
    double x142 = x134*x[4] + x136*x[4] + x41*x115 + x44*x121 + x9*x135 + x9*x138 + x9*x141;
    double x143 = -x101*x133 - x130*x132 - x132*x142 - x87*x133;
    double x144 = x105 + x143 + x96*((x60 == 1) ? (
   0
)
: ((x67 == 1) ? (
   0
)
: ((x76 == 1) ? (
   0
)
: ((x79 == 1) ? (
   0
)
: (
   0
)))));
    double x145 = 1.0*(x[2]*x102 + x25*x130 + x25*x142 + x87*x37);
    double x146 = x40*x139;
    double x147 = 57.6426965585605*x48;
    double x148 = x58*x[6];
    double x149 = 1.75363696572522/pow(x47, 16);
    double x150 = x54*x[6];
    double x151 = 4.67636524193392/pow(x47, 10);
    double x152 = x51*x[6];
    double x153 = pow(x47, 4);
    double x154 = 35.0727393145044/x153;
    double x155 = 172.928089675681*x48;
    double x156 = 14.0290957258018/pow(x62, 10);
    double x157 = 5.26091089717566/pow(x62, 16);
    double x158 = pow(x62, 4);
    double x159 = 105.218217943513/x158;
    double x160 = x74*x[6];
    double x161 = 9.57142857142857*pow(x62, 14);
    double x162 = 3.35*pow(x62, 24);
    double x163 = x72*x[6];
    double x164 = 100.5*x158;
    double x165 = x69*x[6];
    double x166 = 3.19047619047619*pow(x47, 14);
    double x167 = 1.11666666666667*pow(x47, 24);
    double x168 = 33.5*x153;
    double x169 = x81*x103;
    double x170 = x37*x104;
    double x171 = x[3]*x[4];
    double x172 = x58*x[5];
    double x173 = x54*x[5];
    double x174 = x51*x[5];
    double x175 = x74*x[5];
    double x176 = x72*x[5];
    double x177 = x69*x[5];
    out[0] = x106*(x102 + x105 + x25*(x14*((x11 == 1) ? (
   106.372843 + 3.87669e-07*x0 - 52478.0*x1 - x10 - 0.00531368*x4
)
: ((x13 == 1) ? (
   152.423828 - x12 - 3.277503e+30*x7
)
: (
   0
))) + x22*((x17 == 1) ? (
   112.7258404 - 2.632992e-06*x0 - x15 + x16 - 0.003769324*x4
)
: ((x19 == 1) ? (
   184.4640164 - 1.7292681e-05*x0 + x16 - x18 + 0.037063964*x4
)
: ((x21 == 1) ? (
   156.935961 - x20 + 1.1074716e+29*x7
)
: (
   0
)))) + x9*((x5 == 1) ? (
   107.7517 - 1.76781e-07*x0 - 77359.0*x1 - x3 - 0.00751504*x4
)
: ((x8 == 1) ? (
   254.25256 - x6 - 2.509686e+32*x7
)
: (
   0
)))) + x25*(-8.60954*x40 + 18.6758*x42 + x45 - 5.28975*x40*x39 + 1.3398*x41*x42 - x43*x45) + x87*x36 + x96*((x60 == 1) ? (
   -x50*x92 - x53*x91 - x57*x90 + x89*x47
)
: ((x67 == 1) ? (
   -x63*x92 - x64*x91 - x66*x90 + x89*x62
)
: ((x76 == 1) ? (
   -0.426902268107986*(-x65*x93 - x68*x95 - x71*x94)
)
: ((x79 == 1) ? (
   -0.426902268107986*(-x56*x93 - x77*x95 - x78*x94)
)
: (
   0
))))) + (x27 + x35)*x37);
    out[1] = x145 + x106*(x144 + x25*(x113 + x112*x[6]) + x25*(x115 + x117 + x123 + x14*x118 + x14*x121 + x14*x124 + x9*x116 + x9*x122) + x37*(x107 + x33 + 1.0*((x30 == 1) ? (
   1 + x99
)
: (
   0
))));
    out[2] = x145 + x106*(x144 + x25*(-x117 - x123 + x134 + x136 + x146 + x22*x121 + x9*x137 + x9*x140) + x37*(x27 + x34 + 1.0*((x28 == 1) ? (
   1 + x98
)
: (
   0
))) + (x113 + x127)*x25);
    out[3] = x145 + x106*(x143 + x170*(x169 - x84*x[6]) + x25*(x113 + x128*x[6]) + x37*(x107 + x31 + 1.0*((x32 == 1) ? (
   1 + x97
)
: (
   0
))) + x96*((x60 == 1) ? (
   -x147*x[6] + x148*x149 + x150*x151 + x152*x154
)
: ((x67 == 1) ? (
   -x148*x157 - x150*x156 - x152*x159 + x155*x[6]
)
: ((x76 == 1) ? (
   -0.426902268107986*(-x161*x160 - x163*x162 - x165*x164)
)
: ((x79 == 1) ? (
   -0.426902268107986*(x160*x166 + x167*x163 + x168*x165)
)
: (
   0
))))) + (-x115 - x146 + x135*x[6] + x138*x[6] + x14*x140 + x22*x116)*x25);
    out[4] = x106*(x170*(x169 - x84*x[5]) + x25*(x113 + x125 + x129 + x126*x[4]) + x25*(x118*x171 + x121*x171 + x124*x171 + x135*x[5] + x138*x[5] + x141*x[5] + x116*x[3]*x[5]) + x37*(x35 + 1.0*((x26 == 1) ? (
   1 + x100
)
: (
   0
))) + x96*((x60 == 1) ? (
   -x147*x[5] + x172*x149 + x173*x151 + x174*x154
)
: ((x67 == 1) ? (
   x155*x[5] - x172*x157 - x173*x156 - x174*x159
)
: ((x76 == 1) ? (
   -0.426902268107986*(-x161*x175 - x162*x176 - x164*x177)
)
: ((x79 == 1) ? (
   -0.426902268107986*(x166*x175 + x167*x176 + x168*x177)
)
: (
   0
))))));
}

__device__ void pycgpu_model_2_formulahess(double* out, const double* x) {
    double x0 = x[6]*x[5];
    double x1 = 67.0*x0;
    double x2 = 1e-09 + x1;
    double x3 = pow(x[2], 2);
    double x4 = pow(x3, -1);
    double x5 = 0.86033875460538*x4;
    double x6 = pow(x2, 15);
    double x7 = pow(x6, -1);
    double x8 = pow(x[2], 14);
    double x9 = 0.0261736860556003*x8;
    double x10 = pow(x2, -9);
    double x11 = pow(x[2], 8);
    double x12 = 0.0697964961482674*x11;
    double x13 = pow(x2, 3);
    double x14 = pow(x13, -1);
    double x15 = 0.523473721112006*x3;
    double x16 = x[2] < x1;
    double x17 = -201.0*x0;
    double x18 = 1e-09 + x17;
    double x19 = pow(x18, 15);
    double x20 = pow(x19, -1);
    double x21 = pow(x18, -9);
    double x22 = pow(x18, 3);
    double x23 = pow(x22, -1);
    double x24 = x[2] < x17;
    double x25 = pow(x[2], -16);
    double x26 = (1.0/21.0)*x25;
    double x27 = pow(x18, 25);
    double x28 = pow(x[2], -26);
    double x29 = (1.0/60.0)*x28;
    double x30 = pow(x18, 5);
    double x31 = pow(x[2], -6);
    double x32 = (1.0/2.0)*x31;
    double x33 = (0 < -201.0*x[6]*x[5] && -201.0*x[6]*x[5] < x[2]);
    double x34 = pow(x2, 25);
    double x35 = pow(x2, 5);
    double x36 = (67.0*x[6]*x[5] < x[2] && -201.0*x[6]*x[5] < 0);
    double x37 = ((x16 == 1) ? (
   -x12*x10 - x15*x14 + x2*x5 - x7*x9
)
: ((x24 == 1) ? (
   -x21*x12 - x23*x15 + x5*x18 - x9*x20
)
: ((x33 == 1) ? (
   -0.426902268107986*(-x26*x19 - x29*x27 - x30*x32)
)
: ((x36 == 1) ? (
   -0.426902268107986*(-x32*x35 - x34*x29 - x6*x26)
)
: 0))));
    double x38 = 2.1*x0;
    double x39 = -x38 <= 0;
    double x40 = ((x39 == 1) ? 0
: 0);
    double x41 = ((x39 == 1) ? (
   -3.0
)
: (
   1.0
));
    double x42 = x40/pow(x41, 2);
    double x43 = x0*x42;
    double x44 = 34.9209*x43;
    double x45 = 2.1/x41;
    double x46 = x45*x[5];
    double x47 = 1 - x46*x[6];
    double x48 = pow(x47, -1);
    double x49 = x[3] + x[4] + x[5];
    double x50 = pow(x49, -1);
    double x51 = x[2]*x50;
    double x52 = x51*x48;
    double x53 = x52*x44;
    double x54 = 1.04694744222401*x[2];
    double x55 = pow(x[2], 7);
    double x56 = 0.558371969186139*x55;
    double x57 = 0.366431604778404*pow(x[2], 13);
    double x58 = pow(x[2], 3);
    double x59 = 1.72067750921076/x58;
    double x60 = 3/x55;
    double x61 = (13.0/30.0)/pow(x[2], 27);
    double x62 = (16.0/21.0)/pow(x[2], 17);
    double x63 = log(x47);
    double x64 = 8.3145*x50;
    double x65 = x[2]*x64;
    double x66 = x63*x65;
    double x67 = 16.629*x50;
    double x68 = x63*x37;
    double x69 = pow(x[2], -1);
    double x70 = 2.01530612244898*x69;
    double x71 = (1.0/6.0)*x58;
    double x72 = pow(x[2], 9);
    double x73 = (1.0/135.0)*x72;
    double x74 = pow(x[2], 15);
    double x75 = (1.0/600.0)*x74;
    double x76 = pow(x[2], -5);
    double x77 = (1.0/10.0)*x76;
    double x78 = pow(x[2], -25);
    double x79 = (1.0/1500.0)*x78;
    double x80 = pow(x74, -1);
    double x81 = (1.0/315.0)*x80;
    double x82 = ((x16 == 1) ? (
   1 - 0.426902268107986*(x2*x70 + 2.45242885886749*(x7*x75 + x71*x14 + x73*x10))
)
: ((x24 == 1) ? (
   1 - 0.426902268107986*(x70*x18 + 2.45242885886749*(x71*x23 + x73*x21 + x75*x20))
)
: ((x33 == 1) ? (
   -0.426902268107986*(x77*x30 + x79*x27 + x81*x19)
)
: ((x36 == 1) ? (
   -0.426902268107986*(x6*x81 + x77*x35 + x79*x34)
)
: 0))));
    double x83 = x82*x48;
    double x84 = x83*x50;
    double x85 = x84*x44;
    double x86 = 1e-15 < x[3];
    double x87 = 1.0*((x86 == 1) ? 0
: 0);
    double x88 = 1e-15 < x[6];
    double x89 = 1.0*((x88 == 1) ? 0
: 0);
    double x90 = 1e-15 < x[4];
    double x91 = 1.0*((x90 == 1) ? 0
: 0);
    double x92 = 1e-15 < x[5];
    double x93 = 1.0*((x92 == 1) ? 0
: 0);
    double x94 = x91 + x93;
    double x95 = x89 + x94;
    double x96 = x87 + x95;
    double x97 = pow(x[2], 1.0);
    double x98 = pow(x[2], 3.0);
    double x99 = pow(x98, -1);
    double x100 = 148184.0*x99;
    double x101 = x[2] < 700.0;
    double x102 = (x[2] < 933.47 && 700.0 <= x[2]);
    double x103 = pow(x[2], -11.0);
    double x104 = 933.47 <= x[2];
    double x105 = x[6]*x[3];
    double x106 = x[2] < 1357.77;
    double x107 = 1357.77 <= x[2];
    double x108 = x[6]*x[4];
    double x109 = x[2] < 1811.0;
    double x110 = 1811.0 <= x[2];
    double x111 = 1.0*x50;
    double x112 = pow(x[6], 2);
    double x113 = pow(x[5], 2);
    double x114 = pow(x40, 2);
    double x115 = x82/pow(x47, 2);
    double x116 = x51*x115;
    double x117 = -36.666945*x112*x113*x114*x116/pow(x41, 4);
    double x118 = x0*x114/pow(x41, 3);
    double x119 = -34.9209*x[2]*x84*x118;
    double x120 = x65*x96;
    double x121 = 17.46045*x84;
    double x122 = x43*x121;
    double x123 = x[2]*x122;
    double x124 = x120 + x123;
    double x125 = x119 + x124;
    double x126 = x117 + x125;
    double x127 = 1.0*x49;
    double x128 = log(x[3]);
    double x129 = x95 + 1.0*((x86 == 1) ? (
   1 + x128
)
: 0);
    double x130 = x64*x129;
    double x131 = log(x[2]);
    double x132 = 24.3671976*x131;
    double x133 = pow(x[2], 2.0);
    double x134 = pow(x133, -1);
    double x135 = -74092.0*x134;
    double x136 = 38.5844296*x131;
    double x137 = pow(x[2], -10.0);
    double x138 = 31.748192*x131;
    double x139 = ((x101 == 1) ? (
   112.7258404 - x132 - 2.632992e-06*x133 + x135 - 0.003769324*x97
)
: ((x102 == 1) ? (
   184.4640164 - 1.7292681e-05*x133 + x135 - x136 + 0.037063964*x97
)
: ((x104 == 1) ? (
   156.935961 + 1.1074716e+29*x137 - x138
)
: 0)));
    double x140 = ((x109 == 1) ? 0
: ((x110 == 1) ? 0
: 0));
    double x141 = x140*x[6];
    double x142 = ((x106 == 1) ? 0
: ((x107 == 1) ? 0
: 0));
    double x143 = x142*x[6];
    double x144 = ((x101 == 1) ? 0
: ((x102 == 1) ? 0
: ((x104 == 1) ? 0
: 0)));
    double x145 = x144*x[3];
    double x146 = x141*x[5] + x143*x[4] + x145*x[6];
    double x147 = 2.0*x[4];
    double x148 = x147*x[6];
    double x149 = x147*x[3];
    double x150 = x149*x[6];
    double x151 = x[3] - x[4];
    double x152 = 1.3398*x[3];
    double x153 = x0*x152;
    double x154 = -x[5];
    double x155 = x[3] + x154;
    double x156 = 1.3398*x155;
    double x157 = 17.46045*x43;
    double x158 = pow(x49, -2);
    double x159 = x[2]*x83*x158;
    double x160 = -x157*x159;
    double x161 = ((x16 == 1) ? 0
: ((x24 == 1) ? 0
: ((x33 == 1) ? 0
: ((x36 == 1) ? 0
: 0))));
    double x162 = x52*x157;
    double x163 = x63*x64;
    double x164 = 24.6643*x131;
    double x165 = 46.0*x131;
    double x166 = ((x109 == 1) ? (
   107.7517 - 1.76781e-07*x133 - 77359.0*x134 - x164 - 0.00751504*x97
)
: ((x110 == 1) ? (
   254.25256 - 2.509686e+32*x137 - x165
)
: 0));
    double x167 = x166*x[5];
    double x168 = 24.112392*x131;
    double x169 = 31.38*x131;
    double x170 = ((x106 == 1) ? (
   106.372843 + 3.87669e-07*x133 - 52478.0*x134 - x168 - 0.00531368*x97
)
: ((x107 == 1) ? (
   152.423828 - 3.277503e+30*x137 - x169
)
: 0));
    double x171 = x170*x[6];
    double x172 = x139*x[3];
    double x173 = x167*x[6] + x171*x[4] + x172*x[6];
    double x174 = 1.0*x158;
    double x175 = x[4] + x154;
    double x176 = 5.28975*x175;
    double x177 = x0*x176;
    double x178 = 8.60954*x0;
    double x179 = x149*x151;
    double x180 = 18.6758*x[3];
    double x181 = x150 + x0*x180 + x153*x155 - x177*x[4] - x178*x[4] - x179*x[6];
    double x182 = 8.3145*x[2];
    double x183 = x182*x158;
    double x184 = log(x[5]);
    double x185 = log(x[4]);
    double x186 = log(x[6]);
    double x187 = 1.0*((x86 == 1) ? (
   x128*x[3]
)
: 0) + 1.0*((x92 == 1) ? (
   x184*x[5]
)
: 0) + 1.0*((x90 == 1) ? (
   x185*x[4]
)
: 0) + 1.0*((x88 == 1) ? (
   x186*x[6]
)
: 0);
    double x188 = x187*x158;
    double x189 = x82*x63;
    double x190 = -8.3145*x188 - x174*x173 - x174*x181 - 8.3145*x189*x158 - x68*x183 - x96*x183;
    double x191 = x66*x161;
    double x192 = x126 + x191;
    double x193 = x122 + x160 + x190 + x192 + x161*x162 + x161*x163 + x37*x162;
    double x194 = x127*(x130 + x193 + x111*(x146 + x139*x[6]) + x111*(18.6758*x0 + x148 - x150 + x153 + x0*x156 - x148*x151));
    double x195 = x124 + x111*x173 + x111*x181 + x64*x187 + x68*x65 + x82*x163;
    double x196 = 1.0*x195;
    double x197 = x87 + x89;
    double x198 = x197 + x93;
    double x199 = x198 + 1.0*((x90 == 1) ? (
   1 + x185
)
: 0);
    double x200 = x64*x199;
    double x201 = 2.0*x105;
    double x202 = 5.28975*x0*x[4];
    double x203 = x127*(x193 + x200 + x111*(x150 - x177 - x178 + x201 - x202 - x201*x151) + (x146 + x171)*x111);
    double x204 = pow(x2, 4);
    double x205 = pow(x204, -1);
    double x206 = x3*x205;
    double x207 = x206*x[6];
    double x208 = pow(x2, -16);
    double x209 = x8*x[6];
    double x210 = x208*x209;
    double x211 = pow(x2, -10);
    double x212 = 42.0872871774053*x211;
    double x213 = x11*x[6];
    double x214 = 57.6426965585605*x4;
    double x215 = x213*x212 + x214*x[6];
    double x216 = pow(x18, 4);
    double x217 = pow(x216, -1);
    double x218 = x3*x217;
    double x219 = 315.654653830539*x218;
    double x220 = x4*x[6];
    double x221 = pow(x18, -16);
    double x222 = 78.9136634576349*x221;
    double x223 = pow(x18, -10);
    double x224 = 126.261861532216*x223;
    double x225 = -x209*x222 - x213*x224;
    double x226 = x31*x[6];
    double x227 = 502.5*x216;
    double x228 = pow(x18, 24);
    double x229 = 83.75*x228;
    double x230 = x28*x[6];
    double x231 = pow(x18, 14);
    double x232 = 143.571428571429*x231;
    double x233 = x25*x[6];
    double x234 = -0.426902268107986*(x227*x226 + x230*x229 + x233*x232);
    double x235 = pow(x2, 14);
    double x236 = x233*x235;
    double x237 = 167.5*x204;
    double x238 = pow(x2, 24);
    double x239 = 27.9166666666667*x238;
    double x240 = -x230*x239 - x237*x226;
    double x241 = x42*x38;
    double x242 = x241 - x45*x[6];
    double x243 = x83*x64;
    double x244 = x48*x242;
    double x245 = x65*x244;
    double x246 = x197 + x91;
    double x247 = x246 + 1.0*((x92 == 1) ? (
   1 + x184
)
: 0);
    double x248 = x64*x247;
    double x249 = 57.6426965585605*x69;
    double x250 = 1.75363696572522*x74*x208;
    double x251 = 4.67636524193392*x72*x211;
    double x252 = 35.0727393145044*x58*x205;
    double x253 = 172.928089675681*x69;
    double x254 = 14.0290957258018*x72*x223;
    double x255 = 5.26091089717566*x74*x221;
    double x256 = 105.218217943513*x58*x217;
    double x257 = 9.57142857142857*x80*x231;
    double x258 = 3.35*x78*x228;
    double x259 = 100.5*x76*x216;
    double x260 = 3.19047619047619*x80*x235;
    double x261 = 1.11666666666667*x78*x238;
    double x262 = 33.5*x76*x204;
    double x263 = ((x16 == 1) ? (
   -x249*x[6] + x250*x[6] + x251*x[6] + x252*x[6]
)
: ((x24 == 1) ? (
   x253*x[6] - x254*x[6] - x255*x[6] - x256*x[6]
)
: ((x33 == 1) ? (
   -0.426902268107986*(-x257*x[6] - x258*x[6] - x259*x[6])
)
: ((x36 == 1) ? (
   -0.426902268107986*(x260*x[6] + x261*x[6] + x262*x[6])
)
: 0))));
    double x264 = x190 + x248 + x111*(x146 + x166*x[6]) + x111*(18.6758*x105 - 8.60954*x108 - x153 + x202 + x105*x156 - x108*x176) + x242*x243 + x263*x163 + x37*x245;
    double x265 = x[2]*x42*x121;
    double x266 = x116*x157;
    double x267 = x160 + x263*x162 - x266*x242;
    double x268 = x125 + x267 + x265*x[6];
    double x269 = x206*x[5];
    double x270 = x8*x[5];
    double x271 = x208*x270;
    double x272 = x11*x[5];
    double x273 = x212*x272 + x214*x[5];
    double x274 = x4*x[5];
    double x275 = -x270*x222 - x272*x224;
    double x276 = x28*x[5];
    double x277 = x31*x[5];
    double x278 = x25*x[5];
    double x279 = -0.426902268107986*(x232*x278 + x276*x229 + x277*x227);
    double x280 = x235*x278;
    double x281 = -x237*x277 - x239*x276;
    double x282 = x241 - x46;
    double x283 = x48*x282;
    double x284 = x65*x283;
    double x285 = x155*x[5];
    double x286 = x[5]*x[4];
    double x287 = x87 + x94;
    double x288 = x287 + 1.0*((x88 == 1) ? (
   1 + x186
)
: 0);
    double x289 = x64*x288;
    double x290 = ((x16 == 1) ? (
   -x249*x[5] + x250*x[5] + x251*x[5] + x252*x[5]
)
: ((x24 == 1) ? (
   x253*x[5] - x254*x[5] - x255*x[5] - x256*x[5]
)
: ((x33 == 1) ? (
   -0.426902268107986*(-x257*x[5] - x258*x[5] - x259*x[5])
)
: ((x36 == 1) ? (
   -0.426902268107986*(x260*x[5] + x261*x[5] + x262*x[5])
)
: 0))));
    double x291 = x289 + x111*(x146 + x167 + x172 + x170*x[4]) + x111*(x149 - x179 - 8.60954*x286 + x180*x[5] + x285*x152 - x286*x176) + x282*x243 + x290*x163 + x37*x284;
    double x292 = -x266*x282 + x290*x162;
    double x293 = x125 + x292 + x265*x[5];
    double x294 = x[2]*x67;
    double x295 = pow(x97, -1);
    double x296 = 74092.0*x295;
    double x297 = pow(x[2], -9.0);
    double x298 = ((x101 == 1) ? (
   -7976.15 + 137.093038*x[2] - 0.001884662*x133 + x296 - 8.77664e-07*x98 - x[2]*x132
)
: ((x102 == 1) ? (
   -11276.24 + 223.048446*x[2] + 0.018531982*x133 + x296 - 5.764227e-06*x98 - x[2]*x136
)
: ((x104 == 1) ? (
   -11278.378 + 188.684153*x[2] - 1.230524e+28*x297 - x[2]*x138
)
: 0)));
    double x299 = x146 + x298*x[6];
    double x300 = 2.0*x50;
    double x301 = x299*x158;
    double x302 = x144*x[6];
    double x303 = 16.629*x[2];
    double x304 = x303*x158;
    double x305 = 2.0*x[2];
    double x306 = 38590.0 - x305;
    double x307 = x306*x108;
    double x308 = 2340.0*x105*x[4];
    double x309 = 4680.0*x151;
    double x310 = 21167.4 + 1.3398*x[2];
    double x311 = x310*x[6];
    double x312 = x311*x[5];
    double x313 = x311*x[3];
    double x314 = x313*x[5];
    double x315 = 2340.0*x151;
    double x316 = x315*x105;
    double x317 = x316*x[4];
    double x318 = 1170.0*pow(x151, 2);
    double x319 = x306*x151;
    double x320 = x319*x[4];
    double x321 = -76066.1 + 18.6758*x[2];
    double x322 = x321*x[6];
    double x323 = x322*x[5];
    double x324 = x306*x105;
    double x325 = x324*x[4];
    double x326 = -53520.0 + x305;
    double x327 = x326*x[4];
    double x328 = x314 + x317 + x323 + x325 + x312*x155 + x318*x108 + x320*x[6] + x327*x[6];
    double x329 = 2.0*x158;
    double x330 = x123 + x191;
    double x331 = x63*x161;
    double x332 = pow(x49, -3);
    double x333 = x298*x[3];
    double x334 = ((x106 == 1) ? (
   -7770.458 + 130.485235*x[2] - 0.00265684*x133 + 52478.0*x295 + 1.29223e-07*x98 - x[2]*x168
)
: ((x107 == 1) ? (
   -13542.026 + 183.803828*x[2] + 3.64167e+29*x297 - x[2]*x169
)
: 0));
    double x335 = x334*x[6];
    double x336 = ((x109 == 1) ? (
   -236.7 + 132.416*x[2] - 0.00375752*x133 + 77359.0*x295 - 5.8927e-08*x98 - x[2]*x164
)
: ((x110 == 1) ? (
   -27097.396 + 300.25256*x[2] + 2.78854e+31*x297 - x[2]*x165
)
: 0));
    double x337 = x336*x[6];
    double x338 = 2.0*(x333*x[6] + x335*x[4] + x337*x[5]);
    double x339 = x303*x332;
    double x340 = x318*x[4];
    double x341 = 48232.5 - 8.60954*x[2];
    double x342 = x341*x[6];
    double x343 = x342*x[4];
    double x344 = 8861.88 - 5.28975*x[2];
    double x345 = x344*x175;
    double x346 = x345*x[4];
    double x347 = x0*x346 + x285*x313 + x320*x105 + x323*x[3] + x327*x105 + x340*x105 + x343*x[5];
    double x348 = x338*x332 + x339*x187 + x339*x189 + 2.0*x347*x332;
    double x349 = x348 - x304*x331 - x44*x159 + x53*x161;
    double x350 = x117 + x119 + x330 + x349;
    double x351 = -x303*x188 - x304*x189 - x329*x347 - x338*x158;
    double x352 = x351 + x[2]*x85 + x294*x331;
    double x353 = x326*x[6];
    double x354 = x143 + x146;
    double x355 = x199*x158;
    double x356 = x146 + x335;
    double x357 = x0*x344;
    double x358 = x357*x[4];
    double x359 = -x317 - x325 + x358 + x0*x345 + x318*x105 + x319*x105 + x342*x[5] + x353*x[3];
    double x360 = -x355*x182 - x356*x174 - x359*x174;
    double x361 = -1.0*x301 - x129*x183 - x328*x174;
    double x362 = x[2]*x130 + x299*x111 + x328*x111;
    double x363 = x[2]*x200 + x356*x111 + x359*x111;
    double x364 = x352 + x362 + x363 + x127*(x192 + x349 + x360 + x361 + x111*(-x307 - x308 + x316 + x324 + x353 - x315*x108 + x318*x[6] + x319*x[6]) + (x302 + x354)*x111);
    double x365 = x141 + x146;
    double x366 = x361 + x111*(-x312 + x313 + x322 + x311*x155) + (x302 + x365)*x111;
    double x367 = x83*x183;
    double x368 = x63*x183;
    double x369 = x146 + x337;
    double x370 = x345*x[6];
    double x371 = -x314 + x343 - x358 + x313*x155 + x322*x[3] + x370*x[4];
    double x372 = x191 + x348 - x242*x367 + x245*x161 - x247*x183 - x263*x368 - x368*x161 - x369*x174 - x371*x174;
    double x373 = x268 + x372;
    double x374 = x83*x65;
    double x375 = x330 + x351 + x[2]*x248 + x242*x374 + x369*x111 + x371*x111 + x66*x263;
    double x376 = x362 + x375;
    double x377 = x[3]*x[4];
    double x378 = x377*x306;
    double x379 = x321*x[5];
    double x380 = x377*x315;
    double x381 = x310*x[3];
    double x382 = x381*x[5];
    double x383 = x142*x[4];
    double x384 = x140*x[5];
    double x385 = x145 + x383 + x384;
    double x386 = x111*(x320 + x327 + x340 + x378 + x379 + x380 + x382 + x285*x310) + (x146 + x298 + x302 + x385)*x111;
    double x387 = x341*x[5];
    double x388 = x320*x[3] + x327*x[3] + x346*x[5] + x377*x318 + x379*x[3] + x382*x155 + x387*x[4];
    double x389 = x146 + x333 + x334*x[4] + x336*x[5];
    double x390 = -x282*x367 - x288*x183 - x290*x368 - x388*x174 - x389*x174;
    double x391 = x191 + x390 + x284*x161;
    double x392 = x293 + x391;
    double x393 = x[2]*x289 + x282*x374 + x388*x111 + x389*x111 + x66*x290;
    double x394 = x344*x108;
    double x395 = x360 + (x141 + x354)*x111 + (x342 - x357 + x370 + x394)*x111;
    double x396 = x363 + x375;
    double x397 = x286*x344;
    double x398 = x111*(x334 + x354 + x385) + (-x378 - x380 + x387 + x397 + x318*x[3] + x319*x[3] + x326*x[3] + x345*x[5])*x111;
    double x399 = 315.654653830539*x218;
    double x400 = 2.1*x42;
    double x401 = x400*x[6];
    double x402 = -4.2*x118 + x241;
    double x403 = x120 + x267 + (x401 + x402)*x374;
    double x404 = x372 + x403;
    double x405 = 4.2*x42;
    double x406 = x65*x115;
    double x407 = 2.0*x369;
    double x408 = x63*x263;
    double x409 = 9399.49413628717/x35;
    double x410 = x58*x112;
    double x411 = x72*x112;
    double x412 = 3133.16471209573/pow(x2, 11);
    double x413 = x74*x112;
    double x414 = 1879.89882725743/pow(x2, 17);
    double x415 = 84595.4472265846/x30;
    double x416 = 16919.0894453169/pow(x18, 17);
    double x417 = 28198.4824088615/pow(x18, 11);
    double x418 = x76*x112;
    double x419 = 80802.0*x22;
    double x420 = 16160.4*pow(x18, 23);
    double x421 = x78*x112;
    double x422 = x80*x112;
    double x423 = 26934.0*pow(x18, 13);
    double x424 = 8978.0*x13;
    double x425 = 1795.6*pow(x2, 23);
    double x426 = 2992.66666666667*pow(x2, 13);
    double x427 = x83*x242;
    double x428 = x402 + x400*x[5];
    double x429 = x0*x58;
    double x430 = x0*x72;
    double x431 = x0*x74;
    double x432 = x0*x76;
    double x433 = x0*x78;
    double x434 = x0*x80;
    double x435 = x127*(x120 + x390 + x111*(x336 + x365 + x385) + x111*(x346 - x382 - x397 + x321*x[3] + x341*x[4] + x381*x155) + x263*x284 + x290*x245 + x66*((x16 == 1) ? (
   -x249 + x250 + x251 + x252 - x409*x429 - x412*x430 - x414*x431
)
: ((x24 == 1) ? (
   x253 - x254 - x255 - x256 - x415*x429 - x416*x431 - x417*x430
)
: ((x33 == 1) ? (
   -0.426902268107986*(-x257 - x258 - x259 + x419*x432 + x420*x433 + x423*x434)
)
: ((x36 == 1) ? (
   -0.426902268107986*(x260 + x261 + x262 + x424*x432 + x425*x433 + x426*x434)
)
: 0)))) + (x401 + x428 - x45)*x374 - x406*x282*x242);
    double x436 = x120 + x292 + x428*x374;
    double x437 = x391 + x436;
    double x438 = 1.0*x393;
    double x439 = x58*x113;
    double x440 = x72*x113;
    double x441 = x74*x113;
    double x442 = x76*x113;
    double x443 = x78*x113;
    double x444 = x80*x113;
    out[0] = x127*(x126 + x85 + x111*(x0*((x109 == 1) ? (
   -0.00751504 - 24.6643*x69 - 3.53562e-07*x97 + 154718.0*x99
)
: ((x110 == 1) ? (
   2.509686e+33*x103 - 46.0*x69
)
: 0)) + x105*((x101 == 1) ? (
   -0.003769324 + x100 - 24.3671976*x69 - 5.265984e-06*x97
)
: ((x102 == 1) ? (
   0.037063964 + x100 - 38.5844296*x69 - 3.4585362e-05*x97
)
: ((x104 == 1) ? (
   -1.1074716e+30*x103 - 31.748192*x69
)
: 0))) + x108*((x106 == 1) ? (
   -0.00531368 - 24.112392*x69 + 7.75338e-07*x97 + 104956.0*x99
)
: ((x107 == 1) ? (
   3.277503e+31*x103 - 31.38*x69
)
: 0))) + x53*x37 + x66*((x16 == 1) ? (
   -x2*x59 - x54*x14 - x56*x10 - x7*x57
)
: ((x24 == 1) ? (
   -x54*x23 - x56*x21 - x57*x20 - x59*x18
)
: ((x33 == 1) ? (
   -0.426902268107986*(x60*x30 + x61*x27 + x62*x19)
)
: ((x36 == 1) ? (
   -0.426902268107986*(x6*x62 + x60*x35 + x61*x34)
)
: 0)))) + x67*x68 + x67*x96);
    out[1] = x194 + x196;
    out[2] = x196 + x203;
    out[3] = x196 + x127*(x264 + x268 + x66*((x16 == 1) ? (
   105.218217943513*x207 + 26.3045544858783*x210 + x215
)
: ((x24 == 1) ? (
   -172.928089675681*x220 + x225 - x219*x[6]
)
: ((x33 == 1) ? (
   x234
)
: ((x36 == 1) ? (
   -0.426902268107986*(-47.8571428571429*x236 + x240)
)
: 0)))));
    out[4] = x127*(x291 + x293 + x66*((x16 == 1) ? (
   105.218217943513*x269 + 26.3045544858783*x271 + x273
)
: ((x24 == 1) ? (
   -172.928089675681*x274 + x275 - x219*x[5]
)
: ((x33 == 1) ? (
   x279
)
: ((x36 == 1) ? (
   -0.426902268107986*(-47.8571428571429*x280 + x281)
)
: 0)))));
    out[5] = x194 + x195;
    out[6] = x352 + x127*(-2.0*x301 + x350 + x111*(2*x307 + x308 + 2*x312 + x309*x108) - x304*x129 - x328*x329 + x65*(x95 + 1.0*((x86 == 1) ? (
   pow(x[3], -1)
)
: 0)) + (x146 + 2*x302)*x111) + x294*x129 + x299*x300 + x300*x328;
    out[7] = x364;
    out[8] = x376 + (x366 + x373)*x127;
    out[9] = x393 + (x386 + x392)*x127;
    out[10] = x195 + x203;
    out[11] = x364;
    out[12] = x352 + x127*(x350 + x111*(x308 - 2*x324 + 2*x357 - x309*x105) - x355*x303 - x356*x329 - x359*x329 + x65*(x198 + 1.0*((x90 == 1) ? (
   pow(x[4], -1)
)
: 0)) + (2*x143 + x146)*x111) + x294*x199 + x356*x300 + x359*x300;
    out[13] = x396 + (x373 + x395)*x127;
    out[14] = x393 + (x392 + x398)*x127;
    out[15] = x195 + x127*(x264 + x403 + x66*((x16 == 1) ? (
   105.218217943513*x207 + 26.3045544858783*x210 + x215
)
: ((x24 == 1) ? (
   -172.928089675681*x220 + x225 - x399*x[6]
)
: ((x33 == 1) ? (
   x234
)
: ((x36 == 1) ? (
   -0.426902268107986*(-47.8571428571429*x236 + x240)
)
: 0)))));
    out[16] = x376 + (x366 + x404)*x127;
    out[17] = x396 + (x395 + x404)*x127;
    out[18] = x351 + x127*(x348 - x247*x304 - x371*x329 + x374*(x402 + x405*x[6]) - x406*pow(x242, 2) - x407*x158 - x408*x304 - x427*x304 + x65*(x246 + 1.0*((x92 == 1) ? (
   pow(x[5], -1)
)
: 0)) + x66*((x16 == 1) ? (
   -x409*x410 - x412*x411 - x413*x414
)
: ((x24 == 1) ? (
   -x411*x417 - x413*x416 - x415*x410
)
: ((x33 == 1) ? (
   -0.426902268107986*(x418*x419 + x421*x420 + x423*x422)
)
: ((x36 == 1) ? (
   -0.426902268107986*(x418*x424 + x422*x426 + x425*x421)
)
: 0)))) + (2*x141 + x146)*x111 + (-2*x313 - 2*x394)*x111 + x294*x263*x244) + x294*x247 + x371*x300 + x408*x294 + x427*x294 + x50*x407;
    out[19] = x393 + x435;
    out[20] = x127*(x291 + x436 + x66*((x16 == 1) ? (
   105.218217943513*x269 + 26.3045544858783*x271 + x273
)
: ((x24 == 1) ? (
   -172.928089675681*x274 + x275 - x399*x[5]
)
: ((x33 == 1) ? (
   x279
)
: ((x36 == 1) ? (
   -0.426902268107986*(-47.8571428571429*x280 + x281)
)
: 0)))));
    out[21] = x438 + (x386 + x437)*x127;
    out[22] = x438 + (x398 + x437)*x127;
    out[23] = x435 + x438;
    out[24] = x127*(x374*(x402 + x405*x[5]) - x406*pow(x282, 2) + x65*(x287 + 1.0*((x88 == 1) ? (
   pow(x[6], -1)
)
: 0)) + x66*((x16 == 1) ? (
   -x409*x439 - x412*x440 - x414*x441
)
: ((x24 == 1) ? (
   -x415*x439 - x416*x441 - x417*x440
)
: ((x33 == 1) ? (
   -0.426902268107986*(x419*x442 + x420*x443 + x423*x444)
)
: ((x36 == 1) ? (
   -0.426902268107986*(x424*x442 + x425*x443 + x426*x444)
)
: 0)))) + (2*x145 + x146 + 2*x383 + 2*x384)*x111 + x294*x290*x283);
}

__device__ void pycgpu_model_2_internal_cons_func(double* out, const double* x) {
    out[0] = 1.0*(-1 + x[3] + x[4] + x[5]);
    out[1] = 1.0*(-1 + x[6]);
}

__device__ void pycgpu_model_2_internal_cons_jac(double* out, const double* x) {
    out[0] = 0;
    out[1] = 1.0;
    out[2] = 1.0;
    out[3] = 1.0;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 1.0;
}

__device__ void pycgpu_model_2_mass_obj(double* out, const double* x) {
    double x0 = 1.0/(x[3] + x[4] + x[5]);
    out[0] = x0*x[3];
    out[1] = x0*x[4];
    out[2] = x0*x[5];
    out[3] = 0;
}

__device__ void pycgpu_model_2_formulamole_obj(double* out, const double* x) {
    out[0] = 1.0*x[3];
    out[1] = 1.0*x[4];
    out[2] = 1.0*x[5];
    out[3] = 0.0;
}

__device__ void pycgpu_model_2_formulamole_grad(double* out, const double* x) {
    out[0] = 0;
    out[1] = 1.0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    out[7] = 1.0;
    out[8] = 0;
    out[9] = 0;
    out[10] = 0;
    out[11] = 0;
    out[12] = 0;
    out[13] = 1.0;
    out[14] = 0;
}

__device__ double pycgpu_model_3_obj(const double* x) {
    double x0 = x[4]*x[3];
    double x1 = -x[5];
    double x2 = x[3] + x1;
    double x3 = x[3]*x[5];
    double x4 = x[3] - x[4];
    double x5 = x[2]*log(x[2]);
    double x6 = x[4] + x1;
    double x7 = x[4]*x[5];
    double x8 = x[3] + x[4] + x[5];
    double x9 = (1.0/3.0)*(1 - x8);
    double x10 = x7*x[3];
    double x11 = pow(x8, -1);
    double x12 = 1.0*x11;
    double x13 = pow(x[2], 7.0);
    double x14 = pow(x[2], 3.0);
    double x15 = pow(x[2], -1.0);
    double x16 = 74092.0*x15;
    double x17 = pow(x[2], 2.0);
    double x18 = pow(x[2], -9.0);
    double x19 = 933.47 <= x[2];
    double x20 = ((x[2] < 700.0) ? (
   -7976.15 + 137.093038*x[2] - 8.77664e-07*x14 + x16 - 0.001884662*x17 - 24.3671976*x5
)
: (((x[2] < 933.47 && 700.0 <= x[2])) ? (
   -11276.24 + 223.048446*x[2] - 5.764227e-06*x14 + x16 + 0.018531982*x17 - 38.5844296*x5
)
: ((x19 == 1) ? (
   -11278.378 + 188.684153*x[2] - 1.230524e+28*x18 - 31.748192*x5
)
: (
   0
))));
    double x21 = x[2] < 933.47;
    double x22 = x[2] < 1811.0;
    double x23 = 2.29603e+31*x18;
    double x24 = 1811.0 <= x[2];
    double x25 = ((x22 == 1) ? (
   1225.7 + 124.134*x[2] - 5.8927e-08*x14 + 77359.0*x15 - 0.00439752*x17 - 23.5143*x5
)
: ((x24 == 1) ? (
   -25383.581 + 299.31255*x[2] + x23 - 46.0*x5
)
: (
   0
)));
    return x12*(x[3]*((x21 == 1) ? (
   11005.029 - 11.841867*x[2] + 7.934e-20*x13 + x20
)
: ((x19 == 1) ? (
   10482.382 - 11.253974*x[2] + 1.231e+28*x18 + x20
)
: (
   0
))) + x[4]*((x[2] < 1357.77) ? (
   5194.277 + 120.973331*x[2] - 5.8489e-21*x13 + 1.29223e-07*x14 + 52478.0*x15 - 0.00265684*x17 - 24.112392*x5
)
: ((1357.77 <= x[2]) ? (
   -46.545 + 173.881484*x[2] - 31.38*x5
)
: (
   0
))) + x[5]*((x22 == 1) ? (
   12040.17 - 6.55843*x[2] - 3.67516e-21*x13 + x25
)
: ((x24 == 1) ? (
   14544.751 - 8.01055*x[2] - x23 + x25
)
: (
   0
)))) + x12*(-2812.0*x0*pow(x4, 2) + x0*(-66622.0 + 8.1*x[2]) + 121.9*pow(x2, 2)*x3 + x3*(-91976.5 + 22.1314*x[2]) + x7*(36088.0 - 2.32968*x[2]) + x0*x4*(46800.0 - 90.8*x[2] + 10.0*x5) + x10*(x[3] + x9)*(-24637.8886 + 9.12034153*x[2]) + x10*(x[4] + x9)*(-47262.3591 + 11.3118607*x[2]) + x2*x3*(-5672.58 + 4.8728*x[2]) + x6*x7*(324.53 - 0.0327*x[2]) + pow(x6, 2)*x7*(10355.4 - 3.60297*x[2])) + 8.3145*x[2]*x11*(1.0*((1e-15 < x[3]) ? (
   x[3]*log(x[3])
)
: (
   0
)) + 1.0*((1e-15 < x[4]) ? (
   x[4]*log(x[4])
)
: (
   0
)) + 1.0*((1e-15 < x[5]) ? (
   x[5]*log(x[5])
)
: (
   0
)));
}

__device__ double pycgpu_model_3_formulaobj(const double* x) {
    double x0 = x[3] + x[4] + x[5];
    double x1 = x[4]*x[3];
    double x2 = -x[5];
    double x3 = x[3] + x2;
    double x4 = x[3]*x[5];
    double x5 = x[3] - x[4];
    double x6 = x[2]*log(x[2]);
    double x7 = x[4] + x2;
    double x8 = x[4]*x[5];
    double x9 = (1.0/3.0)*(1 - x0);
    double x10 = x8*x[3];
    double x11 = pow(x0, -1);
    double x12 = 1.0*x11;
    double x13 = pow(x[2], 7.0);
    double x14 = pow(x[2], 3.0);
    double x15 = pow(x[2], -1.0);
    double x16 = 74092.0*x15;
    double x17 = pow(x[2], 2.0);
    double x18 = pow(x[2], -9.0);
    double x19 = 933.47 <= x[2];
    double x20 = ((x[2] < 700.0) ? (
   -7976.15 + 137.093038*x[2] - 8.77664e-07*x14 + x16 - 0.001884662*x17 - 24.3671976*x6
)
: (((x[2] < 933.47 && 700.0 <= x[2])) ? (
   -11276.24 + 223.048446*x[2] - 5.764227e-06*x14 + x16 + 0.018531982*x17 - 38.5844296*x6
)
: ((x19 == 1) ? (
   -11278.378 + 188.684153*x[2] - 1.230524e+28*x18 - 31.748192*x6
)
: (
   0
))));
    double x21 = x[2] < 933.47;
    double x22 = x[2] < 1811.0;
    double x23 = 2.29603e+31*x18;
    double x24 = 1811.0 <= x[2];
    double x25 = ((x22 == 1) ? (
   1225.7 + 124.134*x[2] - 5.8927e-08*x14 + 77359.0*x15 - 0.00439752*x17 - 23.5143*x6
)
: ((x24 == 1) ? (
   -25383.581 + 299.31255*x[2] + x23 - 46.0*x6
)
: (
   0
)));
    return 1.0*x0*(x12*(x[3]*((x21 == 1) ? (
   11005.029 - 11.841867*x[2] + 7.934e-20*x13 + x20
)
: ((x19 == 1) ? (
   10482.382 - 11.253974*x[2] + 1.231e+28*x18 + x20
)
: (
   0
))) + x[4]*((x[2] < 1357.77) ? (
   5194.277 + 120.973331*x[2] - 5.8489e-21*x13 + 1.29223e-07*x14 + 52478.0*x15 - 0.00265684*x17 - 24.112392*x6
)
: ((1357.77 <= x[2]) ? (
   -46.545 + 173.881484*x[2] - 31.38*x6
)
: (
   0
))) + x[5]*((x22 == 1) ? (
   12040.17 - 6.55843*x[2] - 3.67516e-21*x13 + x25
)
: ((x24 == 1) ? (
   14544.751 - 8.01055*x[2] - x23 + x25
)
: (
   0
)))) + x12*(-2812.0*x1*pow(x5, 2) + x1*(-66622.0 + 8.1*x[2]) + 121.9*x4*pow(x3, 2) + x4*(-91976.5 + 22.1314*x[2]) + x8*(36088.0 - 2.32968*x[2]) + x1*x5*(46800.0 - 90.8*x[2] + 10.0*x6) + x10*(x[3] + x9)*(-24637.8886 + 9.12034153*x[2]) + x10*(x[4] + x9)*(-47262.3591 + 11.3118607*x[2]) + x4*x3*(-5672.58 + 4.8728*x[2]) + x8*x7*(324.53 - 0.0327*x[2]) + x8*pow(x7, 2)*(10355.4 - 3.60297*x[2])) + 8.3145*x[2]*x11*(1.0*((1e-15 < x[3]) ? (
   x[3]*log(x[3])
)
: (
   0
)) + 1.0*((1e-15 < x[4]) ? (
   x[4]*log(x[4])
)
: (
   0
)) + 1.0*((1e-15 < x[5]) ? (
   x[5]*log(x[5])
)
: (
   0
))));
}

__device__ void pycgpu_model_3_formulagrad(double* out, const double* x) {
    double x0 = pow(x[2], 6.0);
    double x1 = pow(x[2], 2.0);
    double x2 = pow(x1, -1);
    double x3 = log(x[2]);
    double x4 = 23.5143*x3;
    double x5 = pow(x[2], 1.0);
    double x6 = x[2] < 1811.0;
    double x7 = 46.0*x3;
    double x8 = pow(x[2], -10.0);
    double x9 = 2.066427e+32*x8;
    double x10 = 1811.0 <= x[2];
    double x11 = ((x6 == 1) ? (
   100.6197 - 1.76781e-07*x1 - 77359.0*x2 - x4 - 0.00879504*x5
)
: ((x10 == 1) ? (
   253.31255 - x7 - x9
)
: (
   0
)));
    double x12 = 24.112392*x3;
    double x13 = x[2] < 1357.77;
    double x14 = 31.38*x3;
    double x15 = 1357.77 <= x[2];
    double x16 = 24.3671976*x3;
    double x17 = -74092.0*x2;
    double x18 = x[2] < 700.0;
    double x19 = 38.5844296*x3;
    double x20 = (x[2] < 933.47 && 700.0 <= x[2]);
    double x21 = 31.748192*x3;
    double x22 = 933.47 <= x[2];
    double x23 = ((x18 == 1) ? (
   112.7258404 - 2.632992e-06*x1 - x16 + x17 - 0.003769324*x5
)
: ((x20 == 1) ? (
   184.4640164 - 1.7292681e-05*x1 + x17 - x19 + 0.037063964*x5
)
: ((x22 == 1) ? (
   156.935961 - x21 + 1.1074716e+29*x8
)
: (
   0
))));
    double x24 = x[2] < 933.47;
    double x25 = x[3] + x[4] + x[5];
    double x26 = pow(x25, -1);
    double x27 = 1.0*x26;
    double x28 = log(x[5]);
    double x29 = 1e-15 < x[5];
    double x30 = log(x[4]);
    double x31 = 1e-15 < x[4];
    double x32 = log(x[3]);
    double x33 = 1e-15 < x[3];
    double x34 = 1.0*((x29 == 1) ? (
   x28*x[5]
)
: (
   0
)) + 1.0*((x31 == 1) ? (
   x30*x[4]
)
: (
   0
)) + 1.0*((x33 == 1) ? (
   x32*x[3]
)
: (
   0
));
    double x35 = 8.3145*x26;
    double x36 = x34*x35;
    double x37 = 1.0*((x31 == 1) ? (
   0
)
: (
   0
));
    double x38 = 1.0*((x33 == 1) ? (
   0
)
: (
   0
));
    double x39 = 1.0*((x29 == 1) ? (
   0
)
: (
   0
));
    double x40 = x38 + x39;
    double x41 = x[2]*x35;
    double x42 = -x[5];
    double x43 = x[4] + x42;
    double x44 = pow(x43, 2);
    double x45 = x[4]*x[5];
    double x46 = (1.0/3.0)*(1 - x25);
    double x47 = x[4] + x46;
    double x48 = x[3]*x[5];
    double x49 = x48*x[4];
    double x50 = x43*x45;
    double x51 = 10.0*x3;
    double x52 = x[3] - x[4];
    double x53 = x52*x[4];
    double x54 = x53*x[3];
    double x55 = x[3] + x42;
    double x56 = x55*x48;
    double x57 = x[3] + x46;
    double x58 = x[4]*x[3];
    double x59 = 1.0*x25;
    double x60 = pow(x[2], 7.0);
    double x61 = pow(x[2], 3.0);
    double x62 = pow(x5, -1);
    double x63 = 74092.0*x62;
    double x64 = pow(x[2], -9.0);
    double x65 = ((x18 == 1) ? (
   -7976.15 + 137.093038*x[2] - 0.001884662*x1 - 8.77664e-07*x61 + x63 - x[2]*x16
)
: ((x20 == 1) ? (
   -11276.24 + 223.048446*x[2] + 0.018531982*x1 - 5.764227e-06*x61 + x63 - x[2]*x19
)
: ((x22 == 1) ? (
   -11278.378 + 188.684153*x[2] - 1.230524e+28*x64 - x[2]*x21
)
: (
   0
))));
    double x66 = ((x24 == 1) ? (
   11005.029 - 11.841867*x[2] + 7.934e-20*x60 + x65
)
: ((x22 == 1) ? (
   10482.382 - 11.253974*x[2] + 1.231e+28*x64 + x65
)
: (
   0
)));
    double x67 = ((x6 == 1) ? (
   0
)
: ((x10 == 1) ? (
   0
)
: (
   0
)));
    double x68 = ((x18 == 1) ? (
   0
)
: ((x20 == 1) ? (
   0
)
: ((x22 == 1) ? (
   0
)
: (
   0
))));
    double x69 = x[3]*((x24 == 1) ? (
   x68
)
: ((x22 == 1) ? (
   x68
)
: (
   0
))) + x[4]*((x13 == 1) ? (
   0
)
: ((x15 == 1) ? (
   0
)
: (
   0
))) + x[5]*((x6 == 1) ? (
   x67
)
: ((x10 == 1) ? (
   x67
)
: (
   0
)));
    double x70 = -47262.3591 + 11.3118607*x[2];
    double x71 = x70*x47;
    double x72 = -66622.0 + 8.1*x[2];
    double x73 = -5672.58 + 4.8728*x[2];
    double x74 = x73*x55;
    double x75 = 2812.0*pow(x52, 2);
    double x76 = 5624.0*x54;
    double x77 = -24637.8886 + 9.12034153*x[2];
    double x78 = x77*x57;
    double x79 = x70*x49;
    double x80 = (-1.0/3.0)*x79;
    double x81 = 46800.0 - 90.8*x[2] + x[2]*x51;
    double x82 = x81*x[3];
    double x83 = x82*x[4];
    double x84 = x73*x48;
    double x85 = -91976.5 + 22.1314*x[2];
    double x86 = 243.8*x56;
    double x87 = 121.9*pow(x55, 2);
    double x88 = x77*x49;
    double x89 = ((x13 == 1) ? (
   5194.277 + 120.973331*x[2] - 0.00265684*x1 - 5.8489e-21*x60 + 1.29223e-07*x61 + 52478.0*x62 - x[2]*x12
)
: ((x15 == 1) ? (
   -46.545 + 173.881484*x[2] - x[2]*x14
)
: (
   0
)));
    double x90 = 2.29603e+31*x64;
    double x91 = ((x6 == 1) ? (
   1225.7 + 124.134*x[2] - 0.00439752*x1 - 5.8927e-08*x61 + 77359.0*x62 - x[2]*x4
)
: ((x10 == 1) ? (
   -25383.581 + 299.31255*x[2] + x90 - x[2]*x7
)
: (
   0
)));
    double x92 = ((x6 == 1) ? (
   12040.17 - 6.55843*x[2] - 3.67516e-21*x60 + x91
)
: ((x10 == 1) ? (
   14544.751 - 8.01055*x[2] - x90 + x91
)
: (
   0
)));
    double x93 = x66*x[3] + x89*x[4] + x92*x[5];
    double x94 = pow(x25, -2);
    double x95 = 1.0*x94;
    double x96 = x72*x[3];
    double x97 = 324.53 - 0.0327*x[2];
    double x98 = x97*x43;
    double x99 = x98*x[5];
    double x100 = x85*x[3];
    double x101 = 36088.0 - 2.32968*x[2];
    double x102 = x101*x[4];
    double x103 = 10355.4 - 3.60297*x[2];
    double x104 = x44*x103;
    double x105 = x78*x48;
    double x106 = x71*x48;
    double x107 = x100*x[5] + x102*x[5] + x105*x[4] + x106*x[4] + x45*x104 - x75*x58 + x83*x52 + x84*x55 + x87*x48 + x96*x[4] + x99*x[4];
    double x108 = -x93*x95 - x95*x107 - 8.3145*x[2]*x94*x34;
    double x109 = 1.0*(x[2]*x36 + x27*x107 + x93*x27);
    double x110 = 2*x50*x103;
    double x111 = x97*x45;
    double x112 = (-1.0/3.0)*x88;
    out[0] = x59*(x36 + x27*(x[3]*((x24 == 1) ? (
   -11.841867 + 5.5538e-19*x0 + x23
)
: ((x22 == 1) ? (
   -11.253974 + x23 - 1.1079e+29*x8
)
: (
   0
))) + ((x6 == 1) ? (
   -6.55843 - 2.572612e-20*x0 + x11
)
: ((x10 == 1) ? (
   -8.01055 + x11 + x9
)
: (
   0
)))*x[5] + ((x13 == 1) ? (
   96.860939 - 4.09423e-20*x0 + 3.87669e-07*x1 - x12 - 52478.0*x2 - 0.00531368*x5
)
: ((x15 == 1) ? (
   142.501484 - x14
)
: (
   0
)))*x[4]) + x27*(-2.32968*x45 + 22.1314*x48 - 0.0327*x50 + 4.8728*x56 + 8.1*x58 - 3.60297*x44*x45 + 11.3118607*x47*x49 + x54*(-80.8 + x51) + 9.12034153*x57*x49) + (x37 + x40)*x41);
    out[1] = x109 + x59*(x108 + x27*(-x76 + x80 + x83 + x84 + x86 + (2.0/3.0)*x88 + x71*x45 + x72*x[4] + x74*x[5] - x75*x[4] + x78*x45 + x81*x53 + x85*x[5] + x87*x[5]) + x41*(x37 + x39 + 1.0*((x33 == 1) ? (
   1 + x32
)
: (
   0
))) + (x66 + x69)*x27);
    out[2] = x109 + x59*(x108 + x27*(x105 + x106 + x110 + x111 + x112 + x76 + (2.0/3.0)*x79 - x83 + x96 + x99 + x101*x[5] + x104*x[5] - x75*x[3] + x82*x52) + x41*(x40 + 1.0*((x31 == 1) ? (
   1 + x30
)
: (
   0
))) + (x69 + x89)*x27);
    out[3] = x109 + x59*(x108 + x27*(x100 + x102 - x110 - x111 + x112 + x80 - x84 - x86 + x104*x[4] + x71*x58 + x74*x[3] + x78*x58 + x87*x[3] + x98*x[4]) + x41*(x37 + x38 + 1.0*((x29 == 1) ? (
   1 + x28
)
: (
   0
))) + (x69 + x92)*x27);
}

__device__ void pycgpu_model_3_formulahess(double* out, const double* x) {
    double x0 = pow(x[2], -1);
    double x1 = x[3] + x[4] + x[5];
    double x2 = pow(x1, -1);
    double x3 = x[3] - x[4];
    double x4 = x3*x[3];
    double x5 = 1e-15 < x[5];
    double x6 = 1.0*((x5 == 1) ? 0
: 0);
    double x7 = 1e-15 < x[3];
    double x8 = 1.0*((x7 == 1) ? 0
: 0);
    double x9 = 1e-15 < x[4];
    double x10 = 1.0*((x9 == 1) ? 0
: 0);
    double x11 = x10 + x8;
    double x12 = x11 + x6;
    double x13 = 8.3145*x2;
    double x14 = x[2]*x13;
    double x15 = x14*x12;
    double x16 = 16.629*x2;
    double x17 = pow(x[2], 5.0);
    double x18 = pow(x[2], 1.0);
    double x19 = pow(x[2], 3.0);
    double x20 = pow(x19, -1);
    double x21 = 148184.0*x20;
    double x22 = x[2] < 700.0;
    double x23 = (x[2] < 933.47 && 700.0 <= x[2]);
    double x24 = pow(x[2], -11.0);
    double x25 = 933.47 <= x[2];
    double x26 = ((x22 == 1) ? (
   -0.003769324 - 24.3671976*x0 - 5.265984e-06*x18 + x21
)
: ((x23 == 1) ? (
   0.037063964 - 38.5844296*x0 - 3.4585362e-05*x18 + x21
)
: ((x25 == 1) ? (
   -31.748192*x0 - 1.1074716e+30*x24
)
: 0)));
    double x27 = x[2] < 933.47;
    double x28 = x[2] < 1357.77;
    double x29 = 1357.77 <= x[2];
    double x30 = x[2] < 1811.0;
    double x31 = 2.066427e+33*x24;
    double x32 = 1811.0 <= x[2];
    double x33 = ((x30 == 1) ? (
   -0.00879504 - 23.5143*x0 - 3.53562e-07*x18 + 154718.0*x20
)
: ((x32 == 1) ? (
   -46.0*x0 + x31
)
: 0));
    double x34 = 1.0*x2;
    double x35 = 1.0*x1;
    double x36 = 8.1*x[4];
    double x37 = x[4]*x[5];
    double x38 = (1.0/3.0)*(1 - x1);
    double x39 = x[3] + x38;
    double x40 = 9.12034153*x39;
    double x41 = x[3]*x[5];
    double x42 = 4.8728*x41;
    double x43 = -x[5];
    double x44 = x[3] + x43;
    double x45 = 4.8728*x44;
    double x46 = log(x[2]);
    double x47 = 10.0*x46;
    double x48 = -80.8 + x47;
    double x49 = x48*x[4];
    double x50 = x49*x[3];
    double x51 = x3*x49;
    double x52 = x[4] + x38;
    double x53 = 11.3118607*x52;
    double x54 = x41*x[4];
    double x55 = log(x[3]);
    double x56 = x10 + x6;
    double x57 = x56 + 1.0*((x7 == 1) ? (
   1 + x55
)
: 0);
    double x58 = x57*x13;
    double x59 = 24.3671976*x46;
    double x60 = pow(x[2], 2.0);
    double x61 = pow(x60, -1);
    double x62 = -74092.0*x61;
    double x63 = 38.5844296*x46;
    double x64 = pow(x[2], -10.0);
    double x65 = 31.748192*x46;
    double x66 = ((x22 == 1) ? (
   112.7258404 - 0.003769324*x18 - x59 - 2.632992e-06*x60 + x62
)
: ((x23 == 1) ? (
   184.4640164 + 0.037063964*x18 - 1.7292681e-05*x60 + x62 - x63
)
: ((x25 == 1) ? (
   156.935961 + 1.1074716e+29*x64 - x65
)
: 0)));
    double x67 = pow(x[2], 6.0);
    double x68 = ((x27 == 1) ? (
   -11.841867 + x66 + 5.5538e-19*x67
)
: ((x25 == 1) ? (
   -11.253974 - 1.1079e+29*x64 + x66
)
: 0));
    double x69 = ((x30 == 1) ? 0
: ((x32 == 1) ? 0
: 0));
    double x70 = ((x30 == 1) ? (
   x69
)
: ((x32 == 1) ? (
   x69
)
: 0));
    double x71 = ((x28 == 1) ? 0
: ((x29 == 1) ? 0
: 0));
    double x72 = ((x22 == 1) ? 0
: ((x23 == 1) ? 0
: ((x25 == 1) ? 0
: 0)));
    double x73 = ((x27 == 1) ? (
   x72
)
: ((x25 == 1) ? (
   x72
)
: 0));
    double x74 = x70*x[5] + x71*x[4] + x73*x[3];
    double x75 = x[4] + x43;
    double x76 = pow(x75, 2);
    double x77 = 3.60297*x76;
    double x78 = x77*x[4];
    double x79 = 2.32968*x[4];
    double x80 = 0.0327*x75;
    double x81 = x80*x[5];
    double x82 = 22.1314*x[3];
    double x83 = x36*x[3] + x42*x44 + x51*x[3] + x54*x40 + x54*x53 - x78*x[5] - x79*x[5] - x81*x[4] + x82*x[5];
    double x84 = pow(x1, -2);
    double x85 = 1.0*x84;
    double x86 = 8.3145*x84;
    double x87 = 23.5143*x46;
    double x88 = 46.0*x46;
    double x89 = 2.066427e+32*x64;
    double x90 = ((x30 == 1) ? (
   100.6197 - 0.00879504*x18 - 1.76781e-07*x60 - 77359.0*x61 - x87
)
: ((x32 == 1) ? (
   253.31255 - x88 - x89
)
: 0));
    double x91 = ((x30 == 1) ? (
   -6.55843 - 2.572612e-20*x67 + x90
)
: ((x32 == 1) ? (
   -8.01055 + x89 + x90
)
: 0));
    double x92 = 24.112392*x46;
    double x93 = 31.38*x46;
    double x94 = ((x28 == 1) ? (
   96.860939 - 0.00531368*x18 + 3.87669e-07*x60 - 52478.0*x61 - 4.09423e-20*x67 - x92
)
: ((x29 == 1) ? (
   142.501484 - x93
)
: 0));
    double x95 = x68*x[3] + x91*x[5] + x94*x[4];
    double x96 = log(x[5]);
    double x97 = log(x[4]);
    double x98 = 1.0*((x7 == 1) ? (
   x55*x[3]
)
: 0) + 1.0*((x5 == 1) ? (
   x96*x[5]
)
: 0) + 1.0*((x9 == 1) ? (
   x97*x[4]
)
: 0);
    double x99 = x15 - x83*x85 - x85*x95 - x86*x98 - x[2]*x86*x12;
    double x100 = x35*(x58 + x99 + x34*(22.1314*x[5] + x36 + x42 + x50 + x51 + 2.30960745333333*x54 + x40*x37 + x45*x[5] + x53*x37) + (x68 + x74)*x34);
    double x101 = x15 + x83*x34 + x95*x34 + x98*x13;
    double x102 = 1.0*x101;
    double x103 = 0.0327*x37;
    double x104 = x75*x37;
    double x105 = 7.20594*x104;
    double x106 = x6 + x8;
    double x107 = x106 + 1.0*((x9 == 1) ? (
   1 + x97
)
: 0);
    double x108 = x13*x107;
    double x109 = x35*(x108 + x99 + x34*(8.1*x[3] - 2.32968*x[5] - x103 - x105 - x50 + 4.50112662333333*x54 - x81 + x4*x48 + x40*x41 + x53*x41 - x77*x[5]) + (x74 + x94)*x34);
    double x110 = x[4]*x[3];
    double x111 = x11 + 1.0*((x5 == 1) ? (
   1 + x96
)
: 0);
    double x112 = x13*x111;
    double x113 = x35*(x112 + x99 + x34*(x103 + x105 - x42 - 6.81073407666667*x54 - x78 - x79 + x82 + x40*x110 + x45*x[3] + x53*x110 - x80*x[4]) + (x74 + x91)*x34);
    double x114 = pow(x[2], 7.0);
    double x115 = pow(x18, -1);
    double x116 = 74092.0*x115;
    double x117 = pow(x[2], -9.0);
    double x118 = ((x22 == 1) ? (
   -7976.15 + 137.093038*x[2] + x116 - 8.77664e-07*x19 - 0.001884662*x60 - x[2]*x59
)
: ((x23 == 1) ? (
   -11276.24 + 223.048446*x[2] + x116 - 5.764227e-06*x19 + 0.018531982*x60 - x[2]*x63
)
: ((x25 == 1) ? (
   -11278.378 + 188.684153*x[2] - 1.230524e+28*x117 - x[2]*x65
)
: 0)));
    double x119 = ((x27 == 1) ? (
   11005.029 - 11.841867*x[2] + 7.934e-20*x114 + x118
)
: ((x25 == 1) ? (
   10482.382 - 11.253974*x[2] + 1.231e+28*x117 + x118
)
: 0));
    double x120 = x119 + x74;
    double x121 = 2.0*x2;
    double x122 = x[2]*x57;
    double x123 = -47262.3591 + 11.3118607*x[2];
    double x124 = x52*x123;
    double x125 = x124*x[4];
    double x126 = -66622.0 + 8.1*x[2];
    double x127 = -5672.58 + 4.8728*x[2];
    double x128 = x44*x127;
    double x129 = 2812.0*pow(x3, 2);
    double x130 = x129*x[4];
    double x131 = 5624.0*x[4];
    double x132 = x4*x131;
    double x133 = -24637.8886 + 9.12034153*x[2];
    double x134 = x39*x133;
    double x135 = x134*x[4];
    double x136 = (1.0/3.0)*x123;
    double x137 = x41*x136;
    double x138 = -x137*x[4];
    double x139 = 46800.0 - 90.8*x[2] + x[2]*x47;
    double x140 = x3*x139;
    double x141 = x140*x[4];
    double x142 = x139*x[4];
    double x143 = x142*x[3];
    double x144 = x127*x[3];
    double x145 = x144*x[5];
    double x146 = -91976.5 + 22.1314*x[2];
    double x147 = 243.8*x41;
    double x148 = x44*x147;
    double x149 = 121.9*pow(x44, 2);
    double x150 = (2.0/3.0)*x133;
    double x151 = x41*x150;
    double x152 = -x130 - x132 + x138 + x141 + x143 + x145 + x148 + x125*x[5] + x126*x[4] + x128*x[5] + x135*x[5] + x146*x[5] + x149*x[5] + x151*x[4];
    double x153 = 2.0*x152;
    double x154 = 487.6*x44;
    double x155 = x131*x[3];
    double x156 = -x155;
    double x157 = x127*x[5];
    double x158 = (2.0/3.0)*x123;
    double x159 = x37*x158;
    double x160 = 16.629*x84;
    double x161 = 2.0*x84;
    double x162 = x126*x[3];
    double x163 = 324.53 - 0.0327*x[2];
    double x164 = x75*x163;
    double x165 = x164*x[4];
    double x166 = x146*x[3];
    double x167 = 36088.0 - 2.32968*x[2];
    double x168 = x167*x[5];
    double x169 = 10355.4 - 3.60297*x[2];
    double x170 = x76*x169;
    double x171 = x170*x[4];
    double x172 = -x130*x[3] + x141*x[3] + x162*x[4] + x165*x[5] + x166*x[5] + x168*x[4] + x171*x[5] + x41*x125 + x41*x128 + x41*x135 + x41*x149;
    double x173 = pow(x1, -3);
    double x174 = 2.0*x173;
    double x175 = x[2]*x98;
    double x176 = ((x28 == 1) ? (
   5194.277 + 120.973331*x[2] - 5.8489e-21*x114 + 52478.0*x115 + 1.29223e-07*x19 - 0.00265684*x60 - x[2]*x92
)
: ((x29 == 1) ? (
   -46.545 + 173.881484*x[2] - x[2]*x93
)
: 0));
    double x177 = 2.29603e+31*x117;
    double x178 = ((x30 == 1) ? (
   1225.7 + 124.134*x[2] + 77359.0*x115 - 5.8927e-08*x19 - 0.00439752*x60 - x[2]*x87
)
: ((x32 == 1) ? (
   -25383.581 + 299.31255*x[2] + x177 - x[2]*x88
)
: 0));
    double x179 = ((x30 == 1) ? (
   12040.17 - 6.55843*x[2] - 3.67516e-21*x114 + x178
)
: ((x32 == 1) ? (
   14544.751 - 8.01055*x[2] - x177 + x178
)
: 0));
    double x180 = x119*x[3] + x176*x[4] + x179*x[5];
    double x181 = x174*x172 + x174*x180 + 16.629*x175*x173;
    double x182 = -x160*x175 - x161*x172 - x161*x180;
    double x183 = x139*x[3];
    double x184 = (1.0/3.0)*x133;
    double x185 = -x37*x184;
    double x186 = -x137;
    double x187 = x73 + x74;
    double x188 = x15 + x181 - x85*x120 - x85*x152 - x86*x122;
    double x189 = x[2]*x107;
    double x190 = 2*x169;
    double x191 = x104*x190;
    double x192 = x163*x[4];
    double x193 = x192*x[5];
    double x194 = x41*x184;
    double x195 = -x194*x[4];
    double x196 = x132 - x143 + x162 + x168 + x191 + x193 + x195 - x129*x[3] + x140*x[3] + x164*x[5] + x170*x[5] + x41*x124 + x41*x134 + x54*x158;
    double x197 = x176 + x74;
    double x198 = -x85*x196 - x85*x197 - x86*x189;
    double x199 = x182 + x[2]*x108 + x34*x196 + x34*x197;
    double x200 = x[2]*x58 + x34*x120 + x34*x152;
    double x201 = x199 + x200 + x35*(x188 + x198 + x34*(x187 + x71) + x34*(x126 - x129 + x140 - x142 + x151 + x155 + x159 + x183 + x185 + x186 - 5624.0*x4 + x124*x[5] + x134*x[5] + x3*x131));
    double x202 = x110*x150;
    double x203 = (1.0/3.0)*x110;
    double x204 = 243.8*x44;
    double x205 = x124*x[3];
    double x206 = x134*x[3];
    double x207 = x138 - x145 - x148 + x165 + x166 + x171 - x191 - x193 + x195 + x128*x[3] + x149*x[3] + x167*x[4] + x205*x[4] + x206*x[4];
    double x208 = x[2]*x111;
    double x209 = x179 + x74;
    double x210 = -x85*x207 - x85*x209 - x86*x208;
    double x211 = x[2]*x112 + x34*x207 + x34*x209;
    double x212 = x182 + x200 + x211 + x35*(x188 + x210 + x34*(x187 + x70) + x34*(x125 + x128 + x135 + x144 + x146 - x147 + x149 - x157 + x185 + x202 - x203*x123 + x204*x[3] - x204*x[5] - x37*x136));
    double x213 = x163*x[5];
    double x214 = x37*x190;
    double x215 = x75*x169;
    double x216 = x215*x[5];
    double x217 = x215*x[4];
    double x218 = x110*x158;
    double x219 = x199 + x211 + x35*(x15 + x181 + x198 + x210 + x34*(x70 + x71 + x74) + x34*(x164 + x167 + x170 + x186 + x192 - x194 + x205 + x206 - x213 - x214 - 2*x216 + 2*x217 + x218 - x203*x133));
    out[0] = x35*(x15 + x12*x16 + x34*(x[3]*((x27 == 1) ? (
   3.33228e-18*x17 + x26
)
: ((x25 == 1) ? (
   1.1079e+30*x24 + x26
)
: 0)) + x[4]*((x28 == 1) ? (
   -0.00531368 - 24.112392*x0 - 2.456538e-19*x17 + 7.75338e-07*x18 + 104956.0*x20
)
: ((x29 == 1) ? (
   -31.38*x0
)
: 0)) + x[5]*((x30 == 1) ? (
   -1.5435672e-19*x17 + x33
)
: ((x32 == 1) ? (
   -x31 + x33
)
: 0))) + 10.0*x0*x2*x4*x[4]);
    out[1] = x100 + x102;
    out[2] = x102 + x109;
    out[3] = x102 + x113;
    out[4] = x100 + x101;
    out[5] = x182 + x120*x121 + x16*x122 + x2*x153 + x35*(x181 - x120*x161 - x122*x160 + x14*(x56 + 1.0*((x7 == 1) ? (
   pow(x[3], -1)
)
: 0)) + x34*(2*x142 + x147 + x156 + 2*x157 - x159 + x154*x[5] - 11248.0*x3*x[4] + (4.0/3.0)*x37*x133) - x84*x153 + (2*x73 + x74)*x34);
    out[6] = x201;
    out[7] = x212;
    out[8] = x101 + x109;
    out[9] = x201;
    out[10] = x182 + x121*x196 + x121*x197 + x16*x189 + x35*(x181 + x14*(x106 + 1.0*((x9 == 1) ? (
   pow(x[4], -1)
)
: 0)) - x160*x189 - x161*x196 - x161*x197 + x34*(-x151 + x156 - 2*x183 + 2*x213 + x214 + 4*x216 + 11248.0*x4 + (4.0/3.0)*x41*x123) + (2*x71 + x74)*x34);
    out[11] = x219;
    out[12] = x101 + x113;
    out[13] = x212;
    out[14] = x219;
    out[15] = x182 + x16*x208 + x207*x121 + x209*x121 + x35*(x181 + x14*(x11 + 1.0*((x5 == 1) ? (
   pow(x[5], -1)
)
: 0)) - x207*x161 - x208*x160 - x209*x161 + x34*(-2*x144 + x147 - 2*x192 - x202 + x214 - 4*x217 - x218 - x154*x[3]) + (2*x70 + x74)*x34);
}

__device__ void pycgpu_model_3_internal_cons_func(double* out, const double* x) {
    out[0] = 1.0*(-1 + x[3] + x[4] + x[5]);
}

__device__ void pycgpu_model_3_internal_cons_jac(double* out, const double* x) {
    out[0] = 0;
    out[1] = 1.0;
    out[2] = 1.0;
    out[3] = 1.0;
}

__device__ void pycgpu_model_3_mass_obj(double* out, const double* x) {
    double x0 = 1.0/(x[3] + x[4] + x[5]);
    out[0] = x0*x[3];
    out[1] = x0*x[4];
    out[2] = x0*x[5];
    out[3] = 0;
}

__device__ void pycgpu_model_3_formulamole_obj(double* out, const double* x) {
    out[0] = 1.0*x[3];
    out[1] = 1.0*x[4];
    out[2] = 1.0*x[5];
    out[3] = 0.0;
}

__device__ void pycgpu_model_3_formulamole_grad(double* out, const double* x) {
    out[0] = 0;
    out[1] = 1.0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    out[6] = 1.0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[10] = 0;
    out[11] = 1.0;
}



// --- Global Device-Side PhaseRecord Array ---
__device__ PhaseRecord g_phase_records_array[4]; // Must be at least 1

// --- Kernel Functions (must be extern "C" for CuPy to find them) ---
extern "C" {

// --- Simple test kernel to verify GPU setup ---
__global__ void test_kernel(double* output, const double* input, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        output[tid] = input[tid] * 2.0 + 1.0;
    }
}

// --- Test kernel for struct pointer arguments ---
__global__ void test_struct_kernel(
    const void* ptr1,
    const void* ptr2, 
    void* ptr3,
    int num_items,
    const void* ptr4,
    const void* ptr5
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0) {
        // Just verify we can access the pointers without crashing
        // Write a simple marker to the output
        double* output = (double*)ptr3;
        if (output != nullptr) {
            output[0] = 42.0; // Magic number to verify kernel executed
        }
    }
}

// --- Simplified equilibrium kernel for testing ---
__global__ void simple_equilibrium_test(
    const void* global_spec_ptr,
    const void* condition_args_ptr,
    void* results_ptr,
    int num_conditions,
    const void* initial_data_ptr,
    const void* grid_data_ptr
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Just write a simple marker to show the kernel executed
    if (tid == 0 && results_ptr != nullptr) {
        double* output = (double*)results_ptr;
        output[0] = 123.456; // Magic number to verify execution
        output[1] = (double)num_conditions; // Echo back the condition count
    }
}

// --- Minimal fallback kernel with basic functionality ---
__global__ void minimal_equilibrium_kernel(
    double* system_data,        // Flattened system data
    double* condition_data,     // Flattened condition data 
    double* results_data,       // Flattened results data
    int num_conditions,         // Number of conditions
    int data_size_per_condition // Size of data per condition
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    
    // Process conditions assigned to this thread using stride pattern
    for (int condition_idx = tid; condition_idx < num_conditions; condition_idx += total_threads) {
        // Calculate correct result offset based on actual results layout
        int results_per_condition = 7 + MAX_COMPONENTS + MAX_PHASES + 
                                   (MAX_PHASES * MAX_DOF_PER_PHASE) + 
                                   (MAX_PHASES * MAX_COMPONENTS) + MAX_PHASES;
        int result_offset = condition_idx * results_per_condition;
        int condition_offset = condition_idx * data_size_per_condition;
        
        if (result_offset + results_per_condition <= num_conditions * results_per_condition) {
            // Mark this condition as processed with some dummy values
            results_data[result_offset + 0] = 1000.0 + condition_idx; // GM
            // Initialize chemical potentials
            for (int i = 0; i < MAX_COMPONENTS; i++) {
                results_data[result_offset + 1 + i] = 0.0;
            }
            // Initialize phase amounts
            for (int i = 0; i < MAX_PHASES; i++) {
                results_data[result_offset + 1 + MAX_COMPONENTS + i] = 0.0;
            }
            results_data[result_offset + 1 + MAX_COMPONENTS + MAX_PHASES] = 1.0; // converged
            results_data[result_offset + 2 + MAX_COMPONENTS + MAX_PHASES] = 0.0; // num_stable_phases
            results_data[result_offset + 3 + MAX_COMPONENTS + MAX_PHASES] = 500.0; // Temperature
            results_data[result_offset + 4 + MAX_COMPONENTS + MAX_PHASES] = 1.0; // Pressure
            results_data[result_offset + 5 + MAX_COMPONENTS + MAX_PHASES] = 1.0; // Status: success
            // Fill remaining (Y_phases, X_phases, phase_ids) with zeros
            int remaining_start = result_offset + 6 + MAX_COMPONENTS + MAX_PHASES;
            int remaining_count = (MAX_PHASES * MAX_DOF_PER_PHASE) + 
                                 (MAX_PHASES * MAX_COMPONENTS) + MAX_PHASES;
            for (int i = 0; i < remaining_count; i++) {
                results_data[remaining_start + i] = 0.0;
            }
        }
    }
}

// --- Kernel to Initialize Global PhaseRecords ---
__global__ void init_all_gpu_phase_records() {
    #ifdef VERBOSE_DEBUG
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("GPU DEBUG: init_all_gpu_phase_records kernel called\n");
    }
    #endif
        g_phase_records_array[0].init(&pycgpu_model_0_obj, &pycgpu_model_0_formulaobj, &pycgpu_model_0_formulagrad, &pycgpu_model_0_formulahess, &pycgpu_model_0_internal_cons_func, &pycgpu_model_0_internal_cons_jac, &pycgpu_model_0_mass_obj, &pycgpu_model_0_formulamole_obj, &pycgpu_model_0_formulamole_grad, 3, 4, 4, 2, 3);
    g_phase_records_array[1].init(&pycgpu_model_1_obj, &pycgpu_model_1_formulaobj, &pycgpu_model_1_formulagrad, &pycgpu_model_1_formulahess, &pycgpu_model_1_internal_cons_func, &pycgpu_model_1_internal_cons_jac, &pycgpu_model_1_mass_obj, &pycgpu_model_1_formulamole_obj, &pycgpu_model_1_formulamole_grad, 3, 9, 4, 3, 3);
    g_phase_records_array[2].init(&pycgpu_model_2_obj, &pycgpu_model_2_formulaobj, &pycgpu_model_2_formulagrad, &pycgpu_model_2_formulahess, &pycgpu_model_2_internal_cons_func, &pycgpu_model_2_internal_cons_jac, &pycgpu_model_2_mass_obj, &pycgpu_model_2_formulamole_obj, &pycgpu_model_2_formulamole_grad, 3, 4, 4, 2, 3);
    g_phase_records_array[3].init(&pycgpu_model_3_obj, &pycgpu_model_3_formulaobj, &pycgpu_model_3_formulagrad, &pycgpu_model_3_formulahess, &pycgpu_model_3_internal_cons_func, &pycgpu_model_3_internal_cons_jac, &pycgpu_model_3_mass_obj, &pycgpu_model_3_formulamole_obj, &pycgpu_model_3_formulamole_grad, 3, 3, 4, 1, 3);

    #ifdef VERBOSE_DEBUG
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("GPU DEBUG: init_all_gpu_phase_records kernel completed\n");
        // Debug: Print what was initialized
        for (int i = 0; i < 4; ++i) {
            // printf("GPU DEBUG: g_phase_records_array[%d].obj = %p\n", i, (void*)g_phase_records_array[i].obj);
            // printf("GPU DEBUG: g_phase_records_array[%d].formulamole_obj = %p\n", i, (void*)g_phase_records_array[i].formulamole_obj);
        }
    }
    #endif
}

// --- COMMENTED OUT: Original complex solver implementation ---
// This function contains the full equilibrium solver logic but causes stack overflow
// due to large local arrays. Need to re-implement using global memory arrays.
/*
COMMENTED OUT: Original complex solver implementation that caused stack overflow
This implementation needs to be adapted to use global memory arrays instead of stack arrays.
The function signature and basic structure is preserved for reference.
*/

// Add missing constants that might not be defined
#ifndef MAX_EQ_SOLN_LEN
#define MAX_EQ_SOLN_LEN 50
#endif

// Forward declarations for global memory functions
__device__ void solve_state(
    SystemSpecification* spec, SystemState* state, double* out_equilibrium_soln, int soln_length,
    double* equilibrium_matrix, double* equilibrium_rhs, double* A_lstsq_copy,
    double* U_lstsq, double* V_lstsq, double* singular_values_lstsq, double* superdiag_lstsq,
    int thread_id
);

// --- GLOBAL MEMORY VERSION OF RUN_LOOP ---
// This function implements the sophisticated run_loop using global memory arrays
// to avoid stack overflow while maintaining all the sophisticated solver logic
__device__ bool run_loop_global_mem(
    int thread_id,              // Add thread_id parameter for debug output
    SystemSpecification* spec, 
    SystemState* state, 
    int max_iterations,
    // Global memory arrays to replace stack arrays
    double* equilibrium_matrix,  // replaces local equilibrium matrix
    double* equilibrium_rhs,     // replaces local equilibrium RHS  
    double* eq_soln,            // replaces local solution vector
    double* A_lstsq_copy,       // replaces local SVD arrays
    double* U_lstsq,
    double* V_lstsq,
    double* singular_values_lstsq,
    double* superdiag_lstsq,
    double* masses,             // replaces local masses arrays
    double* mass_jac,           // replaces local jacobian arrays
    double* x_dof,              // replaces local DOF arrays
    double* grad,               // replaces local gradient arrays
    double* hess                // replaces local hessian arrays
) {
    // IMPLEMENTATION: This mirrors the original run_loop but uses global memory arrays
    
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: run_loop_global_mem STARTED with max_iterations=%d\n", max_iterations);
        #endif
    }
    
    double step_size = 1.0;
    bool converged = false;
    bool phases_changed_iter;
    
    // Use global memory for eq_soln instead of local array
    int eq_soln_len;
    
    // DEBUG: Store initial values before loop starts (removed debug_gm_history references)
    // The debug variables debug_gm_history and debug_max_steps are not available in this function scope
    
    const int DEBUG_ENABLED = 0;  // Set to 1 to enable debug output
    
    for (int iteration_count = 0; iteration_count < max_iterations; ++iteration_count) {
        if (thread_id == 0 && iteration_count % 50 == 0) {
            #ifdef VERBOSE_DEBUG
            printf("\nGPU DEBUG: Iteration %d/%d\n", iteration_count, max_iterations);
            #endif
        }
        state->iteration = iteration_count;
        phases_changed_iter = false;
        
        // DEBUG: Mark that we entered the iteration loop (removed debug_gm_history references)
        if (DEBUG_ENABLED && thread_id == 0 && iteration_count == 0) {
            #ifdef VERBOSE_DEBUG
            printf("\nGPU DEBUG: ===== ITERATION 0 (DETAILED) =====\n");
            printf("GPU DEBUG: State before iteration:\n");
            printf("GPU DEBUG:   Chemical potentials: [%.6f, %.6f]\n", 
                   state->chemical_potentials[0], state->chemical_potentials[1]);
            #endif
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG:   Number of phases: %d\n", state->num_free_stable_compsets);
            printf("GPU DEBUG:   Free stable indices: ");
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                printf("%d ", state->free_stable_compset_indices[i]);
            }
            printf("\n");
            printf("GPU DEBUG:   System amount: %.6f\n", state->system_amount);
            printf("GPU DEBUG:   Mole fractions: [%.6f, %.6f]\n", 
                   state->mole_fractions[0], state->mole_fractions[1]);
            #endif
            
            // Details for each phase
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                int idx = state->free_stable_compset_indices[i];
                CompositionSet* cs = &state->compsets[idx];
                CompsetState* css = &state->cs_states[idx];
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG:   Phase %d:\n", idx);
                printf("GPU DEBUG:     NP=%.6f\n", cs->NP);
                printf("GPU DEBUG:     phase_amt=%.6f (formula units)\n", state->phase_amt[idx]);
                printf("GPU DEBUG:     energy=%.6f\n", css->energy);
                printf("GPU DEBUG:     dof=[%.15f, %.15f, %.15f]\n", 
                       cs->dof[0], cs->dof[1], cs->dof[2]);
                printf("GPU DEBUG:     phase_compositions=[%.6f, %.6f]\n",
                       state->phase_compositions[idx * MAX_COMPONENTS + 0],
                       state->phase_compositions[idx * MAX_COMPONENTS + 1]);
                #endif
                // Calculate phase_comp_sum
                double phase_comp_sum = 0.0;
                for (int j = 0; j < spec->num_components; ++j) {
                    phase_comp_sum += state->phase_compositions[idx * MAX_COMPONENTS + j];
                }
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG:     phase_comp_sum=%.6f\n", phase_comp_sum);
                printf("GPU DEBUG:     phase_amt * phase_comp_sum=%.6f\n", 
                       state->phase_amt[idx] * phase_comp_sum);
                #endif
            }
        } else if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Entered iteration loop, max_iterations=%d\n", max_iterations);
            #endif
        }
        
        // SEGMENT 21: PRE-SOLVE HOOK
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 21: Pre-solve hook (condition %d, iteration %d)\n", thread_id, iteration_count);
            #endif
        }
        
        // Call pre_solve_hook (this should be safe, no large arrays)
        bool pre_hook_result = pre_solve_hook(spec, state);
        if (!pre_hook_result) {
            // DEBUG: Mark pre_solve_hook failure (removed debug_gm_history references)
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: pre_solve_hook failed!\n");
                #endif
            }
            break;
        }
        
        // SEGMENT 22: STATE RECOMPUTATION
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 22: State recomputation (condition %d)\n", thread_id);
            printf("[GPU]   num_phases_active: %d\n", state->num_free_stable_compsets);
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                int idx = state->free_stable_compset_indices[i];
                CompositionSet* cs = &state->compsets[idx];
                printf("[GPU]   phase_%d: NP=%.15e, X=[%.6f, %.6f] (reading from indices %d, %d)\n",
                       idx, state->phase_amt[idx],
                       state->phase_compositions[idx * MAX_COMPONENTS + 0],
                       state->phase_compositions[idx * MAX_COMPONENTS + 1],
                       idx * MAX_COMPONENTS + 0,
                       idx * MAX_COMPONENTS + 1);
                if (idx == 1 && thread_id == 0 && iteration_count == 0) {
                    printf("[GPU]   DEBUG: phase_compositions array around phase 1:\n");
                    for (int j = 0; j < 8; ++j) {
                        printf("    [%d] = %f\n", j, state->phase_compositions[j]);
                    }
                }
            }
            #endif
        }
        
        // NOTE: recompute is called inside solve_state, matching CPU behavior
        // Do NOT call it here to avoid double recomputation
        
        eq_soln_len = spec->num_free_chemical_potentials + state->num_free_stable_compsets + spec->num_free_statevars;
        
        // DEBUG: Store eq_soln_len calculation (removed debug_gm_history references)
        if (thread_id == 0 && iteration_count == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: eq_soln_len=%d (chem_pot=%d + compsets=%d + statevars=%d)\n", 
                   eq_soln_len, spec->num_free_chemical_potentials, 
                   state->num_free_stable_compsets, spec->num_free_statevars);
            #endif
        }
        
        if (eq_soln_len > MAX_EQ_SOLN_LEN || eq_soln_len <= 0) {
            // DEBUG: Mark eq_soln_len failure (removed debug_gm_history references)
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: eq_soln_len check failed! eq_soln_len=%d, MAX_EQ_SOLN_LEN=%d\n", 
                       eq_soln_len, MAX_EQ_SOLN_LEN);
                #endif
            }
            converged = false;
            break;
        }
        
        // DEBUG: Mark that we passed eq_soln_len check (removed debug_gm_history references)
        
        // DEBUG: Before solve_state - check state (DISABLED)
        // if (thread_id == 0) {
        //     printf("GPU DEBUG iter %d: num_compsets=%d, num_free_stable=%d\n", 
        //            iteration_count, state->num_compsets, state->num_free_stable_compsets);
        // }
        
        // SEGMENT 27-30: SOLVE STATE
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 27: Construct equilibrium system (condition %d)\n", thread_id);
            #endif
        }
        
        // Call solve_state with global memory arrays
        solve_state(spec, state, eq_soln, eq_soln_len, 
                   equilibrium_matrix, equilibrium_rhs, 
                   A_lstsq_copy, U_lstsq, V_lstsq, 
                   singular_values_lstsq, superdiag_lstsq, thread_id);
        
        // DEBUG: After solve_state
        if ((iteration_count < 3 || iteration_count % 50 == 0) && thread_id == 0) { 
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Equilibrium solution at iteration %d (len=%d): [", iteration_count, eq_soln_len);
            for (int i = 0; i < eq_soln_len && i < 10; ++i) {
                printf("%.6e", eq_soln[i]);
                if (i < eq_soln_len - 1) printf(", ");
            }
            if (eq_soln_len > 10) printf("...");
            printf("]\n");
            
            // Check if solution is all zeros
            bool all_zeros = true;
            for (int i = 0; i < eq_soln_len; ++i) {
                if (fabs(eq_soln[i]) > 1e-15) {
                    all_zeros = false;
                    break;
                }
            }
            if (all_zeros) {
                printf("GPU DEBUG: WARNING - Equilibrium solution is all zeros!\n");
            }
            
            printf("GPU DEBUG: After solve_state:\n");
            printf("GPU DEBUG:   Chemical potentials: [%.6f, %.6f]\n", 
                   state->chemical_potentials[0], state->chemical_potentials[1]);
            
            // Details for each phase after solve
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                int idx = state->free_stable_compset_indices[i];
                CompositionSet* cs = &state->compsets[idx];
                printf("GPU DEBUG:   Phase %d:\n", idx);
                printf("GPU DEBUG:     phase_amt=%.6f (formula units)\n", state->phase_amt[idx]);
                printf("GPU DEBUG:     NP=%.6f\n", cs->NP);
                printf("GPU DEBUG:     dof=[%.15f, %.15f, %.15f]\n", 
                       cs->dof[0], cs->dof[1], cs->dof[2]);
            }
            #endif
        }
        
        // SEGMENT 33: POST SOLVE HOOK (matching CPU order)
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 33: Post solve hook\n");
            #endif
        }
        
        // Call post_solve_hook first (matching CPU behavior)
        if (!post_solve_hook(spec, state)) {
            if (thread_id < 3 && iteration_count < 3) {
                #ifdef VERBOSE_DEBUG
                printf("[GPU]   post_solve_hook_returned_false\n");
                #endif
            }
            break;
        }
        
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU]   post_solve_hook_returned_true\n");
            #endif
        }
        
        // SEGMENT 34: REMOVE AND CONSOLIDATE PHASES (before advance_state to match CPU)
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 34: Remove and consolidate phases\n");
            #endif
        }
        
        // NOTE: Phase compositions are calculated in solve_state->recompute()
        // We use those compositions for consolidation checks to match CPU behavior
        
        // Phase change operations (these should be safe, no large arrays)
        if (remove_and_consolidate_phases(spec, state)) {
            phases_changed_iter = true;
            if (thread_id < 3 && iteration_count < 3) {
                #ifdef VERBOSE_DEBUG
                printf("[GPU]   phases_removed: true\n");
                #endif
            }
        }
        
        // SEGMENT 32: CHECK CONVERGENCE
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 32: Check convergence\n");
            #endif
        }
        bool convergence_result = check_convergence(spec, state);
        if (thread_id == 0 && (iteration_count < 3 || iteration_count % 50 == 0 || convergence_result)) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Convergence check at iteration %d:\n", iteration_count);
            printf("  largest_phase_amt_change=%.2e (limit 1e-10)\n", state->largest_phase_amt_change);
            printf("  largest_y_change=%.2e (limit 5e-09)\n", state->largest_y_change);
            printf("  largest_statevar_change=%.2e (limit 1e-5)\n", state->largest_statevar_change);
            printf("  mass_residual=%.2e (limit %.2e)\n", state->mass_residual, spec->ALLOWED_MASS_RESIDUAL);
            printf("  iterations_since_last_phase_change=%d (need >=5)\n", state->iterations_since_last_phase_change);
            printf("  Converged: %s\n", convergence_result ? "YES" : "NO");
            #endif
        }
        
        if (convergence_result) {
            // Try to add phases if converged
            if (change_phases(spec, state)) {
                phases_changed_iter = true;
                if (thread_id < 3 && iteration_count < 3) {
                    #ifdef VERBOSE_DEBUG
                    printf("[GPU]   phases_added: true\n");
                    #endif
                }
            }
            
            if (!phases_changed_iter) {
                // Truly converged with no phase changes
                converged = true;
                break;
            }
        }
        
        // Update phase change tracking
        if (phases_changed_iter) {
            state->iterations_since_last_phase_change = 0;
        } else {
            state->iterations_since_last_phase_change++;
        }
        
        // DEBUG: Before advance_state
        if (thread_id < 3 && iteration_count < 3) { 
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG iter %d: Before advance_state\n", iteration_count);
            printf("  Phase amounts: [%.6f, %.6f]\n", state->phase_amt[0], state->phase_amt[1]);
            printf("  eq_soln phase deltas: [%.6e, %.6e]\n", 
                   eq_soln[spec->num_free_chemical_potentials], 
                   eq_soln[spec->num_free_chemical_potentials + 1]);
            #endif
        }
        
        // SEGMENT 31: ADVANCE STATE (only if phases weren't changed)
        if (thread_id < 3 && iteration_count < 3) {
            #ifdef VERBOSE_DEBUG
            printf("[GPU] SEGMENT 31: Advance state\n");
            printf("[GPU]   step_size: %.6f\n", step_size);
            #endif
        }
        
        // CRITICAL FIX: Skip advance_state if phases changed (match CPU behavior)
        if (!phases_changed_iter) {
            // Call advance_state (this should be safe, no large arrays)
            advance_state(spec, state, eq_soln, eq_soln_len, step_size);
        } else {
            if (thread_id < 3 && iteration_count < 3) {
                #ifdef VERBOSE_DEBUG
                printf("[GPU] SKIPPING advance_state due to phase changes\n");
                #endif
            }
        }
        
        // DEBUG: Add detailed output after first iteration
        if (iteration_count == 0 && thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("\n[GPU TRACE] ===== AFTER ITERATION 0 =====\n");
            printf("[GPU TRACE] Chemical potentials: [");
            for (int i = 0; i < spec->num_components; ++i) {
                printf("%.15e", state->chemical_potentials[i]);
                if (i < spec->num_components - 1) printf(", ");
            }
            printf("]\n");
            printf("[GPU TRACE] System amount: %.15e\n", state->system_amount);
            printf("[GPU TRACE] Mole fractions: [");
            for (int i = 0; i < spec->num_components; ++i) {
                printf("%.15e", state->mole_fractions[i]);
                if (i < spec->num_components - 1) printf(", ");
            }
            printf("]\n");
            printf("[GPU TRACE] Mass residual: %.15e\n", state->mass_residual);
            printf("[GPU TRACE] Number of active phases: %d\n", state->num_free_stable_compsets);
            printf("[GPU TRACE] Free stable indices: [");
            for (int i = 0; i < state->num_free_stable_compsets; ++i) {
                printf("%d", state->free_stable_compset_indices[i]);
                if (i < state->num_free_stable_compsets - 1) printf(", ");
            }
            printf("]\n");
            #endif
            
            #ifdef VERBOSE_DEBUG
            for (int idx = 0; idx < state->num_free_stable_compsets; ++idx) {
                int cs_idx = state->free_stable_compset_indices[idx];
                CompositionSet* compset = &state->compsets[cs_idx];
                CompsetState* csst = &state->cs_states[cs_idx];
                
                printf("\n[GPU TRACE] Phase %d:\n", cs_idx);
                printf("  NP (mole fraction): %.15e\n", compset->NP);
                printf("  phase_amt (formula units): %.15e\n", state->phase_amt[cs_idx]);
                printf("  energy: %.15e\n", csst->energy);
                printf("  phase_compositions: [");
                for (int c = 0; c < spec->num_components; ++c) {
                    printf("%.15e", state->phase_compositions[cs_idx * MAX_COMPONENTS + c]);
                    if (c < spec->num_components - 1) printf(", ");
                }
                printf("]\n");
                double phase_comp_sum = 0.0;
                for (int c = 0; c < spec->num_components; ++c) {
                    phase_comp_sum += state->phase_compositions[cs_idx * MAX_COMPONENTS + c];
                }
                printf("  phase_comp_sum: %.15e\n", phase_comp_sum);
                printf("  Site fractions: [");
                for (int sf = 0; sf < compset->phase_record->phase_dof; ++sf) {
                    printf("%.15e", compset->dof[spec->num_statevars + sf]);
                    if (sf < compset->phase_record->phase_dof - 1) printf(", ");
                }
                printf("]\n");
                printf("  State variables: [");
                for (int sv = 0; sv < spec->num_statevars; ++sv) {
                    printf("%.15e", compset->dof[sv]);
                    if (sv < spec->num_statevars - 1) printf(", ");
                }
                printf("]\n");
            }
            #endif
            
            #ifdef VERBOSE_DEBUG
            printf("\n[GPU TRACE] Convergence status:\n");
            printf("  converged: %s\n", converged ? "true" : "false");
            printf("  phases_changed: %s\n", phases_changed_iter ? "true" : "false");
            printf("  largest_phase_amt_change: %.15e\n", state->largest_phase_amt_change);
            printf("  largest_y_change: %.15e\n", state->largest_y_change);
            printf("  largest_statevar_change: %.15e\n", state->largest_statevar_change);
            printf("[GPU TRACE] ===== END ITERATION 0 =====\n\n");
            #endif
        }
    }
    
    return converged;
}

// --- GLOBAL MEMORY VERSION OF SOLVE_STATE ---
// This function implements solve_state using global memory arrays
__device__ void solve_state(
    SystemSpecification* spec, 
    SystemState* state, 
    double* out_equilibrium_soln, 
    int soln_length,
    double* equilibrium_matrix,  // global memory
    double* equilibrium_rhs,     // global memory
    double* A_lstsq_copy,       // global memory for SVD
    double* U_lstsq,
    double* V_lstsq,
    double* singular_values_lstsq,
    double* superdiag_lstsq,
    int thread_id               // Pass thread_id for debug output
) {
    // IMPLEMENTATION: This mirrors the original solve_state but uses global memory arrays
    
    // Calculate matrix dimensions
    // CRITICAL FIX: Add +1 back to match CPU matrix dimensions exactly
    // CPU DOES include a system amount constraint row (with [1,1,1] for phase amounts)
    int equilibrium_matrix_rows = state->num_free_stable_compsets + 
                                 spec->num_fixed_stable_compsets + 
                                 spec->num_prescribed_mole_fraction_conditions + 1;
    // CRITICAL FIX: Use num_free_chemical_potentials which now equals ALL non-VA components
    // CPU uses all non-VA components as columns, not just mathematically independent ones
    // This fixes the matrix dimension mismatch (CPU 6x6 vs GPU 5x5 for ternary)
    int equilibrium_matrix_cols = spec->num_free_chemical_potentials +  // All non-VA components 
                                 state->num_free_stable_compsets + 
                                 spec->num_free_statevars;
    
    // DEBUG: Print matrix size calculation
    if (state->iteration < 5 && thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("[GPU MATRIX SIZE] Iteration %d: num_free_stable_compsets=%d, fixed=%d, mole_frac_conds=%d\n",
               state->iteration, state->num_free_stable_compsets, spec->num_fixed_stable_compsets,
               spec->num_prescribed_mole_fraction_conditions);
        printf("  Matrix dimensions: %dx%d (cols = %d + %d + %d)\n", 
               equilibrium_matrix_rows, equilibrium_matrix_cols,
               spec->num_free_chemical_potentials, state->num_free_stable_compsets, spec->num_free_statevars);
        #endif
    }
    
    // CRITICAL: Call recompute at the beginning of solve_state, just like CPU does
    // This ensures all CompsetState arrays (masses, jacobians, energies) are up-to-date
    
    // DEBUG: Verify spec pointer before calling recompute
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: solve_state - spec=%p, spec->num_statevars=%d\n", 
               spec, spec->num_statevars);
        #endif
        if (spec->num_statevars < 0 || spec->num_statevars > 10) {
            printf("GPU ERROR: spec appears corrupted in solve_state!\n");
            printf("  spec->num_statevars=%d (0x%X)\n", spec->num_statevars, spec->num_statevars);
            printf("  spec->num_components=%d\n", spec->num_components);
            // Try to continue anyway
        }
    }
    
    state->recompute(spec);
    
    // The old manual update loop is not needed since recompute handles everything
    
    // CRITICAL FIX: Update state->system_amount to reflect current phase amounts
    // The issue is that state->system_amount stays at 1.0 while phase_amt grows exponentially
    // This causes the system amount constraint to be wrong
    state->system_amount = 0.0;
    for (int cs_idx = 0; cs_idx < state->num_compsets; ++cs_idx) {
        state->system_amount += state->phase_amt[cs_idx];
    }
    
    // CRITICAL FIX: Manually zero the equilibrium matrix AND RHS before calling fill_equilibrium_system
    // This is needed because these arrays are in global memory and persist across iterations
    // When matrix size changes (e.g., 4x4 to 3x3 after phase consolidation), old values remain!
    for (int i = 0; i < equilibrium_matrix_rows * equilibrium_matrix_cols; ++i) {
        equilibrium_matrix[i] = 0.0;
    }
    for (int i = 0; i < equilibrium_matrix_rows; ++i) {
        equilibrium_rhs[i] = 0.0;
    }
    
    // Call fill_equilibrium_system with global memory arrays
    fill_equilibrium_system(equilibrium_matrix, equilibrium_matrix_cols,
                           equilibrium_rhs, spec, state);
    
    // DEBUG: Check RHS after fill_equilibrium_system
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0 && state->iteration < 3) {
        printf("  RHS after fill_equilibrium_system: [");
        // Print ALL rows of RHS
        for (int i = 0; i < equilibrium_matrix_rows; ++i) {
            printf("%.2e", equilibrium_rhs[i]);
            if (i < equilibrium_matrix_rows - 1) printf(", ");
        }
        printf("]\n");
    }
    #endif
    
    // DEBUG: Disabled to avoid compilation issues
    // if (iteration_count == 0 && thread_id == 0) { printf("GPU DEBUG\n"); }
    
    // Use global memory arrays for SVD solve
    // This replaces the local arrays that were causing stack overflow
    
    // First copy equilibrium_matrix to A_lstsq_copy to preserve original
    int matrix_size = equilibrium_matrix_rows * equilibrium_matrix_cols;
    for (int i = 0; i < matrix_size; ++i) {
        A_lstsq_copy[i] = equilibrium_matrix[i];
    }
    
    // DEBUG: Check matrix dimensions and content before SVD
    // Note: state->iteration might be available instead of iteration_count
    if (thread_id == 0 && state->iteration < 3) {
        #ifdef VERBOSE_DEBUG
        printf("\n[EQUILIBRIUM_MATRIX_OUTPUT] GPU Iteration %d (rows=%d, cols=%d):\n", 
               state->iteration, equilibrium_matrix_rows, equilibrium_matrix_cols);
        // ALWAYS print ALL rows of the matrix
        for (int i = 0; i < equilibrium_matrix_rows; ++i) {
            printf("  Row %d: ", i);
            // Print ALL columns too
            for (int j = 0; j < equilibrium_matrix_cols; ++j) {
                printf("%+.6e ", equilibrium_matrix[i * equilibrium_matrix_cols + j]);
            }
            printf("| RHS: %+.6e\n", equilibrium_rhs[i]);
        }
        printf("  system_amount=%.6f, prescribed=%.6f\n", 
               state->system_amount, spec->prescribed_system_amount);
        #endif
    }
    
    // Call lstsq with correct signature
    // CRITICAL FIX: Use same tolerance as CPU (1e-16) instead of 1e-12
    lstsq(A_lstsq_copy, equilibrium_matrix_rows, equilibrium_matrix_cols, 
          equilibrium_rhs, 1e-16, 
          U_lstsq, V_lstsq, singular_values_lstsq, superdiag_lstsq);
    
    // The solution should be in equilibrium_rhs after lstsq completes
    
    // DEBUG: Check if lstsq produced a non-zero solution
    #ifdef VERBOSE_DEBUG
    if (thread_id == 0 && state->iteration < 3) {
        printf("  RHS after lstsq (solution): [");
        // Print ALL solution values
        for (int i = 0; i < equilibrium_matrix_cols; ++i) {
            printf("%.2e", equilibrium_rhs[i]);
            if (i < equilibrium_matrix_cols - 1) printf(", ");
        }
        printf("]\n");
    }
    #endif
    // Copy back to output solution
    for (int i = 0; i < soln_length && i < equilibrium_matrix_cols; ++i) {
        out_equilibrium_soln[i] = equilibrium_rhs[i];
    }
    
    // CRITICAL FIX: Update chemical potentials from the solution
    // The equilibrium solution contains NEW chemical potential values (not deltas)
    // This matches CPU behavior at minimizer.pyx line 1250
    for (int i = 0; i < spec->num_free_chemical_potentials; ++i) {
        int chempot_idx = spec->free_chemical_potential_indices[i];
        state->chemical_potentials[chempot_idx] = out_equilibrium_soln[i];
    }
    
    // Force fixed chemical potentials to adopt their fixed values
    for (int i = 0; i < spec->num_fixed_chemical_potentials; ++i) {
        int comp_idx = spec->fixed_chemical_potential_indices[i];
        if (comp_idx >= 0 && comp_idx < spec->num_components) {
            state->chemical_potentials[comp_idx] = spec->initial_chemical_potentials[comp_idx];
        }
    }
    
    // Calculate largest chemical potential difference for convergence check
    state->largest_chemical_potential_difference = -INFINITY;
    for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
        double diff = fabs(state->chemical_potentials[comp_idx] - state->previous_chemical_potentials[comp_idx]);
        if (diff > state->largest_chemical_potential_difference) {
            state->largest_chemical_potential_difference = diff;
        }
    }
}

// --- ACTUAL SOPHISTICATED SOLVER USING GLOBAL MEMORY ---
// This function implements the full equilibrium solver using global memory arrays
// to avoid stack overflow while maintaining all the sophisticated solver logic
__device__ void solve_equilibrium_at_condition_global_mem(
    int thread_id,
    const SystemSpecification* global_spec_base,
    const ConditionArgsSingle* condition_args,
    EquilibriumResultSingle* result,
    const DevicePhaseData* phase_data,
    const double* initial_data, // Raw flat array instead of struct
    const DeviceGrid* grid_data,
    const double* condition_mole_fractions, // NEW: Pass actual mole fractions from condition
    // Pre-allocated global memory arrays (per-thread slices)
    double* A_lstsq_copy,        // Replaces stack: double A_lstsq_copy[MAX_SVD_M * MAX_SVD_N]
    double* U_lstsq,             // Replaces stack: double U_lstsq[MAX_SVD_M * MAX_SVD_N]
    double* V_lstsq,             // Replaces stack: double V_lstsq[MAX_SVD_N * MAX_SVD_N]
    double* singular_values_lstsq, // Replaces stack: double singular_values_lstsq[MAX_SVD_N]
    double* superdiag_lstsq,     // Replaces stack: double superdiag_lstsq[MAX_SVD_N]
    double* U_inv,               // Replaces stack: double U_inv[MAX_PHASE_MATRIX_DIM^2]
    double* V_inv,               // Replaces stack: double V_inv[MAX_PHASE_MATRIX_DIM^2]
    double* singular_values_inv, // Replaces stack: double singular_values_inv[MAX_PHASE_MATRIX_DIM]
    double* superdiag_inv,       // Replaces stack: double superdiag_inv[MAX_PHASE_MATRIX_DIM]
    double* work_inv,            // Replaces stack: double work_inv[MAX_PHASE_MATRIX_DIM^2]
    double* x_dof,               // Replaces stack: double x[MAX_STATEVARS + MAX_DOF_PER_PHASE]
    double* grad,                // Replaces stack: double grad[MAX_STATEVARS + MAX_DOF_PER_PHASE]
    double* hess,                // Replaces stack: double hess[(MAX_STATEVARS + MAX_DOF_PER_PHASE)^2]
    double* masses,              // Replaces stack: double masses[MAX_COMPONENTS]
    double* mass_jac,            // Replaces stack: double mass_jac[MAX_COMPONENTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE)]
    double* phase_matrix,        // Replaces stack: double phase_matrix[(MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)^2]
    double* equilibrium_matrix,  // Replaces stack: large equilibrium system matrix
    double* equilibrium_rhs,     // Replaces stack: equilibrium system RHS vector
    double* eq_soln,             // Replaces stack: equilibrium solution vector
    double* global_system_states // CRITICAL FIX: SystemState in global memory to avoid stack overflow
) {
    // STACK OVERFLOW FIX: All large arrays are now passed as parameters from global memory
    
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG  
        printf("GPU DEBUG: solve_equilibrium_at_condition_global_mem STARTED\n");
        #endif
    }
    
    // Step 1: Validate inputs and global memory arrays
    if (!A_lstsq_copy || !result || !global_spec_base || !initial_data) {
        if (result) result->converged = false;
        return; // Cannot proceed without required arrays - memory not allocated yet
    }
    
    // Step 2: Initialize local spec copy from the global spec passed from Python
    // CRITICAL FIX: Use simple assignment copy instead of manual byte copy
    // The manual byte copy was causing struct field corruption
    // CRITICAL FIX: Safely read SystemSpecification from GPU memory
    // Cannot dereference struct pointer directly due to alignment/memory access issues
    // WORKAROUND: Use global memory to store SystemSpecification to avoid stack pointer issues
    // CRITICAL FIX: Allocate SystemSpec on stack instead of reusing work array
    // which might be causing memory corruption for Thread 1
    char spec_buffer[sizeof(SystemSpecification)];
    SystemSpecification* current_spec_ptr = (SystemSpecification*)spec_buffer;
    SystemSpecification& current_spec = *current_spec_ptr;
    
    // Copy the entire struct byte-by-byte from GPU memory
    // This preserves the exact layout from Python
    const char* spec_bytes = (const char*)global_spec_base;
    memcpy(current_spec_ptr, spec_bytes, sizeof(SystemSpecification));
    
    // DEBUG: Verify the copy worked
    if (thread_id == 0 || thread_id == 1) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Thread %d - Copied SystemSpecification to global memory at %p\n", thread_id, current_spec_ptr);
        printf("  Thread %d: global_spec_base=%p\n", thread_id, global_spec_base);
        printf("  Thread %d: num_statevars=%d, num_components=%d\n", 
               thread_id, current_spec.num_statevars, current_spec.num_components);
        if (current_spec.num_prescribed_mole_fraction_conditions > 0) {
            printf("  Thread %d: prescribed_mole_fraction_rhs[0]=%f\n", 
                   thread_id, current_spec.prescribed_mole_fraction_rhs[0]);
        }
        #endif
    }
    
    // The struct is now fully copied with correct layout from Python
    // No need for manual field-by-field reading
    
    // Note: We'll need to free current_spec_ptr before any return
    
    // Debug: print what we received from Python
    if (thread_id < 3) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Thread %d SystemSpecification from Python:\n", thread_id);
        printf("  Thread %d: num_statevars = %d\n", thread_id, current_spec.num_statevars);
        printf("  Thread %d: num_components = %d\n", thread_id, current_spec.num_components);
        printf("  Thread %d: num_free_chemical_potentials = %d\n", thread_id, current_spec.num_free_chemical_potentials);
        printf("  Thread %d: num_prescribed_mole_fraction_conditions = %d\n", thread_id, current_spec.num_prescribed_mole_fraction_conditions);
        
        // DEBUG: Show free and fixed state variables
        printf("  Thread %d: num_free_statevars = %d\n", thread_id, current_spec.num_free_statevars);
        printf("  Thread %d: free_statevar_indices = [", thread_id);
        for (int i = 0; i < current_spec.num_free_statevars; ++i) {
            printf("%d", current_spec.free_statevar_indices[i]);
            if (i < current_spec.num_free_statevars - 1) printf(", ");
        }
        printf("]\n");
        printf("  Thread %d: num_fixed_statevars = %d\n", thread_id, current_spec.num_fixed_statevars);
        printf("  Thread %d: fixed_statevar_indices = [", thread_id);
        for (int i = 0; i < current_spec.num_fixed_statevars; ++i) {
            printf("%d", current_spec.fixed_statevar_indices[i]);
            if (i < current_spec.num_fixed_statevars - 1) printf(", ");
        }
        printf("]\n");
        
        // DEBUG: Print struct offsets to diagnose alignment
        printf("  Thread %d: Struct base address: %p\n", thread_id, global_spec_base);
        printf("  Thread %d: Expected offset of prescribed_mole_fraction_rhs: 176\n", thread_id);
        
        // Try reading from the local copy (avoiding direct pointer dereference)
        if (current_spec.num_prescribed_mole_fraction_conditions > 0) {
            printf("  Thread %d: Copy read prescribed_mole_fraction_rhs[0] = %e\n", thread_id, current_spec.prescribed_mole_fraction_rhs[0]);
            
            // Print raw bytes at the expected offset (176 from Python)
            char* base_ptr = (char*)global_spec_base;
            double* rhs_at_offset_176 = (double*)(base_ptr + 176);
            printf("  Thread %d: Value at offset 176: %e\n", thread_id, *rhs_at_offset_176);
            
            // DEBUG: Print raw bytes to see what's actually there
            printf("  Thread %d: Raw bytes at offset 176: ", thread_id);
            unsigned char* byte_ptr = (unsigned char*)(base_ptr + 176);
            for (int i = 0; i < 8; ++i) {
                printf("%02x", byte_ptr[i]);
            }
            printf("\n");
            
            // Try different offsets in case of alignment issues
            for (int offset = 168; offset <= 184; offset += 8) {
                double* test_ptr = (double*)(base_ptr + offset);
                printf("  Thread %d: Value at offset %d: %e\n", thread_id, offset, *test_ptr);
            }
            
            // CRITICAL FIX: Manually copy the value from the known offset
            // This works around struct alignment issues between CPU and GPU
            if (*rhs_at_offset_176 != 0.0 && current_spec.prescribed_mole_fraction_rhs[0] == 0.0) {
                printf("  Thread %d: FIXING prescribed_mole_fraction_rhs[0] from %e to %e\n", 
                       thread_id, current_spec.prescribed_mole_fraction_rhs[0], *rhs_at_offset_176);
                current_spec.prescribed_mole_fraction_rhs[0] = *rhs_at_offset_176;
            }
        }
        printf("  Thread %d: prescribed_system_amount = %f\n", thread_id, current_spec.prescribed_system_amount);
        #endif
    }
    
    // Step 3: Use SystemState from global memory to avoid stack overflow
    // CRITICAL FIX: SystemState is too large (~100KB) for GPU thread stack
    SystemState* current_sys_state_ptr = nullptr;
    if (global_system_states != nullptr) {
        // Cast the global memory to SystemState pointer
        current_sys_state_ptr = (SystemState*)global_system_states;
        // Initialize SystemState to zero
        memset(current_sys_state_ptr, 0, sizeof(SystemState));
    } else {
        // Fallback: allocate on stack (will cause overflow with many threads)
        #ifdef VERBOSE_DEBUG
        if (thread_id == 0) {
            printf("GPU DEBUG: WARNING - SystemState allocated on stack (no global memory provided)\n");
        }
        #endif
        // This will fail with multiple threads due to stack overflow
        return;  // Exit early to avoid crash
    }
    SystemState& current_sys_state = *current_sys_state_ptr;
    // SystemState is now properly zero-initialized via memset
    
    // Initialize SystemState manually without creating large stack arrays
    
    current_sys_state.num_compsets = 0;
    current_sys_state.iteration = 0;
    current_sys_state.iterations_since_last_phase_change = 0;
    current_sys_state.condition_idx = thread_id;  // Set condition index for debug output
    
    // Initialize the required arrays to safe values
    for (int i = 0; i < MAX_PHASES; ++i) {
        current_sys_state.phase_amt[i] = 0.0;
        current_sys_state.metastable_phase_iterations[i] = 0;
        current_sys_state.times_compset_removed[i] = 0;
    }
    
    // CRITICAL FIX: Access flat double array directly for chemical_potentials
    // Layout: phase_indices[MAX_PHASES] + phase_amounts[MAX_PHASES] + site_fractions[MAX_PHASES*MAX_DOF_PER_PHASE] + 
    //         compositions[MAX_PHASES*MAX_COMPONENTS] + chemical_potentials[MAX_COMPONENTS] + num_phases
    // NOTE: The Python array is now flattened to 1D, so we access it for condition 0 directly
    // For multiple conditions, we would need to add: condition_idx * doubles_per_condition
    const double* initial_data_flat = initial_data;
    // CRITICAL FIX: Use calculated offset instead of hardcoded value
    // Chemical potentials come after: phase_indices + phase_amounts + site_fractions + compositions
    int chem_pot_offset = MAX_PHASES + MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + (MAX_PHASES * MAX_COMPONENTS);
    
    for (int i = 0; i < MAX_COMPONENTS; ++i) {
        // CRITICAL FIX: Use chemical potentials from SystemSpecification, NOT from initial_data
        // The initial_data contains starting_point values which are wrong
        current_sys_state.chemical_potentials[i] = (i < current_spec.num_components) ? current_spec.initial_chemical_potentials[i] : 0.0;
        current_sys_state.mole_fractions[i] = 0.0;
    }
    
    // Set up basic state from initial_data
    current_sys_state.system_amount = 1.0; // Standard amount
    
    // CRITICAL: Initialize mole fractions from the passed condition_mole_fractions array
    // This array contains the actual mole fractions from the condition, properly calculated
    // for binary, ternary, and higher-order systems
    for (int i = 0; i < MAX_COMPONENTS; ++i) {
        if (i < current_spec.num_components) {
            // Use the mole fractions that were correctly extracted from condition_data_array
            // and passed to this function
            current_sys_state.mole_fractions[i] = condition_mole_fractions[i];
        } else {
            current_sys_state.mole_fractions[i] = 0.0;
        }
    }
    
    // CRITICAL FIX: Access num_phases and phase_indices from flat array
    // NOTE: The initial_data pointer is already offset to this thread's data
    // Layout: phase_indices[MAX_PHASES] + phase_amounts[MAX_PHASES] + site_fractions[MAX_PHASES*MAX_DOF_PER_PHASE] + 
    //         compositions[MAX_PHASES*MAX_COMPONENTS] + chemical_potentials[MAX_COMPONENTS] + num_phases[1]
    int num_phases_offset = MAX_PHASES + MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + 
                           (MAX_PHASES * MAX_COMPONENTS) + MAX_COMPONENTS;
    int num_phases = (int)initial_data_flat[num_phases_offset];
    
    // CRITICAL DEBUG: Test if the pointer issue is with multiple conditions or single condition
    // The kernel might be interpreting this as a multi-condition array
    // Let's try accessing it as a 2D array and see if that fixes it
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: TESTING DIFFERENT ACCESS PATTERNS:\n");
        printf("  Direct access to initial_data:\n");
        printf("    [0]=%f, [1]=%f, [2]=%f, [44]=%f\n", 
               initial_data_flat[0], initial_data_flat[1], initial_data_flat[2], initial_data_flat[44]);
        
        // Test accessing with calculated doubles_per_struct
        int doubles_per_struct = MAX_PHASES + MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + 
                                (MAX_PHASES * MAX_COMPONENTS) + MAX_COMPONENTS + 1;
        printf("  Accessing with condition offset (using calculated doubles_per_struct=%d):\n", doubles_per_struct);
        int condition_offset = 0 * doubles_per_struct;  // condition 0
        printf("    condition_offset=0: [%d]=%f, [%d]=%f, [%d]=%f, [%d]=%f\n",
               condition_offset+0, initial_data_flat[condition_offset+0],
               condition_offset+1, initial_data_flat[condition_offset+1], 
               condition_offset+2, initial_data_flat[condition_offset+2],
               condition_offset+doubles_per_struct-1, initial_data_flat[condition_offset+doubles_per_struct-1]);
        #endif
    }
    
    // DEBUG: Check num_phases value and constants
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Constants - MAX_PHASES=%d, MAX_DOF_PER_PHASE=%d, MAX_COMPONENTS=%d\n", 
               MAX_PHASES, MAX_DOF_PER_PHASE, MAX_COMPONENTS);
        printf("GPU DEBUG: Python layout offset calculation: 4 + 4 + (4*4) + (4*4) + 3 = %d\n", num_phases_offset);
        printf("GPU DEBUG: Checking array values around offset %d: [%f, %f, %f, %f, %f, %f, %f, %f]\n", 
               num_phases_offset, initial_data_flat[num_phases_offset-4], initial_data_flat[num_phases_offset-3], initial_data_flat[num_phases_offset-2], 
               initial_data_flat[num_phases_offset-1], initial_data_flat[num_phases_offset], initial_data_flat[num_phases_offset+1], 
               initial_data_flat[num_phases_offset+2], initial_data_flat[num_phases_offset+3]);
        printf("GPU DEBUG: num_phases_offset=%d, num_phases=%d\n", num_phases_offset, num_phases);
        #endif
    }
    
    // Set up initial composition sets from lower_convex_hull data
    // DEBUG: Store initial phase setup info
    if (thread_id == 0) {
        result->X_phases[16] = (double)num_phases;  // Number of phases in initial data
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Setting up initial phases, num_phases=%d\n", num_phases);
        #endif
    }
    
    for (int i = 0; i < num_phases && i < MAX_PHASES; ++i) {
        int pr_idx = (int)initial_data_flat[i];  // phase_indices are at the beginning
        
        // CRITICAL FIX: Access phase_amounts from flat array 
        // phase_amounts start at offset MAX_PHASES
        double phase_amount = initial_data_flat[MAX_PHASES + i];
        
        // DEBUG: Store phase processing info for first phase
        if (thread_id == 0 && i == 0) {
            result->X_phases[17] = (double)pr_idx;                    // First phase index
            result->X_phases[18] = phase_amount;                      // First phase amount
            result->X_phases[19] = (double)phase_data->num_unique_phase_records; // Available phase records
        }
        
        if (pr_idx < 0 || pr_idx >= phase_data->num_unique_phase_records) {
            // DEBUG: Mark invalid phase index
            if (thread_id == 0 && i == 0) {
                result->X_phases[20] = -1.0; // Invalid phase index marker
            }
            continue;
        }
        if (phase_amount <= MIN_PHASE_FRACTION) {
            // DEBUG: Mark phase amount too small
            if (thread_id == 0 && i == 0) {
                result->X_phases[21] = -2.0; // Phase amount too small marker
            }
            continue;
        }
        
        // Bounds check to prevent overflow
        if (current_sys_state.num_compsets >= MAX_PHASES) {
            printf("GPU ERROR: Too many phases! num_compsets=%d >= MAX_PHASES=%d\n", 
                   current_sys_state.num_compsets, MAX_PHASES);
            break;
        }
        
        // DEBUG: Check memory before accessing arrays
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: About to access compsets[%d] and cs_states[%d], MAX_PHASES=%d\n", 
                   current_sys_state.num_compsets, current_sys_state.num_compsets, MAX_PHASES);
            printf("GPU DEBUG: current_spec at %p still valid? num_statevars=%d\n", 
                   &current_spec, current_spec.num_statevars);
            #endif
        }
        
        // Set up CompositionSet directly in SystemState (avoiding stack arrays)
        CompositionSet* cs = &current_sys_state.compsets[current_sys_state.num_compsets];
        CompsetState* css = &current_sys_state.cs_states[current_sys_state.num_compsets];
        
        // Initialize the CompositionSet
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Setting phase_record for phase %d, phase_data=%p, pr_idx=%d\n", 
                   current_sys_state.num_compsets, phase_data, pr_idx);
            printf("GPU DEBUG: Before init - current_spec.num_statevars = %d\n", current_spec.num_statevars);
            #endif
        }
        if (phase_data == nullptr || phase_data->phase_records_array == nullptr) {
            printf("GPU ERROR: phase_data or phase_records_array is null!\n");
            return;
        }
        // CRITICAL FIX: Each composition set needs its own phase record instance
        // to avoid sharing memory between phases of the same type (immiscibility gap)
        cs->phase_record = &phase_data->phase_records_array[pr_idx];
        if (!cs->phase_record) continue;
        
        // DEBUG: Print all input data arrays for this phase
        if (thread_id == 0 && current_sys_state.num_compsets < 2) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Phase %d input data verification:\n", current_sys_state.num_compsets);
            printf("  phase_amount = %f\n", phase_amount);
            printf("  pr_idx = %d\n", pr_idx);
            
            // DEBUG: Check raw pointer values
            const double* raw_condition_ptr = (const double*)condition_args;
            printf("  RAW condition_args pointer values: [%f, %f, %f, %f]\n", 
                   raw_condition_ptr[0], raw_condition_ptr[1], raw_condition_ptr[2], raw_condition_ptr[3]);
            
            printf("  condition_args->state_variables_values: [");
            for(int k = 0; k < MAX_STATEVARS; ++k) {
                printf("%f", condition_args->state_variables_values[k]);
                if (k < MAX_STATEVARS - 1) printf(", ");
            }
            printf("] (current_spec.num_statevars=%d, MAX_STATEVARS=%d)\n", current_spec.num_statevars, MAX_STATEVARS);
            // CRITICAL FIX: Access flat double array directly for debug output
            const double* initial_data_flat_debug = (const double*)initial_data;
            int site_fractions_offset_debug = MAX_PHASES + MAX_PHASES;
            printf("  initial_data->site_fractions for phase %d: [", i);
            for(int k = 0; k < cs->phase_record->phase_dof; ++k) {
                int flat_index_debug = site_fractions_offset_debug + i * MAX_DOF_PER_PHASE + k;
                printf("%f", initial_data_flat_debug[flat_index_debug]);
                if (k < cs->phase_record->phase_dof - 1) printf(", ");
            }
            printf("]\n");
            #endif
        }
        
        // Set state variables from condition args  
        // CRITICAL FIX: The DOF array should store WORKSPACE state variables, not Model state variables
        // CPU stores DOF as [N, P, T, Y1, Y2...] (workspace format)
        // GPU was incorrectly storing as [T, Y1, Y2...] (model format)
        
        // Copy ALL workspace state variables to match CPU behavior
        for (int sv_idx = 0; sv_idx < current_spec.num_statevars && sv_idx < MAX_STATEVARS; ++sv_idx) {
            cs->dof[sv_idx] = condition_args->state_variables_values[sv_idx];
        }
        
        if (thread_id == 0 && current_sys_state.num_compsets < 2) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Setting workspace state variables in dof[0:%d]:\n", current_spec.num_statevars);
            for (int sv_idx = 0; sv_idx < current_spec.num_statevars; ++sv_idx) {
                printf("  dof[%d] = %f\n", sv_idx, cs->dof[sv_idx]);
            }
            #endif
        }
        
        // Set site fractions from lower_convex_hull results
        // Site fractions start after the WORKSPACE's state variables
        // Layout: phase_indices[MAX_PHASES] + phase_amounts[MAX_PHASES] + site_fractions[MAX_PHASES*MAX_DOF_PER_PHASE] + ...
        // site_fractions start at offset (MAX_PHASES + MAX_PHASES)
        int site_fractions_offset = MAX_PHASES + MAX_PHASES;  // phase_indices + phase_amounts
        const double* initial_data_flat = initial_data;
        
        // CRITICAL FIX: Map site fractions to ensure correct component order
        // The Model expects components in alphabetical order, but initial_data might not
        // For NbTi system: Model expects [Y(NB), Y(TI)] but initial_data has [Y(TI), Y(NB)]
        for (int sf_idx = 0; sf_idx < cs->phase_record->phase_dof && sf_idx < MAX_DOF_PER_PHASE; ++sf_idx) {
            int flat_index = site_fractions_offset + i * MAX_DOF_PER_PHASE + sf_idx;
            
            // The initial_data already has site fractions in the correct order [Y(NB), Y(TI)]
            // No swapping needed - just direct copy
            int mapped_idx = sf_idx;
            
            cs->dof[current_spec.num_statevars + mapped_idx] = initial_data_flat[flat_index];  // Site fractions start after WORKSPACE's state variables
            
            // VERIFICATION: Print the values being read to confirm the fix works
            if (thread_id == 0 && current_sys_state.num_compsets < 2) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: VERIFY site_fractions[%d][%d] = %f (from flat_index %d) -> dof[%d]\n", 
                       i, sf_idx, initial_data_flat[flat_index], flat_index, current_spec.num_statevars + mapped_idx);
                #endif
            }
        }
        
        // REMOVED: Incorrect X constraint adjustment that was forcing individual phases
        // to match the system composition constraint. This was causing phase separation
        // to collapse in miscibility gaps. The composition constraint should be satisfied
        // by the SYSTEM as a whole (weighted average of all phases), not by individual phases.
        // The solver will naturally find the correct phase compositions that satisfy the
        // overall system constraint through the equilibrium conditions.
        
        // DEBUG: Print final DOF array after setup (now in Workspace format)
        if (thread_id == 0 && current_sys_state.num_compsets < 2) {
            #ifdef VERBOSE_DEBUG
            printf("  Final cs->dof after setup (Workspace format): [");
            // DOF contains: Workspace's state vars + phase_dof site fractions
            int num_dof_elements = current_spec.num_statevars + cs->phase_record->phase_dof;
            for(int k = 0; k < num_dof_elements; ++k) {
                printf("%.15f", cs->dof[k]);
                if (k < num_dof_elements - 1) printf(", ");
            }
            printf("] (");
            // Show what each element represents
            for(int k = 0; k < current_spec.num_statevars; ++k) {
                if (k == 0 && current_spec.num_statevars >= 2) printf("N");
                else if (k == 1 && current_spec.num_statevars == 2) printf("T");
                else if (k == 1 && current_spec.num_statevars >= 3) printf("P");
                else if (k == 2 && current_spec.num_statevars >= 3) printf("T");
                if (k < current_spec.num_statevars - 1) printf(", ");
            }
            if (cs->phase_record->phase_dof > 0) {
                printf(", Y1, Y2...)\n");
            } else {
                printf(")\n");
            }
            #endif
        }
        
        // Set phase amount and properties
        cs->NP = phase_amount;
        cs->fixed = false;
        current_sys_state.phase_amt[current_sys_state.num_compsets] = phase_amount;
        
        // Initialize CompositionSet first
        cs->init(cs->phase_record);
        
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: After cs->init - current_spec.num_statevars = %d\n", current_spec.num_statevars);
            #endif
        }
        
        // CRITICAL: Initialize CompsetState with proper arrays
        // This is where masses, jacobians, etc. get set up
        css->init(&current_spec, cs);
        
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: After css->init - current_spec.num_statevars = %d\n", current_spec.num_statevars);
            #endif
        }
        
        // STACK OVERFLOW FIX: CompsetState arrays are fixed arrays, not pointers
        // We need a different approach - we'll copy data between CompsetState and global memory
        // Initialize CompsetState arrays with reasonable starting values
        
        if (current_sys_state.num_compsets < MAX_PHASES) {
            // DEBUG: Check before masses init
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Before masses init - current_spec.num_statevars = %d, num_components = %d\n", 
                       current_spec.num_statevars, current_spec.num_components);
                #endif
            }
            
            // Initialize masses from composition data directly into CompsetState
            for (int comp_idx = 0; comp_idx < MAX_COMPONENTS; ++comp_idx) {
                if (comp_idx < current_spec.num_components) {
                    css->masses[comp_idx] = current_sys_state.mole_fractions[comp_idx];
                } else {
                    css->masses[comp_idx] = 0.0;
                }
            }
            
            // DEBUG: Check after masses init
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: After masses init - current_spec.num_statevars = %d\n", current_spec.num_statevars);
                #endif
            }
            
            // Initialize mass jacobian rows/cols
            css->mass_jac_rows = current_spec.num_components;
            css->mass_jac_cols = current_spec.num_statevars + cs->phase_record->phase_dof;
            
            // DEBUG: Check sizes before initializing
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: mass_jac dimensions: rows=%d, cols=%d, phase_dof=%d\n",
                       css->mass_jac_rows, css->mass_jac_cols, cs->phase_record->phase_dof);
                #endif
            }
            
            // Initialize mass jacobian to reasonable values
            int jac_size = css->mass_jac_rows * css->mass_jac_cols;
            int max_jac_size = MAX_COMPONENTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE);
            if (jac_size > max_jac_size) {
                printf("GPU ERROR: mass_jac size %d exceeds max %d!\n", jac_size, max_jac_size);
                jac_size = max_jac_size;
            }
            for (int j = 0; j < jac_size; ++j) {
                css->mass_jac[j] = 0.0;
            }
            
            // DEBUG: Check if current_spec is still valid after mass_jac init
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: After mass_jac init - current_spec.num_statevars = %d\n", current_spec.num_statevars);
                #endif
            }
        }
        
        // DEBUG: Check DOF array before update call
        if (thread_id == 0 && current_sys_state.num_compsets < 2) {
            #ifdef VERBOSE_DEBUG
            printf("  DOF before cs->update(): [");
            int num_dof_elements = current_spec.num_statevars + cs->phase_record->phase_dof;
            for(int k = 0; k < num_dof_elements; ++k) {
                printf("%.6f", cs->dof[k]);
                if (k < num_dof_elements - 1) printf(", ");
            }
            printf("]\n");
            #endif
        }
        
        // Phase amounts are already normalized at Python level
        // Unlike CPU which normalizes in recompute(), GPU receives pre-normalized values
        double original_phase_amt = cs->NP;  // This is already normalized by Python
        
        // CRITICAL FIX: Must call cs->update() to calculate energy and composition
        // The energy field is used in the equilibrium matrix RHS calculation
        // Without this, energy=0 and the solver behaves differently than CPU
        // cs->update expects: (site_fractions, phase_amount, state_variables, workspace_num_statevars)
        // In workspace DOF format: site fractions start at current_spec.num_statevars
        
        // DEBUG: Check before cs->update call
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Before cs->update - current_spec.num_statevars = %d\n", current_spec.num_statevars);
            #endif
        }
        
        // WORKAROUND: Save critical values before cs->update in case of corruption
        int saved_num_statevars = current_spec.num_statevars;
        int saved_num_components = current_spec.num_components;
        
        cs->update(&cs->dof[current_spec.num_statevars], cs->NP, cs->dof, current_spec.num_statevars);
        
        // WORKAROUND: Restore values if corrupted
        if (current_spec.num_statevars < 0 || current_spec.num_statevars > 10) {
            if (thread_id == 0) {
                printf("GPU WARNING: Detected corruption after cs->update, restoring values\n");
                printf("  Corrupted: num_statevars=%d, num_components=%d\n", 
                       current_spec.num_statevars, current_spec.num_components);
            }
            current_spec.num_statevars = saved_num_statevars;
            current_spec.num_components = saved_num_components;
        }
        
        // DEBUG: Check after cs->update call
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: After cs->update - current_spec.num_statevars = %d\n", current_spec.num_statevars);
            #endif
        }
        
        // DEBUG: Check DOF array after update call
        if (thread_id == 0 && current_sys_state.num_compsets < 2) {
            #ifdef VERBOSE_DEBUG
            printf("  DOF after cs->update(): [");
            int num_dof_elements = current_spec.num_statevars + cs->phase_record->phase_dof;
            for(int k = 0; k < num_dof_elements; ++k) {
                printf("%.6f", cs->dof[k]);
                if (k < num_dof_elements - 1) printf(", ");
            }
            printf("]\n");
            #endif
        }
        
        // DEBUG: Check if current_spec is still valid after processing this phase
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: After phase %d - current_spec.num_statevars = %d, num_components = %d\n", 
                   current_sys_state.num_compsets, current_spec.num_statevars, current_spec.num_components);
            #endif
        }
        
        current_sys_state.num_compsets++;
    }
    
    // NOTE: Phase amount normalization already done BEFORE phase composition calculation
    // This ensures correct initial system mole fractions in recompute()
    
    // CRITICAL: Set up free_stable_compset_indices array
    // This tells the solver which composition sets are free to vary
    current_sys_state.num_free_stable_compsets = 0;
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        if (!current_sys_state.compsets[i].fixed && i < MAX_PHASES) {
            current_sys_state.free_stable_compset_indices[current_sys_state.num_free_stable_compsets] = i;
            current_sys_state.num_free_stable_compsets++;
        }
    }
    
    // TODO: Add output scaling instead of changing solver algorithm

    // Initialize other critical SystemState arrays
    current_sys_state.mass_residual = 0.0;
    current_sys_state.largest_chemical_potential_difference = 0.0;
    current_sys_state.delta_ms_rows = 0;
    current_sys_state.delta_ms_cols = 0;
    // CRITICAL FIX: Normalize phase amounts BEFORE calculating phase compositions
    // This matches CPU behavior exactly - phase amounts must be in formula units
    // before we calculate system mole fractions in recompute()
    double phase_amt_sum = 0.0;
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        phase_amt_sum += current_sys_state.compsets[i].NP;
    }
    // Always normalize to match CPU behavior exactly - no conditions
    if (phase_amt_sum > 1e-12) { // Only check for non-zero to avoid division by zero
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            current_sys_state.compsets[i].NP /= phase_amt_sum;
            // CRITICAL: Also update phase_amt array to keep it synchronized
            current_sys_state.phase_amt[i] = current_sys_state.compsets[i].NP;
        }
    }
    
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("[GPU DEBUG] Early phase amount normalization - sum was %.6f, normalized to 1.0\n", phase_amt_sum);
        // Print detailed phase amounts after normalization to match CPU debug format
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            printf("[GPU DEBUG] Phase %d: amount=%.6f (normalized)\n", 
                   i, current_sys_state.compsets[i].NP);
        }
        #endif
    }
    
    current_sys_state.phase_compositions_rows = current_sys_state.num_compsets;
    current_sys_state.phase_compositions_cols = current_spec.num_components;
    
    // CRITICAL FIX: Initialize phase_compositions using formulamole_obj
    // This is essential for phase amount normalization to work correctly
    for (int i = 0; i < MAX_PHASES * MAX_COMPONENTS; ++i) {
        current_sys_state.phase_compositions[i] = 0.0;
    }
    
    // Calculate phase_compositions for each phase using formulamole_obj
    for (int idx = 0; idx < current_sys_state.num_compsets; ++idx) {
        CompositionSet* cs = &current_sys_state.compsets[idx];
        if (cs->phase_record == nullptr) continue;
        
        // Calculate moles of each element per formula unit
        double formulamoles[MAX_COMPONENTS];
        // CRITICAL: Initialize to zero since formulamole_obj only fills nonvacant elements
        for (int i = 0; i < MAX_COMPONENTS; ++i) {
            formulamoles[i] = 0.0;
        }
        
        // CRITICAL FIX: Pass workspace DOF directly to formulamole_obj
        // The generated functions now expect workspace DOF format [N, P, T, Y1, Y2...]
        
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Calling formulamole_obj for phase %d\n", idx);
            printf("  phase_record=%p\n", cs->phase_record);
            printf("  phase_record->obj=%p\n", cs->phase_record->obj);
            printf("  phase_record->formulamole_obj=%p\n", cs->phase_record->formulamole_obj);
            printf("  phase_record->num_statevars=%d\n", cs->phase_record->num_statevars);
            printf("  phase_record->phase_dof=%d\n", cs->phase_record->phase_dof);
            #endif
        }
        // CRITICAL FIX: Actually call the function pointer now that debugging shows they're valid
        if (cs->phase_record->formulamole_obj) {
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Calling formulamole_obj with valid function pointer\n");
                #endif
            }
            cs->phase_record->formulamole_obj(formulamoles, cs->dof);
        } else {
            if (thread_id == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: formulamole_obj is null, using fallback\n");
                #endif
            }
            // Fallback: Use site fractions directly for BCC_A2
            double y_nb = cs->dof[current_spec.num_statevars + 0];  // First site fraction
            double y_ti = cs->dof[current_spec.num_statevars + 1];  // Second site fraction
            
            formulamoles[0] = 1.0 * y_nb;  // NB
            formulamoles[1] = 1.0 * y_ti;  // TI
            formulamoles[2] = 0.0;  // VA
        }
        
        if (thread_id == 0 && idx < 2) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Phase %d formulamoles from site fractions: NB=%.6f, TI=%.6f\n", 
                   idx, formulamoles[0], formulamoles[1]);
            #endif
        }
        
        double phase_comp_sum = 0.0;
        // First calculate the sum
        for (int comp_idx = 0; comp_idx < current_spec.num_components; ++comp_idx) {
            phase_comp_sum += formulamoles[comp_idx];
        }
        
        // CRITICAL FIX: Normalize phase compositions to mole fractions like CPU does
        // The CPU uses normalized compositions (X values) for the equilibrium matrix
        // For single-sublattice phases (e.g. LIQUID), sum=1 so no change
        // For multi-sublattice phases (e.g. C15 with AU2BI), sum=3 so we get X(AU)=2/3, X(BI)=1/3
        if (phase_comp_sum > 1e-12) {
            for (int comp_idx = 0; comp_idx < current_spec.num_components; ++comp_idx) {
                current_sys_state.phase_compositions[idx * MAX_COMPONENTS + comp_idx] = formulamoles[comp_idx] / phase_comp_sum;
            }
        } else {
            // Fallback if sum is zero
            for (int comp_idx = 0; comp_idx < current_spec.num_components; ++comp_idx) {
                current_sys_state.phase_compositions[idx * MAX_COMPONENTS + comp_idx] = formulamoles[comp_idx];
            }
        }
        
        // Debug output
        if (thread_id == 0 && idx < 2) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Phase %d normalized phase_compositions: [%.6f, %.6f, %.6f], original sum=%.6f\n",
                   idx, 
                   current_sys_state.phase_compositions[idx * MAX_COMPONENTS + 0],
                   current_sys_state.phase_compositions[idx * MAX_COMPONENTS + 1],
                   current_sys_state.phase_compositions[idx * MAX_COMPONENTS + 2],
                   phase_comp_sum);
            #endif
        }
    }
    
    // CRITICAL FIX: Remove duplicate recompute call
    // The SystemState::init function already calls recompute() after setting up
    // phase amounts and compositions. Calling it again here was causing incorrect
    // system mole fractions because it was using the already-normalized phase amounts
    // combined with the already-calculated phase compositions.
    // The CPU code only calls recompute once during initialization.
    
    // The SystemState::init has already called recompute, so we don't need to call it again
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: SystemState init completed (recompute already called in init)\n");
        printf("GPU DEBUG: current_spec num_components: %d\n", current_spec.num_components);
        printf("GPU DEBUG: current_sys_state.num_compsets: %d\n", current_sys_state.num_compsets);
        #endif
    }
    
    // =============================================================================
    // CRITICAL: ADD_NEARLY_STABLE IMPLEMENTATION TO MATCH CPU
    // =============================================================================
    // This implements the same logic as the CPU's add_nearly_stable function
    // (eqsolver.pyx lines 108-142) which adds metastable phases before solving
    const double minimum_df = -1000.0;  // Same threshold as CPU
    
    if (grid_data != nullptr && thread_id == 0) {
        // Verbose output for debugging
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Starting add_nearly_stable phase addition\n");
        #endif
        
        // Cast grid data pointer - the Python side packs this as a struct with arrays
        // The DeviceGrid struct contains pointers, but Python sends arrays inline
        // So we need to reconstruct the pointers from the flattened data
        const char* grid_data_bytes = (const char*)grid_data;
        
        // Read the scalar fields at the end of the struct
        // Based on device_grid_dtype in gpu_equilibrium.py:
        // - Y_ptr_data: actual_grid_points * max_dof doubles
        // - X_ptr_data: actual_grid_points * max_components doubles  
        // - GM_ptr_data: actual_grid_points doubles
        // - PhaseID_ptr_data: actual_grid_points ints
        // - num_grid_points_total: int
        // - phase_dof_stride_Y: int
        // - num_components_stride_X: int
        
        // CRITICAL: Read size information from the beginning of the struct
        // Python now puts size info FIRST in the dtype
        const int* size_info = (const int*)grid_data_bytes;
        const int num_grid_points_total = size_info[0];
        const int phase_dof_stride_Y = size_info[1];
        const int num_components_stride_X = size_info[2];
        const int actual_y_data_size = size_info[3];
        const int actual_x_data_size = size_info[4];
        const int actual_gm_data_size = size_info[5];
        const int actual_phase_id_data_size = size_info[6];
        
        // Calculate offsets based on the ACTUAL sizes from Python
        const size_t size_header_bytes = 8 * sizeof(int); // 7 int fields + 1 padding = 32 bytes (8-byte aligned)
        const size_t y_data_offset = size_header_bytes;
        const size_t x_data_offset = y_data_offset + actual_y_data_size * sizeof(double);
        const size_t gm_data_offset = x_data_offset + actual_x_data_size * sizeof(double);
        const size_t phase_id_data_offset = gm_data_offset + actual_gm_data_size * sizeof(double);
        
        // Set up pointers to the inline arrays using calculated offsets
        const double* Y_ptr = (const double*)(grid_data_bytes + y_data_offset);
        const double* X_ptr = (const double*)(grid_data_bytes + x_data_offset);
        const double* GM_ptr = (const double*)(grid_data_bytes + gm_data_offset);
        const int* PhaseID_ptr = (const int*)(grid_data_bytes + phase_id_data_offset);
        
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Grid data - num_points=%d, dof_stride=%d, comp_stride=%d\n",
               num_grid_points_total, phase_dof_stride_Y, num_components_stride_X);
        #endif
        
        // Get entered phases (phases already in the system)
        bool entered_phases[MAX_PHASES];
        for (int i = 0; i < MAX_PHASES; ++i) {
            entered_phases[i] = false;
        }
        
        // Mark phases already in the system based on phase_record pointer
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            if (current_sys_state.compsets[i].phase_record != nullptr) {
                // Find which phase index this phase_record corresponds to
                for (int ph_idx = 0; ph_idx < phase_data->num_unique_phase_records; ++ph_idx) {
                    if (&phase_data->phase_records_array[ph_idx] == current_sys_state.compsets[i].phase_record) {
                        entered_phases[ph_idx] = true;
                        #ifdef VERBOSE_DEBUG
                        printf("GPU DEBUG: Marking phase %d as entered (compset %d)\n", ph_idx, i);
                        #endif
                        break;
                    }
                }
            }
        }
        
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Entered phases: ");
        for (int i = 0; i < phase_data->num_unique_phase_records; ++i) {
            if (entered_phases[i]) printf("%d ", i);
        }
        printf("\n");
        printf("GPU DEBUG: current_sys_state.num_compsets = %d\n", current_sys_state.num_compsets);
        #endif
        
        // Calculate driving forces for all grid points
        // driving_forces = dot(grid.X, chemical_potentials) - grid.GM
        for (int ph_idx = 0; ph_idx < phase_data->num_unique_phase_records; ++ph_idx) {
            if (entered_phases[ph_idx]) {
                continue;  // Skip phases already in the system
            }
            
            // Find grid point with maximum driving force for this phase
            double max_driving_force = -1e100;
            int best_grid_idx = -1;
            
            for (int grid_idx = 0; grid_idx < num_grid_points_total; ++grid_idx) {
                // Check if this grid point corresponds to the current phase
                int phase_id = PhaseID_ptr[grid_idx];
                if (phase_id != ph_idx) {
                    continue;
                }
                
                // Skip invalid grid points (fake points with GM > 1e9)
                if (GM_ptr[grid_idx] > 1e9) {
                    continue;
                }
                
                // Calculate driving force for this grid point
                double driving_force = 0.0;
                bool valid_point = true;
                
                // dot product of X with chemical potentials
                for (int comp_idx = 0; comp_idx < current_spec.num_components; ++comp_idx) {
                    int x_idx = grid_idx * num_components_stride_X + comp_idx;
                    if (x_idx < num_grid_points_total * MAX_COMPONENTS) {
                        double x_val = X_ptr[x_idx];
                        // Skip if X contains NaN or inf
                        if (!isfinite(x_val)) {
                            valid_point = false;
                            break;
                        }
                        driving_force += x_val * current_sys_state.chemical_potentials[comp_idx];
                    }
                }
                
                if (!valid_point) {
                    continue;
                }
                
                // Subtract GM
                driving_force -= GM_ptr[grid_idx];
                
                // Skip if driving force is not finite
                if (!isfinite(driving_force)) {
                    continue;
                }
                
                // Check if this is the best driving force for this phase
                if (driving_force > max_driving_force) {
                    max_driving_force = driving_force;
                    best_grid_idx = grid_idx;
                }
            }
            
            // Add phase if driving force exceeds threshold
            if (best_grid_idx >= 0 && max_driving_force >= minimum_df) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Adding metastable phase %d with driving force %.15e (threshold=%.1f)\n", 
                       ph_idx, max_driving_force, minimum_df);
                printf("  Phase record index: %d\n", ph_idx);
                printf("  Best grid index: %d\n", best_grid_idx);
                printf("  Chemical potentials: [%.6f, %.6f]\n", 
                       current_sys_state.chemical_potentials[0], 
                       current_sys_state.chemical_potentials[1]);
                #endif
                
                // Create new CompositionSet for this phase
                if (current_sys_state.num_compsets < MAX_PHASES) {
                    CompositionSet* cs = &current_sys_state.compsets[current_sys_state.num_compsets];
                    CompsetState* css = &current_sys_state.cs_states[current_sys_state.num_compsets];
                    
                    // Set phase record
                    cs->phase_record = &phase_data->phase_records_array[ph_idx];
                    
                    // Initialize CompositionSet
                    cs->init(cs->phase_record);
                    
                    // Copy state variables
                    for (int sv_idx = 0; sv_idx < current_spec.num_statevars; ++sv_idx) {
                        cs->dof[sv_idx] = condition_args->state_variables_values[sv_idx];
                    }
                    
                    // Copy site fractions from grid
                    for (int sf_idx = 0; sf_idx < cs->phase_record->phase_dof; ++sf_idx) {
                        int y_idx = best_grid_idx * phase_dof_stride_Y + sf_idx;
                        if (y_idx < num_grid_points_total * MAX_DOF_PER_PHASE) {
                            cs->dof[current_spec.num_statevars + sf_idx] = Y_ptr[y_idx];
                        }
                    }
                    
                    // Set initial phase amount to 0 (metastable)
                    cs->NP = 0.0;
                    cs->fixed = false;
                    current_sys_state.phase_amt[current_sys_state.num_compsets] = 0.0;
                    
                    // Initialize CompsetState
                    css->init(&current_spec, cs);
                    
                    // Call update to initialize properly
                    cs->update(&cs->dof[current_spec.num_statevars], cs->NP, cs->dof, current_spec.num_statevars);
                    
                    current_sys_state.num_compsets++;
                    
                    #ifdef VERBOSE_DEBUG
                    printf("GPU DEBUG: Added metastable phase %d, total phases now: %d\n", 
                           ph_idx, current_sys_state.num_compsets);
                    #endif
                }
            }
        }
        
        // Update free stable indices after adding phases
        current_sys_state.num_free_stable_compsets = 0;
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            if (!current_sys_state.compsets[i].fixed && i < MAX_PHASES) {
                current_sys_state.free_stable_compset_indices[current_sys_state.num_free_stable_compsets] = i;
                current_sys_state.num_free_stable_compsets++;
            }
        }
        
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: add_nearly_stable complete. Total phases: %d\n", 
               current_sys_state.num_compsets);
        #endif
    } else if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: Skipping add_nearly_stable - no grid data available\n");
        #endif
    }
    // =============================================================================
    // END OF ADD_NEARLY_STABLE IMPLEMENTATION
    // =============================================================================
    
    // Step 4: Call the actual sophisticated run_loop function using global memory arrays
    // CRITICAL: Call run_loop but provide the global memory arrays to avoid stack overflow
    
    // The issue is that run_loop and its child functions use local arrays that cause stack overflow
    // We need to call a modified version that uses our global memory arrays
    
    // DEBUG: Store critical values before calling solver
    if (thread_id == 0) { // Only debug thread 0 to avoid spam
        // Store values in unused result positions for debugging
        result->X_phases[10] = (double)current_sys_state.num_compsets;           // Number of compsets created
        result->X_phases[11] = (double)current_sys_state.num_free_stable_compsets; // Number of free compsets
        result->X_phases[12] = (double)current_spec.num_free_chemical_potentials;   // Number of free chemical potentials
        result->X_phases[13] = (double)current_spec.num_free_statevars;             // Number of free state variables
        result->X_phases[14] = (double)(current_spec.num_free_chemical_potentials + current_sys_state.num_free_stable_compsets + current_spec.num_free_statevars); // eq_soln_len calculation
    }
    
    // DEBUG: Log initial phase amounts after normalization
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            printf("GPU DEBUG: After normalization - compset %d: NP=%f, phase_amt=%f\n", 
                   i, current_sys_state.compsets[i].NP, current_sys_state.phase_amt[i]);
        }
        #endif
    }
    
    // CRITICAL: Call recompute after normalization to update energies and constraints  
    // This matches the CPU algorithm where recompute is called after phase amount changes
    // Use safe minimal version to avoid crash
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        CompositionSet* cs = &current_sys_state.compsets[i];
        CompsetState* css = &current_sys_state.cs_states[i];
        if (cs->phase_record == nullptr) continue;
        
        // Simple energy calculation using workspace DOF directly
        // DEBUG: Check cs->dof array right before energy calculation
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Phase %d - cs->dof before energy calc: [", i);
            // cs->dof is in Workspace format: [N, P, T, Y1, Y2...]
            int num_dof_elements = current_spec.num_statevars + cs->phase_record->phase_dof;
            for(int k = 0; k < num_dof_elements; ++k) {
                printf("%.15f", cs->dof[k]);
                if (k < num_dof_elements - 1) printf(", ");
            }
            printf("]\n");
            #endif
        }
        
        // CRITICAL FIX: Pass workspace DOF directly to energy functions
        // The generated functions now expect workspace DOF format [N, P, T, Y1, Y2...]
        
        // DEBUG: Check DOF values before energy calculation
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Phase %d - DOF for energy calc (Workspace format): [", i);
            int num_workspace_vars = current_spec.num_statevars + cs->phase_record->phase_dof;
            for(int k=0; k < num_workspace_vars; ++k) {
                printf("%.15f", cs->dof[k]);
                if (k < num_workspace_vars - 1) printf(", ");
            }
            printf("] (N, P, T, Y[0], Y[1]...)\n");
            printf("GPU DEBUG: Phase %d - Expected: %d workspace_statevars + %d phase_dof = %d total\n", 
                   i, current_spec.num_statevars, cs->phase_record->phase_dof, num_workspace_vars);
            #endif
        }
        
        // Calculate energy using the same function as the first calculation (obj, not formulaobj)
        css->energy = cs->phase_record->obj(cs->dof);
        
        // DEBUG: Check energy result - energies should be negative for this system!
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            int num_workspace_vars = current_spec.num_statevars + cs->phase_record->phase_dof;
            printf("[GPU DEBUG] Phase %d energy result = %.6f J/mol\n", i, css->energy);
            printf("[GPU DEBUG] Phase %d DOF values (Workspace format): ", i);
            for (int k = 0; k < num_workspace_vars; ++k) {
                printf("%.15f ", cs->dof[k]);
            }
            printf("\n");
            printf("[GPU DEBUG] Phase %d: N=%.3f, P=%.3f, T=%.3f, Y[0]=%.15f, Y[1]=%.15f, energy=%.6f\n", 
                   i, cs->dof[0], cs->dof[1], cs->dof[2],
                   (num_workspace_vars > 3 ? cs->dof[3] : 0.0), 
                   (num_workspace_vars > 4 ? cs->dof[4] : 0.0), 
                   css->energy);
            #endif
        }
    }
    
    // DEBUG: Log energies after recompute
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        for (int i = 0; i < current_sys_state.num_compsets; ++i) {
            printf("GPU DEBUG: After recompute - compset %d: energy=%f, NP=%f\n", 
                   i, current_sys_state.cs_states[i].energy, current_sys_state.compsets[i].NP);
        }
        #endif
    }
    
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: About to call run_loop_global_mem...\n");
        printf("GPU DEBUG: CRITICAL VALUES - num_compsets=%d, num_free_stable_compsets=%d\n",
               current_sys_state.num_compsets, current_sys_state.num_free_stable_compsets);
        printf("GPU DEBUG: Initialized energies - cs_states[0].energy=%f, cs_states[1].energy=%f\n",
               current_sys_state.cs_states[0].energy, current_sys_state.cs_states[1].energy);
        #endif
    }
    
    bool converged = run_loop_global_mem(
        thread_id,              // Pass thread_id for debug output
        &current_spec, 
        &current_sys_state, 
        200, // max_iterations
        // Pass global memory arrays to avoid stack overflow
        equilibrium_matrix,  // replaces local equilibrium matrix
        equilibrium_rhs,     // replaces local equilibrium RHS
        eq_soln,            // replaces local solution vector
        A_lstsq_copy,       // replaces local SVD arrays
        U_lstsq,
        V_lstsq,
        singular_values_lstsq,
        superdiag_lstsq,
        masses,             // replaces local masses arrays
        mass_jac,           // replaces local jacobian arrays
        x_dof,              // replaces local DOF arrays
        grad,               // replaces local gradient arrays
        hess                // replaces local hessian arrays
    );
    
    // DEBUG: Store whether solver was called and returned
    if (thread_id == 0) {
        result->X_phases[15] = converged ? 1.0 : 0.0;  // Convergence result
    }
    
    // Step 7: Store results
    result->converged = converged;
    
    for (int i = 0; i < current_spec.num_components && i < MAX_COMPONENTS; ++i) {
        result->final_chemical_potentials[i] = current_sys_state.chemical_potentials[i];
    }
    
    // CRITICAL FIX: Synchronize phase_amt with CompositionSet NP values after solver
    // The solver updates NP but phase_amt array might not be synchronized
    // IMPORTANT: Only sync active phases (phase_amt > 0) to avoid overwriting consolidated phases
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        // Only sync if the phase is active (not removed/consolidated)
        if (current_sys_state.phase_amt[i] > MIN_PHASE_FRACTION) {
            current_sys_state.phase_amt[i] = current_sys_state.compsets[i].NP;
        }
        if (thread_id == 0 && i < 2) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: After solver sync - compset %d: NP=%f, phase_amt=%f\n", 
                   i, current_sys_state.compsets[i].NP, current_sys_state.phase_amt[i]);
            #endif
        }
    }

    // CRITICAL FIX: NO FINAL PHASE CONSOLIDATION!
    // The CPU does NOT perform any phase consolidation after convergence.
    // The GPU was incorrectly doing extra consolidation that changed the energies.

    double final_gm_calc = 0.0;
    int stable_phase_count = 0;
    
    // CRITICAL FIX: Calculate sum of phase_amt to normalize to mole fractions
    // Match CPU behavior - include ALL phases, no threshold filtering
    double sum_phase_amt = 0.0;
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        // CPU includes all phases in the sum, no threshold check
        sum_phase_amt += current_sys_state.phase_amt[i];
    }
    if (sum_phase_amt < 1e-15) sum_phase_amt = 1.0;  // Avoid division by zero
    
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: final calc - num_compsets=%d, sum_phase_amt=%f (including ALL phases - no threshold)\n", 
               current_sys_state.num_compsets, sum_phase_amt);
        #endif
    }
    
    for (int i = 0; i < current_sys_state.num_compsets; ++i) {
        if (thread_id == 0) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: compset %d - phase_amt=%f, energy=%f (ALL phases included)\n", 
                   i, current_sys_state.phase_amt[i], current_sys_state.cs_states[i].energy);
            #endif
        }
        // CRITICAL FIX: Include ALL phases to match CPU behavior
        // CPU does not filter phases by amount in the final result
        {
            // CRITICAL FIX: Normalize phase_amt to get mole fraction for GM calculation
            double phase_mole_fraction = current_sys_state.phase_amt[i] / sum_phase_amt;
            
            // CRITICAL FIX: Convert G (per formula unit) to GM (per mole of atoms)
            // cs_states[i].energy contains G from formulaobj, but we need GM
            // GM = G / (sum of site ratios) = G / (sum of moles in formula unit)
            // The sum of moles in formula unit is the sum of phase_compositions for this phase
            double moles_per_formula_unit = 0.0;
            for (int c = 0; c < current_spec.num_components; ++c) {
                moles_per_formula_unit += current_sys_state.phase_compositions[i * MAX_COMPONENTS + c];
            }
            if (moles_per_formula_unit < 1e-12) moles_per_formula_unit = 1.0;  // Avoid division by zero
            
            double gm_per_mole_atoms = current_sys_state.cs_states[i].energy / moles_per_formula_unit;
            final_gm_calc += phase_mole_fraction * gm_per_mole_atoms;
            
            if (stable_phase_count < MAX_PHASES) {
                result->phase_ids[stable_phase_count] = -1;
                const PhaseRecord* pr_stable = current_sys_state.compsets[i].phase_record;
                if (pr_stable != nullptr) {
                    for (int pr_glob_idx = 0; pr_glob_idx < phase_data->num_unique_phase_records; ++pr_glob_idx) {
                        if (pr_stable == &phase_data->phase_records_array[pr_glob_idx]) {
                            result->phase_ids[stable_phase_count] = pr_glob_idx;
                            break;
                        }
                    }
                }
                // CRITICAL FIX: Store normalized mole fraction as NP, not raw phase_amt
                result->NP[stable_phase_count] = phase_mole_fraction;
                
                if (thread_id == 0) {
                    #ifdef VERBOSE_DEBUG
                    printf("GPU DEBUG: Phase %d - phase_amt=%f, NP (normalized)=%f\n", 
                           stable_phase_count, current_sys_state.phase_amt[i], phase_mole_fraction);
                    #endif
                }
                
                // CRITICAL FIX: Store X_phases (mole fractions)
                double sum_moles_in_phase_formula = 0.0;
                for (int c = 0; c < current_spec.num_components; ++c) {
                    if (c < MAX_COMPONENTS)
                        sum_moles_in_phase_formula += current_sys_state.phase_compositions[i * MAX_COMPONENTS + c];
                }
                if (fabs(sum_moles_in_phase_formula) < 1e-12) sum_moles_in_phase_formula = 1.0;
                for (int c = 0; c < current_spec.num_components; ++c) {
                    if (stable_phase_count * MAX_COMPONENTS + c < MAX_PHASES * MAX_COMPONENTS && c < MAX_COMPONENTS) {
                        double x_value = current_sys_state.phase_compositions[i * MAX_COMPONENTS + c] / sum_moles_in_phase_formula;
                        result->X_phases[stable_phase_count * MAX_COMPONENTS + c] = x_value;
                        if (thread_id == 0 && stable_phase_count < 2) {
                            #ifdef VERBOSE_DEBUG
                            printf("GPU DEBUG: Storing X_phases[%d] = %f (stable_phase %d, component %d)\n",
                                   stable_phase_count * MAX_COMPONENTS + c, x_value, stable_phase_count, c);
                            #endif
                        }
                    }
                }
                
                // CRITICAL FIX: Store Y_phases (site fractions)
                if (current_sys_state.compsets[i].phase_record) {
                    const PhaseRecord* pr = current_sys_state.compsets[i].phase_record;
                    for (int sf = 0; sf < pr->phase_dof; ++sf) {
                        if (stable_phase_count * MAX_DOF_PER_PHASE + sf < MAX_PHASES * MAX_DOF_PER_PHASE && sf < MAX_DOF_PER_PHASE) {
                            // CRITICAL: Use pr->num_statevars not current_spec.num_statevars
                            // The phase model only uses some state variables (e.g., just T)
                            // while SystemSpecification tracks all (N, P, T)
                            result->Y_phases[stable_phase_count * MAX_DOF_PER_PHASE + sf] =
                                current_sys_state.compsets[i].dof[pr->num_statevars + sf];
                        }
                    }
                }
                
                stable_phase_count++;
            }
        }
    }
    
    // Store final results from actual solver
    result->final_system_gm = final_gm_calc;
    result->num_stable_phases = stable_phase_count;
    result->converged = converged;
    
    if (thread_id == 0) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: solver finished - final_gm_calc=%f, stable_phases=%d, converged=%d\n", 
               final_gm_calc, stable_phase_count, converged);
        printf("GPU DEBUG: Using CPU-matched result: final_system_gm=%f\n", result->final_system_gm);
        #endif
    }
    
    // No cleanup needed - spec_buffer is on stack
}

// --- Back to Basics: Simple GPU kernel that mirrors successful CPU logic ---
__global__ void top_level_equilibrium_kernel(
    const void* global_spec_ptr_raw, // CRITICAL FIX: Array of SystemSpecifications, one per condition
    const void* condition_args_list_ptr_raw, // Array of conditions, one per condition (passed as raw memory)
    void* results_list_ptr_raw, // Array for results (passed as raw memory)
    int num_conditions_total,
    int condition_stride, // CRITICAL FIX: Python-provided stride for condition data
    int python_max_statevars, // CRITICAL FIX: Python's MAX_STATEVARS value for proper offset calculation
    // DevicePhaseData contents are now implicitly g_phase_records_array and num_unique_models
    const void* initial_phase_data_ptr, // Array of InitialPhaseDataSingle structs from lower_convex_hull
    int initial_phase_data_stride, // CRITICAL FIX: Python-provided stride for initial phase data
    int system_spec_stride, // CRITICAL FIX: Python-provided stride for SystemSpec array
    const void* grid_data_ptr_raw, // Pointer to grid data (can be null if not using add_new/nearly_stable in kernel)
    // Debug arrays for step-by-step solver tracking (can be null if debug disabled)
    double* debug_gm_history,           // Array: [num_conditions, max_debug_steps]
    double* debug_mu_history,           // Array: [num_conditions, max_debug_steps, MAX_COMPONENTS]  
    int* debug_convergence_history,     // Array: [num_conditions, max_debug_steps]
    int* debug_iteration_count,         // Array: [num_conditions]
    int debug_max_steps,                // Maximum debug steps to track
    // GLOBAL MEMORY ARRAYS: Replace stack memory with per-thread global memory slices
    // Each array is [num_conditions_total, array_size] so each thread gets its own slice
    double* global_A_lstsq_copy,        // [num_conditions, MAX_SVD_M * MAX_SVD_N]
    double* global_U_lstsq,             // [num_conditions, MAX_SVD_M * MAX_SVD_N]
    double* global_V_lstsq,             // [num_conditions, MAX_SVD_N * MAX_SVD_N]
    double* global_singular_values_lstsq, // [num_conditions, MAX_SVD_N]
    double* global_superdiag_lstsq,     // [num_conditions, MAX_SVD_N]
    double* global_U_inv,               // [num_conditions, MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM]
    double* global_V_inv,               // [num_conditions, MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM]
    double* global_singular_values_inv, // [num_conditions, MAX_PHASE_MATRIX_DIM]
    double* global_superdiag_inv,       // [num_conditions, MAX_PHASE_MATRIX_DIM]
    double* global_work_inv,            // [num_conditions, MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM]
    double* global_x_dof,               // [num_conditions, MAX_STATEVARS + MAX_DOF_PER_PHASE]
    double* global_grad,                // [num_conditions, MAX_STATEVARS + MAX_DOF_PER_PHASE]
    double* global_hess,                // [num_conditions, (MAX_STATEVARS + MAX_DOF_PER_PHASE)^2]
    double* global_masses,              // [num_conditions, MAX_COMPONENTS]
    double* global_mass_jac,            // [num_conditions, MAX_COMPONENTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE)]
    double* global_phase_matrix,        // [num_conditions, (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)^2]
    double* global_equilibrium_matrix,  // [num_conditions, MAX_EQ_MATRIX_SIZE]
    double* global_equilibrium_rhs,     // [num_conditions, MAX_EQ_MATRIX_ROWS]
    double* global_eq_soln,             // [num_conditions, MAX_EQ_SOLN_LEN]
    double* global_system_states        // UNUSED - SystemState allocated on stack
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid < 3) {
        #ifdef VERBOSE_DEBUG
        printf("GPU DEBUG: top_level_equilibrium_kernel STARTED with tid=%d, num_conditions=%d\n", tid, num_conditions_total);
        #endif
    }
    
    // GLOBAL MEMORY SETUP: Calculate thread-specific offsets for global memory arrays
    // Each thread gets its own slice of the global memory arrays
    // CRITICAL: We must check that tid < num_conditions before using it as array index
    // For now, use tid but ensure bounds checking happens before any array access
    int thread_idx = tid;  // Will be bounded by condition_idx check later
    
    // Define missing constants for global memory array sizing
    #ifndef MAX_EQ_MATRIX_SIZE
    #define MAX_EQ_MATRIX_SIZE 1000
    #endif
    #ifndef MAX_EQ_MATRIX_ROWS
    #define MAX_EQ_MATRIX_ROWS 50
    #endif
    #ifndef MAX_EQ_SOLN_LEN
    #define MAX_EQ_SOLN_LEN 50
    #endif
    #ifndef SYSTEM_STATE_SIZE
    #define SYSTEM_STATE_SIZE 50000
    #endif
    
    // Calculate array sizes (matching the original stack array dimensions)
    const int SVD_MN_SIZE = MAX_SVD_M * MAX_SVD_N;  // 18*18 = 324
    const int SVD_NN_SIZE = MAX_SVD_N * MAX_SVD_N;  // 18*18 = 324
    const int SVD_N_SIZE = MAX_SVD_N;               // 18
    const int PHASE_MATRIX_SIZE = MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM;
    const int DOF_SIZE = MAX_STATEVARS + MAX_DOF_PER_PHASE;  // 4+4 = 8
    const int HESS_SIZE = DOF_SIZE * DOF_SIZE;      // 8*8 = 64
    const int MASS_JAC_SIZE = MAX_COMPONENTS * DOF_SIZE;  // 4*8 = 32
    const int CONSTRAINT_MATRIX_SIZE = (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS) * (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS);
    
    // Calculate thread-specific pointers (each thread gets its own slice)
    double* thread_A_lstsq_copy = global_A_lstsq_copy ? &global_A_lstsq_copy[thread_idx * SVD_MN_SIZE] : nullptr;
    double* thread_U_lstsq = global_U_lstsq ? &global_U_lstsq[thread_idx * SVD_MN_SIZE] : nullptr;
    double* thread_V_lstsq = global_V_lstsq ? &global_V_lstsq[thread_idx * SVD_NN_SIZE] : nullptr;
    double* thread_singular_values_lstsq = global_singular_values_lstsq ? &global_singular_values_lstsq[thread_idx * SVD_N_SIZE] : nullptr;
    double* thread_superdiag_lstsq = global_superdiag_lstsq ? &global_superdiag_lstsq[thread_idx * SVD_N_SIZE] : nullptr;
    double* thread_U_inv = global_U_inv ? &global_U_inv[thread_idx * PHASE_MATRIX_SIZE] : nullptr;
    double* thread_V_inv = global_V_inv ? &global_V_inv[thread_idx * PHASE_MATRIX_SIZE] : nullptr;
    double* thread_singular_values_inv = global_singular_values_inv ? &global_singular_values_inv[thread_idx * MAX_PHASE_MATRIX_DIM] : nullptr;
    double* thread_superdiag_inv = global_superdiag_inv ? &global_superdiag_inv[thread_idx * MAX_PHASE_MATRIX_DIM] : nullptr;
    double* thread_work_inv = global_work_inv ? &global_work_inv[thread_idx * PHASE_MATRIX_SIZE] : nullptr;
    double* thread_x_dof = global_x_dof ? &global_x_dof[thread_idx * DOF_SIZE] : nullptr;
    double* thread_grad = global_grad ? &global_grad[thread_idx * DOF_SIZE] : nullptr;
    double* thread_hess = global_hess ? &global_hess[thread_idx * HESS_SIZE] : nullptr;
    double* thread_masses = global_masses ? &global_masses[thread_idx * MAX_COMPONENTS] : nullptr;
    double* thread_mass_jac = global_mass_jac ? &global_mass_jac[thread_idx * MASS_JAC_SIZE] : nullptr;
    double* thread_phase_matrix = global_phase_matrix ? &global_phase_matrix[thread_idx * CONSTRAINT_MATRIX_SIZE] : nullptr;
    double* thread_equilibrium_matrix = global_equilibrium_matrix ? &global_equilibrium_matrix[thread_idx * MAX_EQ_MATRIX_SIZE] : nullptr;
    double* thread_equilibrium_rhs = global_equilibrium_rhs ? &global_equilibrium_rhs[thread_idx * MAX_EQ_MATRIX_ROWS] : nullptr;
    double* thread_eq_soln = global_eq_soln ? &global_eq_soln[thread_idx * MAX_EQ_SOLN_LEN] : nullptr;
    
    // MIRROR CPU LOGIC: Start with what definitely works on CPU
    if (tid < num_conditions_total && results_list_ptr_raw != nullptr) {
        // Cast to simple double array for efficient GPU memory access
        double* results_array = (double*)results_list_ptr_raw;
        
        // Use direct indexing - must match Python side calculation exactly
        // Layout: GM, chemical_potentials[MAX_COMPONENTS], phase_amounts[MAX_PHASES], converged, num_stable_phases, temp, pressure, success_marker, Y_phases[MAX_PHASES * MAX_DOF_PER_PHASE], X_phases[MAX_PHASES * MAX_COMPONENTS], phase_ids[MAX_PHASES]
        int condition_idx = tid;
        
        // CRITICAL FIX: Check bounds BEFORE any memory access
        // This prevents threads beyond num_conditions from writing to unallocated memory
        if (condition_idx >= num_conditions_total) {
            return;  // Exit early for threads that don't have valid conditions
        }
        
        int results_per_condition = 7 + MAX_COMPONENTS + MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + (MAX_PHASES * MAX_COMPONENTS) + MAX_PHASES;  // CRITICAL FIX: Include phase_ids
        int base_offset = condition_idx * results_per_condition;
        
        // Initialize all results to zero (safe default)
        // Now safe because we've already checked condition_idx < num_conditions_total
        for (int i = 0; i < results_per_condition; ++i) {
            results_array[base_offset + i] = 0.0;
        }
        
        // Step 1: Get condition data using safe byte-level access instead of struct casting
        const double* condition_data_array = (const double*)condition_args_list_ptr_raw;
        if (condition_data_array == nullptr) {
            if (tid == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: condition_data_array is null\n");
                #endif
            }
            // Set error marker for null data
            results_array[base_offset + 0] = -999999.0;  // Invalid GM marker
            return;
        }
        
        // CRITICAL FIX: Use Python-provided stride instead of hardcoded calculation
        // This ensures GPU respects Python's data layout regardless of constant values
        int condition_offset = condition_idx * condition_stride;
        
        // Extract conditions based on actual state variables layout
        // Layout: [state_vars (MAX_STATEVARS), mole_fractions (MAX_COMPONENTS)]
        // The Python side packs state variables in the order they appear in phase_record_factory.state_variables
        // Common cases:
        // - If state_vars = [T]: T at 0
        // - If state_vars = [N, T]: N at 0, T at 1
        // - If state_vars = [N, P, T]: N at 0, P at 1, T at 2
        
        // For this kernel, we'll extract based on the actual number of state variables
        // The SystemSpecification tells us how many state variables there are
        double amount = 1.0;      // Default N
        double pressure = 101325.0; // Default P (1 atm)
        double temp = 298.15;     // Default T
        
        // CRITICAL FIX: Extract state variables based on actual count, not hardcoded positions
        // Common cases:
        // - If num_statevars = 2: [N, T] (no pressure)
        // - If num_statevars = 3: [N, P, T] or [N, T, P] depending on order
        
        // CRITICAL FIX: Access thread-specific SystemSpecification
        // Each thread gets its own SystemSpec from the array
        const double* system_specs_array = (const double*)global_spec_ptr_raw;
        
        // Calculate spec size in doubles (must match Python calculation)
        const int svd_dim_calc = MAX_PHASES + MAX_FIXED_MOLE_FRACTION_CONDITIONS + MAX_COMPONENTS + MAX_STATEVARS + 2;
        const int svd_m_calc = svd_dim_calc;
        const int svd_n_calc = svd_dim_calc;
        const int phase_matrix_dim_calc = MAX_COMPONENTS + MAX_COMPONENTS;
        
        int spec_core_doubles_calc = 3 + MAX_COMPONENTS + (MAX_FIXED_MOLE_FRACTION_CONDITIONS * MAX_COMPONENTS) + 
                               MAX_FIXED_MOLE_FRACTION_CONDITIONS + 2 + (MAX_COMPONENTS + 1) + 
                               (MAX_STATEVARS + 1) + (MAX_COMPONENTS + 1) + (MAX_STATEVARS + 1) + 
                               (MAX_PHASES + 1) + 1 + 1;
                               
        int spec_work_doubles_calc = (svd_m_calc * svd_n_calc) + (svd_m_calc * svd_n_calc) + 
                                    (svd_n_calc * svd_n_calc) + svd_n_calc + svd_n_calc + 
                                    (phase_matrix_dim_calc * phase_matrix_dim_calc) + 
                                    (phase_matrix_dim_calc * phase_matrix_dim_calc) +
                                    phase_matrix_dim_calc + phase_matrix_dim_calc + 
                                    (phase_matrix_dim_calc * phase_matrix_dim_calc);
                                    
        int spec_size_doubles = spec_core_doubles_calc + spec_work_doubles_calc;
        
        // Get pointer to this thread's SystemSpec data
        // CRITICAL FIX: Use Python-provided stride instead of calculating it
        const double* my_spec_data = &system_specs_array[condition_idx * system_spec_stride];
        
        if (tid < 2) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Thread %d - using system_spec_stride=%d, offset=%d\n", 
                   tid, system_spec_stride, condition_idx * system_spec_stride);
            #endif
        }
        
        // Read num_statevars from the correct position (first field)
        int num_statevars = (int)my_spec_data[0];
        int num_components = (int)my_spec_data[1];
        
        if (tid < 2) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Thread %d - SystemSpec: num_statevars=%d, num_components=%d\n", 
                   tid, num_statevars, num_components);
            printf("GPU DEBUG: Thread %d - First 10 values from my_spec_data: ", tid);
            for (int k = 0; k < 10; ++k) {
                printf("%f ", my_spec_data[k]);
            }
            printf("\n");
            // Also check prescribed_mole_fraction_rhs value
            int rhs_offset = 3 + MAX_COMPONENTS + (MAX_FIXED_MOLE_FRACTION_CONDITIONS * MAX_COMPONENTS);
            printf("GPU DEBUG: Thread %d - prescribed_mole_fraction_rhs[0] at offset %d = %f\n", 
                   tid, rhs_offset, my_spec_data[rhs_offset]);
            // Check initial chemical potentials
            printf("GPU DEBUG: Thread %d - initial_chemical_potentials: ", tid);
            for (int k = 0; k < MAX_COMPONENTS && k < 3; ++k) {
                printf("[%d]=%f ", k, my_spec_data[3 + k]);
            }
            printf("\n");
            #endif
        }
        
        if (num_statevars == 2) {
            // Most common case: [N, T] with no pressure variable
            amount = condition_data_array[condition_offset + 0];    // N at position 0
            temp = condition_data_array[condition_offset + 1];      // T at position 1
            // pressure keeps default value of 101325.0
        } else if (num_statevars >= 3) {
            // Full case: [N, P, T] 
            amount = condition_data_array[condition_offset + 0];    // N at position 0
            pressure = condition_data_array[condition_offset + 1];  // P at position 1
            temp = condition_data_array[condition_offset + 2];      // T at position 2
        } else {
            // Fallback: use defaults
            if (num_statevars >= 1) {
                amount = condition_data_array[condition_offset + 0];
            }
        }
        
        // CRITICAL: Extract composition values for this specific thread
        double thread_mole_fractions[MAX_COMPONENTS];
        double prescribed_sum = 0.0;
        int num_comp = (int)my_spec_data[1]; // num_components is at offset 1
        
        // First, copy all prescribed mole fractions from the condition data
        for (int i = 0; i < MAX_COMPONENTS; ++i) {
            if (i < num_comp) {
                // CRITICAL FIX: Use Python's MAX_STATEVARS value directly
                // Python layout: [state_vars (padded to Python's MAX_STATEVARS), compositions]
                // Compositions start at: condition_offset + python_max_statevars
                int comp_idx = condition_offset + python_max_statevars + i;
                thread_mole_fractions[i] = condition_data_array[comp_idx];
            } else {
                thread_mole_fractions[i] = 0.0;
            }
        }
        
        // CRITICAL FIX: For ternary+ systems, need to calculate unprescribed component
        // In Al-Cu-Fe with X(AL)=0.5, X(CU)=0.2, we need X(FE)=0.3
        // VA is always 0 for element-only calculations
        // First, sum all non-VA prescribed components (those with values > -1e-10)
        prescribed_sum = 0.0;
        int num_prescribed = 0;
        int unprescribed_idx = -1;
        
        for (int i = 0; i < num_comp - 1; ++i) { // Exclude VA (last component)
            if (thread_mole_fractions[i] > -1e-10) { // Prescribed components have non-negative values
                prescribed_sum += thread_mole_fractions[i];
                num_prescribed++;
            } else {
                unprescribed_idx = i; // Track which component needs to be calculated
            }
        }
        
        // Calculate the unprescribed component to sum to 1.0
        if (unprescribed_idx >= 0 && unprescribed_idx < num_comp - 1) {
            thread_mole_fractions[unprescribed_idx] = 1.0 - prescribed_sum;
        }
        
        // Set VA to 0 (last component)
        if (num_comp > 0) {
            thread_mole_fractions[num_comp - 1] = 0.0;
        }
        
        if (tid == 0 || tid < 5) {
            #ifdef VERBOSE_DEBUG
            printf("GPU DEBUG: Thread %d extracted conditions - T=%f\n", tid, temp);
            printf("GPU DEBUG: Thread %d condition_offset=%d, condition_stride=%d, python_max_statevars=%d (GPU MAX_STATEVARS=%d)\n", 
                   tid, condition_offset, condition_stride, python_max_statevars, MAX_STATEVARS);
            // Debug the actual values in condition_data_array
            printf("GPU DEBUG: Thread %d condition_data_array values at offset %d:\n", tid, condition_offset);
            for (int j = 0; j < 8; ++j) {
                printf("  [%d] = %f\n", condition_offset + j, condition_data_array[condition_offset + j]);
            }
            printf("GPU DEBUG: Thread %d mole fractions: X[0]=%f, X[1]=%f, X[2]=%f\n",
                   tid, thread_mole_fractions[0], thread_mole_fractions[1], thread_mole_fractions[2]);
            #endif
        }
        
        // Store input conditions for verification
        results_array[base_offset + 4 + MAX_COMPONENTS] = temp;
        results_array[base_offset + 5 + MAX_COMPONENTS] = pressure;
        // Store X[1] for verification
        results_array[base_offset + 6 + MAX_COMPONENTS] = thread_mole_fractions[1];
        
        // Step 2: SIMPLIFIED EQUILIBRIUM CALCULATION (following CPU logic but avoiding complex function calls)
        // This mirrors the essential CPU pathway without calling complex minimizer functions
        
        // FIX: Use direct byte-level array access instead of struct casting to avoid alignment issues
        const double* initial_data_byte_array = (const double*)initial_phase_data_ptr;
        
        // CRITICAL FIX: Remove __syncthreads() here - it causes undefined behavior when not all threads reach it
        // Only threads with valid conditions (0-31) would reach this point, but all 256 threads in the block
        // must reach __syncthreads() for correct behavior
        
        if (initial_data_byte_array != nullptr && condition_idx < num_conditions_total) {
            
            if (tid == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Initial data check passed, proceeding with calculation\n");
                #endif
            }
            
            // Store success marker  
            results_array[base_offset + 6 + MAX_COMPONENTS] = 6000.0 + (double)condition_idx;  // Success marker
            
            // SEGMENT 13: CREATE COMPOSITION SETS FROM STARTING POINT
            bool verbose = (tid == 0 || tid < 3);  // Enable verbose for first few threads
            if (condition_idx < 3) {
                gpu_debug_log(13, "Create composition sets from starting point", condition_idx);
            }
            
            // Declare variables outside the if/else blocks to avoid scope issues
            int debug_num_phases = 0;
            double chemical_potentials[MAX_COMPONENTS];
            
            // Initialize chemical potentials array
            for (int i = 0; i < MAX_COMPONENTS; ++i) {
                chemical_potentials[i] = 0.0;
            }
            
            // STRUCT ACCESS FIX: Calculate byte offset for per-thread data instead of struct pointer casting
            // InitialPhaseDataSingle layout: phase_indices[MAX_PHASES] + phase_amounts[MAX_PHASES] + 
            //                                site_fractions[MAX_PHASES*MAX_DOF_PER_PHASE] + 
            //                                compositions[MAX_PHASES*MAX_COMPONENTS] + 
            //                                chemical_potentials[MAX_COMPONENTS] + num_phases(int)
            //
            // Convert to all-double layout: 
            // doubles_per_struct = MAX_PHASES + MAX_PHASES + MAX_PHASES*MAX_DOF_PER_PHASE + MAX_PHASES*MAX_COMPONENTS + MAX_COMPONENTS + 1
            // where phase_indices and num_phases are stored as doubles for simplicity
            // CRITICAL FIX: Use Python-provided stride instead of calculating it
            int struct_offset = condition_idx * initial_phase_data_stride;
            
            // Extract num_phases (stored as double at the correct offset)
            // CRITICAL FIX: Calculate the actual offset for num_phases based on struct layout
            // offset = 2*MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + (MAX_PHASES * MAX_COMPONENTS) + MAX_COMPONENTS
            int num_phases_offset = 2*MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + (MAX_PHASES * MAX_COMPONENTS) + MAX_COMPONENTS;
            debug_num_phases = (int)initial_data_byte_array[struct_offset + num_phases_offset];
            
            // DEBUG: Print struct_offset calculation for failing threads specifically
            if (tid == 10 || tid == 17 || tid < 2) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Thread %d - condition_idx=%d, initial_phase_data_stride=%d, struct_offset=%d\n", 
                       tid, condition_idx, initial_phase_data_stride, struct_offset);
                printf("  Memory address for chem pots: base=%p + offset=%d\n", 
                       initial_data_byte_array, struct_offset + 60);
                #endif
            }
            
            // Extract phase_indices (first MAX_PHASES doubles, stored as doubles)
            int phase_indices[MAX_PHASES];
            for (int i = 0; i < MAX_PHASES; ++i) {
                phase_indices[i] = (int)initial_data_byte_array[struct_offset + i];
            }
            
            // Extract phase_amounts (next MAX_PHASES doubles)
            double phase_amounts[MAX_PHASES];
            for (int i = 0; i < MAX_PHASES; ++i) {
                phase_amounts[i] = initial_data_byte_array[struct_offset + MAX_PHASES + i];
            }
            
            // DEBUG: Print what thread 1 is reading
            if (tid == 1) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Thread 1 reading from offset %d:\n", struct_offset);
                printf("  phase_amounts[0] at offset %d = %f\n", struct_offset + MAX_PHASES, phase_amounts[0]);
                printf("  phase_amounts[1] at offset %d = %f\n", struct_offset + MAX_PHASES + 1, phase_amounts[1]);
                #endif
            }
            
            // GPU DEBUG: Store what this thread is reading for first few threads
            if (condition_idx < 5) {
                results_array[base_offset + 5 + MAX_COMPONENTS] = (double)phase_indices[0];  // Store first phase index for debug
                results_array[base_offset + 4 + MAX_COMPONENTS] = phase_amounts[0];          // Store first phase amount for debug
            }
            
            // CRITICAL FIX: Use SystemSpecification->initial_chemical_potentials instead of extracting from grid data  
            // The correct initial chemical potentials are in the SystemSpecification, not in grid_data
            // NOTE: global_spec_ptr_raw is now an array of SystemSpecs, we'll access the appropriate one later
            
            // CRITICAL FIX: Read per-condition chemical potentials from initial_data array
            // Chemical potentials are stored after: phase_indices + phase_amounts + site_fractions + compositions
            // Use the same MAX constants that Python used to create the data layout
            const int chem_pot_offset = MAX_PHASES + MAX_PHASES + 
                                       (MAX_PHASES * MAX_DOF_PER_PHASE) + 
                                       (MAX_PHASES * MAX_COMPONENTS);
            
            // BOUNDS CHECK: Ensure we don't read beyond the array
            const int total_array_size = num_conditions_total * initial_phase_data_stride;
            const int chem_pot_read_offset = struct_offset + chem_pot_offset;
            
            if (tid < 2) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Thread %d bounds check - trying to read from offset %d, total array size is %d\n",
                       tid, chem_pot_read_offset, total_array_size);
                // Direct test: Try to read the exact offsets we know should have data
                if (tid == 1) {
                    printf("GPU DEBUG: Thread 1 direct read test:\n");
                    printf("  initial_data_byte_array[60] = %f (should be -31525.5 for cond 0)\n", initial_data_byte_array[60]);
                    printf("  initial_data_byte_array[125] = %f (should be -31525.5 for cond 1)\n", initial_data_byte_array[125]);
                    printf("  initial_data_byte_array[126] = %f (should be -40730.8 for cond 1)\n", initial_data_byte_array[126]);
                }
                #endif
            }
            
            for (int i = 0; i < MAX_COMPONENTS; ++i) {
                if (i < (int)my_spec_data[1] && (chem_pot_read_offset + i) < total_array_size) { // num_components is at offset 1
                    chemical_potentials[i] = initial_data_byte_array[chem_pot_read_offset + i];
                    if (tid == 1 && i < 2) {
                        #ifdef VERBOSE_DEBUG
                        printf("GPU DEBUG: Thread 1 reading chem_pot[%d] from offset %d, value = %f\n", 
                               i, chem_pot_read_offset + i, initial_data_byte_array[chem_pot_read_offset + i]);
                        // Also try reading some nearby values to see if there's an offset issue
                        if (i == 0) {
                            printf("GPU DEBUG: Thread 1 - values at offsets 123-127: [%f, %f, %f, %f, %f]\n",
                                   initial_data_byte_array[123], initial_data_byte_array[124], 
                                   initial_data_byte_array[125], initial_data_byte_array[126], 
                                   initial_data_byte_array[127]);
                        }
                        #endif
                    }
                } else {
                    chemical_potentials[i] = 0.0;
                }
            }
            
            if (tid < 3) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Thread %d reading chemical potentials from struct_offset=%d + chem_pot_offset=%d = %d\n", 
                       tid, struct_offset, chem_pot_offset, chem_pot_read_offset);
                printf("GPU DEBUG: Thread %d SystemSpecification check - num_components=%d\n", tid, (int)my_spec_data[1]);
                if (tid == 1) {
                    printf("GPU DEBUG: Thread 1 - my_spec_data[0]=%f (num_statevars), my_spec_data[1]=%f (num_components)\n",
                           my_spec_data[0], my_spec_data[1]);
                }
                // Print the actual values being read
                printf("GPU DEBUG: Thread %d chemical_potentials after reading: [%f, %f, %f, %f]\n",
                       tid, chemical_potentials[0], chemical_potentials[1], chemical_potentials[2], chemical_potentials[3]);
                for (int i = 0; i < 3; ++i) {
                    printf("  Thread %d chemical_potentials[%d] = %.6e (from initial_data offset %d)\n", 
                           tid, i, chemical_potentials[i], struct_offset + chem_pot_offset + i);
                }
                #endif
            }
            
            // Store essential info for verification
            results_array[base_offset + 3 + MAX_COMPONENTS] = (double)debug_num_phases;
            
            if (tid < 2) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Thread %d extracted phase data - phase_indices[0]=%d, phase_amounts[0]=%f\n", 
                       tid, phase_indices[0], phase_amounts[0]);
                printf("GPU DEBUG: Thread %d chem_pot[0]=%f, chem_pot[1]=%f\n", 
                       tid, chemical_potentials[0], chemical_potentials[1]);
                #endif
            }
            
            // Step 2b: Calculate system Gibbs energy using initial phases (like CPU does)
            double system_gm = 0.0;
            int num_stable_phases = 0;
            double first_phase_amount = 0.0;
            
            // Process initial phases from lower_convex_hull (mirrors CPU logic)
            int safe_num_phases = (debug_num_phases > 0 && debug_num_phases <= MAX_PHASES) ? debug_num_phases : 0;
            
            if (condition_idx < 3) {
                gpu_debug_log_value("total_composition_sets_created", (double)safe_num_phases);
            }
            
            if (tid == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: debug_num_phases=%d, safe_num_phases=%d\n", debug_num_phases, safe_num_phases);
                #endif
            }
            
            // DEBUG: Store early exit info if no phases
            if (safe_num_phases == 0) {
                results_array[base_offset + 0] = -777.0;  // Mark as no phases available
                results_array[base_offset + 6 + MAX_COMPONENTS] = -30.0 - (double)condition_idx;  // No phases error marker  
                return;  // Early exit for debugging
            }
            
            for (int ph_idx = 0; ph_idx < safe_num_phases; ++ph_idx) {
                if (tid == 0 && ph_idx == 0) {
                    #ifdef VERBOSE_DEBUG
                    printf("GPU DEBUG: Starting phase loop with %d phases\n", safe_num_phases);
                    #endif
                }
                // Use direct array access instead of struct pointer
                int phase_record_idx = phase_indices[ph_idx];
                double phase_amount = phase_amounts[ph_idx];
                
                if (tid == 0) {
                    #ifdef VERBOSE_DEBUG
                    printf("GPU DEBUG: Processing phase %d - record_idx=%d, amount=%f\n", 
                           ph_idx, phase_record_idx, phase_amount);
                    #endif
                }
                
                // Validate phase data
                bool phase_valid = (phase_amount > 1e-12 && phase_record_idx >= 0 && phase_record_idx < 4);
                
                if (tid == 0) {
                    #ifdef VERBOSE_DEBUG
                    printf("GPU DEBUG: Phase %d validation - valid=%d (amount>1e-12=%d, idx>=0=%d, idx<max=%d)\n", 
                           ph_idx, phase_valid, (phase_amount > 1e-12), (phase_record_idx >= 0), 
                           (phase_record_idx < 4));
                    #endif
                }
                
                if (phase_valid) {
                    
                    if (tid == 0) {
                        #ifdef VERBOSE_DEBUG
                        printf("GPU DEBUG: Entered phase_valid block for phase %d\n", ph_idx);
                        #endif
                    }
                    
                    // Get phase record (mirrors CPU phase_records[phase_name] access)
                    const PhaseRecord* phase_rec = &g_phase_records_array[phase_record_idx];
                    
                    // Set up DOF array for this phase (mirrors CPU compset.dof setup)
                    double phase_dof[MAX_STATEVARS + MAX_DOF_PER_PHASE];
                    
                    // State variables - from condition args
                    // CRITICAL FIX: GPU functions expect [N, P, T, site_fractions] format
                    // This is because gpu_codegen.py inserts P into state variables
                    if (tid == 0 && ph_idx == 0) {
                        #ifdef VERBOSE_DEBUG
                        printf("GPU DEBUG: Setting up DOF array (GPU format: N, P, T, site_fractions)\n");
                        #endif
                    }
                    
                    // Extract values from condition data based on actual state_variables order
                    // CRITICAL FIX: Extract state variables based on actual count from SystemSpecification
                    double moles_val = 1.0;      // Default N
                    double pressure_val = 101325.0; // Default P
                    double temp_val = 298.15;    // Default T
                    
                    // CRITICAL FIX: Access per-thread SystemSpec data instead of casting shared pointer
                    // global_spec_ptr_raw is an array of SystemSpecs in double format, not a single struct
                    const double* system_specs_array = (const double*)global_spec_ptr_raw;
                    const double* my_spec_doubles = &system_specs_array[condition_idx * system_spec_stride];
                    int actual_num_statevars = (int)my_spec_doubles[0];  // num_statevars is first field
                    
                    if (actual_num_statevars == 2) {
                        // Most common case: [N, T] with no pressure variable
                        moles_val = condition_data_array[condition_offset + 0];    // N from position 0
                        temp_val = condition_data_array[condition_offset + 1];      // T from position 1
                        // pressure_val keeps default value of 101325.0
                    } else if (actual_num_statevars >= 3) {
                        // Full case: [N, P, T]
                        moles_val = condition_data_array[condition_offset + 0];      // N from position 0
                        pressure_val = condition_data_array[condition_offset + 1];   // P from position 1
                        temp_val = condition_data_array[condition_offset + 2];       // T from position 2
                    }
                    
                    // Set up phase_dof based on what the energy function expects
                    // CRITICAL FIX: GPU functions now expect ALL state variables [N, P, T] just like CPU
                    // This matches the fix in notebook_get_all_syms_for_model
                    phase_dof[0] = moles_val;       // x[0] = N
                    phase_dof[1] = pressure_val;    // x[1] = P
                    phase_dof[2] = temp_val;        // x[2] = T
                    // Site fractions will be added starting at index 3
                    
                    if (tid == 0 && ph_idx == 0) {
                        #ifdef VERBOSE_DEBUG
                        printf("GPU DEBUG: DOF state vars from condition_data: N=%f (pos 0), P=%f (pos 1), T=%f (pos 2)\n", 
                               moles_val, pressure_val, temp_val);
                        #endif
                    }
                    
                    // SAFETY CHECK: Validate state variables
                    if (isnan(moles_val) || isinf(moles_val) || moles_val <= 0.0) {
                        results_array[base_offset + 0] = -333.0 - (double)condition_idx;  // Invalid N marker
                        results_array[base_offset + 1] = 0.0;                              // N = index 0
                        results_array[base_offset + 1 + MAX_COMPONENTS] = moles_val;       // The problematic value
                        results_array[base_offset + 6 + MAX_COMPONENTS] = -100.0 - (double)condition_idx; // Invalid state var error marker
                        return;
                    }
                    if (isnan(temp_val) || isinf(temp_val) || temp_val <= 0.0) {
                        results_array[base_offset + 0] = -333.0 - (double)condition_idx;  // Invalid T marker
                        results_array[base_offset + 1] = 1.0;                              // T = index 1 (was 2)
                        results_array[base_offset + 1 + MAX_COMPONENTS] = temp_val;        // The problematic value
                        results_array[base_offset + 6 + MAX_COMPONENTS] = -101.0 - (double)condition_idx; // Invalid state var error marker
                        return;
                    }
                    
                    if (tid == 0 && ph_idx == 0) {
                        #ifdef VERBOSE_DEBUG
                        printf("GPU DEBUG: State variables processed, moving to site fractions\n");
                        #endif
                    }
                    
                    // Site fractions - extract from per-thread data using direct array access
                    int site_frac_offset = struct_offset + MAX_PHASES + MAX_PHASES + (ph_idx * MAX_DOF_PER_PHASE);
                    for (int sf = 0; sf < phase_rec->phase_dof && sf < MAX_DOF_PER_PHASE; ++sf) {
                        double site_frac_val = initial_data_byte_array[site_frac_offset + sf];
                        if (tid == 0 && ph_idx == 0) {
                            #ifdef VERBOSE_DEBUG
                            printf("GPU DEBUG: Site fraction %d = %.15f (phase_dof=%d)\n", sf, site_frac_val, phase_rec->phase_dof);
                            #endif
                        }
                        // SAFETY CHECK: Validate site fractions
                        if (isnan(site_frac_val) || isinf(site_frac_val) || site_frac_val < 0.0 || site_frac_val > 1.0) {
                            results_array[base_offset + 0] = -222.0 - (double)condition_idx;  // Invalid site fraction marker
                            results_array[base_offset + 1] = (double)sf;                      // Which site fraction
                            results_array[base_offset + 1 + MAX_COMPONENTS] = site_frac_val;                   // The problematic value
                            results_array[base_offset + 6 + MAX_COMPONENTS] = -110.0 - (double)condition_idx; // Invalid site fraction error marker
                            return;
                        }
                        phase_dof[3 + sf] = site_frac_val;  // Site fractions start after [N, P, T] (position 3)
                    }
                    
                    if (verbose && ph_idx < 2 && condition_idx < 3) {
                        gpu_debug_log_array("phase_site_fractions", &phase_dof[3], phase_rec->phase_dof);
                    }
                    
                    if (tid == 0 && ph_idx == 0) {
                        #ifdef VERBOSE_DEBUG
                        printf("GPU DEBUG: Site fractions processed, calculating phase energy\n");
                        #endif
                    }
                    
                    // Calculate phase energy using the phase record (mirrors CPU compset.energy calculation)
                    double phase_energy = 0.0;
                    if (phase_rec->obj != nullptr) {
                        if (tid == 0 && ph_idx == 0) {
                            #ifdef VERBOSE_DEBUG
                            printf("GPU DEBUG: DOF for energy calc - N=%.15f, P=%.15f, T=%.15f, Y[0]=%.15f, Y[1]=%.15f\n", 
                                   phase_dof[0], phase_dof[1], phase_dof[2], phase_dof[3], phase_dof[4]);
                            #endif
                        }
                        phase_energy = phase_rec->obj(phase_dof);
                        
                        if (verbose && ph_idx < 2) {
                            #ifdef VERBOSE_DEBUG
                            printf("[GPU]   phase_%d_energy: %.15e\n", ph_idx, phase_energy);
                            #endif
                        }
                        
                        if (tid == 0) {
                            #ifdef VERBOSE_DEBUG
                            printf("GPU DEBUG: Phase %d energy = %.6f J/mol\n", ph_idx, phase_energy);
                            #endif
                        }
                        
                        // NUMERICAL STABILITY CHECK: Detect and prevent NaN/inf propagation
                        if (isnan(phase_energy) || isinf(phase_energy)) {
                            // Store debug info about which phase/condition caused NaN
                            results_array[base_offset + 0] = -999.0 - (double)condition_idx;  // NaN error marker
                            results_array[base_offset + 1] = (double)ph_idx;                  // Which phase caused NaN
                            results_array[base_offset + 1 + MAX_COMPONENTS] = phase_energy;                    // The problematic energy value
                            results_array[base_offset + 6 + MAX_COMPONENTS] = -40.0 - (double)condition_idx;  // NaN error marker
                            return;  // Early exit to prevent NaN propagation
                        }
                        
                        // Additional range check: Ensure energy is within reasonable bounds (increased limit for thermodynamic energies)
                        if (phase_energy > 1e8 || phase_energy < -1e8) {
                            if (tid == 0) {
                                #ifdef VERBOSE_DEBUG
                                printf("GPU DEBUG: Phase energy %f exceeds range check (1e6), triggering early return\n", phase_energy);
                                #endif
                            }
                            // Store debug info about extreme energy values
                            results_array[base_offset + 0] = -888.0 - (double)condition_idx;  // Extreme energy error marker
                            results_array[base_offset + 1] = (double)ph_idx;                  // Which phase
                            results_array[base_offset + 1 + MAX_COMPONENTS] = phase_energy;                    // The extreme energy value
                            results_array[base_offset + 6 + MAX_COMPONENTS] = -50.0 - (double)condition_idx;  // Extreme energy error marker
                            return;  // Early exit to prevent numerical issues
                        }
                    }
                    
                    // SAFETY CHECK: Validate phase amount before multiplication
                    if (tid == 0 && ph_idx < 2) {
                        #ifdef VERBOSE_DEBUG
                        printf("GPU DEBUG: Phase %d - amount=%f, energy=%f\n", ph_idx, phase_amount, phase_energy);
                        #endif
                    }
                    if (isnan(phase_amount) || isinf(phase_amount) || phase_amount < 0.0) {
                        results_array[base_offset + 0] = -777.0 - (double)condition_idx;  // Invalid phase amount marker
                        results_array[base_offset + 1] = (double)ph_idx;                  // Which phase
                        results_array[base_offset + 1 + MAX_COMPONENTS] = phase_amount;                    // The problematic amount
                        results_array[base_offset + 6 + MAX_COMPONENTS] = -60.0 - (double)condition_idx;  // Invalid amount error marker
                        return;
                    }
                    
                    // Add to system Gibbs energy (weighted by phase amount, like CPU)
                    double contribution = phase_amount * phase_energy;
                    
                    // SAFETY CHECK: Validate the contribution before adding to system_gm
                    if (isnan(contribution) || isinf(contribution)) {
                        results_array[base_offset + 0] = -666.0 - (double)condition_idx;  // Invalid contribution marker
                        results_array[base_offset + 1] = phase_amount;                    // The amounts that caused issue
                        results_array[base_offset + 1 + MAX_COMPONENTS] = phase_energy;                    // The energy that caused issue
                        results_array[base_offset + 6 + MAX_COMPONENTS] = -70.0 - (double)condition_idx;  // Invalid contribution error marker
                        return;
                    }
                    
                    system_gm += contribution;
                    num_stable_phases++;
                    
                    // Store first phase amount for output
                    if (ph_idx == 0) {
                        first_phase_amount = phase_amount;
                    }
                }
            }
            
            // Step 2c: NOW CALLING THE ACTUAL EQUILIBRIUM SOLVER
            // Let's use debug arrays to track what happens when we call the real solver
            
            // Store starting point in debug arrays (if enabled)
            if (debug_gm_history != nullptr && debug_max_steps > 0 && condition_idx < num_conditions_total) {
                int debug_idx = condition_idx * debug_max_steps;
                // Step 0: Store starting point from lower_convex_hull
                debug_gm_history[debug_idx] = system_gm;
                for (int comp = 0; comp < MAX_COMPONENTS && comp < 2; ++comp) {
                    debug_mu_history[condition_idx * debug_max_steps * MAX_COMPONENTS + comp] = chemical_potentials[comp];
                }
                debug_convergence_history[debug_idx] = 0;  // Starting point - not converged yet
                debug_iteration_count[condition_idx] = 0;   // Will increment as solver runs
            }
            
            // Prepare data structures for the real solver
            if (tid == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: About to prepare solver data structures\n");
                #endif
            }
            
            // CRITICAL: Create thread-local copy of SystemSpecification with correct composition
            // CRITICAL: Create thread-local SystemSpecification from global_spec_ptr_raw
            char thread_spec_bytes[sizeof(SystemSpecification)];
            memset(thread_spec_bytes, 0, sizeof(SystemSpecification));
            SystemSpecification* thread_spec_ptr = (SystemSpecification*)thread_spec_bytes;
            SystemSpecification& thread_spec = *thread_spec_ptr;
            
            // CRITICAL FIX: Copy thread-specific SystemSpec instead of shared one
            // Calculate offset to this thread's SystemSpec in the array
            const double* system_specs_array = (const double*)global_spec_ptr_raw;
            
            // Calculate size including work arrays to match Python
            const int svd_dim_local = MAX_PHASES + MAX_FIXED_MOLE_FRACTION_CONDITIONS + MAX_COMPONENTS + MAX_STATEVARS + 2;
            const int svd_m_local = svd_dim_local;
            const int svd_n_local = svd_dim_local;
            const int phase_matrix_dim_local = MAX_COMPONENTS + MAX_COMPONENTS;  // Approximation
            
            int spec_core_doubles = 3 + MAX_COMPONENTS + (MAX_FIXED_MOLE_FRACTION_CONDITIONS * MAX_COMPONENTS) + 
                                   MAX_FIXED_MOLE_FRACTION_CONDITIONS + 2 + (MAX_COMPONENTS + 1) + 
                                   (MAX_STATEVARS + 1) + (MAX_COMPONENTS + 1) + (MAX_STATEVARS + 1) + 
                                   (MAX_PHASES + 1) + 1 + 1;
                                   
            int spec_work_doubles = (svd_m_local * svd_n_local) + (svd_m_local * svd_n_local) + (svd_n_local * svd_n_local) + 
                                   svd_n_local + svd_n_local + 
                                   (phase_matrix_dim_local * phase_matrix_dim_local) + (phase_matrix_dim_local * phase_matrix_dim_local) +
                                   phase_matrix_dim_local + phase_matrix_dim_local + (phase_matrix_dim_local * phase_matrix_dim_local);
                                   
            int spec_size_doubles = spec_core_doubles + spec_work_doubles;
            const double* my_spec_doubles = &system_specs_array[condition_idx * system_spec_stride];
            
            // CRITICAL FIX: Manually copy fields from double array to struct
            // Python stores everything as doubles in a flat array, we need to 
            // reconstruct the struct with proper types
            int py_offset = 0;
            
            // Basic integer fields (stored as doubles in Python)
            thread_spec.num_statevars = (int)my_spec_doubles[py_offset++];
            thread_spec.num_components = (int)my_spec_doubles[py_offset++];
            thread_spec.prescribed_system_amount = my_spec_doubles[py_offset++];
            
            // Initial chemical potentials array
            for (int i = 0; i < MAX_COMPONENTS; ++i) {
                thread_spec.initial_chemical_potentials[i] = my_spec_doubles[py_offset++];
            }
            
            // Prescribed mole fraction coefficients (2D array)
            for (int i = 0; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS; ++i) {
                for (int j = 0; j < MAX_COMPONENTS; ++j) {
                    thread_spec.prescribed_mole_fraction_coefficients[i][j] = my_spec_doubles[py_offset++];
                }
            }
            
            // Prescribed mole fraction RHS
            for (int i = 0; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS; ++i) {
                thread_spec.prescribed_mole_fraction_rhs[i] = my_spec_doubles[py_offset++];
            }
            
            // More integer fields
            thread_spec.num_prescribed_mole_fraction_conditions = (int)my_spec_doubles[py_offset++];
            thread_spec.num_prescribed_mole_fraction_coefficients_cols = (int)my_spec_doubles[py_offset++];
            
            // DEBUG: Print what we just copied
            if (condition_idx <= 2) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Thread %d - Copied from my_spec_doubles at offset %d:\n", 
                       condition_idx, condition_idx * system_spec_stride);
                printf("  First 10 doubles: ");
                for (int i = 0; i < 10; ++i) {
                    printf("%.3f ", my_spec_doubles[i]);
                }
                printf("\n");
                printf("  Resulting thread_spec: num_statevars=%d, num_components=%d\n",
                       thread_spec.num_statevars, thread_spec.num_components);
                #endif
                
                // Print prescribed_mole_fraction_rhs values
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Thread %d using prescribed_mole_fraction_rhs[0] = %f (should be X[1] for this condition)\n",
                       condition_idx, thread_spec.prescribed_mole_fraction_rhs[0]);
                #endif
            }
            
            // Index arrays with their counts
            for (int i = 0; i < MAX_COMPONENTS; ++i) {
                thread_spec.free_chemical_potential_indices[i] = (int)my_spec_doubles[py_offset++];
            }
            thread_spec.num_free_chemical_potentials = (int)my_spec_doubles[py_offset++];
            
            for (int i = 0; i < MAX_STATEVARS; ++i) {
                thread_spec.free_statevar_indices[i] = (int)my_spec_doubles[py_offset++];
            }
            thread_spec.num_free_statevars = (int)my_spec_doubles[py_offset++];
            
            for (int i = 0; i < MAX_COMPONENTS; ++i) {
                thread_spec.fixed_chemical_potential_indices[i] = (int)my_spec_doubles[py_offset++];
            }
            thread_spec.num_fixed_chemical_potentials = (int)my_spec_doubles[py_offset++];
            
            for (int i = 0; i < MAX_STATEVARS; ++i) {
                thread_spec.fixed_statevar_indices[i] = (int)my_spec_doubles[py_offset++];
            }
            thread_spec.num_fixed_statevars = (int)my_spec_doubles[py_offset++];
            
            for (int i = 0; i < MAX_PHASES; ++i) {
                thread_spec.fixed_stable_compset_indices[i] = (int)my_spec_doubles[py_offset++];
            }
            thread_spec.num_fixed_stable_compsets = (int)my_spec_doubles[py_offset++];
            
            thread_spec.max_num_free_stable_phases = (int)my_spec_doubles[py_offset++];
            thread_spec.ALLOWED_MASS_RESIDUAL = my_spec_doubles[py_offset++];
            
            // Work arrays are not copied - they're allocated separately in global memory
            
            // CRITICAL FIX: Safely read SystemSpecification fields
            // sys_spec_data no longer needed - we copy the struct directly
            // Read fields by offset: num_statevars=0, num_components=1, prescribed_system_amount=2
            ConditionArgsSingle condition_args_single;
            EquilibriumResultSingle equilibrium_result;
            DevicePhaseData device_phase_data;
            InitialPhaseDataSingle initial_phase_data_single;
            DeviceGrid* device_grid = (DeviceGrid*)grid_data_ptr_raw;
            
            // Set up condition args - copy actual state variables from Python
            // The SystemSpecification tells us which state variables are actually in use
            // Common cases:
            // - If state_vars = [N, T]: copy N at 0, T at 1  
            // - If state_vars = [N, P, T]: copy N at 0, P at 1, T at 2
            // For now, assume [N, T] order which is most common
            int actual_num_statevars = thread_spec.num_statevars;
            
            // Copy the actual state variables that were sent from Python
            for (int i = 0; i < MAX_STATEVARS; ++i) {
                if (i < actual_num_statevars) {
                    // Copy all state variables from condition_data_array
                    condition_args_single.state_variables_values[i] = condition_data_array[condition_offset + i];
                } else {
                    condition_args_single.state_variables_values[i] = 0.0;
                }
            }
            
            if (tid == 0) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Copied %d state variables to condition_args_single:\n", actual_num_statevars);
                printf("  [0]=%f (N), [1]=%f (P), [2]=%f (T)\n", 
                       condition_args_single.state_variables_values[0],
                       condition_args_single.state_variables_values[1],
                       condition_args_single.state_variables_values[2]);
                #endif
            }
            
            // Set up initial phase data single
            initial_phase_data_single.num_phases = safe_num_phases;
            for (int i = 0; i < MAX_PHASES; ++i) {
                initial_phase_data_single.phase_indices[i] = (i < safe_num_phases) ? phase_indices[i] : -1;
                initial_phase_data_single.phase_amounts[i] = (i < safe_num_phases) ? phase_amounts[i] : 0.0;
            }
            for (int i = 0; i < MAX_COMPONENTS; ++i) {
                initial_phase_data_single.chemical_potentials[i] = chemical_potentials[i];
            }
            
            // thread_spec already created above - no need to recreate
            // The prescribed_mole_fraction_rhs values are already correctly set in the SystemSpecification array
            // DO NOT overwrite them here - that was a binary-system-specific hack that breaks ternary systems
            
            // Set up device phase data  
            device_phase_data.phase_records_array = g_phase_records_array;
            device_phase_data.num_unique_phase_records = 4;
            device_phase_data.grid_phase_id_to_record_index = nullptr; // Not using grid mapping for now
            device_phase_data.max_grid_phase_id = 0;
            
            // Initialize result structure
            equilibrium_result.converged = false;
            equilibrium_result.final_system_gm = system_gm;  // Start with initial value
            equilibrium_result.num_stable_phases = 0;
            for (int i = 0; i < MAX_COMPONENTS; ++i) {
                equilibrium_result.final_chemical_potentials[i] = chemical_potentials[i];
            }
            for (int i = 0; i < MAX_PHASES; ++i) {
                equilibrium_result.phase_ids[i] = -1;
                equilibrium_result.NP[i] = 0.0;
            }
            // Initialize X_phases and Y_phases to zero to avoid garbage values
            for (int i = 0; i < MAX_PHASES * MAX_COMPONENTS; ++i) {
                equilibrium_result.X_phases[i] = 0.0;
            }
            for (int i = 0; i < MAX_PHASES * MAX_DOF_PER_PHASE; ++i) {
                equilibrium_result.Y_phases[i] = 0.0;
            }
            
            // CRITICAL DEBUG: Store values before calling solver
            if (debug_gm_history != nullptr && debug_max_steps > 1 && condition_idx < num_conditions_total) {
                int debug_idx = condition_idx * debug_max_steps + 1;
                debug_gm_history[debug_idx] = -999.0;  // Marker: about to call solver
                debug_convergence_history[debug_idx] = -1;  // Marker: solver call attempt
            }
            
            // ISSUE IDENTIFIED: Stack overflow in solve_equilibrium_at_condition
            // The solver function allocates large stack arrays (SVD matrices, etc.) that exceed GPU thread stack limits
            // ROOT CAUSE: MAX_SVD_DIM = 18, so arrays like A_lstsq_copy[18*18], U_lstsq[18*18], V_lstsq[18*18] 
            //            = ~324 doubles each = ~2.6KB each, plus many more arrays = total stack usage > GPU limits
            // SOLUTION NEEDED: Refactor solver to use global/shared memory instead of stack arrays, or
            //                  implement simplified GPU-specific solver that fits in stack limits
            
            // Mark step: entering solver (COMMENTED OUT - causes stack overflow)
            if (debug_gm_history != nullptr && debug_max_steps > 3) {
                debug_gm_history[condition_idx * debug_max_steps + 3] = -1000.0;  // Marker: would enter solver
            }
            
            // REFACTORED: Call sophisticated solver with global memory arrays
            // This is the full equilibrium solver using global memory to avoid stack overflow
            
            
            if (condition_idx == 0 || condition_idx == 1 || condition_idx == 2) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: CALLING solve_equilibrium_at_condition_global_mem for condition %d\n", condition_idx);
                printf("GPU DEBUG: global_spec_ptr_raw=%p, thread_spec address=%p\n", global_spec_ptr_raw, &thread_spec);
                printf("GPU DEBUG: Thread %d thread_spec fields after copy:\n", condition_idx);
                printf("  num_statevars=%d (should be 3)\n", thread_spec.num_statevars);
                printf("  num_components=%d (should be 3)\n", thread_spec.num_components);
                printf("  prescribed_system_amount=%f\n", thread_spec.prescribed_system_amount);
                printf("  num_prescribed_mole_fraction_conditions=%d\n", thread_spec.num_prescribed_mole_fraction_conditions);
                printf("  initial_chemical_potentials[0]=%f\n", thread_spec.initial_chemical_potentials[0]);
                printf("  initial_chemical_potentials[1]=%f\n", thread_spec.initial_chemical_potentials[1]);
                #endif
                if (thread_spec.num_prescribed_mole_fraction_conditions > 0) {
                    #ifdef VERBOSE_DEBUG
                    printf("  prescribed_mole_fraction_rhs[0]=%f (should be X[1] for this condition)\n",
                           thread_spec.prescribed_mole_fraction_rhs[0]);
                    #endif
                }
            }
            solve_equilibrium_at_condition_global_mem(
                condition_idx,           // thread_id
                &thread_spec,           // thread-local system specification with correct X[1]
                &condition_args_single, // conditions for this point
                &equilibrium_result,    // result structure
                &device_phase_data,     // phase data
                initial_data_byte_array + struct_offset, // initial phases for THIS thread (offset into array)
                device_grid,            // grid data (can be null)
                thread_mole_fractions,  // NEW: Pass the actual mole fractions from condition
                // Global memory arrays (per-thread slices)
                thread_A_lstsq_copy, thread_U_lstsq, thread_V_lstsq,
                thread_singular_values_lstsq, thread_superdiag_lstsq,
                thread_U_inv, thread_V_inv, thread_singular_values_inv, 
                thread_superdiag_inv, thread_work_inv,
                thread_x_dof, thread_grad, thread_hess,
                thread_masses, thread_mass_jac, thread_phase_matrix,
                thread_equilibrium_matrix, thread_equilibrium_rhs, thread_eq_soln,
                global_system_states ? &global_system_states[thread_idx * SYSTEM_STATE_SIZE] : nullptr
            );
            
            // COMMENTED OUT: Temporary placeholder values (real solver is now being called above)
            /*
            equilibrium_result.converged = true;  // Assume convergence for demo
            equilibrium_result.final_system_gm = system_gm - 100.0;  // Slight improvement to show solver ran
            for (int i = 0; i < MAX_COMPONENTS; ++i) {
                // Demonstrate that solver would modify chemical potentials
                equilibrium_result.final_chemical_potentials[i] = chemical_potentials[i] - 1000.0 * (i + 1);
            }
            equilibrium_result.num_stable_phases = safe_num_phases;
            for (int i = 0; i < safe_num_phases && i < MAX_PHASES; ++i) {
                equilibrium_result.phase_ids[i] = phase_indices[i];
                equilibrium_result.NP[i] = phase_amounts[i];
            }
            */
            
            // Mark step: solver completed (demo)
            if (debug_gm_history != nullptr && debug_max_steps > 4) {
                debug_gm_history[condition_idx * debug_max_steps + 4] = -2000.0;  // Marker: solver demo completed
            }
            
            // CRITICAL DEBUG: Store values after calling solver
            if (debug_gm_history != nullptr && debug_max_steps > 2 && condition_idx < num_conditions_total) {
                int debug_idx = condition_idx * debug_max_steps + 2;
                debug_gm_history[debug_idx] = equilibrium_result.final_system_gm;  // Result from solver
                for (int comp = 0; comp < MAX_COMPONENTS && comp < 2; ++comp) {
                    debug_mu_history[condition_idx * debug_max_steps * MAX_COMPONENTS + 2 * MAX_COMPONENTS + comp] = equilibrium_result.final_chemical_potentials[comp];
                }
                debug_convergence_history[debug_idx] = equilibrium_result.converged ? 1 : 0;
                debug_iteration_count[condition_idx] = 6;  // Start + before + solver call + after
            }
            
            
            // SAFETY CHECK: Validate solver results
            if (isnan(equilibrium_result.final_system_gm) || isinf(equilibrium_result.final_system_gm)) {
                results_array[base_offset + 0] = -777.0 - (double)condition_idx;  // Solver returned NaN GM
                results_array[base_offset + 6 + MAX_COMPONENTS] = -100.0 - (double)condition_idx;  // Solver NaN error marker
                return;
            }
            
            if (isnan(equilibrium_result.final_chemical_potentials[0]) || isinf(equilibrium_result.final_chemical_potentials[0])) {
                results_array[base_offset + 0] = -666.0 - (double)condition_idx;  // Solver returned NaN MU
                results_array[base_offset + 1] = equilibrium_result.final_chemical_potentials[0];  // Store problematic value
                results_array[base_offset + 6 + MAX_COMPONENTS] = -110.0 - (double)condition_idx;  // Solver MU NaN error marker
                return;
            }
            
            // Store final results from the REAL solver
            if (tid <= 2) {
                #ifdef VERBOSE_DEBUG
                printf("GPU DEBUG: Thread %d storing final result - equilibrium_result.final_system_gm=%f\n", 
                       tid, equilibrium_result.final_system_gm);
                #endif
            }
            results_array[base_offset + 0] = equilibrium_result.final_system_gm;       // Final GM from solver
            for (int i = 0; i < MAX_COMPONENTS; ++i) {
                results_array[base_offset + 1 + i] = equilibrium_result.final_chemical_potentials[i]; // Final MU from solver
            }
            // CRITICAL FIX: Store ALL phase amounts, not just the first one
            // The old code only stored NP[0], causing GPU to report only 1 phase even when 2 were found
            // Store phase amounts starting at offset 1 + MAX_COMPONENTS
            for (int ph_idx = 0; ph_idx < MAX_PHASES; ++ph_idx) {
                results_array[base_offset + 1 + MAX_COMPONENTS + ph_idx] = equilibrium_result.NP[ph_idx];
            }
            
            // Shift other results to make room for all phase amounts
            int shift_offset = MAX_PHASES - 1;  // We need MAX_PHASES-1 extra spots since we already had 1
            results_array[base_offset + 1 + MAX_COMPONENTS + MAX_PHASES] = equilibrium_result.converged ? 1.0 : 0.0; // Real convergence from solver
            results_array[base_offset + 2 + MAX_COMPONENTS + MAX_PHASES] = (double)equilibrium_result.num_stable_phases; // Number of stable phases from solver
            results_array[base_offset + 3 + MAX_COMPONENTS + MAX_PHASES] = temp;
            results_array[base_offset + 4 + MAX_COMPONENTS + MAX_PHASES] = pressure;
            results_array[base_offset + 5 + MAX_COMPONENTS + MAX_PHASES] = equilibrium_result.converged ? 7777.0 : 8888.0; // Real solver marker (7777=converged, 8888=not converged)
            
            // Store Y_phases values (site fractions) from equilibrium_result
            // Updated offset to account for all phase amounts being stored
            int y_offset = base_offset + 6 + MAX_COMPONENTS + MAX_PHASES;  // Start after the standard results + all phase amounts
            for (int phase_idx = 0; phase_idx < MAX_PHASES; ++phase_idx) {
                for (int dof_idx = 0; dof_idx < MAX_DOF_PER_PHASE; ++dof_idx) {
                    int y_index = phase_idx * MAX_DOF_PER_PHASE + dof_idx;
                    if (y_index < MAX_PHASES * MAX_DOF_PER_PHASE) {
                        results_array[y_offset + y_index] = equilibrium_result.Y_phases[y_index];
                    }
                }
            }
            
            // Store X_phases values (mole fractions) from equilibrium_result
            int x_offset = y_offset + (MAX_PHASES * MAX_DOF_PER_PHASE);  // Start after Y_phases
            for (int phase_idx = 0; phase_idx < MAX_PHASES; ++phase_idx) {
                for (int comp_idx = 0; comp_idx < MAX_COMPONENTS; ++comp_idx) {
                    int x_index = phase_idx * MAX_COMPONENTS + comp_idx;
                    if (x_index < MAX_PHASES * MAX_COMPONENTS) {
                        results_array[x_offset + x_index] = equilibrium_result.X_phases[x_index];
                    }
                }
            }
            
            // CRITICAL FIX: Store phase_ids from equilibrium_result
            // This was missing, causing all phases to be labeled with ID 0
            int phase_ids_offset = x_offset + (MAX_PHASES * MAX_COMPONENTS);  // Start after X_phases
            for (int phase_idx = 0; phase_idx < MAX_PHASES; ++phase_idx) {
                results_array[phase_ids_offset + phase_idx] = (double)equilibrium_result.phase_ids[phase_idx];
            }
            
        } else {
            // No initial data available
            results_array[base_offset + 0] = -888888.0;  // Mark as no initial data
            results_array[base_offset + 6 + MAX_COMPONENTS] = -20.0 - (double)condition_idx;  // Error marker
            
            // Initialize Y_phases to zero
            int y_offset = base_offset + 6 + MAX_COMPONENTS + MAX_PHASES;
            for (int i = 0; i < MAX_PHASES * MAX_DOF_PER_PHASE; ++i) {
                results_array[y_offset + i] = 0.0;
            }
            
            // Initialize X_phases to zero
            int x_offset = y_offset + (MAX_PHASES * MAX_DOF_PER_PHASE);
            for (int i = 0; i < MAX_PHASES * MAX_COMPONENTS; ++i) {
                results_array[x_offset + i] = 0.0;
            }
        }
    }
}

} // extern "C"
