import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import util.MathUtil;

import java.util.Random;

import static util.MathUtil.generateRandomMat;
import static util.MathUtil.generateRandomVec;
import static util.MathUtil.lu;

public class MathUtilTest {

    public static void main(String args[]){

        // factors a matrix into upper and lower triangular matrices by
        // gaussian elimination. On entry, a[0..n-1][0..n-1] is the
        // matrix to be factored. ipvt[] returns an integer vector of pivot
        // indices, used in the lu_solve() routine.
        //        double mat[][] = {
        //                {4,2,-1,3},
        //                {3,-4,2,5},
        //                {-2,6,-5,-2},
        //                {5,1,6,-3}
        //        };
        //
        //        double rs[]={16.9,-14,25,9.4};
        //        double x[]={4.5,1.6,-3.8,-2.7};
        //
        //        int ipvt[]={0,0,0,0};
        //
        //        int n=4;

        double mat[][] = generateRandomMat(500);

        int [] lst = {4,4,4,8,8,8,16,16,16,32,32,32,64,64,64,128,128,128,512,512,512};
        for(int n : lst){
//            int n = (int) Math.round(Math.pow(2,i));
//            DoubleMatrix jblasmat = DoubleMatrix.randn(n,n);
            FloatMatrix jblasmat = FloatMatrix.randn(n,n);
            double inmat_o[][] = new double[n][n];
            double inmat[][] = new double[n][n];
            int ipiv_o[] = new int[n];
            int ipiv[] = new int[n];

            double xvec[] = generateRandomVec(n);
            double xvec_o[] = new double[n];

            for(int k=0;k<n;k++){
                for(int j=0;j<n;j++){
                    inmat[k][j]=inmat_o[k][j]=jblasmat.get(k,j);
                }
                xvec_o[k]=xvec[k];
            }


            long startTime = System.nanoTime();

            //TODO change this method
//            MathUtil.lu_factor(inmat_o,n,ipiv_o);
//            MathUtil.apacheLUD(inmat,n,ipiv);

            long luorig=System.nanoTime();
            MathUtil.lu_factor(inmat_o,n,ipiv_o);
            luorig=Math.round((System.nanoTime()-luorig)/1000.);

            long lublasnoadmin=System.nanoTime();
//            MathUtil.jblasLUD(jblasmat);
            lu(jblasmat,ipiv);
            lublasnoadmin=Math.round((System.nanoTime()-lublasnoadmin)/1000.);

            long lublasadmin=System.nanoTime();
//            MathUtil.jblasLUD(inmat,n,ipiv);
            lublasadmin=Math.round((System.nanoTime()-lublasadmin)/1000.);


//            long luTime = System.nanoTime();
//
//            //TODO change this method
//            MathUtil.lu_solve(inmat,n,ipiv,xvec);
//            MathUtil.lu_solve(inmat_o,n,ipiv_o,xvec_o);
//
//            long curTime = System.nanoTime();
//            long elapsedTimeLUFact = Math.round((luTime-startTime)/1000.);
//            long elapsedTimeTotal = Math.round((curTime - startTime)/1000.);
//            long elapsedTimeLUSolve = Math.round((curTime-luTime)/1000.);
//            System.out.printf("For n=%d LU_fact time = %dus LU_solve time = %dus Total %dus",n,elapsedTimeLUFact,elapsedTimeLUSolve,elapsedTimeTotal);

            System.out.printf("For n=%d LU_fact_orig time = %dus LU_blas_native time = %dus LU_blas_converted %dus",n,luorig,lublasnoadmin,lublasadmin);
            System.out.println();
        }

    }


}
