import util.MathUtil;

import java.util.Random;

public class MathUtilTest {

    public static double[][] generateRandomMat(int n){
        Random rand = new Random();
        double[][] retMat=new double[n][n];
        int[] ipiv = new int[n];

        do {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    retMat[i][j] = rand.nextGaussian();
                }
            }
        }while(!MathUtil.lu_factor(retMat,n,ipiv));

        return retMat;
    }

    public static double[] generateRandomVec(int n){
        Random rand = new Random();
        double[] retVec = new double[n];
        for(int i=0;i<n;i++){
            retVec[i] = rand.nextGaussian();
        }
        return retVec;
    }

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


        for(int i=2;i<12;i++){
            int n = (int) Math.round(Math.pow(2,i));
            double inmat[][] = generateRandomMat(n);
            double xvec[] = generateRandomVec(n);
            int ipiv[] = new int[n];

            long startTime = System.currentTimeMillis();

            //TODO change this method
            MathUtil.lu_factor(inmat,n,ipiv);
//            MathUtil.lud(inmat,n,ipiv);

            long luTime = System.currentTimeMillis();

            //TODO change this method
            MathUtil.lu_solve(inmat,n,ipiv,xvec);

            long curTime = System.currentTimeMillis();
            long elapsedTimeLUFact = luTime-startTime;
            long elapsedTimeTotal = curTime - startTime;
            long elapsedTimeLUSolve = curTime-luTime;

            System.out.printf("For n=%d LU_fact time = %dms LU_solve time = %dms Total %dms",n,elapsedTimeLUFact,elapsedTimeLUSolve,elapsedTimeTotal);
            System.out.println();
        }

    }


}
