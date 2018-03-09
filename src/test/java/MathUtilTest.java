import util.MathUtil;

import java.util.Arrays;

public class MathUtilTest {


    public static void main(String args[]){

        // factors a matrix into upper and lower triangular matrices by
        // gaussian elimination. On entry, a[0..n-1][0..n-1] is the
        // matrix to be factored. ipvt[] returns an integer vector of pivot
        // indices, used in the lu_solve() routine.
        double mat[][] = {
                {4,2,-1,3},
                {3,-4,2,5},
                {-2,6,-5,-2},
                {5,1,6,-3}
        };

        double rs[]={16.9,-14,25,9.4};
        double x[]={4.5,1.6,-3.8,-2.7};

        int ipvt[]={0,0,0,0};

        int n=4;

        //boolean lu_factor(double a[][], int n, int ipvt[])
        MathUtil.lu_factor(mat,n,ipvt);

        //void lu_solve(double a[][], int n, int ipvt[], double b[]) {
        MathUtil.lu_solve(mat, n, ipvt, rs);

        System.out.println("Solution: "+Arrays.toString(rs));
        System.out.println("Expected: "+Arrays.toString(x));

    }


}
