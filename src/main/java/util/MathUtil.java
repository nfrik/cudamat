package util;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.jblas.Decompose;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.NativeBlas;
import org.jblas.util.Functions;
import org.jblas.util.Permutations;
import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.blas.CpuLapack;
import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.linalg.jcublas.blas.JcublasLapack;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.util.Random;


public class MathUtil {

    private static Random random = new Random();

    private static Logger logger = LoggerFactory.getLogger(BaseLapack.class);

        private static CpuLapack lapack;
//    private static JcublasLapack lapack;


    public static int getrand(int x) {
        int q = random.nextInt();
        if (q < 0)
            q = -q;
        return q % x;
    }

    //ND4J CPU lapack
    public static CpuLapack getLapack() {
        if(lapack==null){
//            lapack = new JcublasLapack();
            lapack = new CpuLapack();
        }
        return lapack;
    }


    // Solves the set of n linear equations using a LU factorization
    // previously performed by lu_factor. On input, b[0..n-1] is the right
    // hand side of the equations, and on output, contains the solution.
    public static void lu_solve(double a[][], int n, int ipvt[], double b[]) {
        int i;

        // find first nonzero b element
        for (i = 0; i != n; i++) {
            int row = ipvt[i];

            double swap = b[row];
            b[row] = b[i];
            b[i] = swap;
            if (swap != 0)
                break;
        }

        int bi = i++;
        for (; i < n; i++) {
            int row = ipvt[i];
            int j;
            double tot = b[row];

            b[row] = b[i];
            // forward substitution using the lower triangular matrix
            for (j = bi; j < i; j++)
                tot -= a[i][j] * b[j];
            b[i] = tot;
        }
        for (i = n - 1; i >= 0; i--) {
            double tot = b[i];

            // back-substitution using the upper triangular matrix
            int j;
            for (j = i + 1; j != n; j++)
                tot -= a[i][j] * b[j];
            b[i] = tot / a[i][i];
        }
    }

    // factors a matrix into upper and lower triangular matrices by
    // gaussian elimination. On entry, a[0..n-1][0..n-1] is the
    // matrix to be factored. ipvt[] returns an integer vector of pivot
    // indices, used in the lu_solve() routine.
    public static boolean lu_factor(double a[][], int n, int ipvt[]) {
        double scaleFactors[];
        int i, j, k;

        scaleFactors = new double[n];

        // divide each row by its largest element, keeping track of the
        // scaling factors
        for (i = 0; i != n; i++) {
            double largest = 0;
            for (j = 0; j != n; j++) {
                double x = Math.abs(a[i][j]);
                if (x > largest)
                    largest = x;
            }
            // if all zeros, it's a singular matrix
            if (largest == 0)
                return false;
            scaleFactors[i] = 1.0 / largest;
        }

        // use Crout's method; loop through the columns
        for (j = 0; j != n; j++) {

            // calculate upper triangular elements for this column
            for (i = 0; i != j; i++) {
                double q = a[i][j];
                for (k = 0; k != i; k++)
                    q -= a[i][k] * a[k][j];
                a[i][j] = q;
            }

            // calculate lower triangular elements for this column
            double largest = 0;
            int largestRow = -1;
            for (i = j; i != n; i++) {
                double q = a[i][j];
                for (k = 0; k != j; k++)
                    q -= a[i][k] * a[k][j];
                a[i][j] = q;
                double x = Math.abs(q);
                if (x >= largest) {
                    largest = x;
                    largestRow = i;
                }
            }

            // pivoting
            if (j != largestRow) {
                double x;
                for (k = 0; k != n; k++) {
                    x = a[largestRow][k];
                    a[largestRow][k] = a[j][k];
                    a[j][k] = x;
                }
                scaleFactors[largestRow] = scaleFactors[j];
            }

            // keep track of row interchanges
            ipvt[j] = largestRow;

            // avoid zeros
            if (a[j][j] == 0.0) {
                // System.out.println("avoided zero");
                a[j][j] = 1e-18;
            }

            if (j != n - 1) {
                double mult = 1.0 / a[j][j];
                for (i = j + 1; i != n; i++)
                    a[i][j] *= mult;
            }
        }
        return true;
    }



    public static boolean getrf(INDArray A, int n, INDArray IPIV){
        //// TODO: 3/5/2017  create info once, big overhead

        INDArray INFO = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createInt(1),
                Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {1, 1}));

//        INDArray IPIV = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createInt(n),
//                Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {1, n}));

        getLapack().sgetrf(n, n, A, IPIV, INFO);

        if(INFO.getInt(new int[]{0}) < 0) {
            throw new Error("Parameter #" + INFO.getInt(new int[]{0}) + " to getrf() was not valid");
        } else {
            if(INFO.getInt(new int[]{0}) > 0) {
                logger.warn("The matrix is singular - cannot be used for inverse op. Check L matrix at row " + INFO.getInt(new int[]{0}));
                return false;
            }

            return true;
        }

    }

    // Solves the set of n linear equations using a LU factorization
    // previously performed by lu_factor. On input, b[0..n-1] is the right
    // hand side of the equations, and on output, contains the solution.
    public static void lu_solve(INDArray a, int n, INDArray ipvt, double b[]) {
        int i;

        // find first nonzero b element
        for (i = 0; i != n; i++) {
            int row = ipvt.getInt(i)-1;

            double swap = b[row];
            b[row] = b[i];
            b[i] = swap;
            if (swap != 0)
                break;
        }

        int bi = i++;
        for (; i < n; i++) {
            int row = ipvt.getInt(i)-1;
            int j;
            double tot = b[row];

            b[row] = b[i];
            // forward substitution using the lower triangular matrix
            for (j = bi; j < i; j++)
                tot -= a.getDouble(i,j) * b[j];
            b[i] = tot;
        }
        for (i = n - 1; i >= 0; i--) {
            double tot = b[i];

            // back-substitution using the upper triangular matrix
            int j;
            for (j = i + 1; j != n; j++)
                tot -= a.getDouble(i,j) * b[j];
            b[i] = tot / a.getDouble(i,i);
        }
    }



    public static void copy_1Darray(int src[], int dest[]){
        for(int i=0;i<src.length;i++){
            dest[i]=src[i];
        }
    }

    public static void construct_LU(double L[][], double U[][], double LU[][], int n){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(j>=i){
                    LU[i][j]=U[i][j];
                }else{
                    LU[i][j]=L[i][j];
                }
            }
        }
    }

    public static void copy_2D(double src[][], double dest[][], int n){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                dest[i][j]=src[i][j];
            }
        }
    }

    public static boolean apacheLUD(double a[][], int n, int ipvt[]){

//        double[][] data = {{1, 2, 4, 6}, {5, 2, 3, 3}, {1, 2, 3, 5}, {7, 8, 3, 115}};

        RealMatrix matrix = new Array2DRowRealMatrix(a);
        LUDecomposition lud = new LUDecomposition(matrix);
        RealMatrix u = lud.getU();
        if(u == null){
            return false;
        }
        RealMatrix l = lud.getL();
        RealMatrix p = lud.getP();

        int[] pivot=lud.getPivot();

        double[][] lu = new double[n][n];
        copy_1Darray(pivot,ipvt);
        construct_LU(l.getData(),u.getData(),lu,ipvt.length);
        copy_2D(lu,a,ipvt.length);



//        System.out.println(l);
//        System.out.println(u);
//        System.out.println(p);
//        System.out.println(pivot.length);
        return true;
    }

    public static boolean lu(DoubleMatrix A,int[] piv) {
        int info = NativeBlas.dgetrf(A.rows, A.columns, A.data, 0, A.rows, piv, 0);
        if(info!=0){
            return false;
        }
        return true;
    }

    public static boolean lu(FloatMatrix A,int[] piv) {
        int info = NativeBlas.sgetrf(A.rows, A.columns, A.data, 0, A.rows, piv, 0);
        if(info!=0){
            return false;
        }
        return true;
    }

    private static void decomposeLowerUpper(DoubleMatrix A, DoubleMatrix L, DoubleMatrix U) {
        for(int i = 0; i < A.rows; ++i) {
            for(int j = 0; j < A.columns; ++j) {
                if (i < j) {
                    U.put(i, j, A.get(i, j));
                } else if (i == j) {
                    U.put(i, i, A.get(i, i));
                    L.put(i, i, 1.0D);
                } else {
                    L.put(i, j, A.get(i, j));
                }
            }
        }

    }

    public static boolean jblasLUD(double a[][],int n, int ipvt[]){
        DoubleMatrix matrix = new DoubleMatrix(a);
//        LUDecomposition lud = new LUDecomposition(matrix);
        Decompose.LUDecomposition<DoubleMatrix> output = Decompose.lu(matrix);


        for(int i=0;i<n;i++){

            for(int j=0;j<n;j++){
                a[i][j]=output.l.get(i,j)+output.u.get(i,j);
                if(output.p.get(i,j)>0){
                    ipvt[i]=j;
                }

            }
            a[i][i]=a[i][i]-1.;
        }

        return true;
    }

    public static Decompose.LUDecomposition<DoubleMatrix> jblasLUD(DoubleMatrix dm){

        return Decompose.lu(dm);
    }

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


    public static int distanceSq(int x1, int y1, int x2, int y2) {
        x2 -= x1;
        y2 -= y1;
        return x2 * x2 + y2 * y2;
    }

//    public static void main(String[] arsg){
//        lapack=getLapack();
//        double mat[][] = {
//                {4,2,-1,3},
//                {3,-4,2,5},
//                {-2,6,-5,-2},
//                {5,1,6,-3}
//        };
//
//        double mat2[][] = {
//                {4,2,-1,3},
//                {3,-4,2,5},
//                {-2,6,-5,-2},
//                {5,1,6,-3}
//        };
//
//        int n =4;
//        int ipvt[]={0,0,0,0};
//        int ipvt2[]=ipvt.clone();
//
//        MathUtil.lu_factor(mat,n,ipvt);
//
//        apacheLUD(mat2,n,ipvt2);
//    }

    public static void main(String[] args){
        double mat[][] = {
                {1,2,3},
                {1,2,3},
                {1,2,3}
        };

        DoubleMatrix dm = new DoubleMatrix(mat);

        int [] ipiv = new int[3];

//        lu_factor(mat,3,ipiv);

        lu(dm,ipiv);
//        jblasLUD(dm,3,ipiv);

    }
}
