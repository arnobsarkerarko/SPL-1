import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Eigenfaces (PCA): mean face, NxN covariance trick, Jacobi eigen-solver,
 * projection into face space and nearest-neighbour matching by Euclidean distance.
 */
final class Eigenfaces implements Recognizer {
    private final double varianceKept;
    private double[] mean;
    private double[][] faces;      // k x D eigenfaces (unit length)
    private double[][] projected;  // N x k projections of the training images
    private List<String> labels;

    Eigenfaces() { this(0.95); }
    Eigenfaces(double varianceKept) { this.varianceKept = varianceKept; }

    @Override public String name() { return "Eigenfaces"; }

    int components() { return faces == null ? 0 : faces.length; }

    @Override
    public void train(List<int[]> images, List<String> labels) {
        int n = images.size(), d = images.get(0).length;
        this.labels = new ArrayList<>(labels);

        mean = new double[d];
        for (int[] im : images) for (int j = 0; j < d; j++) mean[j] += im[j];
        for (int j = 0; j < d; j++) mean[j] /= n;

        double[][] a = new double[n][d]; // mean-subtracted faces
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) a[i][j] = images.get(i)[j] - mean[j];

        // Covariance trick: eigenvectors of L = A A^T (NxN) instead of A^T A (DxD)
        double[][] l = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = i; j < n; j++) {
                double s = 0;
                for (int k = 0; k < d; k++) s += a[i][k] * a[j][k];
                l[i][j] = l[j][i] = s;
            }

        double[] vals = new double[n];
        double[][] vecs = new double[n][n];
        Jacobi.eigen(l, vals, vecs);

        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (x, y) -> Double.compare(vals[y], vals[x]));

        double total = 0;
        for (double v : vals) if (v > 0) total += v;
        double top = vals[order[0]];
        List<double[]> kept = new ArrayList<>();
        double cum = 0;
        for (int idx = 0; idx < n; idx++) {
            int o = order[idx];
            if (vals[o] <= 1e-9 * top) break;
            // u = A^T v, then normalise
            double[] u = new double[d];
            for (int i = 0; i < n; i++) {
                double w = vecs[o][i];
                for (int j = 0; j < d; j++) u[j] += w * a[i][j];
            }
            double norm = 0;
            for (double x : u) norm += x * x;
            norm = Math.sqrt(norm);
            if (norm == 0) continue;
            for (int j = 0; j < d; j++) u[j] /= norm;
            kept.add(u);
            cum += vals[o];
            if (cum >= varianceKept * total) break;
        }
        faces = kept.toArray(new double[0][]);

        projected = new double[n][];
        for (int i = 0; i < n; i++) projected[i] = project(a[i]);
    }

    private double[] project(double[] centered) {
        double[] w = new double[faces.length];
        for (int c = 0; c < faces.length; c++) {
            double s = 0;
            double[] u = faces[c];
            for (int j = 0; j < u.length; j++) s += u[j] * centered[j];
            w[c] = s;
        }
        return w;
    }

    @Override
    public Match identify(int[] image) {
        double[] x = new double[image.length];
        for (int j = 0; j < x.length; j++) x[j] = image[j] - mean[j];
        double[] w = project(x);
        double best = Double.MAX_VALUE;
        int bi = -1;
        for (int i = 0; i < projected.length; i++) {
            double s = 0;
            for (int c = 0; c < w.length; c++) {
                double diff = w[c] - projected[i][c];
                s += diff * diff;
            }
            if (s < best) { best = s; bi = i; }
        }
        return new Match(labels.get(bi), Math.sqrt(best));
    }
}
