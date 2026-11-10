/** Hand-written cyclic Jacobi eigenvalue solver for real symmetric matrices. */
final class Jacobi {
    /**
     * Diagonalises symmetric matrix a (destroyed in the process).
     * @param values  output: eigenvalues (length n)
     * @param vectors output: vectors[k] is the unit eigenvector for values[k] (n x n)
     */
    static void eigen(double[][] a, double[] values, double[][] vectors) {
        int n = a.length;
        for (int i = 0; i < n; i++) {
            java.util.Arrays.fill(vectors[i], 0);
            vectors[i][i] = 1;
        }
        for (int sweep = 0; sweep < 60; sweep++) {
            double off = 0, diag = 0;
            for (int p = 0; p < n; p++) {
                diag += a[p][p] * a[p][p];
                for (int q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
            }
            if (off <= 1e-22 * Math.max(diag, 1e-300)) break;

            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double apq = a[p][q];
                    if (Math.abs(apq) < 1e-300) continue;
                    double theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                    double t = Math.signum(theta == 0 ? 1 : theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                    double c = 1.0 / Math.sqrt(t * t + 1), s = t * c;

                    double app = a[p][p], aqq = a[q][q];
                    a[p][p] = app - t * apq;
                    a[q][q] = aqq + t * apq;
                    a[p][q] = a[q][p] = 0;
                    for (int r = 0; r < n; r++) {
                        if (r == p || r == q) continue;
                        double arp = a[r][p], arq = a[r][q];
                        a[r][p] = a[p][r] = c * arp - s * arq;
                        a[r][q] = a[q][r] = c * arq + s * arp;
                    }
                    // vectors are stored as rows (vectors[k] = k-th eigenvector)
                    double[] vp = vectors[p], vq = vectors[q];
                    for (int r = 0; r < n; r++) {
                        double x = vp[r], y = vq[r];
                        vp[r] = c * x - s * y;
                        vq[r] = s * x + c * y;
                    }
                }
            }
        }
        for (int i = 0; i < n; i++) values[i] = a[i][i];
    }
}
