import java.util.ArrayList;
import java.util.List;

/**
 * Local Binary Pattern Histograms: 8-neighbour LBP codes, 8x8 grid of cell histograms,
 * chi-square distance, nearest-neighbour matching.
 */
final class LBPH implements Recognizer {
    private static final int GRID = 8;
    private static final int[] DY = {-1, -1, -1, 0, 1, 1, 1, 0};
    private static final int[] DX = {-1, 0, 1, 1, 1, 0, -1, -1};

    private final List<double[]> features = new ArrayList<>();
    private List<String> labels;

    @Override public String name() { return "LBPH"; }

    static double[] features(int[] img) {
        int s = ImageUtil.SIZE, cs = s / GRID;
        double[] f = new double[GRID * GRID * 256];
        for (int y = 0; y < s; y++)
            for (int x = 0; x < s; x++) {
                int c = img[y * s + x], code = 0;
                for (int k = 0; k < 8; k++) {
                    int ny = Math.min(Math.max(y + DY[k], 0), s - 1);
                    int nx = Math.min(Math.max(x + DX[k], 0), s - 1);
                    if (img[ny * s + nx] >= c) code |= 1 << k;
                }
                int cell = (y / cs) * GRID + (x / cs);
                f[cell * 256 + code]++;
            }
        double norm = cs * cs;
        for (int i = 0; i < f.length; i++) f[i] /= norm;
        return f;
    }

    static double chiSquare(double[] a, double[] b) {
        double d = 0;
        for (int i = 0; i < a.length; i++) {
            double sum = a[i] + b[i];
            if (sum > 0) { double diff = a[i] - b[i]; d += diff * diff / sum; }
        }
        return d;
    }

    @Override
    public void train(List<int[]> images, List<String> labels) {
        this.labels = new ArrayList<>(labels);
        features.clear();
        for (int[] im : images) features.add(features(im));
    }

    @Override
    public Match identify(int[] image) {
        double[] f = features(image);
        double best = Double.MAX_VALUE;
        int bi = -1;
        for (int i = 0; i < features.size(); i++) {
            double d = chiSquare(f, features.get(i));
            if (d < best) { best = d; bi = i; }
        }
        return new Match(labels.get(bi), best);
    }
}
