import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Image loading and preprocessing, written from scratch:
 * grayscale conversion, bilinear resize to 64x64, histogram equalisation.
 * (ImageIO is used ONLY to decode PNG/JPG/BMP files into raw pixels; PGM is parsed by hand.)
 */
final class ImageUtil {
    static final int SIZE = 64;

    static final class Gray {
        final int w, h;
        final int[] px; // 0..255
        Gray(int w, int h, int[] px) { this.w = w; this.h = h; this.px = px; }
    }

    // ---------------------------------------------------------------- loading

    static Gray load(Path p) throws IOException {
        String n = p.getFileName().toString().toLowerCase();
        if (n.endsWith(".pgm") || n.endsWith(".ppm")) return readPnm(p);
        BufferedImage img = ImageIO.read(p.toFile());
        if (img == null) throw new IOException("Unsupported or corrupt image: " + p);
        int w = img.getWidth(), h = img.getHeight();
        int[] g = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 255, gg = (rgb >> 8) & 255, b = rgb & 255;
                g[y * w + x] = toGray(r, gg, b);
            }
        return new Gray(w, h, g);
    }

    /** Grayscale conversion (ITU-R BT.601 luma). */
    static int toGray(int r, int g, int b) {
        return (int) Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }

    private static Gray readPnm(Path p) throws IOException {
        byte[] d = Files.readAllBytes(p);
        int[] pos = {0};
        String magic = token(d, pos);
        int w = Integer.parseInt(token(d, pos));
        int h = Integer.parseInt(token(d, pos));
        int max = Integer.parseInt(token(d, pos));
        int[] out = new int[w * h];
        boolean binary = magic.equals("P5") || magic.equals("P6");
        boolean color = magic.equals("P3") || magic.equals("P6");
        if (!binary && !magic.equals("P2") && !magic.equals("P3"))
            throw new IOException("Unsupported PNM format " + magic + ": " + p);
        if (binary) pos[0]++; // single whitespace after maxval
        int ch = color ? 3 : 1;
        for (int i = 0; i < w * h; i++) {
            int[] v = new int[ch];
            for (int c = 0; c < ch; c++) {
                int val;
                if (binary) {
                    if (max > 255) { val = ((d[pos[0]] & 255) << 8) | (d[pos[0] + 1] & 255); pos[0] += 2; }
                    else val = d[pos[0]++] & 255;
                } else val = Integer.parseInt(token(d, pos));
                v[c] = val * 255 / max;
            }
            out[i] = color ? toGray(v[0], v[1], v[2]) : v[0];
        }
        return new Gray(w, h, out);
    }

    private static String token(byte[] d, int[] pos) {
        int i = pos[0];
        while (i < d.length) {
            if (d[i] == '#') { while (i < d.length && d[i] != '\n') i++; }
            else if (Character.isWhitespace(d[i])) i++;
            else break;
        }
        int s = i;
        while (i < d.length && !Character.isWhitespace(d[i])) i++;
        pos[0] = i;
        return new String(d, s, i - s, StandardCharsets.US_ASCII);
    }

    static void savePgm(int[] px, int w, int h, Path p) throws IOException {
        byte[] head = ("P5\n" + w + " " + h + "\n255\n").getBytes(StandardCharsets.US_ASCII);
        byte[] out = new byte[head.length + px.length];
        System.arraycopy(head, 0, out, 0, head.length);
        for (int i = 0; i < px.length; i++) out[head.length + i] = (byte) px[i];
        Files.write(p, out);
    }

    // ---------------------------------------------------------- preprocessing

    /** Bilinear resize. */
    static int[] resize(Gray g, int nw, int nh) {
        int[] out = new int[nw * nh];
        double sx = (double) g.w / nw, sy = (double) g.h / nh;
        for (int y = 0; y < nh; y++) {
            double fy = (y + 0.5) * sy - 0.5;
            int y0 = (int) Math.floor(fy);
            double wy = fy - y0;
            int y1 = Math.min(Math.max(y0 + 1, 0), g.h - 1);
            y0 = Math.min(Math.max(y0, 0), g.h - 1);
            for (int x = 0; x < nw; x++) {
                double fx = (x + 0.5) * sx - 0.5;
                int x0 = (int) Math.floor(fx);
                double wx = fx - x0;
                int x1 = Math.min(Math.max(x0 + 1, 0), g.w - 1);
                x0 = Math.min(Math.max(x0, 0), g.w - 1);
                double top = g.px[y0 * g.w + x0] * (1 - wx) + g.px[y0 * g.w + x1] * wx;
                double bot = g.px[y1 * g.w + x0] * (1 - wx) + g.px[y1 * g.w + x1] * wx;
                out[y * nw + x] = (int) Math.round(top * (1 - wy) + bot * wy);
            }
        }
        return out;
    }

    /** Histogram equalisation (CDF-based, 256 levels). */
    static int[] equalize(int[] px) {
        int[] hist = new int[256];
        for (int v : px) hist[Math.min(255, Math.max(0, v))]++;
        int[] cdf = new int[256];
        int run = 0, cdfMin = 0;
        for (int i = 0; i < 256; i++) {
            run += hist[i];
            cdf[i] = run;
            if (cdfMin == 0 && hist[i] > 0) cdfMin = hist[i];
        }
        int n = px.length;
        if (n == cdfMin) return px.clone(); // flat image
        int[] lut = new int[256];
        for (int i = 0; i < 256; i++)
            lut[i] = (int) Math.round((double) Math.max(0, cdf[i] - cdfMin) / (n - cdfMin) * 255.0);
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = lut[Math.min(255, Math.max(0, px[i]))];
        return out;
    }

    /** Full pipeline: load -> grayscale -> 64x64 bilinear -> histogram equalisation. */
    static int[] preprocess(Path p) throws IOException {
        return equalize(resize(load(p), SIZE, SIZE));
    }

    /** Load an already-preprocessed 64x64 PGM from the face store. */
    static int[] loadPreprocessed(Path p) throws IOException {
        Gray g = readPnm(p);
        if (g.w != SIZE || g.h != SIZE) throw new IOException("Bad stored face: " + p);
        return g.px;
    }
}
