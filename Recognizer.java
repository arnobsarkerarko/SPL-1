import java.util.List;

/** Common interface for the two face-recognition algorithms. */
interface Recognizer {
    String name();

    /** Train on preprocessed 64x64 images (row-major, length 4096) with their person IDs. */
    void train(List<int[]> images, List<String> labels);

    /** Nearest-neighbour match; smaller distance = more similar. */
    Match identify(int[] image);

    record Match(String label, double distance) {}
}
