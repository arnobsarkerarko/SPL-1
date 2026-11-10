import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;

/**
 * File-based storage.
 *   <root>/people.tsv            id \t name \t department \t info
 *   <root>/faces/<id>/<n>.pgm    preprocessed 64x64 face images
 */
final class PersonDB {
    record Person(String id, String name, String department, String info) {}

    private final Path root, peopleFile, facesDir;

    PersonDB(Path root) throws IOException {
        this.root = root;
        this.peopleFile = root.resolve("people.tsv");
        this.facesDir = root.resolve("faces");
        Files.createDirectories(facesDir);
    }

    private static String clean(String s) { return s == null ? "" : s.replaceAll("[\\t\\r\\n]+", " ").trim(); }

    Map<String, Person> people() throws IOException {
        Map<String, Person> m = new LinkedHashMap<>();
        if (!Files.exists(peopleFile)) return m;
        for (String line : Files.readAllLines(peopleFile, StandardCharsets.UTF_8)) {
            if (line.isBlank()) continue;
            String[] f = line.split("\t", -1);
            if (f.length < 3) continue;
            m.put(f[0], new Person(f[0], f[1], f[2], f.length > 3 ? f[3] : ""));
        }
        return m;
    }

    private void save(Map<String, Person> m) throws IOException {
        List<String> lines = new ArrayList<>();
        for (Person p : m.values())
            lines.add(String.join("\t", p.id(), p.name(), p.department(), p.info()));
        Files.write(peopleFile, lines, StandardCharsets.UTF_8);
    }

    void upsert(Person p) throws IOException {
        Map<String, Person> m = people();
        m.put(p.id(), new Person(clean(p.id()), clean(p.name()), clean(p.department()), clean(p.info())));
        save(m);
    }

    List<Path> faceFiles(String id) throws IOException {
        Path dir = facesDir.resolve(id);
        if (!Files.isDirectory(dir)) return List.of();
        try (Stream<Path> s = Files.list(dir)) {
            return s.filter(p -> p.toString().endsWith(".pgm")).sorted().toList();
        }
    }

    void addFace(String id, int[] px) throws IOException {
        Path dir = facesDir.resolve(id);
        Files.createDirectories(dir);
        int n = faceFiles(id).size() + 1;
        Path p;
        while (Files.exists(p = dir.resolve(String.format("%03d.pgm", n)))) n++;
        ImageUtil.savePgm(px, ImageUtil.SIZE, ImageUtil.SIZE, p);
    }

    boolean remove(String id) throws IOException {
        Map<String, Person> m = people();
        if (m.remove(id) == null) return false;
        save(m);
        Path dir = facesDir.resolve(id);
        if (Files.isDirectory(dir)) {
            try (Stream<Path> s = Files.walk(dir)) {
                for (Path p : s.sorted(Comparator.reverseOrder()).toList()) Files.delete(p);
            }
        }
        return true;
    }

    /** Loads every stored face with its person ID. */
    void loadSamples(List<int[]> images, List<String> labels) throws IOException {
        for (String id : people().keySet())
            for (Path f : faceFiles(id)) {
                images.add(ImageUtil.loadPreprocessed(f));
                labels.add(id);
            }
    }
}
