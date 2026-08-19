import os
import numpy as np
import matplotlib.pyplot as plt

from voiceverification.core.asvspoof import compute_score
from voiceverification.core.calibration import find_eer_threshold


def collect_scores(folder, label):
    """
    Mengambil spoof probability dari seluruh file audio
    pada folder yang diberikan.
    """

    scores = []

    print("=" * 70)
    print(f"Memproses {label}")
    print(f"Folder: {folder}")
    print("=" * 70)

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
    ])

    print(f"Jumlah file: {len(files)}")

    for filename in files:
        filepath = os.path.join(folder, filename)

        score, features = compute_score(filepath)

        scores.append(score)

        print(
            f"{filename}: "
            f"spoof_probability={score:.4f}, "
            f"flat={features['flatness']:.4f}, "
            f"var={features['temporal_var']:.4f}, "
            f"high={features['highband']:.4f}"
        )

    return np.array(scores)


def calculate_statistics(scores, label):
    """
    Menghitung statistik sederhana dari distribusi score.
    """

    print("\n" + "-" * 70)
    print(f"STATISTIK {label}")
    print("-" * 70)

    print(f"Jumlah sampel : {len(scores)}")
    print(f"Minimum       : {np.min(scores):.4f}")
    print(f"Maksimum       : {np.max(scores):.4f}")
    print(f"Rata-rata      : {np.mean(scores):.4f}")
    print(f"Median         : {np.median(scores):.4f}")
    print(f"Std. Deviasi   : {np.std(scores):.4f}")


def plot_distribution(genuine_scores, spoof_scores):
    """
    Membuat grafik distribusi spoof probability
    Genuine vs Spoof/Replay.
    """

    plt.figure(figsize=(10, 6))

    plt.hist(
        genuine_scores,
        bins=10,
        alpha=0.6,
        label="Genuine",
        density=True
    )

    plt.hist(
        spoof_scores,
        bins=10,
        alpha=0.6,
        label="Spoof / Replay",
        density=True
    )

    plt.xlabel("Spoof Probability")
    plt.ylabel("Density")
    plt.title("Distribusi Spoof Probability")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    output = "distribution_spoof_probability.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"\nGrafik distribusi disimpan:")
    print(os.path.abspath(output))


def plot_distribution_with_threshold(
    genuine_scores,
    spoof_scores,
    threshold
):
    """
    Membuat grafik distribusi spoof probability
    dengan garis threshold hasil kalibrasi EER.
    """

    plt.figure(figsize=(10, 6))

    plt.hist(
        genuine_scores,
        bins=10,
        alpha=0.6,
        label="Genuine",
        density=True
    )

    plt.hist(
        spoof_scores,
        bins=10,
        alpha=0.6,
        label="Spoof / Replay",
        density=True
    )

    # Garis threshold
    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"EER Threshold = {threshold:.4f}"
    )

    plt.xlabel("Spoof Probability")
    plt.ylabel("Density")
    plt.title(
        "Distribusi Spoof Probability dengan Threshold EER"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    output = "distribution_spoof_probability_threshold.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"\nGrafik distribusi + threshold disimpan:")
    print(os.path.abspath(output))


def plot_scores(genuine_scores, spoof_scores, threshold):
    """
    Membuat grafik scatter setiap sampel
    untuk memperlihatkan posisi score terhadap threshold.
    """

    plt.figure(figsize=(12, 6))

    genuine_x = np.arange(len(genuine_scores))
    spoof_x = np.arange(len(spoof_scores))

    plt.scatter(
        genuine_x,
        genuine_scores,
        label="Genuine",
        s=60
    )

    plt.scatter(
        spoof_x,
        spoof_scores,
        label="Spoof / Replay",
        s=60
    )

    plt.axhline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.4f}"
    )

    plt.xlabel("Sample Index")
    plt.ylabel("Spoof Probability")
    plt.title("Spoof Probability Setiap Sampel")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    output = "scatter_spoof_probability_threshold.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"\nGrafik scatter disimpan:")
    print(os.path.abspath(output))


def calibrate_spoof(genuine_dir, spoof_dir):

    print("\n")
    print("=" * 70)
    print("ANTI-SPOOF CALIBRATION")
    print("=" * 70)

    # ==========================================================
    # 1. Ambil score genuine
    # ==========================================================

    genuine_scores = collect_scores(
        genuine_dir,
        "GENUINE"
    )

    # ==========================================================
    # 2. Ambil score spoof/replay
    # ==========================================================

    spoof_scores = collect_scores(
        spoof_dir,
        "SPOOF / REPLAY"
    )

    # ==========================================================
    # 3. Statistik
    # ==========================================================

    calculate_statistics(
        genuine_scores,
        "GENUINE"
    )

    calculate_statistics(
        spoof_scores,
        "SPOOF / REPLAY"
    )

    # ==========================================================
    # 4. Cari threshold EER
    # ==========================================================

    threshold, eer = find_eer_threshold(
        scores_genuine=genuine_scores.tolist(),
        scores_impostor=spoof_scores.tolist()
    )

    print("\n")
    print("=" * 70)
    print("HASIL KALIBRASI")
    print("=" * 70)

    print(f"Threshold : {threshold:.4f}")
    print(f"EER       : {eer * 100:.2f}%")

    # ==========================================================
    # 5. Buat grafik distribusi
    # ==========================================================

    plot_distribution(
        genuine_scores,
        spoof_scores
    )

    # ==========================================================
    # 6. Buat grafik distribusi + threshold
    # ==========================================================

    plot_distribution_with_threshold(
        genuine_scores,
        spoof_scores,
        threshold
    )

    # ==========================================================
    # 7. Buat scatter score + threshold
    # ==========================================================

    plot_scores(
        genuine_scores,
        spoof_scores,
        threshold
    )

    print("\n")
    print("=" * 70)
    print("SELESAI")
    print("=" * 70)

    return threshold


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 3:
        print("Usage:")
        print(
            "python -m voiceverification.services.calibrate_spoof "
            "<genuine_dir> <spoof_dir>"
        )
        sys.exit(1)

    genuine_dir = sys.argv[1]
    spoof_dir = sys.argv[2]

    calibrate_spoof(
        genuine_dir,
        spoof_dir
    )