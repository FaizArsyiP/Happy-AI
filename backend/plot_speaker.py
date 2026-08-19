import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from voiceverification.models.speaker_verifier import SpeakerVerifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "voiceverification/dataset")
ENROLL_EMBEDDING = os.path.join(DATASET_DIR, "enroll_embedding.npy")
GENUINE_DIR = os.path.join(DATASET_DIR, "genuine")
IMPOSTOR_DIR = os.path.join(DATASET_DIR, "impostor")

OUT_DIST = os.path.join(DATASET_DIR, "distribution_speaker_similarity.png")
OUT_THRESH = os.path.join(DATASET_DIR, "distribution_speaker_similarity_threshold.png")
OUT_SCATTER = os.path.join(DATASET_DIR, "scatter_speaker_similarity_threshold.png")


def audio_files(folder):
    files = []
    for ext in ("*.wav", "*.WAV", "*.flac", "*.FLAC", "*.mp3", "*.MP3"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def load_embedding(path):
    emb = np.asarray(np.load(path)).reshape(-1)
    n = np.linalg.norm(emb)
    if n == 0:
        raise ValueError(f"Embedding norm 0: {path}")
    return emb / n


def get_scores(speaker, enroll_emb, files, label):
    scores = []
    for path in files:
        emb = speaker.extract_embedding(path)
        score = speaker.compare_embeddings(enroll_emb, emb)
        scores.append(score)
        print(f"[{label:8s}] {os.path.basename(path):35s} {score:.4f}")
    return np.asarray(scores, dtype=float)


def eer_threshold(genuine, impostor):
    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    s = np.r_[genuine, impostor]
    fpr, tpr, thresholds = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    return float(thresholds[i]), float((fpr[i] + fnr[i]) / 2.0)


def stats(name, x):
    print(f"\n{'-' * 65}\nSTATISTIK {name.upper()}\n{'-' * 65}")
    print(f"Jumlah sampel : {len(x)}")
    print(f"Minimum       : {np.min(x):.4f}")
    print(f"Maksimum      : {np.max(x):.4f}")
    print(f"Rata-rata     : {np.mean(x):.4f}")
    print(f"Median        : {np.median(x):.4f}")
    print(f"Std. Deviasi  : {np.std(x):.4f}")


def main():
    if not os.path.exists(ENROLL_EMBEDDING):
        raise FileNotFoundError(ENROLL_EMBEDDING)
    if not os.path.isdir(GENUINE_DIR):
        raise FileNotFoundError(GENUINE_DIR)
    if not os.path.isdir(IMPOSTOR_DIR):
        raise FileNotFoundError(IMPOSTOR_DIR)

    enroll = load_embedding(ENROLL_EMBEDDING)
    genuine_files = audio_files(GENUINE_DIR)
    impostor_files = audio_files(IMPOSTOR_DIR)

    print("=" * 65)
    print("DISTRIBUSI SPEAKER SIMILARITY - ECAPA-TDNN")
    print("=" * 65)
    print(f"Enrollment embedding : {ENROLL_EMBEDDING}")
    print(f"Dimension            : {len(enroll)}")
    print(f"Genuine              : {len(genuine_files)}")
    print(f"Impostor             : {len(impostor_files)}")

    speaker = SpeakerVerifier(device="cpu")

    genuine = get_scores(speaker, enroll, genuine_files, "Genuine")
    impostor = get_scores(speaker, enroll, impostor_files, "Impostor")

    stats("Genuine", genuine)
    stats("Impostor", impostor)

    threshold, eer = eer_threshold(genuine, impostor)
    print("\n" + "=" * 65)
    print("HASIL KALIBRASI")
    print("=" * 65)
    print(f"Threshold EER : {threshold:.4f}")
    print(f"EER           : {eer * 100:.2f}%")

    all_scores = np.r_[genuine, impostor]
    margin = max((all_scores.max() - all_scores.min()) * 0.05, 0.01)
    bins = np.linspace(all_scores.min() - margin, all_scores.max() + margin, 12)

    # 1. Distribusi
    plt.figure(figsize=(12, 7))
    plt.hist(genuine, bins=bins, density=True, alpha=0.6, label="Genuine")
    plt.hist(impostor, bins=bins, density=True, alpha=0.6, label="Impostor")
    plt.xlabel("Speaker Similarity")
    plt.ylabel("Density")
    plt.title("Distribusi Speaker Similarity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIST, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Distribusi + threshold
    plt.figure(figsize=(12, 7))
    plt.hist(genuine, bins=bins, density=True, alpha=0.6, label="Genuine")
    plt.hist(impostor, bins=bins, density=True, alpha=0.6, label="Impostor")
    plt.axvline(threshold, linestyle="--", linewidth=2,
                label=f"EER Threshold = {threshold:.4f}")
    plt.xlabel("Speaker Similarity")
    plt.ylabel("Density")
    plt.title("Distribusi Speaker Similarity dengan Threshold EER")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_THRESH, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Scatter
    plt.figure(figsize=(12, 7))
    plt.scatter(np.arange(len(genuine)), genuine, s=70, label="Genuine")
    plt.scatter(np.arange(len(impostor)), impostor, s=70, label="Impostor")
    plt.axhline(threshold, linestyle="--", linewidth=2,
                label=f"Threshold = {threshold:.4f}")
    plt.xlabel("Sample Index")
    plt.ylabel("Speaker Similarity")
    plt.title("Speaker Similarity Setiap Sampel")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_SCATTER, dpi=300, bbox_inches="tight")
    plt.close()

    np.save(os.path.join(DATASET_DIR, "genuine_speaker_similarity.npy"), genuine)
    np.save(os.path.join(DATASET_DIR, "impostor_speaker_similarity.npy"), impostor)

    print("\nOutput:")
    print(OUT_DIST)
    print(OUT_THRESH)
    print(OUT_SCATTER)


if __name__ == "__main__":
    main()