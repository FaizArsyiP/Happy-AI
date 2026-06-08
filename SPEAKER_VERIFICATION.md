# Speaker Verification (ECAPA‑TDNN) — Penjelasan Pipeline

Dokumen ini merangkum bagian speaker verification pada backend, berdasarkan implementasi di `backend/voiceverification/models/speaker_verifier.py`, pemakaian di `BiometricService`, dan aturan pengambilan keputusan.

## 1. Ringkasan Singkat

- Model utama: ECAPA‑TDNN melalui `speechbrain/spkrec-ecapa-voxceleb`.
- Output utama: embedding vektor ter‑normalisasi (L2) dan skor kesamaan (cosine similarity).
- Keputusan akhir digabung dengan sinyal anti‑spoof (`spoof_prob`) dan aturan threshold di `decision_engine.py`.

## 2. Ekstraksi Embedding

Fungsi utama: `SpeakerVerifier.extract_embedding(wav_path)`

Langkah:

1. Load audio lewat model helper: `waveform = self.model.load_audio(wav_path)`.
2. Pastikan batch dimensi: jika 1D → `unsqueeze(0)`.
3. Forward ke encoder: `emb = self.model.encode_batch(waveform)`.
4. Dikonversi ke NumPy, squeeze, lalu dinormalisasi L2:

$$
\mathbf{e} \leftarrow \frac{\mathbf{e}}{\|\mathbf{e}\|_2}
$$

Normalisasi ini membuat embedding sebanding dan memungkinkan perhitungan cosine similarity langsung sebagai dot product.

## 3. Perbandingan Embedding (Scoring)

Ada dua cara yang dipakai di kode:

- `compare_embeddings(emb1, emb2)` — menghitung cosine similarity secara langsung:

$$
\text{score} = \mathbf{e}_1 \cdot \mathbf{e}_2
$$

  Karena embedding sudah L2‑normal, dot product berkisar antara -1 dan 1; pada praktik ini nilainya diinterpretasikan di rentang 0..1 (implementasi model dan data biasanya menghasilkan nilai positif untuk kecocokan).

- `verify(live, enroll)` — memanggil helper `self.model.verify_files(live, enroll)` dari SpeechBrain yang mengembalikan skor probabilistik/compatibility (dibungkus menjadi float).

Di `BiometricService.verify_against_multiple_embeddings()`, alur yang dipakai adalah:

1. Ekstrak `live_emb` dari audio live.
2. Bandingkan `live_emb` dengan setiap embedding enroll (yang disimpan sebagai `embedding` di database/file).
3. Ambil skor tertinggi (`best_score`) dan label terkait sebagai best match.

## 4. Interpretasi Skor dan Threshold

Threshold dan aturan keputusan utama berada di `core/decision_engine.py`:

- `abs_min_speaker = 0.35` — ambang minimum mutlak; jika `speaker_score < abs_min_speaker` → `DENIED`.
- `voice_accept = 0.45` — skor di atas ini dianggap cukup untuk `VERIFIED` (jika tidak ada replay/spoof).
- `voice_repeat = 0.30` — rentang antara `voice_repeat` dan `voice_accept` → `REPEAT` (permintaan ulang).

Catatan tambahan untuk update profil tepercaya di `TrustedUpdatePolicy`:

- `min_speaker_score = 0.65` — untuk mengizinkan pembaruan profil perilaku, speaker score harus sangat tinggi.

Perlu diketahui: nilai threshold dapat disesuaikan berdasarkan hasil ROC/EER dataset. Ada skrip analisis ROC di `utils/roc_analysis.py` yang menghitung EER dan threshold untuk speaker-only ROC.

## 5. Kalibrasi & Evaluasi

Untuk menganalisis performa speaker verification:

- `utils/roc_analysis.py` mengumpulkan `speaker_scores` dan label dari folder `dataset/` (genuine, impostor, spoof) dan menghitung ROC.
- EER dihitung mencari threshold di mana `FPR` hampir sama dengan `1 - TPR`.

Rumus umum pada ROC/EER:

$$
\text{FPR}(t) = \mathrm{mean}(\text{scores\_neg} \ge t)
$$

$$
\text{TPR}(t) = \mathrm{mean}(\text{scores\_pos} \ge t)
$$

EER dipilih di threshold terdekat saat `FPR(t) \approx 1 - TPR(t)`.

## 6. Penyimpanan Enrollment & Integrasi

- Embedding enroll bisa disimpan sebagai `enroll_embedding.npy` atau di database (`speaker_profiles` pada Supabase) bersama `label` dan metadata.
- Di endpoint `verify-voice`, backend memuat semua embedding enroll untuk user, membandingkannya, lalu memilih yang terbaik.

## 7. Interaksi Dengan Anti‑Spoof & Behavior

- Setelah `best_score` diperoleh, service memanggil `compute_score(live_wav)` untuk mendapatkan `spoof_prob`.
- Keputusan akhir menggunakan `speaker_score` dan `replay_prob` (spoof_prob) bersama aturan di `decision_engine`.
- Jika `decision == VERIFIED`, sistem juga menghitung fitur perilaku (`pitch`, `rate`) dan membandingkannya dengan `BehaviorProfile` pengguna. Jika memenuhi kebijakan `TrustedUpdatePolicy`, profil perilaku bisa diperbarui otomatis.

## 8. Rumus Penting

- Cosine similarity (dengan embedding L2 norm):

$$
\text{cos\_sim}(\mathbf{e}_1, \mathbf{e}_2) = \mathbf{e}_1 \cdot \mathbf{e}_2
$$

- Normalisasi L2:

$$
\mathbf{e} \leftarrow \frac{\mathbf{e}}{\|\mathbf{e}\|_2}
$$

## 9. File dan Fungsi Kunci

- `backend/voiceverification/models/speaker_verifier.py` — `SpeakerVerifier` class (`extract_embedding`, `compare_embeddings`, `verify`).
- `backend/voiceverification/services/biometric_service.py` — integrasi end‑to‑end (extract → compare → antispoof → decide → behavior).
- `backend/voiceverification/core/decision_engine.py` — aturan keputusan dan threshold default.
- `backend/voiceverification/core/trusted_update.py` — kebijakan aman untuk pembaruan profil pengguna.
- `backend/voiceverification/utils/roc_analysis.py` — analisis ROC/EER untuk kalibrasi threshold.

## 10. Saran untuk Kalibrasi

- Jalankan `utils/roc_analysis.py` pada dataset valid (genuine/impostor) untuk mendapatkan EER dan threshold speaker yang sesuai.
- Pertimbangkan adaptasi threshold per pengguna (lihat fungsi `build_decision_config` yang dikomentari di `decision_engine.py`) saat tersedia cukup sampel.

---

File ini dibuat sebagai dokumentasi ringkas pipeline speaker verification di proyek; ingin saya tambahkan contoh perhitungan numerik atau visualisasi ROC? 