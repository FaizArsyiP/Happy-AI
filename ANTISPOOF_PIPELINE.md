# Pipeline Anti-Spoof di Proyek Ini

Dokumen ini menjelaskan seluruh alur anti-spoofing yang dipakai di backend voice verification, berdasarkan implementasi di `backend/voiceverification/core/asvspoof.py`, service kalibrasi, dan integrasinya di `BiometricService`.

## 1. Posisi Anti-Spoof di Pipeline Verifikasi

Saat user mengirim audio untuk verifikasi, backend menjalankan beberapa tahap utama:

1. Ekstraksi embedding speaker dari audio live.
2. Perbandingan embedding live dengan embedding enroll.
3. Perhitungan skor anti-spoof dari audio live.
4. Pengambilan keputusan akhir dengan aturan ambang di decision engine.
5. Jika lolos, sistem juga bisa menghitung perilaku suara seperti pitch dan speaking rate.

Artinya, anti-spoof bukan berdiri sendiri, tetapi menjadi salah satu sinyal risiko yang dipakai untuk menolak rekaman palsu, replay attack, atau audio sintetis.

## 2. File dan Modul yang Terlibat

- `backend/voiceverification/core/asvspoof.py` — inti ekstraksi fitur anti-spoof dan scoring.
- `backend/voiceverification/services/biometric_service.py` — memanggil `compute_score()` lalu meneruskan hasilnya ke decision engine.
- `backend/voiceverification/core/decision_engine.py` — memakai `replay_prob` sebagai sinyal anti-spoof dalam keputusan final.
- `backend/voiceverification/services/calibrate_spoof.py` — kalibrasi threshold menggunakan data genuine dan spoof.
- `backend/voiceverification/core/calibration.py` — mencari threshold Equal Error Rate (EER).
- `backend/voiceverification/services/train_asvspoof_lite_v2.py` — skrip training ringan untuk menghasilkan parameter model yang dipakai di `asvspoof.py`.

## 3. Input yang Diproses

Fungsi utama adalah `compute_score(input_data, sr=16000)`.

Input bisa berupa:

- path file audio (`str`), atau
- array audio hasil loading sendiri.

Jika input berupa path file, audio akan:

1. Diload dengan `librosa.load(..., sr=16000, mono=True)`.
2. Di-trim silence-nya menggunakan `librosa.effects.trim(y, top_db=25)`.
3. Jika audio kosong atau terlalu pendek, sistem mengembalikan array nol.

Jika maksimum amplitudo audio sangat kecil (`< 1e-6`), fungsi langsung mengembalikan skor `0.0` dan fitur kosong.

## 4. Fitur yang Diekstrak dari Suara User

Pipeline anti-spoof ini tidak memakai fitur speaker embedding seperti ECAPA. Yang diekstrak adalah fitur akustik sederhana dari sinyal audio mentah:

### 4.1 STFT Magnitude

Sinyal diubah ke domain frekuensi menggunakan Short-Time Fourier Transform:

$$
X(f, t) = |\mathrm{STFT}(y)| + 10^{-9}
$$

Parameter yang dipakai:

- `n_fft = 1024`
- `hop_length = 512`

Penambahan `1e-9` dipakai untuk menghindari nilai nol saat perhitungan berikutnya.

### 4.2 Spectral Flatness

Fitur ini mengukur apakah spektrum terdengar lebih seperti noise atau lebih seperti sinyal tonalan/terstruktur.

Secara praktis, library menghitung flatness dari spektrum STFT, lalu diambil rata-ratanya:

$$
\text{flat} = \mathrm{mean}(\mathrm{spectral\_flatness}(X))
$$

Interpretasi umum:

- flatness tinggi: spektrum lebih datar, sering mirip noise.
- flatness rendah: spektrum lebih terstruktur, sering lebih mirip suara manusia natural.

### 4.3 Temporal Energy Variance

Energi setiap frame dihitung dari rata-rata magnitudo STFT per frame:

$$
E_t = \mathrm{mean}(X_{:, t})
$$

Kemudian energi dinormalisasi:

$$
E_t \leftarrow \frac{E_t}{\max(E)}
$$

Jika maksimum energi nol, pembagian tidak dilakukan secara langsung; sistem memakai `1` sebagai pembagi aman.

Lalu dihitung varians temporal:

$$
\text{temp\_var} = \mathrm{var}(E)
$$

Makna fitur ini:

- varians tinggi: energi berubah-ubah antar frame.
- varians rendah: pola energi lebih datar / konstan.

### 4.4 High-Band Energy Ratio

Frekuensi FFT dihitung dengan:

$$
f = \mathrm{fft\_frequencies}(sr=16000, n\_fft=1024)
$$

Lalu spektrum dibagi menjadi dua band:

- voice band: `300 < f < 3400` Hz
- high band: `10000 < f < 16000` Hz

Rata-rata energi masing-masing band:

$$
V = \mathrm{mean}(X_{voice})
$$

$$
H = \mathrm{mean}(X_{high})
$$

Rasio high-band terhadap voice-band:

$$
\text{highband} = \frac{H}{V}
$$

Jika `V` terlalu kecil (`<= 1e-6`), rasio dibuat `0.0` agar tidak terjadi pembagian tidak stabil.

Makna fitur ini:

- jika energi tinggi di band atas terlalu besar dibanding band suara, sinyal bisa tampak tidak natural.
- ini sering berguna untuk mendeteksi replay, artefak kompresi, atau sintesis yang meninggalkan jejak spektral.

## 5. Vektor Fitur Final

Tiga fitur yang dipakai model adalah:

1. `flat` = spectral flatness
2. `temp_var` = temporal energy variance
3. `highband` = rasio energi high band terhadap voice band

Vektor fitur:

$$
\mathbf{x} = [\text{flat},\ \text{temp\_var},\ \text{highband}]
$$

## 6. Normalisasi dan Model Klasifikasi

Setelah fitur diekstrak, pipeline memakai standardisasi z-score per fitur:

$$
\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}
$$

Parameter model yang tertanam di `asvspoof.py` adalah:

- `MODEL_MEANS = [0.011605034616271345, 0.03527778138717016, 0.0]`
- `MODEL_SCALES = [0.010305995506340091, 0.014120233215618267, 1.0]`
- `MODEL_COEFFS = [0.6451673888241342, 0.08283186796791177, 0.0]`
- `MODEL_BIAS = 0.10828528294764354`

Model yang dipakai adalah bentuk logistic regression ringan.

Skor linear:

$$
s = \mathbf{w} \cdot \mathbf{z} + b
$$

Lalu dipetakan ke probabilitas dengan sigmoid:

$$
\sigma(s) = \frac{1}{1 + e^{-s}}
$$

Output inilah yang dikembalikan sebagai `score` dan juga diberi nama `ml_prob`.

## 7. Arti Skor Output

Fungsi `compute_score()` mengembalikan:

- `score` / `ml_prob` — probabilitas dari model anti-spoof.
- `flatness` — spectral flatness.
- `temporal_var` — varians energi antar frame.
- `highband` — rasio energi high-band terhadap voice-band.

Contoh print yang dihasilkan:

$$
\text{ASVspoof Score: } 0.9231 \text{ (flat: 0.0123, var: 0.0412, high: 0.0088)}
$$

Secara implementasi, nilai yang dipakai di service verifikasi adalah `spoof_prob = compute_score(live_wav)[0]`.

## 8. Kalibrasi Threshold

Script `calibrate_spoof.py` dipakai untuk mencari threshold yang cocok dari data genuine dan spoof.

Alurnya:

1. Audio genuine dibaca satu per satu.
2. Audio spoof dibaca satu per satu.
3. Masing-masing file dihitung skornya dengan `compute_score()`.
4. Semua skor genuine dan spoof dimasukkan ke `find_eer_threshold()`.

Di `find_eer_threshold()` dilakukan pencarian threshold pada semua nilai skor unik.

Untuk setiap threshold `t`:

$$
\text{FAR}(t) = \mathrm{mean}(\text{scores\_impostor} \ge t)
$$

$$
\text{FRR}(t) = \mathrm{mean}(\text{scores\_genuine} < t)
$$

Kemudian threshold EER dipilih saat selisih `FAR` dan `FRR` paling kecil.

Equal Error Rate dihitung sebagai:

$$
\text{EER} = \frac{\text{FAR}(t) + \text{FRR}(t)}{2}
$$

Output kalibrasi:

- `threshold` — threshold terbaik.
- `eer` — nilai EER.

## 9. Training Model Ringan

Skrip `train_asvspoof_lite_v2.py` menunjukkan bagaimana parameter model pada `asvspoof.py` dibuat.

Langkah training:

1. Load data dari folder `genuine`, `impostor`, dan `spoof`.
2. Ekstrak tiga fitur yang sama seperti inference.
3. Split data train/test dengan `train_test_split(test_size=0.3, stratify=y, random_state=42)`.
4. Train pipeline:
    - `StandardScaler()`
    - `LogisticRegression(max_iter=1000, class_weight="balanced")`
5. Evaluasi menggunakan confusion matrix dan classification report.
6. Cetak parameter scaler dan classifier untuk disalin ke `asvspoof.py`.

## 10. Integrasi Ke Keputusan Final

Di `BiometricService.verify_against_multiple_embeddings()`:

1. Embedding live diekstrak dari audio.
2. Setiap embedding enroll dibandingkan dengan cosine similarity.
3. Skor terbaik diambil sebagai `best_score`.
4. Anti-spoof dihitung:

$$
\text{spoof\_prob} = \text{compute\_score}(\text{live\_wav})
$$

5. Decision engine menerima dua sinyal utama:
    - `speaker_score = best_score`
    - `replay_prob = spoof_prob`

Aturan keputusan di `decision_engine.py` adalah:

- jika `speaker_score < abs_min_speaker = 0.35` → `DENIED`
- jika `replay_prob >= replay_deny = 0.75` → `DENIED`
- jika `replay_prob >= replay_warn = 0.60` → `REPEAT`
- jika `speaker_score >= voice_accept = 0.45` → `VERIFIED`
- jika `speaker_score >= voice_repeat = 0.30` → `REPEAT`
- selain itu → `DENIED`

Jadi, anti-spoof bisa langsung menggagalkan verifikasi walaupun speaker score cukup bagus.

## 11. Output yang Dikirim ke Frontend / API

Hasil verifikasi dari service membawa informasi berikut:

- `verified`
- `decision`
- `reason`
- `score`
- `spoof_prob`
- `best_index`
- `best_label`
- `all_scores`

Ini berarti frontend atau layer API bisa tahu bukan hanya apakah user lolos, tetapi juga seberapa kuat skor speaker dan seberapa mencurigakan sinyal anti-spoof-nya.

## 12. Fitur Perilaku yang Dihitung Setelah Lolos

Walau bukan bagian inti anti-spoof, pipeline verifikasi juga bisa menghitung:

- `pitch` dengan `librosa.yin(y, fmin=50, fmax=300, sr=16000)`
- `rate` sebagai durasi audio: `len(y) / sr`
- `behavior_score` dari z-score pitch dan rate terhadap profil pengguna

Artinya, setelah audio dinilai cukup aman, sistem juga bisa membangun profil perilaku untuk deteksi anomali lanjutan.

## 13. Catatan Tentang `replay_heuristic.py`

Ada modul terpisah bernama `core/replay_heuristic.py` yang menghitung skor replay berbasis:

- varians spectral centroid
- varians spectral rolloff
- varians amplitude modulation
- rasio frekuensi modulasi rendah terhadap tinggi

Skor akhirnya dinormalisasi ke rentang `0` sampai `1`.

Namun, pada jalur verifikasi utama yang aktif di `BiometricService`, anti-spoof yang dipakai adalah `compute_score()` dari `asvspoof.py`. Modul replay heuristic lebih terlihat sebagai eksperimen atau alat analisis tambahan.

## 14. Ringkasan Singkat

Intinya, pipeline anti-spoof di proyek ini bekerja seperti ini:

1. Audio user dibersihkan dan dinormalisasi ke 16 kHz mono.
2. Audio dipecah ke STFT.
3. Tiga ciri diekstrak: spectral flatness, temporal energy variance, dan high-band energy ratio.
4. Ciri distandardisasi dengan mean dan std model.
5. Logistic regression ringan menghasilkan probabilitas spoof.
6. Probabilitas ini dipakai bersama skor speaker untuk menentukan VERIFIED, REPEAT, atau DENIED.

Dengan kata lain, sistem ini menilai apakah suara tersebut **masuk akal sebagai suara manusia yang natural**, bukan hanya apakah cocok dengan identitas speaker.
