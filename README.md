#  Bidirectional LSTM — Time Series Forecasting

**Part 6 of Deep Learning Series** — Memprediksi suhu minimum harian menggunakan Bidirectional LSTM (BiLSTM) dengan PyTorch.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Selesai-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

##  Tentang Proyek

Proyek ini adalah kelanjutan dari seri pembelajaran *Recurrent Neural Network*, kali ini membahas **Bidirectional LSTM (BiLSTM)** — varian LSTM yang memproses sequence dari dua arah sekaligus (maju & mundur) untuk menangkap konteks temporal yang lebih kaya.

Studi kasus yang digunakan adalah **forecasting deret waktu (time series)**: memprediksi suhu minimum harian berdasarkan pola historis 10 tahun data cuaca.

>  **Kenapa Bidirectional?**
> LSTM biasa hanya "membaca" data dari masa lalu ke masa depan. BiLSTM menambahkan satu layer LSTM lagi yang membaca dari arah sebaliknya, lalu menggabungkan kedua representasi tersebut — sehingga model punya konteks yang lebih utuh saat memahami pola dalam sequence.

---

##  Dataset

| Detail | Keterangan |
|---|---|
| **Nama** | Daily Minimum Temperatures |
| **Fitur** | `Temp` (suhu minimum harian, °C) |
| **Total data** | 3.650 baris (10 tahun observasi harian) |
| **Split** | 80% train (2.920) / 20% test (730) — *tanpa shuffle*, karena data time series |
| **Sequence length** | 14 hari (2 minggu) sebagai window input |

Split dilakukan **tanpa pengacakan (`shuffle=False`)** agar urutan waktu tetap terjaga — prinsip penting dalam time series forecasting.

---

##  Arsitektur Model

```python
class BiLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True,
            bidirectional=True   #  kunci utama BiLSTM
        )
        # Output dikali 2 karena ada 2 arah (forward + backward)
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x)
        return x, hidden
```

**Poin penting arsitektur:**
- `bidirectional=True` membuat LSTM memproses sequence dari dua arah.
- Karena outputnya digabung dari kedua arah, ukuran input `fc` (fully connected) menjadi **2× hidden_size**.

###  Konfigurasi Model

| Parameter | Nilai | Keterangan |
|---|---|---|
| `hidden_size` | 64 | Kapasitas memori model |
| `num_layers` | 2 | Kedalaman layer LSTM |
| `dropout` | 0 | Regularisasi (tidak dipakai di run ini) |
| `seq_len` | 14 | Panjang window input |
| `batch_size` | 32 | — |
| `optimizer` | AdamW (lr=0.0005) | — |
| `loss function` | MSE Loss | Cocok untuk regresi nilai kontinu |
| `early stopping` | patience-based | Otomatis berhenti saat model berhenti membaik |

---

##  Hasil Training

Model dilatih dengan **early stopping** dan berhenti secara otomatis di:

| Metrik | Nilai |
|---|---|
| **Epoch berhenti** | 178 |
| **Best Test Cost (MSE)** | **0.3059** |
| **Train Cost awal → akhir** | 135.96 → ~0.55 |
| **Test Cost awal → akhir** | 137.06 → 0.3059 |

Penurunan loss dari ratusan menjadi di bawah 1.0 menunjukkan model berhasil belajar pola musiman dan tren suhu harian dengan baik, tanpa mengalami overfitting parah berkat mekanisme early stopping.

---

##  Eksperimen Forecasting Multi-Step

Bagian paling menarik dari proyek ini: menguji seberapa jauh model bisa "meramal" ke depan.

| Eksperimen | Context (`n_prior`) | Prediksi ke depan (`n_forecast`) | Observasi |
|---|---|---|---|
| Percobaan 1 | 10 hari | 140 hari | Akurat di awal, makin meleset seiring horizon prediksi menjauh |
| Percobaan 2 | 30 hari | 120 hari | Konteks lebih panjang → prediksi lebih stabil dan mendekati pola asli |

---

##  Insight & Pembelajaran

Beberapa catatan reflektif dari eksplorasi notebook ini:

1. **Sequence length yang tepat itu penting.** Dengan `seq_len=14`, model punya cukup informasi historis untuk mengenali pola — intinya, machine learning hanya mencari pola; tugas kita adalah memastikan fitur yang masuk cukup bermakna.
2. **Semakin jauh horizon prediksi, semakin besar ketidakpastian.** Ini hal yang wajar dalam forecasting — bukan berarti model "gagal", tapi memang sifat alami dari time series di masa depan.
3. **Menambah konteks (`n_prior`) membantu.** Menaikkan context dari 10 ke 30 hari terbukti membuat prediksi jangka panjang menjadi lebih masuk akal.
4. **Overfitting adalah "default" dalam time series forecasting.** Jangan berkecil hati saat prediksi meleset — pengelolaan ekspektasi terhadap ketidakpastian adalah bagian dari proses.

---

##  Tech Stack

- **Python 3.12**
- **PyTorch** — arsitektur dan training loop
- **jcopdl** — `Callback`, `set_config`, `TimeSeriesDataset` untuk mempercepat eksperimen
- **Pandas & NumPy** — manipulasi data
- **Matplotlib** — visualisasi hasil prediksi & forecasting
- **scikit-learn** — `train_test_split`
- **tqdm** — progress bar training

---

##  Cara Menjalankan

```bash
# 1. Clone repository
git clone <repo-url>
cd <repo-folder>

# 2. Install dependencies
pip install torch pandas numpy matplotlib scikit-learn jcopdl tqdm

# 3. Jalankan notebook
jupyter notebook "Part_6_-_Bidirectional.ipynb"
```

>  Sesuaikan path dataset (`daily_min_temp.csv`) dengan lokasi file di komputer kamu.

---

##  Struktur Proyek

```
├── Part_6_-_Bidirectional.ipynb   # Notebook utama
├── model/
│   └── bilstm/                     # Checkpoint model terbaik (auto-saved)
├── utils.py                        # Helper: data4pred, pred4pred
└── data/
    └── daily_min_temp.csv          # Dataset suhu harian
```

---

##  Bagian dari Seri

Notebook ini adalah **Part 6** dari seri pembelajaran Deep Learning — melanjutkan pembahasan sebelumnya tentang RNN dan LSTM standar, kali ini dengan fokus pada arsitektur bidirectional untuk kasus forecasting deret waktu.

---

##  Catatan

Proyek ini dibuat sebagai bagian dari eksplorasi pembelajaran deep learning, dengan fokus pada pemahaman konsep bidirectional recurrent networks dan tantangan nyata dalam time series forecasting (ketidakpastian jangka panjang, pemilihan context length, dan trade-off overfitting).

---

