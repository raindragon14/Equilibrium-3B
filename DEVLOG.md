# Developer Log: Building Equilibrium-3B

Jurnal ini mendokumentasikan keputusan teknis, kegagalan, dan terobosan selama pengembangan SLM ini.

## Phase 1: Architecture Design (Weeks 1-2)
**Goal:** Menentukan arsitektur yang efisien untuk konteks panjang (Laporan Keuangan/Sejarah Ekonomi).

* **Eksperimen 1:** Full Transformer.
    * *Masalah:* VRAM meledak saat konteks > 32k token.
* **Eksperimen 2:** Pure Mamba-2.
    * *Masalah:* Kemampuan "In-Context Learning" (ICL) lemah untuk tugas *recall* jarum-dalam-jerami.
* **Keputusan Akhir:** Hybrid Jamba-Style. Saya menyisipkan 1 layer Transformer setiap 7 layer Mamba.
    * *Implementasi:* Menggunakan blok `MambaBlock` dari `mamba_ssm` dikombinasikan dengan `FlashAttention-3`.

## Phase 2: The Data Pipeline (Weeks 3-5)
**Goal:** Membuat data sintetis berkualitas "Buku Teks".

* **Tantangan Ekonomi:** Data web terlalu *noisy*.
* **Solusi:** Membangun `EconAgent` pipeline. Saya menggunakan LLM besar untuk mensimulasikan interaksi pasar (Supply/Demand shocks) dan merekam log-nya sebagai narasi pelatihan.
* **Verifikasi Matematika:** Menerapkan loop "Generator-Verifier". Kode Python dieksekusi untuk memvalidasi solusi matematika sebelum masuk ke dataset. Lihat `/data_pipeline/verify_math.py`.

## Phase 3: Training Stability & Optimizers (Weeks 6-8)
**Goal:** Konvergensi stabil pada skala 3 Miliar parameter.

* **Kegagalan AdamW:** Loss spike terjadi di pertengahan pelatihan. Penyetelan LR schedule sangat sulit.
* **Migrasi ke Muon:** Beralih ke optimizer Muon (Momentum Orthogonalized) untuk matriks 2D.
    * *Hasil:* Konvergensi 30% lebih cepat. Tidak perlu *warmup* yang rumit.
    * *Catatan:* Embedding layer tetap menggunakan SOAP optimizer agar stabil.

## Phase 4: Alignment & "System 2" Thinking (Weeks 9-10)
**Goal:** Memicu kemampuan penalaran (Chain-of-Thought).

* Mengimplementasikan **GRPO (Group Relative Policy Optimization)**.
* Model dipaksa menghasilkan tag `<think>` sebelum menjawab.
* *Observasi:* Model mulai melakukan "self-correction" pada derivasi kalkulus yang panjang.
