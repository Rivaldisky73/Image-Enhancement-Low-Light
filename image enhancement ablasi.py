# face_detection_ablation.py
# ============================================================
# Face Detection + Ablation Study (Custom Enhancement v2)
# Menampilkan: Custom (asli) lalu 6 varian ablasi
# (NoRetinex, NoAGC, NoCLAHE, NoFusion, NoDenoise, ...)
# ============================================================

import math
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from retinaface import RetinaFace
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# ------------------------------------------------------------
# fungsi logging kecil untuk menampilkan info (flush segera)
# ------------------------------------------------------------
def log_info(method_name):
    """Print info message about running method (flush segera)."""
    print(f"[INFO] Menjalankan metode: {method_name}", flush=True)


# ============================================================
# 1. RETINEX (MSRCR Adaptif)
# ============================================================
def MSRCR_adaptive(img, scales=[15, 80, 250]):
    """
    Multi-scale Retinex (adaptif) - memperbaiki pencahayaan.
    Input: BGR uint8 image.
    Output: BGR uint8 image (retinex result).
    """
    img = img.astype(np.float32) + 1.0
    log_img = np.log(img)
    retinex = np.zeros_like(img)
    for scale in scales:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=scale, sigmaY=scale)
        retinex += (log_img - np.log(blur + 1.0))
    retinex = retinex / len(scales)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(retinex)


# ============================================================
# 2. Adaptive Gamma Correction (AGC)
# ============================================================
def adaptive_gamma_correction(y_channel):
    """
    Koreksi gamma adaptif berdasarkan CDF pada channel Y.
    Input: single-channel uint8 (0..255)
    Output: uint8 channel
    """
    hist, _ = np.histogram(y_channel.flatten(), 256, [0, 256])
    cdf = hist.cumsum() / hist.sum()
    mean_intensity = int(y_channel.mean())
    gamma = 1.0 + (0.5 - cdf[mean_intensity])
    gamma = max(0.7, min(2.0, gamma))
    y_gamma = np.array(255 * (y_channel / 255) ** (1 / gamma), dtype='uint8')
    return y_gamma


# ============================================================
# 3. Gray World Balance
# ============================================================
def gray_world_balance(img):
    """
    Simple Gray-World white balance.
    Input/Output: BGR uint8
    """
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    kb = avg_gray / avg_b if avg_b != 0 else 1.0
    kg = avg_gray / avg_g if avg_g != 0 else 1.0
    kr = avg_gray / avg_r if avg_r != 0 else 1.0

    balanced = np.zeros_like(img, dtype=np.float32)
    balanced[:, :, 0] = img[:, :, 0] * kb
    balanced[:, :, 1] = img[:, :, 1] * kg
    balanced[:, :, 2] = img[:, :, 2] * kr
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    return balanced


# ============================================================
# 4. CUSTOM ENHANCEMENT (ASLI)
#    Urutan langkah:
#    1) Retinex -> 2) YCrCb split -> 3) AGC -> 4) CLAHE -> 5) Gray World
#    6) Fusion with original -> 7) Gaussian + Sharpen -> 8) Denoise
# ============================================================
def custom_enhancement_v2(img):
    """Full pipeline (original) - tidak diubah (include denoise)."""
    # 1) Retinex
    retinex_img = MSRCR_adaptive(img)

    # 2) Convert to YCrCb and split
    ycrcb = cv2.cvtColor(retinex_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # 3) Adaptive Gamma Correction (AGC)
    y_gamma = adaptive_gamma_correction(y)

    # 4) CLAHE pada channel Y
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y_gamma)

    # 5) Merge dan kembali ke BGR
    ycrcb_enhanced = cv2.merge((y_clahe, cr, cb))
    img_enhanced = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)

    # 6) Gray world balance
    img_balanced = gray_world_balance(img_enhanced)

    # 7) Fusion dengan citra asli untuk mempertahankan naturalitas
    fusion = cv2.addWeighted(img_balanced, 0.7, img, 0.3, 0)

    # 8) Gaussian blur + unsharp-like sharpening
    gaussian = cv2.GaussianBlur(fusion, (0, 0), 2)
    sharpened = cv2.addWeighted(fusion, 1.3, gaussian, -0.3, 0)

    # 9) Denoise akhir
    final = cv2.fastNlMeansDenoisingColored(sharpened, None, 7, 7, 7, 15)
    return final


# ============================================================
# 4b. CUSTOM - NO DENOISE (untuk Ablasi "NoDenoise")
#    Sama seperti original tetapi TIDAK melakukan langkah denoise akhir.
# ============================================================
def custom_no_denoise(img):
    """Full pipeline but skip final denoising step (return sharpened image)."""
    # 1) Retinex
    retinex_img = MSRCR_adaptive(img)

    # 2) Convert to YCrCb and split
    ycrcb = cv2.cvtColor(retinex_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # 3) Adaptive Gamma Correction (AGC)
    y_gamma = adaptive_gamma_correction(y)

    # 4) CLAHE pada channel Y
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y_gamma)

    # 5) Merge dan kembali ke BGR
    ycrcb_enhanced = cv2.merge((y_clahe, cr, cb))
    img_enhanced = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)

    # 6) Gray world balance
    img_balanced = gray_world_balance(img_enhanced)

    # 7) Fusion dengan citra asli untuk mempertahankan naturalitas
    fusion = cv2.addWeighted(img_balanced, 0.7, img, 0.3, 0)

    # 8) Gaussian blur + unsharp-like sharpening
    gaussian = cv2.GaussianBlur(fusion, (0, 0), 2)
    sharpened = cv2.addWeighted(fusion, 1.3, gaussian, -0.3, 0)

    # NOTE: skip denoising intentionally
    final = sharpened
    return final


# ============================================================
# 5. VARIAN ABLASI
#    - custom_no_retinex: hapus Retinex (mulai dari input langsung)
#    - custom_no_agc: hapus adaptive_gamma_correction (langsung ke CLAHE dari Y)
#    - custom_no_clahe: hapus CLAHE (gunakan y_gamma apa adanya)
#    - custom_no_grayworld: hapus gray_world_balance (opsional)
#    - custom_no_fusion: hapus fusion (lanjutkan dari img_balanced langsung)
#    - custom_no_denoise: hapus denoise akhir (baru ditambahkan)
# ============================================================

def custom_no_retinex(img):
    """
    Ablasi 1: Tanpa Retinex.
    Langkah lain tetap sama; Retinex dilewatkan (gunakan img asli untuk Y channel).
    """
    # langsung konversi dari citra input
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # AGC
    y_gamma = adaptive_gamma_correction(y)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y_gamma)

    ycrcb_enhanced = cv2.merge((y_clahe, cr, cb))
    img_enhanced = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)

    # Gray world
    img_balanced = gray_world_balance(img_enhanced)

    # Fusion + sharpening + denoise
    fusion = cv2.addWeighted(img_balanced, 0.7, img, 0.3, 0)
    gaussian = cv2.GaussianBlur(fusion, (0, 0), 2)
    sharpened = cv2.addWeighted(fusion, 1.3, gaussian, -0.3, 0)
    final = cv2.fastNlMeansDenoisingColored(sharpened, None, 7, 7, 7, 15)
    return final


def custom_no_agc(img):
    """
    Ablasi 2: Tanpa Adaptive Gamma Correction (AGC).
    Gunakan Retinex, tapi lewati langkah AGC; langsung CLAHE dari Y.
    """
    # Retinex tetap dipakai
    retinex_img = MSRCR_adaptive(img)

    # Convert & split
    ycrcb = cv2.cvtColor(retinex_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # <-- AGC dilewatkan: y_gamma = adaptive_gamma_correction(y) tidak dipanggil
    # langsung apply CLAHE ke y
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)

    ycrcb_enhanced = cv2.merge((y_clahe, cr, cb))
    img_enhanced = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)

    # Gray world
    img_balanced = gray_world_balance(img_enhanced)

    # Fusion + sharpening + denoise
    fusion = cv2.addWeighted(img_balanced, 0.7, img, 0.3, 0)
    gaussian = cv2.GaussianBlur(fusion, (0, 0), 2)
    sharpened = cv2.addWeighted(fusion, 1.3, gaussian, -0.3, 0)
    final = cv2.fastNlMeansDenoisingColored(sharpened, None, 7, 7, 7, 15)
    return final


def custom_no_clahe(img):
    """
    Ablasi 3: Tanpa CLAHE.
    Gunakan Retinex dan AGC, tapi lewati CLAHE (gunakan y_gamma langsung).
    """
    retinex_img = MSRCR_adaptive(img)

    # Convert & split
    ycrcb = cv2.cvtColor(retinex_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # AGC tetap
    y_gamma = adaptive_gamma_correction(y)

    # <-- CLAHE dilewatkan: gunakan y_gamma langsung
    y_clahe = y_gamma

    ycrcb_enhanced = cv2.merge((y_clahe, cr, cb))
    img_enhanced = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)

    # Gray world
    img_balanced = gray_world_balance(img_enhanced)

    # Fusion + sharpening + denoise
    fusion = cv2.addWeighted(img_balanced, 0.7, img, 0.3, 0)
    gaussian = cv2.GaussianBlur(fusion, (0, 0), 2)
    sharpened = cv2.addWeighted(fusion, 1.3, gaussian, -0.3, 0)
    final = cv2.fastNlMeansDenoisingColored(sharpened, None, 7, 7, 7, 15)
    return final


def custom_no_grayworld(img):
    """
    Ablasi 4: Tanpa Gray World Balance.
    Semua langkah sampai ycrcb -> img_enhanced tetap, tapi skip gray_world_balance.
    (fungsi disediakan jika nanti ingin mengaktifkan)
    """
    retinex_img = MSRCR_adaptive(img)

    # Convert & split
    ycrcb = cv2.cvtColor(retinex_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # AGC + CLAHE
    y_gamma = adaptive_gamma_correction(y)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y_gamma)

    ycrcb_enhanced = cv2.merge((y_clahe, cr, cb))
    img_enhanced = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)

    # <-- Gray world dilewatkan: img_balanced = img_enhanced
    img_balanced = img_enhanced

    # Fusion + sharpening + denoise
    fusion = cv2.addWeighted(img_balanced, 0.7, img, 0.3, 0)
    gaussian = cv2.GaussianBlur(fusion, (0, 0), 2)
    sharpened = cv2.addWeighted(fusion, 1.3, gaussian, -0.3, 0)
    final = cv2.fastNlMeansDenoisingColored(sharpened, None, 7, 7, 7, 15)
    return final


def custom_no_fusion(img):
    """
    Ablasi 5: Tanpa Fusion (gabungan dengan citra asli).
    Semua langkah sampai img_balanced tetap, tapi lanjut sharpening dari img_balanced
    (tidak menambahkan citra asli ke hasil).
    """
    retinex_img = MSRCR_adaptive(img)

    # Convert & split
    ycrcb = cv2.cvtColor(retinex_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # AGC + CLAHE
    y_gamma = adaptive_gamma_correction(y)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y_gamma)

    ycrcb_enhanced = cv2.merge((y_clahe, cr, cb))
    img_enhanced = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)

    # Gray world
    img_balanced = gray_world_balance(img_enhanced)

    # <-- Fusion dilewatkan: langsung sharpening pada img_balanced
    gaussian = cv2.GaussianBlur(img_balanced, (0, 0), 2)
    sharpened = cv2.addWeighted(img_balanced, 1.3, gaussian, -0.3, 0)
    final = cv2.fastNlMeansDenoisingColored(sharpened, None, 7, 7, 7, 15)
    return final


# custom_no_denoise already defined above (skips final denoise)


# ============================================================
# 6. FACE DETECTION (RetinaFace wrapper)
# ============================================================
def face_detection_retina(img):
    """
    Menggunakan RetinaFace (mengharapkan input RGB di dalam library),
    tapi RetinaFace.detect_faces menerima RGB arrays, jadi kita konversi dari BGR.
    Mengembalikan list bounding boxes (x,y,w,h) dan list confidence scores.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = RetinaFace.detect_faces(rgb)

    faces, confs = [], []
    if isinstance(results, dict):
        for key in results.keys():
            x1, y1, x2, y2 = results[key]['facial_area']
            score = results[key].get('score', 0.0)
            faces.append((x1, y1, x2 - x1, y2 - y1))
            confs.append(score)
    return faces, confs


# ============================================================
# 7. GUI & Utility: NIQE & BRISQUE (manual/simple)
# ============================================================
class FaceDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Detection Ablation Study")

        self.file_list = []
        self.current_index = -1

        # Buttons dasar
        tk.Button(master, text="Load Image", command=self.load_image).pack(
            side=tk.LEFT, padx=10, pady=10
        )
        tk.Button(master, text="Next Image", command=self.next_image).pack(
            side=tk.LEFT, padx=10, pady=10
        )
        tk.Button(master, text="Exit", command=master.quit).pack(
            side=tk.LEFT, padx=10, pady=10
        )

    # File dialog: load awal
    def load_image(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png")]
        filename = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        if filename:
            self.file_list = [filename]
            self.current_index = 0
            self.run_detection(filename)

    # Tambah gambar (next)
    def next_image(self):
        if not self.file_list:
            messagebox.showinfo("Info", "Please load an image first.")
            return
        filename = filedialog.askopenfilename(
            title="Select Next Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if filename:
            self.file_list.append(filename)
            self.current_index += 1
            self.run_detection(filename)

    # ------------------ NIQE & BRISQUE -----------------
    def normalize_mscn(self, img):
        mu = cv2.GaussianBlur(img, (7, 7), 1.166)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(img * img, (7, 7), 1.166)
        sigma = np.sqrt(np.abs(sigma - mu_sq))
        sigma[sigma < 1e-6] = 1e-6
        return (img - mu) / sigma

    def gamma_array(self, x):
        return np.array([math.gamma(float(val)) for val in x])

    def estimate_ggd_param(self, mscn):
        gam = np.arange(0.2, 10, 0.001)
        g1 = self.gamma_array(1.0 / gam)
        g2 = self.gamma_array(2.0 / gam)
        g3 = self.gamma_array(3.0 / gam)
        r_gam = (g1 * g3) / (g2 ** 2 + 1e-12)

        sigma_sq = np.mean(mscn ** 2)
        E = np.mean(np.abs(mscn))
        rho = sigma_sq / (E ** 2 + 1e-12)

        alpha = gam[np.argmin(np.abs(rho - r_gam))]
        beta = np.sqrt(sigma_sq)
        return alpha, beta

    def estimate_aggd_param(self, pair):
        neg_mask = pair < 0
        pos_mask = pair > 0

        left_sq_mean = np.mean(pair[neg_mask] ** 2) if np.any(neg_mask) else 0.0
        right_sq_mean = np.mean(pair[pos_mask] ** 2) if np.any(pos_mask) else 0.0
        left_std = np.sqrt(left_sq_mean) if left_sq_mean > 0 else 0.0
        right_std = np.sqrt(right_sq_mean) if right_sq_mean > 0 else 0.0

        gamma_hat = left_std / (right_std + 1e-12) if right_std != 0 else 1.0
        rhat = (np.mean(np.abs(pair)) ** 2) / (np.mean(pair ** 2) + 1e-12)

        gam = np.arange(0.2, 10, 0.001)
        g1 = self.gamma_array(1.0 / gam)
        g2 = self.gamma_array(2.0 / gam)
        g3 = self.gamma_array(3.0 / gam)
        r_gam = (g2 ** 2) / (g1 * g3 + 1e-12)

        term = r_gam * ((gamma_hat ** 3 + 1) * (gamma_hat + 1)) / ((gamma_hat ** 2 + 1) ** 2 + 1e-12)
        alpha = gam[np.argmin(np.abs(term - rhat))]
        beta = np.sqrt(np.mean(pair ** 2))
        return alpha, beta, left_std, right_std

    def extract_brisque_features(self, img):
        mscn = self.normalize_mscn(img)
        feats = []

        alpha_mscn, beta_mscn = self.estimate_ggd_param(mscn)
        feats.extend([alpha_mscn, beta_mscn])

        shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for shift in shifts:
            pair = mscn * np.roll(mscn, shift, axis=(0, 1))
            alpha, beta, ls, rs = self.estimate_aggd_param(pair)
            feats.extend([alpha, (ls + rs) / 2.0, ls, rs])

        return np.array(feats)

    def brisque_manual(self, img):
        if img is None:
            return 0.0
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        img_f = img_gray.astype(np.float32) / 255.0
        feats = self.extract_brisque_features(img_f)
        return float(np.mean(feats) * 10.0)

    def niqe_manual(self, img):
        if img is None:
            return 0.0
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        img_f = img_gray.astype(np.float32) / 255.0
        mscn = self.normalize_mscn(img_f)
        mean, std = np.mean(mscn), np.std(mscn) + 1e-12
        skew = np.mean(((mscn - mean) / std) ** 3)
        kurt = np.mean(((mscn - mean) / std) ** 4)
        return float(abs(skew) * 10 + abs(kurt - 3) * 5)

    # ------------------ RUN DETECTION & VISUAL -----------------
    def run_detection(self, filename):
        """
        Main pipeline per gambar:
         - Buat versi enhancement: Custom (asli) + 6 Ablasi
         - Jalankan RetinaFace per versi
         - Hitung NIQE & BRISQUE per versi
         - Visualisasikan semua (gambar + grafik + tabel)
        """
        img = cv2.imread(filename)
        if img is None:
            messagebox.showerror("Error", f"File {filename} not found!")
            return

        # optional ground-truth jumlah wajah (user input)
        user_total = simpledialog.askstring("Ground Truth", "Masukkan jumlah wajah ground-truth (opsional):")
        manual_total = int(user_total) if user_total and user_total.isdigit() else None

        # --- Buat semua versi: Custom asli diikuti 6 ablasi
        methods = {
            "Custom": custom_enhancement_v2(img),
            "Ablasi-NoRetinex": custom_no_retinex(img),
            "Ablasi-NoAGC": custom_no_agc(img),
            "Ablasi-NoCLAHE": custom_no_clahe(img),
            # "Ablasi-NoGrayWorld": custom_no_grayworld(img), # jika ingin aktifkan kembali, uncomment
            "Ablasi-NoFusion": custom_no_fusion(img),
            "Ablasi-NoDenoise": custom_no_denoise(img),
        }

        # Containers untuk hasil evaluasi
        counts, confs, times, imgs = {}, {}, {}, {}
        quality_scores = {}

        # Loop evaluasi tiap metode
        for name, im in methods.items():
            log_info(name)
            start = time.time()
            faces, conf_list = face_detection_retina(im.copy())
            duration = time.time() - start

            counts[name] = len(faces)
            confs[name] = np.mean(conf_list) if conf_list else 0.0
            times[name] = duration
            imgs[name] = im.copy()

            # gambar bounding box pada salinan
            for (x, y, w, h) in faces:
                cv2.rectangle(imgs[name], (x, y), (x + w, y + h), (0, 255, 0), 2)

            # quality metrics
            try:
                niqe_score = self.niqe_manual(im)
                brisque_score = self.brisque_manual(im)
            except Exception:
                niqe_score, brisque_score = 0.0, 0.0
            quality_scores[name] = (niqe_score, brisque_score)

        # tentukan total ground-truth (manual jika ada, else estimasi max)
        estimated_total = max(counts.values()) if counts else 0
        total_used = manual_total if manual_total else estimated_total

        def percent_str(count, total):
            return "0%" if total == 0 else f"{int(round((count / total) * 100))}%"

        # ---------------- VISUALISASI semua metode dalam 1 figure ----------------
        num_methods = len(imgs)
        base_width, base_height = 3.2, 2.8
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(base_width * num_methods, base_height * 2)
        gs = gridspec.GridSpec(2, num_methods, height_ratios=[1.6, 1], figure=fig)

        plt.suptitle(f"Face Detection Ablation Study\n{filename}", fontsize=18, fontweight="bold", y=0.98)

        # Baris atas: gambar + title (NIQE & BRISQUE, tiga angka di belakang koma)
        for i, (name, im) in enumerate(imgs.items()):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            niqe_s, brisque_s = quality_scores.get(name, (0.0, 0.0))
            ax.set_title(
                f"{name}\nDetected: {counts[name]}/{total_used} ({percent_str(counts[name], total_used)})\nNIQE:{niqe_s:.3f} BRISQUE:{brisque_s:.3f}",
                fontsize=10,
            )
            ax.axis("off")

        # Baris bawah kiri: Deteksi (%) per method (horizontal bar)
        # split area: gunakan separasi setengah
        left_cols = max(1, num_methods // 2)
        ax1 = fig.add_subplot(gs[1, :left_cols])
        labels_plot = list(methods.keys())
        values_pct = [int(round((counts[n] / total_used) * 100)) if total_used > 0 else 0 for n in labels_plot]

        default_colors = ["gray", "royalblue", "orange", "limegreen", "purple", "brown", "cyan"]
        colors_pct = default_colors[: len(labels_plot)]
        bars = ax1.barh(labels_plot, values_pct, edgecolor="black", color=colors_pct[: len(labels_plot)])
        ax1.set_xlim(0, 100)
        ax1.set_xlabel("Deteksi (%)")
        ax1.set_title("Deteksi (%) per Method")
        for bar, val in zip(bars, values_pct):
            ax1.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val}%", va="center", fontsize=9, fontweight="bold")

        # Baris bawah kanan: Confidence per method
        ax2 = fig.add_subplot(gs[1, left_cols :])
        conf_vals = [confs[n] for n in labels_plot]
        bars2 = ax2.bar(labels_plot, conf_vals, color=colors_pct[: len(labels_plot)])
        ax2.set_ylabel("Mean Confidence")
        ax2.set_title("Confidence per Method")
        for bar, val in zip(bars2, conf_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.001, f"{val:.4f}", ha="center", va="bottom", fontsize=8)

        if conf_vals:
            ax2.set_ylim(max(0, min(conf_vals) - 0.01), max(conf_vals) + 0.01)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # ----------------- POPUP TABEL (text report) -----------------
        report_lines = []
        report_lines.append("============================================")
        report_lines.append(f"File: {filename}")
        report_lines.append(f"Ground Truth Faces : {total_used}")
        report_lines.append("============================================")
        report_lines.append(f"{'Method':<20} | {'Faces':<5} | {'%':<6} | {'Confidence':<10} | {'Time(s)':<8} | {'NIQE':<7} | {'BRISQUE':<8}")
        report_lines.append("-" * 110)

        for name in methods.keys():
            niqe_v, brisque_v = quality_scores.get(name, (0.0, 0.0))
            report_lines.append(f"{name:<20} | {counts[name]:<5} | {percent_str(counts[name], total_used):<6} | {confs[name]:<10.4f} | {times[name]:<8.2f} | {niqe_v:<7.3f} | {brisque_v:<8.3f}")

        report_lines.append("============================================")
        report_text = "\n".join(report_lines)
        print(report_text)
        messagebox.showinfo("Detection Report", report_text)

        # ----------------- POPUP GRAFIK NIQE & BRISQUE -----------------
        fig2, ax = plt.subplots(figsize=(8, 5))
        index = np.arange(len(methods))
        bar_width = 0.35

        niqe_vals = [quality_scores[n][0] for n in labels_plot]
        brisque_vals = [quality_scores[n][1] for n in labels_plot]

        bars1 = ax.bar(index, niqe_vals, bar_width, label="NIQE", edgecolor="black")
        bars2 = ax.bar(index + bar_width, brisque_vals, bar_width, label="BRISQUE", edgecolor="black")

        ax.set_xlabel("Enhancement Method", fontsize=12)
        ax.set_ylabel("Quality Score", fontsize=12)
        ax.set_title("Comparison of NIQE & BRISQUE Scores", fontsize=14, fontweight="bold")
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(labels_plot)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        for i in range(len(labels_plot)):
            ax.text(i, niqe_vals[i] + 0.5, f"{niqe_vals[i]:.3f}", ha="center", va="bottom", fontsize=9)
            ax.text(i + bar_width, brisque_vals[i] + 0.5, f"{brisque_vals[i]:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
