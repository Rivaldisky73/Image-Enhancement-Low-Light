# ============================================================
# IMPORT LIBRARY
# ============================================================
import cv2                           # OpenCV
import numpy as np                   # Operasi numerik
import matplotlib.pyplot as plt      # Visualisasi
import time                          # Menghitung waktu proses
from retinaface import RetinaFace    # Deteksi wajah
import tkinter as tk                 # GUI
from tkinter import filedialog, simpledialog, messagebox
import matplotlib.gridspec as gridspec
import math

# ------------------------------------------------------------
# fungsi logging kecil untuk menampilkan info
# ------------------------------------------------------------
def log_info(method_name):
    """Print info message about running method (flush segera)."""
    print(f"[INFO] Menjalankan metode: {method_name}", flush=True)

# ============================================================
# 1. RETINEX (MSRCR Adaptif)
# ============================================================
def MSRCR_adaptive(img, scales=[15, 80, 250]):
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
# 2. Adaptive Gamma Correction
# ============================================================
def adaptive_gamma_correction(y_channel):
    hist, _ = np.histogram(y_channel.flatten(), 256, [0, 256])
    cdf = hist.cumsum() / hist.sum()
    mean_intensity = int(y_channel.mean())
    gamma = 1.0 + (0.5 - cdf[mean_intensity])
    gamma = max(0.7, min(2.0, gamma))
    y_gamma = np.array(255 * (y_channel / 255) ** (1 / gamma), dtype='uint8')
    return y_gamma


# ============================================================
# 3. CUSTOM ENHANCEMENT
# ============================================================
def custom_enhancement_v2(img):
    retinex_img = MSRCR_adaptive(img)
    ycrcb = cv2.cvtColor(retinex_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_gamma = adaptive_gamma_correction(y)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y_gamma)
    ycrcb_enhanced = cv2.merge((y_clahe, cr, cb))
    img_enhanced = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)
    fusion = cv2.addWeighted(img_enhanced, 0.7, img, 0.3, 0)
    gaussian = cv2.GaussianBlur(fusion, (0, 0), 2)
    sharpened = cv2.addWeighted(fusion, 1.3, gaussian, -0.3, 0)
    final = cv2.fastNlMeansDenoisingColored(sharpened, None, 7, 7, 7, 15)
    return final

# ============================================================
# 4. LIME Enhancement
# ============================================================
def box_filter(img, r):
    ksize = (2*r+1, 2*r+1)
    return cv2.boxFilter(img, ddepth=-1, ksize=ksize, normalize=True)

def guided_filter(I, p, r=15, eps=1e-3):
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    mean_Ip = box_filter(I*p, r)
    cov_Ip = mean_Ip - mean_I*mean_p
    mean_II = box_filter(I*I, r)
    var_I = mean_II - mean_I*mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a*mean_I
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)
    q = mean_a*I + mean_b
    return q

def lime_enhancement(img, r=15, eps=1e-3, gamma=0.9):
    img_f = img.astype(np.float32)/255.0
    T_init = np.max(img_f, axis=2)
    T_ref = guided_filter(T_init, T_init, r, eps)
    T_ref = np.clip(T_ref, eps, 1.0)
    R = img_f / T_ref[:,:,None]
    R = np.power(np.clip(R,0.0,1.0), gamma)
    enhanced = (R*255.0).astype(np.uint8)
    return enhanced

# ============================================================
# 5. FACE DETECTION
# ============================================================
def face_detection_retina(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = RetinaFace.detect_faces(rgb)
    faces, confs = [], []
    if isinstance(results, dict):
        for key in results.keys():
            x1, y1, x2, y2 = results[key]['facial_area']
            score = results[key]['score']
            faces.append((x1, y1, x2-x1, y2-y1))
            confs.append(score)
    return faces, confs

# ============================================================
# 6. GUI & Main App
# ============================================================
class FaceDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Detection Comparison")
        self.file_list = []
        self.current_index = -1
        tk.Button(master, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(master, text="Next Image", command=self.next_image).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(master, text="Exit", command=master.quit).pack(side=tk.LEFT, padx=10, pady=10)

    # --------------------------------------------------------
    def load_image(self):
        filetypes = [("Image files","*.jpg *.jpeg *.png")]
        filename = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        if filename:
            self.file_list = [filename]
            self.current_index = 0
            self.run_detection(filename)

    def next_image(self):
        if not self.file_list:
            messagebox.showinfo("Info","Please load an image first.")
            return
        filename = filedialog.askopenfilename(title="Select Next Image",filetypes=[("Image files","*.jpg *.jpeg *.png")])
        if filename:
            self.file_list.append(filename)
            self.current_index += 1
            self.run_detection(filename)

    # ------------------ NIQE & BRISQUE -----------------
    def normalize_mscn(self,img):
        mu = cv2.GaussianBlur(img,(7,7),1.166)
        mu_sq = mu*mu
        sigma = cv2.GaussianBlur(img*img,(7,7),1.166)
        sigma = np.sqrt(np.abs(sigma-mu_sq))
        sigma[sigma<1e-6] = 1e-6
        return (img - mu)/sigma

    def gamma_array(self,x):
        return np.array([math.gamma(float(val)) for val in x])

    def estimate_ggd_param(self, mscn):
        gam = np.arange(0.2,10,0.001)
        g1 = self.gamma_array(1.0/gam)
        g2 = self.gamma_array(2.0/gam)
        g3 = self.gamma_array(3.0/gam)
        r_gam = (g1*g3)/(g2**2 + 1e-12)
        sigma_sq = np.mean(mscn**2)
        E = np.mean(np.abs(mscn))
        rho = sigma_sq/(E**2+1e-12)
        alpha = gam[np.argmin(np.abs(rho - r_gam))]
        beta = np.sqrt(sigma_sq)
        return alpha, beta

    def estimate_aggd_param(self, pair):
        neg_mask = pair<0
        pos_mask = pair>0
        left_sq_mean = np.mean(pair[neg_mask]**2) if np.any(neg_mask) else 0.0
        right_sq_mean = np.mean(pair[pos_mask]**2) if np.any(pos_mask) else 0.0
        left_std = np.sqrt(left_sq_mean) if left_sq_mean>0 else 0.0
        right_std = np.sqrt(right_sq_mean) if right_sq_mean>0 else 0.0
        gamma_hat = left_std/(right_std+1e-12) if right_std!=0 else 1.0
        rhat = (np.mean(np.abs(pair))**2)/(np.mean(pair**2)+1e-12)
        gam = np.arange(0.2,10,0.001)
        g1 = self.gamma_array(1.0/gam)
        g2 = self.gamma_array(2.0/gam)
        g3 = self.gamma_array(3.0/gam)
        r_gam = (g2**2)/(g1*g3 +1e-12)
        term = r_gam*((gamma_hat**3+1)*(gamma_hat+1))/((gamma_hat**2+1)**2 + 1e-12)
        alpha = gam[np.argmin(np.abs(term - rhat))]
        beta = np.sqrt(np.mean(pair**2))
        return alpha, beta, left_std, right_std

    def extract_brisque_features(self, img):
        mscn = self.normalize_mscn(img)
        feats = []
        alpha_mscn, beta_mscn = self.estimate_ggd_param(mscn)
        feats.extend([alpha_mscn,beta_mscn])
        shifts = [(0,1),(1,0),(1,1),(1,-1)]
        for shift in shifts:
            pair = mscn*np.roll(mscn,shift,axis=(0,1))
            alpha,beta,ls,rs = self.estimate_aggd_param(pair)
            feats.extend([alpha,(ls+rs)/2.0,ls,rs])
        return np.array(feats)

    def brisque_manual(self,img):
        if img is None: return 0.0
        if len(img.shape)==3: img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else: img_gray = img
        img_f = img_gray.astype(np.float32)/255.0
        feats = self.extract_brisque_features(img_f)
        return float(np.mean(feats)*10.0)

    def niqe_manual(self,img):
        if img is None: return 0.0
        if len(img.shape)==3: img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else: img_gray = img
        img_f = img_gray.astype(np.float32)/255.0
        mscn = self.normalize_mscn(img_f)
        mean,std = np.mean(mscn), np.std(mscn)+1e-12
        skew = np.mean(((mscn-mean)/std)**3)
        kurt = np.mean(((mscn-mean)/std)**4)
        return float(abs(skew)*10 + abs(kurt-3)*5)

    # ------------------ RUN DETECTION & VISUAL -----------------
    def run_detection(self, filename):
        img = cv2.imread(filename)
        if img is None:
            messagebox.showerror("Error", f"File {filename} not found!")
            return

        user_total = simpledialog.askstring("Ground Truth", "Masukkan jumlah wajah ground-truth (opsional):")
        manual_total = int(user_total) if user_total and user_total.isdigit() else None

        # --- Enhancement
        enhanced_clahe = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(enhanced_clahe)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        y_clahe = clahe.apply(y)
        enhanced_clahe = cv2.cvtColor(cv2.merge((y_clahe,cr,cb)), cv2.COLOR_YCrCb2BGR)

        enhanced_custom = custom_enhancement_v2(img)

        enhanced_he = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_he, cr_he, cb_he = cv2.split(enhanced_he)
        y_he = cv2.equalizeHist(y_he)
        enhanced_he = cv2.cvtColor(cv2.merge((y_he,cr_he,cb_he)), cv2.COLOR_YCrCb2BGR)

        enhanced_lime = lime_enhancement(img,r=15,eps=1e-3,gamma=0.9)

        methods = {
            "Raw": img,
            "CLAHE": enhanced_clahe,
            "HE": enhanced_he,
            "LIME": enhanced_lime,
            "Custom": enhanced_custom
        }

        counts, confs, times, imgs = {}, {}, {}, {}
        quality_scores = {}

        for name, im in methods.items():
            # panggil fungsi logging yang baru ditambahkan
            log_info(name)

            start = time.time()
            faces, conf_list = face_detection_retina(im.copy())
            duration = time.time()-start
            counts[name]=len(faces)
            confs[name]=np.mean(conf_list) if conf_list else 0.0
            times[name]=duration
            imgs[name]=im.copy()
            for (x,y,w,h) in faces:
                cv2.rectangle(imgs[name],(x,y),(x+w,y+h),(0,255,0),2)
            try:
                niqe_score = self.niqe_manual(im)
                brisque_score = self.brisque_manual(im)
            except:
                niqe_score, brisque_score = 0.0,0.0
            quality_scores[name]=(niqe_score,brisque_score)

        estimated_total = max(counts.values()) if counts else 0
        total_used = manual_total if manual_total else estimated_total
        def percent_str(count,total):
            return "0%" if total==0 else f"{int(round((count/total)*100))}%"

        # ---------------- VISUALISASI ----------------
        num_methods = len(imgs)
        base_width, base_height = 3.2,2.8
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(base_width*num_methods, base_height*2)
        gs = gridspec.GridSpec(2,num_methods,height_ratios=[1.6,1], figure=fig)
        plt.suptitle(f"Face Detection Comparison\n{filename}", fontsize=18, fontweight="bold", y=0.98)
        for i,(name,im) in enumerate(imgs.items()):
            ax=fig.add_subplot(gs[0,i])
            ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            niqe_s, brisque_s = quality_scores.get(name,(0.0,0.0))
            ax.set_title(f"{name}\nDetected: {counts[name]}/{total_used} ({percent_str(counts[name],total_used)})\nNIQE:{niqe_s:.2f} BRISQUE:{brisque_s:.2f}", fontsize=11)
            ax.axis("off")
        ax1 = fig.add_subplot(gs[1,:max(1,num_methods//2)])
        labels_plot=list(methods.keys())
        values_pct=[int(round((counts[n]/total_used)*100)) if total_used>0 else 0 for n in labels_plot]
        colors_pct=["gray","royalblue","orange","limegreen","purple"]
        bars=ax1.barh(labels_plot,values_pct,edgecolor="black",color=colors_pct[:len(labels_plot)])
        ax1.set_xlim(0,100)
        ax1.set_xlabel("Deteksi (%)")
        ax1.set_title("Deteksi (%) per Method")
        for bar,val in zip(bars,values_pct):
            ax1.text(val+1,bar.get_y()+bar.get_height()/2,f"{val}%",va='center',fontsize=10,fontweight="bold")
        ax2=fig.add_subplot(gs[1,max(1,num_methods//2):])
        conf_vals=[confs[n] for n in labels_plot]
        bars2=ax2.bar(labels_plot,conf_vals,color=colors_pct[:len(labels_plot)])
        ax2.set_ylabel("Mean Confidence")
        ax2.set_title("Confidence per Method")
        for bar,val in zip(bars2,conf_vals):
            ax2.text(bar.get_x()+bar.get_width()/2,val+0.001,f"{val:.4f}",ha='center',va='bottom',fontsize=9,fontweight="bold")
        if conf_vals:
            ax2.set_ylim(max(0,min(conf_vals)-0.01), max(conf_vals)+0.01)
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show()

        # ----------------- POPUP TABEL -----------------
        report_lines=[]
        report_lines.append("============================================")
        report_lines.append(f"File: {filename}")
        report_lines.append(f"Ground Truth Faces : {total_used}")
        report_lines.append("============================================")
        report_lines.append(f"{'Method':<10} | {'Faces':<5} | {'%':<6} | {'Confidence':<10} | {'Time(s)':<8} | {'NIQE':<7} | {'BRISQUE':<8}")
        report_lines.append("-"*95)
        for name in methods.keys():
            niqe_v, brisque_v = quality_scores.get(name,(0.0,0.0))
            report_lines.append(f"{name:<10} | {counts[name]:<5} | {percent_str(counts[name],total_used):<6} | {confs[name]:<10.4f} | {times[name]:<8.2f} | {niqe_v:<7.3f} | {brisque_v:<8.3f}")
        report_lines.append("============================================")
        report_text="\n".join(report_lines)
        print(report_text)
        messagebox.showinfo("Detection Report",report_text)

        # ----------------- POPUP GRAFIK NIQE & BRISQUE -----------------
        fig2, ax = plt.subplots(figsize=(7,5))
        index = np.arange(len(methods))
        bar_width = 0.35
        niqe_vals=[quality_scores[n][0] for n in labels_plot]
        brisque_vals=[quality_scores[n][1] for n in labels_plot]
        bars1=ax.bar(index,niqe_vals,bar_width,label='NIQE',color='skyblue',edgecolor='black')
        bars2=ax.bar(index+bar_width,brisque_vals,bar_width,label='BRISQUE',color='salmon',edgecolor='black')
        ax.set_xlabel('Enhancement Method',fontsize=12)
        ax.set_ylabel('Quality Score',fontsize=12)
        ax.set_title('Comparison of NIQE & BRISQUE Scores',fontsize=14,fontweight='bold')
        ax.set_xticks(index + bar_width/2)
        ax.set_xticklabels(labels_plot)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for i in range(len(labels_plot)):
            ax.text(i,niqe_vals[i]+0.5,f"{niqe_vals[i]:.2f}",ha='center',va='bottom',fontsize=9)
            ax.text(i+bar_width,brisque_vals[i]+0.5,f"{brisque_vals[i]:.2f}",ha='center',va='bottom',fontsize=9)
        plt.tight_layout()
        plt.show()

# ============================================================
# MAIN
# ============================================================
if __name__=="__main__":
    root=tk.Tk()
    app=FaceDetectionApp(root)
    root.mainloop()
