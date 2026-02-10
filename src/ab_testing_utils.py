import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests

# ==========================================
# 1. FUNGSI ANALISIS MENTAH (GENERATE SUMMARY)
# ==========================================
def generate_text_summary(df, variant_col='variant', control_variant=None, alpha=0.05):
    """
    Menghasilkan SUMMARY TABLE berbasis TEKS (Console Friendly).
    PLUS: Mengembalikan dictionary berisi hasil analisis (P-Value, Lift, dll).
    """
    
    # --- 1. SETUP & DETEKSI VARIANT ---
    try:
        var_col_idx = df.columns.get_loc(variant_col)
        metric_cols = df.columns[var_col_idx + 1:]
    except KeyError:
        print("❌ Error: Kolom variant tidak ditemukan.")
        return {} 

    unique_variants = sorted(df[variant_col].unique())
    num_variants = len(unique_variants)
    
    if control_variant is None:
        control_variant = unique_variants[0]
    elif control_variant not in unique_variants:
        print(f"❌ Error: Control '{control_variant}' tidak ada di data.")
        return {}

    # --- 2. HEADER TABEL ---
    print("\n" + "="*105)
    print(f"📊 EXPERIMENT SUMMARY REPORT")
    print(f"   • Variants: {num_variants} ({', '.join(unique_variants)})")
    print(f"   • Control : {control_variant}")
    print(f"   • Alpha   : {alpha}")
    print("="*105)
    
    header = f"| {'METRIC NAME':<25} | {'TYPE':<12} | {'TEST USED':<25} | {'LIFT (%)':<10} | {'P-VALUE':<10} | {'SIG?':<5} |"
    print(header)
    print("|" + "-"*27 + "|" + "-"*14 + "|" + "-"*27 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*7 + "|")

    # KERANJANG PENYIMPANAN
    stored_results = {}

    # --- 3. LOOPING METRIK & ANALISIS ---
    for metric in metric_cols:
        clean_data = df[[variant_col, metric]].dropna()
        groups = [clean_data[clean_data[variant_col] == v][metric] for v in unique_variants]
        ctrl_data = clean_data[clean_data[variant_col] == control_variant][metric]
        unique_vals = clean_data[metric].unique()
        
        # Variabel Default
        metric_type = "Unknown"
        test_name = "Unknown"
        p_val = 1.0
        
        # A. BINARY
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            metric_type = "Binary"
            successes = [g.sum() for g in groups]
            nobs = [g.count() for g in groups]
            
            if num_variants == 2:
                test_name = "Z-Test (Prop)"
                _, p_val = proportions_ztest(successes, nobs)
            else:
                test_name = "Chi-Square"
                obs = [[s, n-s] for s, n in zip(successes, nobs)]
                _, p_val, _, _ = stats.chi2_contingency(obs)

        # B. COUNT & CONTINUOUS
        else:
            is_count = np.all(clean_data[metric] % 1 == 0) and (clean_data[metric].min() >= 0)
            
            if is_count:
                metric_type = "Count"
                mean_v = ctrl_data.mean()
                var_v = ctrl_data.var()
                ratio = var_v / mean_v if mean_v > 0 else 0
                test_mode = "Stable" if ratio <= 1.5 else "Skewed"
            else:
                metric_type = "Continuous"
                if len(ctrl_data) > 5000:
                    stat_jb, p_jb = stats.jarque_bera(ctrl_data)
                    test_mode = "Normal" if p_jb > 0.05 else "Skewed"
                elif len(ctrl_data) >= 3:
                    stat_sh, p_shapiro = stats.shapiro(ctrl_data)
                    test_mode = "Normal" if p_shapiro > 0.05 else "Skewed"
                else:
                    test_mode = "SmallSample"

            if num_variants == 2:
                if test_mode in ["Stable", "Normal"]:
                    test_name = "T-Test (Welch)" if metric_type == "Continuous" else "Poisson Test"
                    _, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                else:
                    test_name = "Mann-Whitney U"
                    _, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            else:
                if test_mode in ["Stable", "Normal"]:
                    test_name = "ANOVA"
                    _, p_val = stats.f_oneway(*groups)
                else:
                    test_name = "Kruskal-Wallis"
                    _, p_val = stats.kruskal(*groups)

        # Hitung Lift
        mean_ctrl = ctrl_data.mean()
        other_means = [clean_data[clean_data[variant_col] == v][metric].mean() for v in unique_variants if v != control_variant]
        best_trt_mean = max(other_means) if other_means else 0
        
        lift_val = 0.0
        lift_str = "N/A"
        if mean_ctrl != 0:
            lift_val = (best_trt_mean - mean_ctrl) / mean_ctrl 
            lift_str = f"{lift_val*100:+.2f}%"

        # Simpan ke Dictionary
        stored_results[metric] = {
            "p_value": p_val,
            "lift": lift_val,
            "significant": p_val < alpha
        }

        # Print Row
        sig_str = "YES" if p_val < alpha else "NO"
        row = f"| {metric:<25} | {metric_type:<12} | {test_name:<25} | {lift_str:<10} | {p_val:<10.5f} | {sig_str:<5} |"
        print(row)

    # --- 4. FOOTER ---
    print("|" + "-"*103 + "|")
    
    # !!! INI YANG TADI HILANG !!!
    return stored_results


# ==========================================
# 2. FUNGSI KOREKSI (HOLM-BONFERRONI)
# ==========================================
def apply_holm_correction(stored_results, alpha=0.05):
    """
    Fungsi Tahap 2: Menerima dictionary hasil 'generate_text_summary',
    lalu melakukan Multiple Testing Correction (Holm-Bonferroni).
    """
    
    # Cek Safety kalau input None
    if stored_results is None:
        print("❌ Error: Input data kosong (None). Pastikan fungsi sebelumnya me-return data.")
        return

    # 1. AMBIL DATA
    metric_names = list(stored_results.keys())
    raw_p_values = [stored_results[m]['p_value'] for m in metric_names]
    
    if not raw_p_values:
        print("⚠️  Tidak ada data P-Value untuk dikoreksi.")
        return

    # 2. HITUNG KOREKSI
    reject, p_adjusted, _, _ = multipletests(raw_p_values, alpha=alpha, method='holm')
    
    # 3. TAMPILKAN HASIL
    print(f"\n{'='*80}")
    print(f"⚖️  FINAL VERDICT: HOLM-BONFERRONI CORRECTION")
    print(f"   (Mengoreksi {len(metric_names)} metrik sekaligus)")
    print(f"{'='*80}")
    
    header = f"| {'METRIC NAME':<25} | {'RAW P-VAL':<10} | {'ADJ P-VAL':<10} | {'CONCLUSION':<10} |"
    print(header)
    print("|" + "-"*27 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|")
    
    for i, metric in enumerate(metric_names):
        raw_p = raw_p_values[i]
        adj_p = p_adjusted[i]
        is_sig = reject[i] # True jika Adj P < Alpha
        
        # Logika visual
        if is_sig:
            sig_str = "VALID"
        else:
            if raw_p < alpha:
                sig_str = "FALSE*" # Kena koreksi
            else:
                sig_str = "FAIL"
        
        print(f"| {metric:<25} | {raw_p:<10.5f} | {adj_p:<10.5f} | {sig_str:<10} |")
        
    print("|" + "-"*67 + "|")
    print("   *FAIL: Signifikan secara mentah, tapi False Positive setelah koreksi.")
    print("\n")