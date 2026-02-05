# Project Analysis With Master Hilmi

### Business Understanding

At this stage, the main focus is to understand the business problem being addressed.
The dataset comes from an e-commerce platform that aims to evaluate whether the changes they implemented actually produce a positive impact on performance.

To answer this question, the company conducted five experiments, which are provided as five separate files (test_1 to test_5).
Each file represents a different experiment that will be analyzed to assess the effectiveness of the applied changes.

### Data Understanding

The next step is understanding the structure and characteristics of the data.

Key observations:

- No duplicate cleaning is required, as each session_id and user_id combination is unique, with no identical records found.

- Null values are retained, since they are still valid and can be included in the analysis without compromising data integrity.

### Data Validation

The following step focuses on data validation, ensuring that the data aligns with expected variation and distribution.  
Key statistical formulas and checks:

- **Sample Ratio Mismatch (SRM)**  
  Used to check whether the allocation of samples matches the expected proportions.

\[
\chi^2 = \sum\_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
\]

where \(O_i\) = observed count, \(E_i\) = expected count, \(k\) = number of groups.  
 If the chi‑square test p‑value < α, SRM is detected.

- **Standardized Mean Difference (SMD)**  
  Measures the difference in means between groups relative to variation.

\[
SMD = \frac{\bar{x}\_1 - \bar{x}\_2}{s_p}
\]

with

\[
s_p = \sqrt{\frac{s_1^2 + s_2^2}{2}}
\]

Values of |SMD| > 0.1 are often considered imbalanced.

- **Temporal Stability Check (Coefficient of Variation)**  
  Evaluates stability of data distribution over time.

\[
CV = \frac{\sigma}{\mu}
\]

where \(\sigma\) = standard deviation, \(\mu\) = mean.  
 If CV > 0.2 → considered **unstable allocation**.

- **Bonferroni Correction**  
  Adjusts p‑values for multiple hypothesis testing.

\[
p^\text{corr} = \min(p \cdot m, 1)
\]

where \(m\) = number of tests.

- **Holm–Bonferroni Correction**  
  Step‑down method: order p‑values \(p*{(1)} \leq p*{(2)} \leq \dots \leq p*{(m)}\).  
  Compare each \(p*{(i)}\) with \(\frac{\alpha}{m - i + 1}\).  
  Reject H₀ sequentially until a test fails.

- **Benjamini–Hochberg False Discovery Rate (FDR)**  
  Controls the expected proportion of false discoveries.  
  Ordered p‑values: \(p*{(1)} \leq p*{(2)} \leq \dots \leq p\_{(m)}\).  
  Corrected p‑value:

\[
p^\text{corr}_{(i)} = \min\left(\frac{m}{i} \cdot p_{(i)}, 1\right)
\]

Significance determined by comparing with FDR threshold.
