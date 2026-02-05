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

$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}$$

Where $O_i = \text{observed count}$ and $E_i = \text{expected count}$ for group $i$.

- **Standardized Mean Difference (SMD)**  

$$\text{SMD} = \frac{|\bar{X}_{\text{treatment}} - \bar{X}_{\text{control}}|}{\sqrt{\frac{s_{\text{treatment}}^2 + s_{\text{control}}^2}{2}}}$$

- **Temporal Stability Check (Coefficient of Variation)**  

$$CV = \frac{\text{std(daily counts)}}{\text{mean(daily counts)}}$$

If $CV > 0.2$: Unstable allocation $\rightarrow$ investigate system.

- **Bonferroni Correction**  

$$p^{\text{corr}} = \min(p \cdot m, 1)$$

- **Holm–Bonferroni Correction**  

Order p-values $p_{(1)} \leq p_{(2)} \leq \dots \leq p_{(m)}$.  
Compare each $p_{(i)}$ with:

$$\frac{\alpha}{m - i + 1}$$

- **Benjamini–Hochberg False Discovery Rate (FDR)**  

Corrected p-value:

$$p^{\text{corr}}_{(i)} = \min\left(\frac{m}{i} \cdot p_{(i)}, 1\right)$$
