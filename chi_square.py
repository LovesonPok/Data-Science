import pandas as pd
from scipy.stats import chi2_contingency

# Load your cleaned dataset
df = pd.read_csv("/Users/lovesonpokhrel/Documents/Data Science/cleaned_dataset.csv")

# Cross-tab habit × risk
ct = pd.crosstab(df["habit"], df["risk"], dropna=False)
print("Contingency Table:")
print(ct)

# Only try chi-square if table has >1 nonzero column
if ct.shape[1] > 1:
    chi2, p, dof, expected = chi2_contingency(ct)
    print("\nChi-Square Test Results:")
    print(f"Chi² statistic : {chi2:.4f}")
    print(f"Degrees of freedom : {dof}")
    print(f"P-value : {p:.6f}")
else:
    print("\nChi-square test not possible: only one risk category present.")
