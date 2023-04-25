import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('results/chexpert_val_auc_focal_31_3_2023_17_16_19.csv')
df2 = pd.read_csv('results/chexpert_val_auc_wce_28_3_2023_2_27_58.csv')
plt.plot(df1['Mean'], label='focal')
plt.plot(df2['Mean'], label='wce')
plt.title('Val AUC')
plt.legend()
plt.savefig('img9.png')