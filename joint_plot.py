import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(style="darkgrid", font_scale=2)

label_pred_S1R = pd.read_excel(r"label_pred_S1R_EL_MSE.xlsx")


sns.jointplot(x=label_pred_S1R["Ensemble"], y=label_pred_S1R["label"], kind="reg", color="m")
plt.savefig(r"label_pred_ensemble.jpg", dpi=600)
#
# sns.jointplot(x=label_pred_S1R["Morgan_PseudoAAC"], y=label_pred_S1R["y_label"], kind="reg", color="m")
# plt.savefig(r"output_files/label_pred_Morgan_PseudoAAC.jpg", dpi=600)
#
# sns.jointplot(x=label_pred_S1R["Morgan_AAC"], y=label_pred_S1R["y_label"], kind="reg", color="m")
# plt.savefig(r"output_files/label_pred_Morgan_AAC.jpg", dpi=600)
#
# sns.jointplot(x=label_pred_S1R["Morgan_CNN"], y=label_pred_S1R["y_label"], kind="reg", color="m")
# plt.savefig(r"output_files/label_pred_Morgan_CNN.jpg", dpi=600)
#
# sns.jointplot(x=label_pred_S1R["Morgan_Conjoint_triad"], y=label_pred_S1R["y_label"], kind="reg", color="m")
# plt.savefig(r"output_files/label_pred_Morgan_Conjoint_triad.jpg", dpi=600)
#
# sns.jointplot(x=label_pred_S1R["Morgan_GRU"], y=label_pred_S1R["y_label"], kind="reg", color="m")
# plt.savefig(r"output_files/label_pred_Morgan_GRU.jpg", dpi=600)
#
# sns.jointplot(x=label_pred_S1R["Morgan_LSTM"], y=label_pred_S1R["y_label"], kind="reg", color="m")
# plt.savefig(r"output_files/label_pred_Morgan_LSTM.jpg", dpi=600)
#
# sns.jointplot(x=label_pred_S1R["Morgan_ESPF"], y=label_pred_S1R["y_label"], kind="reg", color="m")
# plt.savefig(r"output_files/label_pred_Morgan_ESPF.jpg", dpi=600)
#
# sns.jointplot(x=label_pred_S1R["average"], y=label_pred_S1R["y_label"], kind="reg", color="m")
# plt.savefig(r"output_files/label_pred_average.jpg", dpi=600)

print("ok")
