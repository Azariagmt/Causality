import seaborn as sns
import matplotlib.pyplot as plt
def plot_cm(df_cm):
    # df_cm = pd.DataFrame(array, index=["stage 1", "stage 2", "stage 3", "stagte 4"], columns=["stage 1", "stage 2", "stage 3", "stagte 4"])
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(cm_2,annot=True,fmt="d")

    return plt.subplots()