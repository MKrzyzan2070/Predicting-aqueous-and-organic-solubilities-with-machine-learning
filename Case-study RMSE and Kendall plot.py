import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

df = pd.read_csv("Case-study metrics.csv")

data = np.vstack([df["Kendall_Distance"], df["RMSE"]])
bw_method = 0.40
kde = gaussian_kde(data, bw_method=bw_method)

xmin, xmax = 0, 0.7
ymin, ymax = 0, 1.4
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])

f = np.reshape(kde(positions).T, xx.shape)

f_norm = (f - f.min()) / (f.max() - f.min())

colors = ["#FFFFFF", "#daf0ff", "#b5e2ff", "#8fd3fe", "#6ac5fe", "#45b6fe"]
cmap = LinearSegmentedColormap.from_list("whatever", colors, N=6)

fig, ax_main = plt.subplots(figsize=(7, 7))
fig.subplots_adjust(left=0.15, top=0.85, right=0.85, bottom=0.15)

cf = ax_main.contourf(xx, yy, f_norm, levels=np.linspace(0, 1, 6), cmap=cmap)

ax_main.set_xlabel("Kendall's Tau", fontsize=18)
ax_main.set_ylabel('RMSE (log(x))', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
ax_main.set_xlim(xmin, xmax)
ax_main.set_ylim(ymin, ymax)

highlight_list = ["FPYJFEHAWHCUMM-UHFFFAOYSA-N", "JYGFTBXVXVMTGB-UHFFFAOYSA-N"]
mask = df['InChIKey'].isin(highlight_list)
ax_main.scatter(df["Kendall_Distance"], df["RMSE"], s=35, color="black")
ax_main.scatter(df.loc[mask, "Kendall_Distance"], df.loc[mask, "RMSE"], s=100, color="red")


ticks_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
ticks_y = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

ax_main.set_xticks(ticks_x)
ax_main.set_yticks(ticks_y)

ax_histx = fig.add_axes([ax_main.get_position().x0, ax_main.get_position().y1 + 0.005,
                         ax_main.get_position().width, 0.13])
kde_x = gaussian_kde(df["Kendall_Distance"], bw_method=bw_method)
x_range = np.linspace(xmin, xmax, 400)
ax_histx.fill_between(x_range, 0, kde_x(x_range), color='#8fd3fe')
ax_histx.set_yticks([])
ax_histx.set_xticks([])
for spine in ax_histx.spines.values():
    spine.set_visible(False)

ax_histy = fig.add_axes([ax_main.get_position().x1 + 0.005, ax_main.get_position().y0-0.02, 0.13,
                         ax_main.get_position().height])
kde_y = gaussian_kde(df["RMSE"], bw_method=bw_method)
y_range = np.linspace(ymin-0.02, ymax, 400)
ax_histy.fill_betweenx(y_range, 0, kde_y(y_range), color='#8fd3fe')
ax_histy.set_xticks([])
ax_histy.set_yticks([])
for spine in ax_histy.spines.values():
    spine.set_visible(False)

plt.savefig("RMSE vs Kendall.png")