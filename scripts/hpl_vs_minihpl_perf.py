import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

mini_path = Path("m2_nbscan_perf.csv")
hpl_path  = Path("hpl_standard_nbscan_perf.csv")

mini = pd.read_csv(mini_path)
hpl  = pd.read_csv(hpl_path)


mini_std = mini.rename(columns={"total_s": "mini_time_s", "GFLOPS": "mini_gflops"}).copy()
hpl_std  = hpl.rename(columns={"Time_s": "hpl_time_s", "GFLOPS": "hpl_gflops"}).copy()
hpl_std["Grid"] = hpl_std["P"].astype(str) + "x" + hpl_std["Q"].astype(str)


mini_cols = ["NB", "N", "Grid", "ranks", "mini_time_s", "mini_gflops"]
hpl_cols  = ["NB", "N", "Grid", "hpl_time_s", "hpl_gflops"]
mini_std = mini_std[mini_cols]
hpl_std  = hpl_std[hpl_cols]


compare = pd.merge(
    mini_std,
    hpl_std,
    on=["NB", "N"],
    how="inner",
    suffixes=("_mini", "_hpl"),
)


compare["Grid"] = compare["Grid_mini"].combine_first(compare["Grid_hpl"])
compare.drop(columns=["Grid_mini", "Grid_hpl"], inplace=True)


compare["gflops_ratio_mini_over_hpl"] = compare["mini_gflops"] / compare["hpl_gflops"]
compare["time_ratio_mini_over_hpl"]   = compare["mini_time_s"] / compare["hpl_time_s"]
compare["gflops_gap_hpl_minus_mini"]  = compare["hpl_gflops"] - compare["mini_gflops"]

compare.sort_values(by="NB", inplace=True)

compare.to_csv("hpl_minihpl_comparison.csv", index=False)

plt.figure()
plt.plot(compare["NB"], compare["hpl_gflops"], marker="o", label="HPL (official)")
plt.plot(compare["NB"], compare["mini_gflops"], marker="o", label="MiniHPL (ours)")
plt.title(f"GFLOPS vs NB (N={int(compare['N'].iloc[0])}, Grid={compare['Grid'].iloc[0]})")
plt.xlabel("NB")
plt.ylabel("GFLOPS")
plt.legend()
plt.grid(True)
plt.savefig("hpl_vs_minihpl_gflops.png", dpi=200, bbox_inches="tight")
plt.close()


plt.figure()
plt.plot(compare["NB"], compare["gflops_ratio_mini_over_hpl"], marker="o")
plt.title("MiniHPL / HPL GFLOPS ratio")
plt.xlabel("NB")
plt.ylabel("Ratio")
plt.grid(True)
plt.savefig("minihpl_over_hpl_ratio.png", dpi=200, bbox_inches="tight")
plt.close()

print("Wrote: hpl_minihpl_comparison.csv, hpl_vs_minihpl_gflops.png, minihpl_over_hpl_ratio.png")
