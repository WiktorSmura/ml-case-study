"""
Czech Financial Dataset - Part 1 Presentation Visualizations
=============================================================
Generates all charts, statistics, and diagrams for the dataset description section.
Run:  python visualize_dataset.py
Output: figures/ directory with all PNG files + stats printed to console.
"""

import warnings
from pathlib import Path
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.sankey import Sankey
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "blue":   "#3B82F6",
    "indigo": "#6366F1",
    "purple": "#8B5CF6",
    "teal":   "#14B8A6",
    "green":  "#22C55E",
    "amber":  "#F59E0B",
    "red":    "#EF4444",
    "slate":  "#64748B",
    "bg":     "#0F172A",   # dark background
    "surface":"#1E293B",   # card surface
    "text":   "#F1F5F9",
    "muted":  "#94A3B8",
}

DARK = False    # set False for a white-background version

def style_ax(ax, title="", xlabel="", ylabel="", legend=True):
    if DARK:
        ax.set_facecolor(C["surface"])
        ax.figure.set_facecolor(C["bg"])
        ax.tick_params(colors=C["muted"])
        ax.xaxis.label.set_color(C["muted"])
        ax.yaxis.label.set_color(C["muted"])
        ax.title.set_color(C["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["surface"])
        ax.tick_params(axis="both", which="both", color=C["surface"])
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if legend and ax.get_legend_handles_labels()[0]:
        leg = ax.legend(fontsize=8, framealpha=0.2,
                        facecolor=C["surface"] if DARK else "white",
                        edgecolor="none", labelcolor=C["text"] if DARK else "black")

OUT = Path("slides/figures")
OUT.mkdir(exist_ok=True, parents=True)

# ── Load raw CSVs ──────────────────────────────────────────────────────────────
DATA = Path(".")

def read_semi(name):
    return pd.read_csv(DATA / f"{name}.csv", sep=";", engine="python")

print("Loading tables …")
loan     = read_semi("loan")
account  = read_semi("account")
client   = read_semi("client")
disp     = read_semi("disp")
district = read_semi("district")
trans    = read_semi("trans")
card     = read_semi("card")
order    = read_semi("order")

# ── Basic cleanup ──────────────────────────────────────────────────────────────
def parse_ymd(s):
    """Parse YYMMDD or YYYYMMDD integers/strings."""
    s2 = s.astype(str).str.strip().str.split(" ").str[0]  # drop time part
    parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    m6 = s2.str.match(r"^\d{6}$", na=False)
    if m6.any():
        parsed.loc[m6] = pd.to_datetime(s2[m6], format="%y%m%d", errors="coerce")
        future = m6 & (parsed.dt.year > 1999)
        parsed.loc[future] -= pd.DateOffset(years=100)
    m8 = s2.str.match(r"^\d{8}$", na=False) & parsed.isna()
    if m8.any():
        parsed.loc[m8] = pd.to_datetime(s2[m8], format="%Y%m%d", errors="coerce")
    return parsed

loan["loan_date"]    = parse_ymd(loan["date"])
account["acc_date"]  = parse_ymd(account["date"])
trans["trans_date"]  = parse_ymd(trans["date"])
card["issued_date"]  = parse_ymd(card["issued"].astype(str).str.split(" ").str[0])

# birth_number → birth_date + gender
def parse_birth(bn):
    s = bn.astype(str).str.extract(r"(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})")
    yy = pd.to_numeric(s["yy"], errors="coerce")
    mm_raw = pd.to_numeric(s["mm"], errors="coerce")
    dd = pd.to_numeric(s["dd"], errors="coerce")
    gender = np.where(mm_raw > 50, "F", "M")
    mm = np.where(mm_raw > 50, mm_raw - 50, mm_raw)
    dates = pd.to_datetime(
        pd.DataFrame({"year": 1900 + yy, "month": mm, "day": dd}), errors="coerce"
    )
    return dates, gender

client["birth_date"], client["gender"] = parse_birth(client["birth_number"])

# district column rename
dist_map = {
    "A1":"district_id","A2":"district_name","A3":"region",
    "A4":"num_inhabitants","A5":"n_mun_lt499","A6":"n_mun_500_1999",
    "A7":"n_mun_2000_9999","A8":"n_mun_gt10000","A9":"num_cities",
    "A10":"ratio_urban","A11":"average_salary","A12":"unemp95",
    "A13":"unemp96","A14":"entrep_per1000","A15":"crimes95","A16":"crimes96",
}
district.columns = [dist_map.get(c, c) for c in district.columns]
for col in ["num_inhabitants","average_salary","unemp96","ratio_urban","crimes96","entrep_per1000"]:
    district[col] = pd.to_numeric(district[col], errors="coerce")
district["crime_rate96"] = district["crimes96"] / district["num_inhabitants"] * 1000

loan["amount"] = pd.to_numeric(loan["amount"], errors="coerce")
loan["duration"] = pd.to_numeric(loan["duration"], errors="coerce")
loan["payments"] = pd.to_numeric(loan["payments"], errors="coerce")
trans["amount"] = pd.to_numeric(trans["amount"], errors="coerce")
trans["balance"] = pd.to_numeric(trans["balance"], errors="coerce")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1 · Dataset overview / entity-relationship summary (infographic)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/8] Dataset overview …")

table_counts = {
    "Transactions\n(trans)": len(trans),
    "Clients\n(client)": len(client),
    "Accounts\n(account)": len(account),
    "Dispositions\n(disp)": len(disp),
    "Orders\n(order)": len(order),
    "Cards\n(card)": len(card),
    "Loans\n(loan)": len(loan),
    "Districts\n(district)": len(district),
}

colors_bar = [C["blue"], C["indigo"], C["purple"], C["teal"],
              C["green"], C["amber"], C["red"], C["slate"]]

fig, ax = plt.subplots(figsize=(12, 5))
if DARK:
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["surface"])

labels = list(table_counts.keys())
vals   = list(table_counts.values())
bars = ax.barh(labels, vals, color=colors_bar, height=0.6, edgecolor="none")
for bar, val in zip(bars, vals):
    ax.text(val + 5000, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", ha="left",
            fontsize=10, color=C["text"] if DARK else "black", fontweight="bold")

ax.set_xlim(0, max(vals) * 1.22)
style_ax(ax, title="Czech Financial Dataset — Table Sizes",
         xlabel="Number of rows", ylabel="")
ax.invert_yaxis()
ax.tick_params(axis="y", labelsize=9, colors=C["text"] if DARK else "black")
plt.tight_layout()
fig.savefig(OUT / "01_table_sizes.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2 · Schema / ER-like relationship diagram (visual)
# ─────────────────────────────────────────────────────────────────────────────
print("[2/8] ER diagram …")

fig, ax = plt.subplots(figsize=(10, 6))
if DARK:
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

# node positions: (x_center, y_center, title, columns_text)
nodes = [
    ("Loan", 1.5, 6, "account_id"),
    ("Permanent\norder", 1.5, 4, "account_id"),
    ("Transactions", 1.5, 2, "account_id"),
    ("Account", 4, 4, "account_id\ndistrict_id"),
    ("Disposition", 6.5, 4, "disp_id\nclient_id\naccount_id"),
    ("Credit Card", 6.5, 6, "disp_id"),
    ("Demograph.", 6.5, 2, "district_id"),
    ("Client", 8.8, 4.5, "client_id\ndistrict_id")
]

box_w, box_h = 1.6, 1.2
node_map = {n[0]: (n[1], n[2]) for n in nodes}

for title, x, y, cols in nodes:
    rect = mpatches.Rectangle((x - box_w/2, y - box_h/2), box_w, box_h,
                              edgecolor="black" if not DARK else C["text"], facecolor="white" if not DARK else C["surface"], lw=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x - box_w/2 + 0.1, y + box_h/2 - 0.2, title, ha="left", va="top",
            fontsize=10, fontweight="bold", fontstyle="italic" if "Demograph" not in title and "Permanent" not in title else "normal", color="black" if not DARK else C["text"], zorder=4)
    ax.text(x - box_w/2 + 0.1, y - box_h/2 + 0.1, cols, ha="left", va="bottom",
            fontsize=9, color="black" if not DARK else C["text"], zorder=4, linespacing=1.5)

# edges: (from_node, to_node)
edges = [
    ("Loan", "Account"),
    ("Permanent\norder", "Account"),
    ("Transactions", "Account"),
    ("Account", "Disposition"),
    ("Account", "Demograph."),
    ("Disposition", "Credit Card"),
    ("Disposition", "Client"),
    ("Client", "Demograph.")
]

for src, dst in edges:
    x0, y0 = node_map[src]
    x1, y1 = node_map[dst]
    
    # Simple orthogonal-like lines or direct lines
    if src in ["Loan", "Permanent\norder", "Transactions"] and dst == "Account":
        # Draw from right edge of src to left edge of dst
        ax.plot([x0 + box_w/2, 2.75, 2.75, x1 - box_w/2], [y0, y0, y1, y1], color="black" if not DARK else C["text"], lw=1.5, zorder=2)
    elif src == "Account" and dst == "Disposition":
        ax.plot([x0 + box_w/2, x1 - box_w/2], [y0, y1], color="black" if not DARK else C["text"], lw=1.5, zorder=2)
    elif src == "Account" and dst == "Demograph.":
        ax.plot([x0 + box_w/2, 5.25, 5.25, x1 - box_w/2], [y0 - 0.2, y0 - 0.2, y1, y1], color="black" if not DARK else C["text"], lw=1.5, zorder=2)
    elif src == "Disposition" and dst == "Credit Card":
        ax.plot([x0, x0, x1 - box_w/2], [y0 + box_h/2, 5.8, y1 - 0.2], color="black" if not DARK else C["text"], lw=1.5, zorder=2)
    elif src == "Disposition" and dst == "Client":
        ax.plot([x0 + box_w/2, x1 - box_w/2], [y0, y0], color="black" if not DARK else C["text"], lw=1.5, zorder=2)
    elif src == "Client" and dst == "Demograph.":
        ax.plot([x0 - box_w/2 + 0.2, 7.5, 7.5, x1 + box_w/2], [y0 - box_h/2, 3.0, y1, y1], color="black" if not DARK else C["text"], lw=1.5, zorder=2)

ax.set_title("Entity-Relationship Overview — Czech Financial Dataset",
             fontsize=13, fontweight="bold",
             color=C["text"] if DARK else "black", pad=12)
plt.tight_layout()
fig.savefig(OUT / "02_er_diagram.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3 · Loan status & target distribution
# ─────────────────────────────────────────────────────────────────────────────
print("[3/8] Loan status …")

status_labels = {
    "A": "A — Finished\n(no problems)",
    "B": "B — Finished\n(problems)",
    "C": "C — Running\n(no problems)",
    "D": "D — Running\n(problems)",
}
status_counts = loan["status"].str.strip().value_counts().reindex(["A","B","C","D"])
status_colors = [C["green"], C["red"], C["teal"], C["amber"]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
if DARK:
    fig.patch.set_facecolor(C["bg"])

# Bar chart
ax = axes[0]
if DARK: ax.set_facecolor(C["surface"])
bars = ax.bar([status_labels[s] for s in ["A","B","C","D"]],
              status_counts.values, color=status_colors,
              edgecolor="none", width=0.6)
for bar, val in zip(bars, status_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f"{val}\n({val/len(loan)*100:.1f}%)",
            ha="center", va="bottom", fontsize=9,
            color=C["text"] if DARK else "black", fontweight="bold")
style_ax(ax, title="Loan Status Distribution", ylabel="Count")
ax.set_ylim(0, status_counts.max() * 1.2)
ax.tick_params(axis="x", labelsize=8.5, colors=C["text"] if DARK else "black")

# Donut — finished_binary target
ax2 = axes[1]
if DARK: ax2.set_facecolor(C["surface"])
finished = loan[loan["status"].isin(["A","B"])]
target_counts = finished["status"].value_counts().reindex(["A","B"])
wedge_colors = [C["green"], C["red"]]
wedges, texts, autotexts = ax2.pie(
    target_counts.values,
    colors=wedge_colors, autopct="%1.1f%%",
    startangle=90, pctdistance=0.7,
    wedgeprops=dict(width=0.55, edgecolor=C["bg"] if DARK else "white", linewidth=2)
)
for at in autotexts:
    at.set_fontsize(12); at.set_fontweight("bold")
    at.set_color(C["text"] if DARK else "black")
ax2.legend(
    handles=[mpatches.Patch(color=c, label=l) for c, l in
     zip(wedge_colors, ["A — Good (0)", "B — Bad (1)"])],
    loc="lower center", fontsize=9,
    facecolor=C["surface"] if DARK else "white",
    edgecolor="none",
    labelcolor=C["text"] if DARK else "black"
)
ax2.set_title("Finished Loans — Binary Target\n(Class Balance)",
              fontsize=12, fontweight="bold",
              color=C["text"] if DARK else "black")
center_text = f"n = {len(finished)}"
ax2.text(0, 0, center_text, ha="center", va="center", fontsize=11,
         color=C["text"] if DARK else "black", fontweight="bold")

plt.tight_layout()
fig.savefig(OUT / "03_loan_status.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4 · Loan amount & duration distributions
# ─────────────────────────────────────────────────────────────────────────────
print("[4/8] Loan distributions …")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
if DARK: fig.patch.set_facecolor(C["bg"])

# Amount histogram
ax = axes[0]
if DARK: ax.set_facecolor(C["surface"])
good = loan[loan["status"]=="A"]["amount"].dropna()
bad  = loan[loan["status"]=="B"]["amount"].dropna()
bins = np.linspace(0, loan["amount"].quantile(0.98), 30)
ax.hist(good, bins=bins, alpha=0.7, color=C["green"], label="Good (A)", edgecolor="none")
ax.hist(bad,  bins=bins, alpha=0.7, color=C["red"],   label="Bad (B)",  edgecolor="none")
style_ax(ax, title="Loan Amount Distribution", xlabel="Amount (CZK)", ylabel="Count", legend=True)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.tick_params(colors=C["text"] if DARK else "black")

# Duration bar
ax = axes[1]
if DARK: ax.set_facecolor(C["surface"])
dur_vals = sorted(loan["duration"].dropna().unique())
good_d = loan[loan["status"]=="A"]["duration"].value_counts().reindex(dur_vals, fill_value=0)
bad_d  = loan[loan["status"]=="B"]["duration"].value_counts().reindex(dur_vals, fill_value=0)
x = np.arange(len(dur_vals))
w = 0.38
ax.bar(x - w/2, good_d.values, w, color=C["green"], label="Good (A)", edgecolor="none")
ax.bar(x + w/2, bad_d.values,  w, color=C["red"],   label="Bad (B)",  edgecolor="none")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(d)}m" for d in dur_vals], fontsize=8,
                   color=C["text"] if DARK else "black")
style_ax(ax, title="Loan Duration (months)", xlabel="Duration", ylabel="Count", legend=True)

# Monthly payment distribution
ax = axes[2]
if DARK: ax.set_facecolor(C["surface"])
good_p = loan[loan["status"]=="A"]["payments"].dropna()
bad_p  = loan[loan["status"]=="B"]["payments"].dropna()
bins2 = np.linspace(0, loan["payments"].quantile(0.98), 28)
ax.hist(good_p, bins=bins2, alpha=0.7, color=C["green"], label="Good (A)", edgecolor="none")
ax.hist(bad_p,  bins=bins2, alpha=0.7, color=C["red"],   label="Bad (B)",  edgecolor="none")
style_ax(ax, title="Monthly Payment Distribution", xlabel="Monthly Payment (CZK)", ylabel="Count", legend=True)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.tick_params(colors=C["text"] if DARK else "black")

plt.tight_layout()
fig.savefig(OUT / "04_loan_distributions.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 5 · Client demographics (age, gender)
# ─────────────────────────────────────────────────────────────────────────────
print("[5/8] Client demographics …")

# link clients to their loans via disp
ref_date = pd.Timestamp("1998-01-01")  # roughly midpoint of dataset
client["age"] = (ref_date - client["birth_date"]).dt.days / 365.25

disp_clean = disp.rename(columns={"type": "disp_type"})
loan_disp = loan[["loan_id","account_id","status"]].merge(
    disp_clean[["account_id","client_id","disp_type"]], on="account_id", how="left"
)
loan_clients = loan_disp.merge(client[["client_id","age","gender"]], on="client_id", how="left")
owners = loan_clients[loan_clients["disp_type"].str.upper().str.strip() == "OWNER"]
owners = owners[owners["status"].isin(["A","B"])]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
if DARK: fig.patch.set_facecolor(C["bg"])

# Age histogram
ax = axes[0]
if DARK: ax.set_facecolor(C["surface"])
age_bins = np.arange(15, 75, 5)
ag = owners[owners["status"]=="A"]["age"].dropna()
ab = owners[owners["status"]=="B"]["age"].dropna()
ax.hist(ag, bins=age_bins, alpha=0.75, color=C["green"], label="Good (A)", edgecolor="none")
ax.hist(ab, bins=age_bins, alpha=0.75, color=C["red"],   label="Bad (B)",  edgecolor="none")
style_ax(ax, title="Owner Age at Loan Date", xlabel="Age (years)", ylabel="Count", legend=True)
ax.tick_params(colors=C["text"] if DARK else "black")

# Gender stacked bar
ax = axes[1]
if DARK: ax.set_facecolor(C["surface"])
gender_status = owners.groupby(["status","gender"]).size().unstack(fill_value=0)
categories = gender_status.index.tolist()
m_vals = gender_status.get("M", pd.Series([0]*len(categories))).values
f_vals = gender_status.get("F", pd.Series([0]*len(categories))).values
x = np.arange(len(categories))
ax.bar(x, m_vals, color=C["blue"],   label="Male",   edgecolor="none")
ax.bar(x, f_vals, color=C["purple"], label="Female", edgecolor="none", bottom=m_vals)
ax.set_xticks(x)
ax.set_xticklabels(["Good (A)", "Bad (B)"], fontsize=10,
                   color=C["text"] if DARK else "black")
for i, (mv, fv) in enumerate(zip(m_vals, f_vals)):
    total = mv + fv
    ax.text(i, mv/2, f"M {mv}", ha="center", va="center", fontsize=9,
            color=C["text"], fontweight="bold")
    ax.text(i, mv + fv/2, f"F {fv}", ha="center", va="center", fontsize=9,
            color=C["text"], fontweight="bold")
style_ax(ax, title="Owner Gender vs Loan Outcome", ylabel="Count", legend=True)

plt.tight_layout()
fig.savefig(OUT / "05_client_demographics.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 6 · Transactions — volume, type mix, temporal activity
# ─────────────────────────────────────────────────────────────────────────────
print("[6/8] Transaction patterns …")

# Normalize type
type_map_t = {"PRIJEM":"Credit","VYDAJ":"Debit","VYBER":"Cash Withdrawal",
               "CREDIT":"Credit","DEBIT":"Debit","WITHDRAWAL":"Cash Withdrawal"}
trans["type_clean"] = trans["type"].str.strip().map(type_map_t).fillna(trans["type"].str.strip())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
if DARK: fig.patch.set_facecolor(C["bg"])

# Transaction type donut
ax = axes[0]
if DARK: ax.set_facecolor(C["surface"])
tc = trans["type_clean"].value_counts()
wedge_c = [C["green"], C["red"], C["amber"]]
wedges, _, autotexts = ax.pie(
    tc.values, labels=tc.index, colors=wedge_c[:len(tc)],
    autopct="%1.1f%%", startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor=C["bg"] if DARK else "white", linewidth=2),
    textprops=dict(color=C["text"] if DARK else "black", fontsize=9)
)
for at in autotexts:
    at.set_fontsize(9); at.set_fontweight("bold")
    at.set_color(C["text"] if DARK else "black")
ax.set_title("Transaction Type Mix", fontsize=12, fontweight="bold",
             color=C["text"] if DARK else "black")
ax.text(0, 0, f"{len(trans):,}\ntransactions", ha="center", va="center",
        fontsize=9, color=C["text"] if DARK else "black")

# Monthly transaction volume over time
ax = axes[1]
if DARK: ax.set_facecolor(C["surface"])
trans["ym"] = trans["trans_date"].dt.to_period("Q")
vol = trans.groupby("ym").size().reset_index(name="count")
vol = vol.dropna(subset=["ym"])
vol["ym_ts"] = vol["ym"].dt.to_timestamp()
ax.fill_between(vol["ym_ts"], vol["count"], alpha=0.35, color=C["blue"])
ax.plot(vol["ym_ts"], vol["count"], color=C["blue"], lw=2)
style_ax(ax, title="Quarterly Transaction Volume", xlabel="Quarter", ylabel="Transactions")
ax.tick_params(axis="x", rotation=30, colors=C["text"] if DARK else "black")
ax.tick_params(axis="y", colors=C["text"] if DARK else "black")

# Average transaction amount by type
ax = axes[2]
if DARK: ax.set_facecolor(C["surface"])
amt_by_type = trans.groupby("type_clean")["amount"].agg(["mean","median"]).reset_index()
x = np.arange(len(amt_by_type))
w = 0.35
ax.bar(x - w/2, amt_by_type["mean"],   w, color=C["blue"],   label="Mean",   edgecolor="none")
ax.bar(x + w/2, amt_by_type["median"], w, color=C["indigo"], label="Median", edgecolor="none")
ax.set_xticks(x)
ax.set_xticklabels(amt_by_type["type_clean"], fontsize=8,
                   color=C["text"] if DARK else "black")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f"{v/1000:.0f}k"))
style_ax(ax, title="Transaction Amount by Type (CZK)",
         xlabel="", ylabel="Amount (CZK)", legend=True)
ax.tick_params(axis="y", colors=C["text"] if DARK else "black")

plt.tight_layout()
fig.savefig(OUT / "06_transactions.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 7 · District / geographic context
# ─────────────────────────────────────────────────────────────────────────────
print("[7/8] District statistics …")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
if DARK: fig.patch.set_facecolor(C["bg"])

# Salary by region
ax = axes[0]
if DARK: ax.set_facecolor(C["surface"])
reg_sal = district.groupby("region")["average_salary"].mean().sort_values(ascending=True)
bars = ax.barh(reg_sal.index, reg_sal.values, color=C["teal"], edgecolor="none")
for bar, v in zip(bars, reg_sal.values):
    ax.text(v + 50, bar.get_y() + bar.get_height()/2,
            f"{v:,.0f}", va="center", fontsize=8,
            color=C["text"] if DARK else "black")
style_ax(ax, title="Avg Salary by Region (CZK)", xlabel="CZK / month")
ax.tick_params(axis="y", labelsize=8, colors=C["text"] if DARK else "black")
ax.tick_params(axis="x", colors=C["text"] if DARK else "black")
ax.set_xlim(0, reg_sal.max() * 1.18)

# Unemployment rate distribution
ax = axes[1]
if DARK: ax.set_facecolor(C["surface"])
ax.hist(district["unemp96"].dropna(), bins=16, color=C["amber"], edgecolor="none")
style_ax(ax, title="Unemployment Rate (1996)", xlabel="%", ylabel="Districts", legend=False)
ax.tick_params(colors=C["text"] if DARK else "black")

# Crime rate vs salary scatter
ax = axes[2]
if DARK: ax.set_facecolor(C["surface"])
d_clean = district.dropna(subset=["crime_rate96","average_salary","region"])
regions = d_clean["region"].unique()
pal = [C["blue"],C["indigo"],C["purple"],C["teal"],C["green"],C["amber"],C["red"],C["slate"]]
for i, reg in enumerate(regions):
    sub = d_clean[d_clean["region"]==reg]
    ax.scatter(sub["average_salary"], sub["crime_rate96"],
               color=pal[i % len(pal)], label=reg, s=50, alpha=0.85, edgecolors="none")
style_ax(ax, title="Crime Rate vs Avg Salary", xlabel="Avg Salary (CZK)",
         ylabel="Crimes / 1000 inhabitants", legend=True)
ax.tick_params(colors=C["text"] if DARK else "black")
if ax.get_legend():
    ax.get_legend().set_title("Region", prop={"size":7})
    for lh in ax.get_legend().legend_handles:
        try:
            lh.set_sizes([25])
        except AttributeError:
            pass

plt.tight_layout()
fig.savefig(OUT / "07_district.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 8 · Data leakage concept + timeline diagram
# ─────────────────────────────────────────────────────────────────────────────
print("[8/8] Data leakage concept diagram …")

fig, ax = plt.subplots(figsize=(14, 6.5))
if DARK:
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Timeline arrow
ax.annotate("", xy=(9.8, 2.2), xytext=(0.2, 2.2),
            arrowprops=dict(arrowstyle="-|>", color="black" if not DARK else C["muted"], lw=3, mutation_scale=25))
ax.text(5, 0.4, "Time", ha="center", fontsize=15, color="black" if not DARK else C["muted"], style="italic", fontweight="bold")

# Key events on timeline
events = [
    (1.5, "Account\nOpened", C["blue"], "account.date"),
    (4.8, "Loan\nGranted", C["red"],  "loan.date\n(Decision Point)"),
    (8.2, "Transaction\nPost-Loan", C["amber"], "trans.date > loan.date\nLEAKAGE RISK"),
]
for x, lbl, col, sublbl in events:
    # Dashed line connecting box to timeline and below
    ax.plot([x, x], [1.3, 2.7], color=col, lw=3, ls="--", zorder=3)
    
    # Event Box
    box = mpatches.FancyBboxPatch(
        (x - 1.3, 2.7), 2.6, 1.4,
        boxstyle="round,pad=0.05", linewidth=2.5,
        edgecolor=col, facecolor="white" if not DARK else "#F0F4FF", zorder=4
    )
    ax.add_patch(box)
    
    # Text inside Event Box
    ax.text(x, 3.4, lbl, ha="center", va="center",
            fontsize=13, fontweight="bold", color=col, zorder=5)
            
    # Text below timeline
    ax.text(x, 0.9, sublbl, ha="center", va="center",
            fontsize=11, color="black" if not DARK else "#666", zorder=5, linespacing=1.4, fontweight="bold")
            
    # Dot on timeline
    ax.plot(x, 2.2, "o", color=col, ms=14, zorder=6)

# Shaded safe zone
ax.axvspan(0.2, 4.75, alpha=0.15, color=C["green"], label="Safe zone")
ax.axvspan(4.85, 9.7, alpha=0.15, color=C["red"],   label="Danger zone")
ax.text(2.5, 5.5, "✓ Safe to use", fontsize=14, color="darkgreen" if not DARK else C["green"], fontweight="bold")
ax.text(7.2, 5.5, "✗ Data leakage!", fontsize=14, color="darkred" if not DARK else C["red"],  fontweight="bold")

# Rule box
rule_text = (
    "Preprocessing Rule:\n"
    "• trans.date < loan.date\n"
    "• card.issued < loan.date\n"
)
rule_box = mpatches.FancyBboxPatch(
    (0.2, 4.4), 3.8, 1.0, boxstyle="round,pad=0.05",
    facecolor="#FFF8E1" if not DARK else C["surface"],
    edgecolor=C["amber"], lw=2.5, zorder=6
)
ax.add_patch(rule_box)
ax.text(2.1, 4.9, rule_text, ha="center", va="center",
        fontsize=12, color="black" if not DARK else "#444",
        fontfamily="monospace", linespacing=1.6, fontweight="bold", zorder=7)

ax.set_title("Temporal Data Leakage",
             fontsize=20, fontweight="bold",
             color="black" if not DARK else C["text"], y=0.98)
plt.tight_layout()
fig.savefig(OUT / "08_leakage_concept.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 9 · Correlation Matrix
# ─────────────────────────────────────────────────────────────────────────────
print("[9/9] Correlation Matrix …")

# Merge a flat dataset just for correlation purposes (1 row = 1 loan)
corr_df = loan[["loan_id", "amount", "duration", "payments", "account_id"]].copy()
# Add age
corr_df = corr_df.merge(owners[["account_id", "age"]], on="account_id", how="left")
# Add district stats
acc_dist = account[["account_id", "district_id"]].merge(
    district[["district_id", "average_salary", "unemp96", "crime_rate96"]], 
    on="district_id", how="left"
)
corr_df = corr_df.merge(acc_dist, on="account_id", how="left")
# Add some transaction stats
trans_agg = trans.groupby("account_id")["amount"].agg(["mean", "count"]).reset_index()
trans_agg.columns = ["account_id", "trans_mean_amt", "trans_count"]
corr_df = corr_df.merge(trans_agg, on="account_id", how="left")

# Select numeric columns for correlation
cols_to_corr = [
    "amount", "duration", "payments", 
    "age", "average_salary", "unemp96", 
    "crime_rate96", "trans_mean_amt", "trans_count"
]
labels_corr = [
    "Loan Amount", "Duration", "Payments",
    "Client Age", "Dist. Salary", "Dist. Unemp.",
    "Dist. Crime", "Avg Trans Amt", "Trans Count"
]

corr_matrix = corr_df[cols_to_corr].corr()

fig, ax = plt.subplots(figsize=(8, 7))
if DARK:
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

cax = ax.matshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
fig.colorbar(cax, shrink=0.8)

ax.set_xticks(np.arange(len(labels_corr)))
ax.set_yticks(np.arange(len(labels_corr)))
ax.set_xticklabels(labels_corr, rotation=45, ha="left", color="black" if not DARK else C["text"], fontsize=10)
ax.set_yticklabels(labels_corr, color="black" if not DARK else C["text"], fontsize=10)

# Add text annotations
for i in range(len(labels_corr)):
    for j in range(len(labels_corr)):
        val = corr_matrix.iloc[i, j]
        text_col = "white" if abs(val) > 0.5 else ("black" if not DARK else C["text"])
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_col, fontsize=9)

ax.set_title("Feature Correlations across merged tables", pad=20, fontsize=14, fontweight="bold", color="black" if not DARK else C["text"])
plt.tight_layout()
fig.savefig(OUT / "09_correlation.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  Summary statistics dashboard (printable table)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating summary stats table …")

rows = [
    ("Accounts", f"{len(account):,}", "Unique bank accounts"),
    ("Clients", f"{len(client):,}", "Unique clients"),
    ("Loans", f"{len(loan):,}", "Loan contracts (682 total)"),
    ("→ Finished A (Good)", f"{(loan['status']=='A').sum():,}", f"  {(loan['status']=='A').mean()*100:.1f}% of all loans"),
    ("→ Finished B (Bad)", f"{(loan['status']=='B').sum():,}", f"  {(loan['status']=='B').mean()*100:.1f}% of all loans"),
    ("→ Running C (Good)", f"{(loan['status']=='C').sum():,}", "  Active / no problems"),
    ("→ Running D (Bad)", f"{(loan['status']=='D').sum():,}", "  Active / problems"),
    ("Transactions", f"{len(trans):,}", "Individual bank transactions"),
    ("Orders", f"{len(order):,}", "Standing payment orders"),
    ("Cards", f"{len(card):,}", "Issued credit/debit cards"),
    ("Districts", f"{len(district):,}", "Unique geographic districts"),
    ("Loan amount mean", f"CZK {loan['amount'].mean():,.0f}", f"Std: CZK {loan['amount'].std():,.0f}"),
    ("Loan amount range", f"CZK {loan['amount'].min():,.0f} – {loan['amount'].max():,.0f}", ""),
    ("Loan duration", f"{sorted(loan['duration'].dropna().unique().astype(int).tolist())}", "months (6 levels)"),
    ("Owner age (median)", f"{owners['age'].median():.1f} years", "at time of loan"),
    ("Gender split (owners)", f"M {(owners['gender']=='M').mean()*100:.0f}% / F {(owners['gender']=='F').mean()*100:.0f}%", "finished loans subset"),
    ("Time span", "Jan 1993 – Dec 1998", "approx (from transaction dates)"),
]

print("\n" + "="*70)
print(f"{'Metric':<30} {'Value':<22} {'Notes'}")
print("="*70)
for r, v, n in rows:
    print(f"{r:<30} {v:<22} {n}")
print("="*70)

# Save to text
with open(OUT / "summary_stats.txt", "w") as f:
    f.write("Czech Financial Dataset — Summary Statistics\n")
    f.write("="*70 + "\n")
    f.write(f"{'Metric':<30} {'Value':<22} {'Notes'}\n")
    f.write("="*70 + "\n")
    for r, v, n in rows:
        f.write(f"{r:<30} {v:<22} {n}\n")
    f.write("="*70 + "\n")

print(f"\nAll figures saved to: {OUT.resolve()}/")
print("Files: " + ", ".join(sorted(str(p.name) for p in OUT.glob("*.png"))))
