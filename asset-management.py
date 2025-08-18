# save as app_rcm.py
import os, re
import numpy as np
import pandas as pd
import streamlit as st

# ==================== Data locations (GitHub-first, local fallback) ====================
import os, pandas as pd, streamlit as st

# Your repo/branch where the CSVs live
GH_OWNER  = "camwcolby"
GH_REPO   = "rcm-streamlit"
GH_BRANCH = "data"  # <- you put the CSVs on the 'data' branch
def gh_raw(path: str) -> str:
    return f"https://raw.githubusercontent.com/{GH_OWNER}/{GH_REPO}/{GH_BRANCH}/{path}"

# Optional: let local dev override via env var; safe to leave as your Windows path
LOCAL_OUT_DIR = os.environ.get(
    "RCM_LOCAL_OUT_DIR",
    r"C:\Users\cacolby\Desktop\Asset Management Data Analysis\Gap Analysis"
)

# File names (must match what you committed to the data branch)
ASSET_FILE = "rcm_asset_scores_v2.csv"
ACT_FILE   = "rcm_activity_recommendations_v2.csv"
GAP_FILE   = "gap_activity_level_exact.csv"  # optional for baseline backfill

def _local_path(name: str):
    if LOCAL_OUT_DIR:
        return os.path.join(LOCAL_OUT_DIR, name)
    return None

@st.cache_data(show_spinner=False)
def load_csv_smart(file_name: str) -> tuple[pd.DataFrame, str]:
    """
    Try local path (if it exists) then GitHub raw URL. Returns (df, source_used).
    """
    candidates = []
    lp = _local_path(file_name)
    if lp and os.path.exists(lp):
        candidates.append(lp)  # local (when running on your own PC)
    candidates.append(gh_raw(file_name))  # GitHub (for Streamlit Cloud)

    last_err = None
    for src in candidates:
        try:
            df = pd.read_csv(src, low_memory=False)
            df.columns = df.columns.astype(str).str.strip()
            return df, src
        except Exception as e:
            last_err = e
            continue
    # If both fail, stop the app with a clear message
    st.error(f"Could not load '{file_name}'. Last error:\n{last_err}")
    st.stop()

# ==================== Load CSVs ====================
assets, assets_src = load_csv_smart(ASSET_FILE)
acts,   acts_src   = load_csv_smart(ACT_FILE)

# GAP is optional; try to load but don't fail if missing on GitHub
gap = None
try:
    gap, gap_src = load_csv_smart(GAP_FILE)
except Exception:
    gap_src = None

st.caption(f"Data sources → Assets: {assets_src} | Activities: {acts_src}" + (f" | Gap: {gap_src}" if gap_src else ""))


# Match your notebook window for baseline rate
WINDOW_START = pd.Timestamp("2022-01-01")
WINDOW_END   = pd.Timestamp("2024-12-31")
YEARS = (WINDOW_END - WINDOW_START).days / 365.25

st.set_page_config(page_title="RCM Explorer", layout="wide")
st.title("RCM Explorer — Portfolio")

# ==================== Helpers ====================
def norm_text(s):
    s = "" if pd.isna(s) else str(s).upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def norm_id_str(x):
    s = "" if pd.isna(x) else str(x)
    s = s.replace(",", "")
    s = re.sub(r"\.0$", "", s)         # remove Excel float tails like 12345.0
    s = re.sub(r"\s+", "", s)
    return s.upper()

def coerce_keys_str(df):
    """Make join keys consistent strings (prevents merge dtype errors)."""
    if "__AssetID_norm" in df.columns:
        df["__AssetID_norm"] = df["__AssetID_norm"].map(norm_id_str).astype(str)
    if "__Area" in df.columns:
        df["__Area"] = df["__Area"].astype(str).str.strip()
    return df

def minmax01(s):
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / (hi - lo)).fillna(0.0)

def sigmoid(x, k=1.0, bias=0.0):
    return 1.0 / (1.0 + np.exp(-k*(x + bias)))

def coerce_year_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return float year Series aligned to df.index from many formats."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        y = pd.to_numeric(s, errors="coerce")
    else:
        y = pd.to_datetime(s, errors="coerce").dt.year.astype("float64")
        y_fallback = s.astype(str).str.extract(r"(\d{4})", expand=False).astype("float64")
        y = y.fillna(y_fallback)
    return pd.Series(y.values, index=df.index, dtype="float64")

def ensure_series(x, like_index, dtype="float64"):
    return x if isinstance(x, pd.Series) else pd.Series(x, index=like_index, dtype=dtype)

def num_series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    """Numeric Series for a column, or a default Series aligned to df.index."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")

def _pick_col(df, *cands):
    """Case/space/punct-insensitive column matcher."""
    norm = {re.sub(r"[^a-z0-9]+","", c.lower()): c for c in df.columns}
    for cand in cands:
        key = re.sub(r"[^a-z0-9]+","", str(cand).lower())
        if key in norm:
            return norm[key]
    return None

# ==================== Load CSVs ====================
assets = pd.read_csv(ASSET_CSV, low_memory=False)
acts   = pd.read_csv(ACT_CSV,   low_memory=False)

# Strip headers (important if Excel/ETL left spaces/BOM)
assets.columns = assets.columns.astype(str).str.strip()
acts.columns   = acts.columns.astype(str).str.strip()

# Normalize keys immediately
assets = coerce_keys_str(assets)
acts   = coerce_keys_str(acts)

# ==================== Rebuild baseline_per_year only if ABSENT ====================
acts["Activity_norm_app"] = acts.get("Activity", "").astype(str).map(norm_text)

has_baseline_col = "baseline_per_year" in acts.columns
if not has_baseline_col:
    # Only rebuild from gap when the column is missing
    if os.path.exists(GAP_CSV):
        gap = pd.read_csv(GAP_CSV, low_memory=False)
        gap.columns = gap.columns.astype(str).str.strip()
        gap = coerce_keys_str(gap)
        gap["Activity_norm_app"] = gap.get("Activity", "").astype(str).map(norm_text)

        base = (gap.rename(columns={"Completed_Count": "_cmpl"})
                    .groupby(["__Area","__AssetID_norm","Activity_norm_app"], dropna=False)["_cmpl"]
                    .sum()
                    .reset_index())
        base = coerce_keys_str(base)
        base["baseline_per_year"] = base["_cmpl"] / YEARS

        acts = acts.merge(
            base[["__Area","__AssetID_norm","Activity_norm_app","baseline_per_year"]],
            on=["__Area","__AssetID_norm","Activity_norm_app"], how="left"
        )
        st.info("baseline_per_year rebuilt from gap_activity_level_exact.csv (column was missing in ACT_CSV)")
    else:
        acts["baseline_per_year"] = 0.0
        st.warning("baseline_per_year missing and gap_activity_level_exact.csv not found; defaulting to 0.0")

# Ensure numeric and non-negative
acts["baseline_per_year"] = pd.to_numeric(acts["baseline_per_year"], errors="coerce").fillna(0.0).clip(lower=0.0)

# ==================== Sidebar: filters & knobs ====================
st.sidebar.header("Filters")
areas = sorted(assets["__Area"].dropna().unique().tolist()) if "__Area" in assets.columns else []
area_sel = st.sidebar.multiselect("Area(s)", areas)

risk_min = st.sidebar.number_input("Min Annual Risk (USD)", 0.0, 1e8, 0.0, step=1000.0)

st.sidebar.header("CoF knobs")
use_csv_cof = st.sidebar.checkbox("Use CoF from CSV (Jupyter)", value=True)
usd_per_downtime_hour = st.sidebar.number_input("Downtime $/hr", 0.0, 100000.0, 1500.0, 50.0)
penalty_unit_usd      = st.sidebar.number_input("Penalty unit $ (env/safety)", 0.0, 1e6, 50000.0, 1000.0)
rep_cost_pct          = st.sidebar.slider("Replacement cost factor", 0.0, 0.10, 0.01, 0.005)
downtime_default_hr   = st.sidebar.number_input("Default downtime hours", 0.0, 48.0, 4.0, 0.5)

st.sidebar.header("PM cost knobs")
labor_rate_override   = st.sidebar.number_input("Labor rate $/hr", 0.0, 500.0, 75.0, 1.0)
material_adder        = st.sidebar.number_input("Material adder per PM ($)", 0.0, 5000.0, 0.0, 25.0)

st.sidebar.header("Decision line")
k = st.sidebar.slider("Increase approval factor (RRV ≥ k × ΔCost)", 0.0, 3.0, 1.0, 0.1)

st.sidebar.header("PoF recompute (optional)")
recompute_pof = st.sidebar.checkbox("Recompute PoF in app (uses Age/Condition/Risk/Compliance if present)", value=False)
sigmoid_k_ui  = st.sidebar.slider("PoF steepness (k)", 0.1, 3.0, 1.1, 0.1)
pof_bias      = st.sidebar.slider("PoF bias", -1.0, 1.0, 0.0, 0.1)
st.sidebar.caption("Weights (normalized to sum to 1)")
w_age   = st.sidebar.slider("w_age",    0.0, 1.0, 0.30, 0.05)
w_cond  = st.sidebar.slider("w_cond",   0.0, 1.0, 0.30, 0.05)
w_ncomp = st.sidebar.slider("w_noncomp",0.0, 1.0, 0.20, 0.05)
w_risk  = st.sidebar.slider("w_prior",  0.0, 1.0, 0.10, 0.05)
w_rx    = st.sidebar.slider("w_rehab",  0.0, 1.0, 0.10, 0.05)
rx_horizon_years = st.sidebar.slider("Rehab horizon (yrs)", 1, 20, 5, 1)

# Apply area filter early so all downstream views match
if area_sel:
    assets = assets[assets["__Area"].isin(area_sel)].copy()
    acts   = acts[acts["__Area"].isin(area_sel)].copy()

# ==================== Assets: CoF/PoF & annual risk ====================
assets_calc = assets.copy()

if use_csv_cof and "CoF_USD_v2" in assets.columns:
    assets_calc["CoF_USD_app"] = pd.to_numeric(assets["CoF_USD_v2"], errors="coerce").fillna(0.0)
else:
    dt_hours = assets_calc["__Downtime_hours_est"] if "__Downtime_hours_est" in assets_calc.columns \
               else pd.Series(downtime_default_hr, index=assets_calc.index)
    cof_score = num_series(assets_calc, "Governing COF Score", 0.0).fillna(0.0)
    rep_best  = num_series(assets_calc, "RepCost_best", 0.0).fillna(0.0)
    assets_calc["CoF_USD_app"] = (
        pd.to_numeric(dt_hours, errors="coerce").fillna(downtime_default_hr) * usd_per_downtime_hour
        + cof_score * penalty_unit_usd
        + rep_best * rep_cost_pct
    )

if recompute_pof:
    age01   = minmax01(assets_calc.get("Age_years", 0))
    cond01  = minmax01(assets_calc.get("Condition Score", 0))
    comp    = pd.to_numeric(assets_calc.get("Compliance_rate", np.nan), errors="coerce")
    nonc01  = (1.0 - comp).clip(0,1) if isinstance(comp, pd.Series) else pd.Series(0.0, index=assets_calc.index)
    risk01  = minmax01(assets_calc.get("Asset Risk Score", 0))
    next_y = pd.Series(np.nan, index=assets_calc.index, dtype=float)
    for cand in ["__NextInvestYear", "Governing Rehab Replace Year", "Governing Rehab/Replace Year"]:
        if cand in assets_calc.columns:
            next_y = coerce_year_col(assets_calc, cand)
            break
    yrs_to = next_y - float(pd.Timestamp.now().year)
    urg01  = ensure_series((rx_horizon_years - yrs_to) / rx_horizon_years, assets_calc.index).clip(0.0, 1.0).fillna(0.0)
    tot = max(1e-9, (w_age + w_cond + w_ncomp + w_risk + w_rx))
    w_age_n, w_cond_n, w_ncomp_n, w_risk_n, w_rx_n = (w_age/tot, w_cond/tot, w_ncomp/tot, w_risk/tot, w_rx/tot)
    lin = (w_age_n*age01 + w_cond_n*cond01 + w_ncomp_n*nonc01 + w_risk_n*risk01 + w_rx_n*urg01)
    assets_calc["PoF_app"] = sigmoid(lin, k=sigmoid_k_ui, bias=pof_bias).clip(0,1)
else:
    assets_calc["PoF_app"] = pd.to_numeric(assets_calc.get("PoF", 0.5), errors="coerce").clip(0,1).fillna(0.5)

assets_calc["Annual_Risk_USD_app"] = assets_calc["PoF_app"] * assets_calc["CoF_USD_app"]
assets_view = assets_calc[assets_calc["Annual_Risk_USD_app"] >= risk_min].copy()

# ==================== Activities: harden PM cost & recompute decisions ====================
acts_calc = acts.copy()

# Baseline per year already ensured. Now robust proposed_per_year detection.
acts_calc["baseline_per_year"] = num_series(acts_calc, "baseline_per_year", 0.0).fillna(0.0).clip(lower=0.0)

prop_col = _pick_col(
    acts_calc,
    "proposed_per_year", "Proposed_per_year", "proposed rate", "proposed",
    "proposed_rate", "proposedperyear"
)
if prop_col is None:
    acts_calc["proposed_per_year"] = acts_calc["baseline_per_year"]
    st.info("No proposed_per_year-like column found in ACT_CSV; using baseline_per_year as proposed.")
else:
    acts_calc["proposed_per_year"] = pd.to_numeric(acts_calc[prop_col], errors="coerce")
    if acts_calc["proposed_per_year"].isna().all():
        acts_calc["proposed_per_year"] = acts_calc["baseline_per_year"]
        st.info(f"Column '{prop_col}' is empty/NaN; using baseline_per_year as proposed.")
    acts_calc["proposed_per_year"] = acts_calc["proposed_per_year"].fillna(0.0).clip(lower=0.0)

# PM hours (fallback 2.0)
acts_calc["PM_hours"] = num_series(acts_calc, "PM_hours", np.nan).fillna(2.0).clip(lower=0.0)

# PM cost each = CSV if >0 else PM_hours * labor_rate_override (+ material adder)
pmc = num_series(acts_calc, "PM_cost_each", np.nan)
pmc_fallback = acts_calc["PM_hours"] * float(labor_rate_override)
acts_calc["PM_cost_each"] = pmc.where(pmc > 0, pmc_fallback).fillna(pmc_fallback)
if float(material_adder) > 0:
    acts_calc["PM_cost_each"] = acts_calc["PM_cost_each"] + float(material_adder)

# Join CoF & PoF from assets_calc (ensures decisions align with app knobs)
key_cols = ["__Area","__AssetID_norm"]
if all(c in assets_calc.columns for c in key_cols) and all(c in acts_calc.columns for c in key_cols):
    acts_calc = acts_calc.merge(
        assets_calc[key_cols + ["CoF_USD_app","PoF_app"]],
        on=key_cols, how="left"
    )
else:
    acts_calc["CoF_USD_app"] = pd.to_numeric(acts_calc.get("CoF_USD_v2", 0), errors="coerce").fillna(0.0)
    acts_calc["PoF_app"]     = pd.to_numeric(acts_calc.get("PoF", 0.5), errors="coerce").clip(0,1).fillna(0.5)

# Costs & RRV with tuned PM cost
bpy = acts_calc["baseline_per_year"]
ppy = acts_calc["proposed_per_year"]
eff = pd.to_numeric(acts_calc.get("eff_per_event", 0.05), errors="coerce").fillna(0.05)

acts_calc["cost_baseline_per_year"] = bpy * acts_calc["PM_cost_each"]
acts_calc["cost_proposed_per_year"] = ppy * acts_calc["PM_cost_each"]
acts_calc["cost_delta_per_year"]    = acts_calc["cost_proposed_per_year"] - acts_calc["cost_baseline_per_year"]

raw_delta_pof = eff * (ppy - bpy)
delta_pof = np.where(
    raw_delta_pof >= 0, np.minimum(raw_delta_pof, acts_calc["PoF_app"]),
    -np.minimum(-raw_delta_pof, 1.0 - acts_calc["PoF_app"])
)
acts_calc["Delta_PoF_app"] = delta_pof
acts_calc["RRV_USD_app"]   = acts_calc["Delta_PoF_app"] * acts_calc["CoF_USD_app"]

# Tuned decision
acts_calc["Decision_tuned"] = np.where(
    acts_calc["cost_delta_per_year"] > 0,
    np.where(acts_calc["RRV_USD_app"] >= k * acts_calc["cost_delta_per_year"], "INCREASE", "KEEP_BASELINE"),
    np.where(acts_calc["cost_delta_per_year"] < 0,
             np.where((-acts_calc["RRV_USD_app"]) <= (-acts_calc["cost_delta_per_year"]), "REDUCE", "KEEP_BASELINE"),
             "KEEP_BASELINE")
)

# --- Approved plan vs requested proposal (clarifies the "contradiction") ---

# What we actually plan to spend after decisions
acts_calc["Approved_cost_per_year"] = np.where(
    acts_calc["Decision_tuned"] == "INCREASE",
    acts_calc["cost_proposed_per_year"],                            # fund the increase
    np.where(
        acts_calc["Decision_tuned"] == "REDUCE",
        acts_calc["cost_proposed_per_year"],                        # take the reduction
        acts_calc["cost_baseline_per_year"]                         # keep baseline
    )
)

# Approved delta vs baseline
acts_calc["Approved_delta_per_year"] = (
    acts_calc["Approved_cost_per_year"] - acts_calc["cost_baseline_per_year"]
)

# Approved risk reduction (KEEP_BASELINE contributes zero)
acts_calc["Approved_RRV_USD"] = np.where(
    acts_calc["Decision_tuned"] == "KEEP_BASELINE", 0.0, acts_calc["RRV_USD_app"]
)

# Area × decision rollup (approved)
roll = (acts_calc
        .groupby(["__Area","Decision_tuned"], as_index=False)[
            ["cost_baseline_per_year","Approved_cost_per_year","Approved_delta_per_year","Approved_RRV_USD"]
        ].sum())

roll = roll.rename(columns={
    "Approved_cost_per_year": "approved_cost_per_year",
    "Approved_delta_per_year": "approved_net_delta",
    "Approved_RRV_USD":       "approved_risk_reduction_usd"
})

# Area net summary (single line per area)
area_net = (acts_calc
            .groupby("__Area", as_index=False)[
                ["cost_baseline_per_year","Approved_cost_per_year","Approved_delta_per_year","Approved_RRV_USD"]
            ].sum()
            .rename(columns={
                "cost_baseline_per_year":"baseline_cost_total",
                "Approved_cost_per_year":"approved_cost_total",
                "Approved_delta_per_year":"approved_net_delta",
                "Approved_RRV_USD":"approved_risk_reduction_usd"
            }))

# Optional ROI at area level for the funded increases only
inc_mask = acts_calc["Approved_delta_per_year"] > 0
roi_area = (acts_calc[inc_mask]
            .groupby("__Area", as_index=False)[["Approved_delta_per_year","Approved_RRV_USD"]]
            .sum()
            .rename(columns={"Approved_delta_per_year":"inc_delta_sum",
                             "Approved_RRV_USD":"inc_rrv_sum"}))
area_net = area_net.merge(roi_area, on="__Area", how="left")
area_net["ROI_for_increases"] = np.where(
    area_net["inc_delta_sum"] > 0, area_net["inc_rrv_sum"] / area_net["inc_delta_sum"], np.nan
)

st.subheader("Activities — decisions & ROI (approved)")
st.caption("Approved = what will actually be done after decision rules, not just what was requested.")
st.dataframe(
    roll.sort_values(["__Area","Decision_tuned"]),
    use_container_width=True
)

st.markdown("**Area net (single verdict per area):**")
st.dataframe(
    area_net.sort_values("approved_net_delta", ascending=False),
    use_container_width=True
)

# ==================== Quick sanity ====================
st.caption("Quick debug: nonzero counts in activity inputs (post-harden)")
st.write({
    "baseline_per_year>0": int((acts_calc["baseline_per_year"] > 0).sum()),
    "proposed_per_year>0": int((acts_calc["proposed_per_year"] > 0).sum()),
    "PM_hours>0":          int((acts_calc["PM_hours"] > 0).sum()),
    "PM_cost_each>0":      int((acts_calc["PM_cost_each"] > 0).sum()),
    "cost_baseline>0":     int((acts_calc["cost_baseline_per_year"] > 0).sum()),
    "cost_proposed>0":     int((acts_calc["cost_proposed_per_year"] > 0).sum()),
})

# ==================== Views ====================
st.subheader("Assets — ranked by annual risk (using app knobs)")

assets_cols_show = [c for c in [
    "__Area","__AssetID_norm","Asset Type","PoF_app","CoF_USD_app","Annual_Risk_USD_app",
    "Age_years","Condition Score","RepCost_best","Governing COF Score"
] if c in assets_calc.columns]

# Build view df
assets_df = (
    assets_view[assets_cols_show]
    .sort_values("Annual_Risk_USD_app", ascending=False)
    .head(500)
    .copy()
)

# Helper: find cost-like numeric columns and apply $ formatting with 2 decimals
def format_money_styler(df: pd.DataFrame, extra_keys=("cost","usd")) -> pd.io.formats.style.Styler:
    money_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in extra_keys)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not money_cols:
        return df.style  # nothing to format
    fmt_map = {c: "${:,.2f}" for c in money_cols}
    return df.style.format(fmt_map)

# Show assets with currency formatting on cost-like columns
st.dataframe(format_money_styler(assets_df), use_container_width=True)

st.divider()
st.subheader("Activities — decisions & ROI")

# Rollup table
roll = (acts_calc
        .assign(Risk_Reduction_USD = pd.to_numeric(acts_calc.get("RRV_USD_app", 0), errors="coerce").fillna(0.0))
        .groupby(["__Area","Decision_tuned"], as_index=False)[
            ["cost_baseline_per_year","cost_proposed_per_year","Risk_Reduction_USD"]
        ].sum()
        .copy())
roll["Net_Cost_Delta"] = roll["cost_proposed_per_year"] - roll["cost_baseline_per_year"]

st.write("Rollup (tuned decisions):")
st.dataframe(format_money_styler(roll), use_container_width=True)

# Detailed activities table
view_cols = [c for c in [
    "__Area","__AssetID_norm","Activity","Asset Type",
    "baseline_per_year","proposed_per_year","PM_hours","PM_cost_each",
    "cost_baseline_per_year","cost_proposed_per_year","cost_delta_per_year",
    "CoF_USD_app","PoF_app","Delta_PoF_app","RRV_USD_app","Decision_tuned"
] if c in acts_calc.columns]

acts_df = (
    acts_calc[view_cols]
    .sort_values(["Decision_tuned","RRV_USD_app"], ascending=[True, False])
    .head(2000)
    .copy()
)

st.dataframe(format_money_styler(acts_df), use_container_width=True)


# ==================== Downloads ====================
st.download_button("⬇️ Download assets (filtered)",
                   data=assets_view.to_csv(index=False).encode("utf-8-sig"),
                   file_name="assets_filtered.csv", mime="text/csv")
st.download_button("⬇️ Download activities (tuned)",
                   data=acts_calc[view_cols].to_csv(index=False).encode("utf-8-sig"),
                   file_name="activities_tuned.csv", mime="text/csv")


# ==================== Help / Explainer ====================
st.divider()
st.header("What this tool is doing — quick explainer")

# (Optional) live status about proposed_per_year
status_bits = []
if "proposed_per_year" in acts_calc.columns:
    prop_sum = pd.to_numeric(acts_calc["proposed_per_year"], errors="coerce").fillna(0).sum()
    if prop_sum > 0:
        status_bits.append("**Proposed data:** found")
    else:
        status_bits.append("**Proposed data:** present but zeros → falls back to baseline")
else:
    status_bits.append("**Proposed data:** not found → using baseline as proposed")

st.caption(" • ".join(status_bits))

with st.expander("What this thing does (in 60 seconds)", expanded=False):
    st.markdown("""
You already have a **baseline PM program** (what you do now) and a **proposed PM program** (what you might do next year).

The tool estimates:

- **Probability of Failure (PoF)** for each asset (how likely it is to fail).
- **Consequence of Failure (CoF)** in dollars (what it costs when it fails).
- **Annual Risk ($) = PoF × CoF.**

For each maintenance activity, it compares baseline vs proposed:

- **Cost change** (how much more/less you’d spend).
- **Risk Reduction Value (RRV)** — the dollar value of risk you reduce by doing more/better PM.

It then decides **INCREASE / KEEP_BASELINE / REDUCE** using an ROI-style rule.
""")

with st.expander("What’s under the hood (light version)", expanded=False):
    st.markdown("""
**The Jupyter pipeline (offline)**

- **Read & clean:** Pulls asset lists, PM frequencies, CMMS work history; fixes messy headers/IDs.
- **Match activities to tasks:** Uses text matching to tie your program activities to real CMMS task text (with safety checks).
- **Baseline rate:** Counts completed tasks in your history window → `baseline_per_year`.
- **Proposed rate:** Reads the adjusted program file → `proposed_per_year`.
- **PoF:** Combines condition/age/compliance/risk and any rehab/replace timing.
- **CoF:** Composes downtime cost + penalty terms (+ optional CM event cost).
- **Outputs CSVs** the Streamlit app reads:
  - Assets ranked by risk
  - Activities with baseline/proposed, costs, and decisions

**The Streamlit app (interactive)**

- Loads those CSVs.
- Lets you turn **knobs** and instantly recompute costs, risk, ROI, and decisions.
- Shows a **portfolio view** (assets by risk) and an **activities view** (what to increase/keep/reduce), plus **area rollups**.
""")

with st.expander("Key terms (no jargon)", expanded=False):
    st.markdown("""
- **PoF:** Likelihood an asset fails within a period (0–1). Higher = riskier.
- **CoF ($):** Dollar impact when it fails (lost production, compliance penalties, etc.).
- **Annual Risk ($):** `PoF × CoF`. Your risk exposure in money terms.
- **`baseline_per_year`:** How many times you currently do the activity each year (from history).
- **`proposed_per_year`:** How many times you plan to do the activity (from the adjusted program).
- **`PM_cost_each` ($):** Cost for one activity execution (labor/materials).
- **RRV ($):** Risk Reduction Value = (change in PoF from the activity) × CoF.

**Decision rule**
- **INCREASE:** Extra PM is worth it (`RRV ≥ k × ΔCost`).
- **KEEP_BASELINE:** Proposal didn’t clear the bar.
- **REDUCE:** Savings outweigh added risk.
""")

with st.expander("The knobs (what happens when you move them)", expanded=False):
    st.markdown("""
**Filters**
- **Area(s):** Just limits what’s on screen. No math changes.

**CoF knobs**
- **Downtime $/hr ↑** → CoF ↑ → Annual Risk ↑ → RRV ↑ → more **INCREASE** decisions.
- **Penalty unit $ ↑** (env/safety penalties) → same effect as above.
- **Replacement-cost factor ↑** → CoF ↑ (if included) → more **INCREASE**.

**PM cost knobs**
- **Labor rate $/hr** or **Material adder ↑** → `PM_cost_each` ↑ → ΔCost ↑ → ROI falls → fewer **INCREASE** / more **KEEP_BASELINE**.

**Decision line**
- **k (approval factor):** Higher k = tougher bar (RRV must be much bigger than ΔCost) → fewer **INCREASE**. Lower k → more **INCREASE**.

**PoF recompute (advanced)**
- Toggle **Recompute PoF** to use the app’s live model (age, condition, compliance, prior risk, rehab urgency).
- **Weights** (`w_age`, `w_cond`, `w_noncomp`, `w_prior`, `w_rehab`) shift what drives PoF.
  - Example: raise `w_cond` if inspections are trustworthy; raise `w_rehab` if end-of-life is near.
- **PoF steepness (k) / bias:** Shape PoF’s S-curve.
- **Rehab horizon (yrs):** Shorter horizon makes soon-to-be-replaced assets look urgent → higher PoF now.
""")

with st.expander("How to read the views", expanded=False):
    st.markdown("""
**1) Assets — ranked by annual risk**
- Sorts assets by Annual Risk ($) using your current knob settings.
- Use this to see where the risk lives before choosing PM changes.

**2) Activities — decisions & ROI**
- Each row = one activity on an asset.
- Shows baseline vs proposed cost, RRV, and `Decision_tuned`.
- If you see **KEEP_BASELINE** with a large proposed cost: it means you requested an increase, but it didn’t pass the ROI bar. Approved plan stays at baseline.

**3) Area rollups**
- Sums baseline cost, proposed cost, risk reduction, and net cost delta by area & decision.
- (Optional improvement) Switch to **approved rollup** (only INCREASE/REDUCE that passed the gate) so budget impact reflects what you’d actually do.
""")

with st.expander("A practical workflow (5–10 mins)", expanded=False):
    st.markdown("""
1. **Scan assets by risk** to see hotspots.
2. **Set CoF knobs** to reflect reality (downtime $/hr, penalties). Start with defaults; sensitivity-test ±25%.
3. **Tune the decision line:** start **k = 1.0** (break-even). Raise to **1.2–1.5** for a stricter budget; lower to **0.8–0.9** to be aggressive.
4. **Check activities:** 
   - **INCREASE** buckets with high ROI → good candidates to fund.
   - **REDUCE** buckets → savings opportunities with small modeled risk upsides.
5. *(Optional)* **Recompute PoF** if you think CSV PoF is off. Adjust weights until the top-risk assets match field intuition.
6. **Export** the tuned tables for review/budget.
""")

with st.expander("Common gotchas (and what they mean)", expanded=False):
    st.markdown("""
- **Costs identical baseline vs proposed:** your activity CSV is missing `proposed_per_year`; the app uses baseline as fallback. Re-run the notebook to write `proposed_per_year` (it pulls `__per_year_adj` from the adjusted expected file).
- **Lots of KEEP_BASELINE:** either **k** is high, PM costs are high, or CoF is too low. Try lowering **k**, revisiting CoF, or confirming `PM_cost_each`.
- **Very high risk in one process:** penalty/downtime settings or rehab horizon might be aggressive — double-check those assumptions.
""")

with st.expander("One-pager cheat sheet", expanded=False):
    st.markdown("""
- **Annual Risk = PoF × CoF** (both set by you or the CSV).
- **RRV = ΔPoF × CoF** (benefit of more/less PM).
- **Decision rule:** If `RRV ≥ k × ΔCost` → **INCREASE**; if saving and risk bump is small → **REDUCE**; else **KEEP_BASELINE**.
- **Raising CoF → more INCREASE.**  
- **Raising PM cost or k → fewer INCREASE.**  
- **Shorter rehab horizon / higher rehab weight → higher PoF now.**
""")






