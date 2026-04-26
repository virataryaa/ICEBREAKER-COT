"""
cot_backfill.py — COT backfill from ICE Connect
================================================
Produces two parquets:

  cot_cit.parquet   (CIT / CFTC supplemental — US softs)
    Commodity, Date, Comm Long, Comm Short, Spec Long, Spec Short,
    Spec Spread, Index Long, Index Short, Non Rep Long, Non Rep Short,
    Total OI, Px

  cot_disagg.parquet  (Disaggregated — ICE London)
    Commodity, Date, Swap Long, Swap Short, Swap Spread,
    MM Long, MM Short, MM Spread,
    Other Long, Other Short, Non Rep Long, Non Rep Short,
    Total OI, Px

Usage:
    python cot_backfill.py                        # incremental both
    python cot_backfill.py --full                 # full rebuild both
    python cot_backfill.py --full --commodity KC  # single CIT commodity
    python cot_backfill.py --full --commodity RC  # single disagg commodity
    python cot_backfill.py --cit                  # CIT only
    python cot_backfill.py --disagg               # disagg only
"""

import argparse
import icepython as ice
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

DB_DIR       = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT\Database")
CIT_PATH     = DB_DIR / "cot_cit.parquet"
DISAGG_PATH  = DB_DIR / "cot_disagg.parquet"
ROLLEX_SRC   = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\Rollex\Database")
ROLLEX_DEST  = DB_DIR / "Rollex"
ROLLEX_COMMS = ["KC", "RC", "CC", "SB", "CT", "LCC", "LSU"]
START_DATE   = "2010-01-01"

# ── CIT config ────────────────────────────────────────────────────────────────
CIT_FIELDS = [
    'Open Interest All Close',
    'Comm Positions Long All Nocit Close',
    'Comm Positions Short All Nocit Close',
    'Ncomm Positions Long All Nocit Close',
    'Ncomm Positions Short All Nocit Close',
    'Ncomm Positions Spread All Nocit Close',
    'Cit Positions Long All Close',
    'Cit Positions Short All Close',
]

CIT_COL_MAP = {
    'Open Interest All Close':               'Total OI',
    'Comm Positions Long All Nocit Close':   'Comm Long',
    'Comm Positions Short All Nocit Close':  'Comm Short',
    'Ncomm Positions Long All Nocit Close':  'Spec Long',
    'Ncomm Positions Short All Nocit Close': 'Spec Short',
    'Ncomm Positions Spread All Nocit Close':'Spec Spread',
    'Cit Positions Long All Close':          'Index Long',
    'Cit Positions Short All Close':         'Index Short',
}

CIT_FINAL_COLS = ['Commodity', 'Date', 'Comm Long', 'Comm Short',
                  'Spec Long', 'Spec Short', 'Spec Spread',
                  'Index Long', 'Index Short',
                  'Non Rep Long', 'Non Rep Short', 'Total OI', 'Px']

CIT_INT_COLS = ['Comm Long', 'Comm Short', 'Spec Long', 'Spec Short', 'Spec Spread',
                'Index Long', 'Index Short', 'Non Rep Long', 'Non Rep Short', 'Total OI']

CIT_COMMODITIES = {
    "KC": {"cot_sym": "KC #COMB-CFTC", "px_sym": "%KC 1!"},
    "CC": {"cot_sym": "CC #COMB-CFTC", "px_sym": "%CC 1!"},
    "SB": {"cot_sym": "SB #COMB-CFTC", "px_sym": "%SB 1!"},
    "CT": {"cot_sym": "CT #COMB-CFTC", "px_sym": "%CT 1!"},
}

# ── Disagg config ─────────────────────────────────────────────────────────────
DISAGG_FIELDS = [
    'Open Interest All Close',
    'Prod Merc Positions Long All Close',
    'Prod Merc Positions Short All Close',
    'Swap Positions Long All Close',
    'Swap Positions Short All Close',
    'Swap Positions Spread All Close',
    'M Money Positions Long All Close',
    'M Money Positions Short All Close',
    'M Money Positions Spread All Close',
    'Other Rept Positions Long All Close',
    'Other Rept Positions Short All Close',
    'Nonrept Positions Long All Close',
    'Nonrept Positions Short All Close',
]

DISAGG_COL_MAP = {
    'Open Interest All Close':              'Total OI',
    'Prod Merc Positions Long All Close':   'Comm Long',
    'Prod Merc Positions Short All Close':  'Comm Short',
    'Swap Positions Long All Close':        'Swap Long',
    'Swap Positions Short All Close':       'Swap Short',
    'Swap Positions Spread All Close':      'Swap Spread',
    'M Money Positions Long All Close':     'MM Long',
    'M Money Positions Short All Close':    'MM Short',
    'M Money Positions Spread All Close':   'MM Spread',
    'Other Rept Positions Long All Close':  'Other Long',
    'Other Rept Positions Short All Close': 'Other Short',
    'Nonrept Positions Long All Close':     'Non Rep Long',
    'Nonrept Positions Short All Close':    'Non Rep Short',
}

DISAGG_FINAL_COLS = ['Commodity', 'Date',
                     'Comm Long', 'Comm Short',
                     'Swap Long', 'Swap Short', 'Swap Spread',
                     'MM Long', 'MM Short', 'MM Spread',
                     'Other Long', 'Other Short',
                     'Non Rep Long', 'Non Rep Short', 'Total OI', 'Px']

DISAGG_INT_COLS = ['Comm Long', 'Comm Short',
                   'Swap Long', 'Swap Short', 'Swap Spread',
                   'MM Long', 'MM Short', 'MM Spread',
                   'Other Long', 'Other Short',
                   'Non Rep Long', 'Non Rep Short', 'Total OI']

DISAGG_COMMODITIES = {
    "RC":  {"cot_sym": "RC.ICE #COMB-CFTC", "px_sym": "%RC 1!-ICE"},
    "LCC": {"cot_sym": "C.ICE #COMB-CFTC",  "px_sym": "%C 1!-ICE"},
}


# ── Shared helpers ────────────────────────────────────────────────────────────
def fetch_timeseries(symbol, fields, start, end):
    try:
        data = ice.get_timeseries(symbol, fields, granularity='D',
                                  start_date=start, end_date=end)
        df = pd.DataFrame(list(data))
        if df.empty or 'Error' in str(df.iloc[0, 0]):
            return None
        df.columns = ['Date'] + fields
        df = df[1:].reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        for f in fields:
            df[f] = pd.to_numeric(df[f], errors='coerce')
        return df.set_index('Date')
    except Exception as e:
        print(f"  ERROR fetching {symbol}: {e}")
        return None


def fetch_px(symbol, start, end):
    try:
        data = ice.get_timeseries(symbol, ['Settle'], granularity='D',
                                  start_date=start, end_date=end)
        df = pd.DataFrame(list(data))
        if df.empty or 'Error' in str(df.iloc[0, 0]):
            return pd.Series(dtype=float)
        df.columns = ['Date', 'Px']
        df = df[1:].reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Px'] = pd.to_numeric(df['Px'], errors='coerce')
        return df.set_index('Date')['Px']
    except Exception as e:
        print(f"  ERROR fetching price {symbol}: {e}")
        return pd.Series(dtype=float)


def attach_px(df, px_sym, start, end):
    print(f"  Fetching Px:  {px_sym}...", end=" ", flush=True)
    px = fetch_px(px_sym, start, end)
    print(f"{len(px)} rows")
    px_filled = px.reindex(px.index.union(df.index)).ffill()
    df['Px'] = px_filled.reindex(df.index)
    return df


def upsert(db_path, new_data, fetch_start):
    if db_path.exists():
        existing = pd.read_parquet(db_path)
        existing['Date'] = pd.to_datetime(existing['Date'])
        mask = ~(
            existing['Commodity'].isin(new_data['Commodity'].unique()) &
            (existing['Date'] >= pd.Timestamp(fetch_start))
        )
        final = pd.concat([existing[mask], new_data], ignore_index=True)
    else:
        final = new_data
    final = final.sort_values(['Commodity', 'Date']).reset_index(drop=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(db_path, index=False)
    return final


def incremental_start(db_path):
    existing = pd.read_parquet(db_path, columns=['Date'])
    existing['Date'] = pd.to_datetime(existing['Date'])
    latest = existing['Date'].max()
    return (latest - pd.Timedelta(days=30)).strftime('%Y-%m-%d')


# ── CIT builder ───────────────────────────────────────────────────────────────
def build_cit(comm, cfg, start, end):
    print(f"  Fetching COT: {cfg['cot_sym']}...", end=" ", flush=True)
    df = fetch_timeseries(cfg['cot_sym'], CIT_FIELDS, start, end)
    if df is None or df.empty:
        print("no data"); return None
    print(f"{len(df)} rows")

    df = df.rename(columns=CIT_COL_MAP)
    # Non-reportable = residual after accounting for all reportable groups
    df['Non Rep Long']  = df['Total OI'] - df['Comm Long']  - df['Spec Long']  - df['Spec Spread'] - df['Index Long']
    df['Non Rep Short'] = df['Total OI'] - df['Comm Short'] - df['Spec Short'] - df['Spec Spread'] - df['Index Short']

    df = attach_px(df, cfg['px_sym'], start, end)
    df['Commodity'] = comm
    df = df.reset_index()
    df = df[[c for c in CIT_FINAL_COLS if c in df.columns]]

    for c in CIT_INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    df['Px'] = pd.to_numeric(df['Px'], errors='coerce').astype('Float64')
    return df


# ── Disagg builder ────────────────────────────────────────────────────────────
def build_disagg(comm, cfg, start, end):
    print(f"  Fetching COT: {cfg['cot_sym']}...", end=" ", flush=True)
    df = fetch_timeseries(cfg['cot_sym'], DISAGG_FIELDS, start, end)
    if df is None or df.empty:
        print("no data"); return None
    print(f"{len(df)} rows")

    df = df.rename(columns=DISAGG_COL_MAP)
    df = attach_px(df, cfg['px_sym'], start, end)
    df['Commodity'] = comm
    df = df.reset_index()
    df = df[[c for c in DISAGG_FINAL_COLS if c in df.columns]]

    for c in DISAGG_INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    df['Px'] = pd.to_numeric(df['Px'], errors='coerce').astype('Float64')
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--full",      action="store_true")
parser.add_argument("--cit",       action="store_true", help="CIT only")
parser.add_argument("--disagg",    action="store_true", help="Disagg only")
parser.add_argument("--commodity", type=str, default=None)
args = parser.parse_args()

run_cit    = not args.disagg
run_disagg = not args.cit
END_DATE   = datetime.today().strftime('%Y-%m-%d')

# Filter by --commodity if given
if args.commodity:
    comm = args.commodity.upper()
    if comm in CIT_COMMODITIES:
        run_disagg = False
        CIT_COMMODITIES = {comm: CIT_COMMODITIES[comm]}
    elif comm in DISAGG_COMMODITIES:
        run_cit = False
        DISAGG_COMMODITIES = {comm: DISAGG_COMMODITIES[comm]}
    else:
        print(f"Unknown commodity: {comm}"); exit(1)

# ── CIT run ───────────────────────────────────────────────────────────────────
if run_cit:
    if not args.full and CIT_PATH.exists():
        fetch_start = incremental_start(CIT_PATH)
        print(f"CIT  | INCREMENTAL from {fetch_start}\n")
    else:
        fetch_start = START_DATE
        print(f"CIT  | FULL from {fetch_start}\n")

    all_cit = []
    for comm, cfg in CIT_COMMODITIES.items():
        print(f"-- {comm} --")
        df = build_cit(comm, cfg, fetch_start, END_DATE)
        if df is not None:
            all_cit.append(df)

    if all_cit:
        new_cit = pd.concat(all_cit, ignore_index=True)
        final_cit = upsert(CIT_PATH, new_cit, fetch_start)
        print(f"\nCIT saved -> {CIT_PATH}  |  {len(final_cit)} rows")
        print(final_cit.groupby('Commodity').agg(
            rows=('Date','count'), from_=('Date','min'), to=('Date','max')).to_string())
    else:
        print("CIT: no data retrieved.")

# ── Disagg run ────────────────────────────────────────────────────────────────
if run_disagg:
    print()
    if not args.full and DISAGG_PATH.exists():
        fetch_start = incremental_start(DISAGG_PATH)
        print(f"DISAGG | INCREMENTAL from {fetch_start}\n")
    else:
        fetch_start = START_DATE
        print(f"DISAGG | FULL from {fetch_start}\n")

    all_disagg = []
    for comm, cfg in DISAGG_COMMODITIES.items():
        print(f"-- {comm} --")
        df = build_disagg(comm, cfg, fetch_start, END_DATE)
        if df is not None:
            all_disagg.append(df)

    if all_disagg:
        new_disagg = pd.concat(all_disagg, ignore_index=True)
        final_disagg = upsert(DISAGG_PATH, new_disagg, fetch_start)
        print(f"\nDisagg saved -> {DISAGG_PATH}  |  {len(final_disagg)} rows")
        print(final_disagg.groupby('Commodity').agg(
            rows=('Date','count'), from_=('Date','min'), to=('Date','max')).to_string())
    else:
        print("Disagg: no data retrieved.")

# ── Rollex sync ───────────────────────────────────────────────────────────────
print("\nSyncing Rollex parquets...")
ROLLEX_DEST.mkdir(parents=True, exist_ok=True)
for comm in ROLLEX_COMMS:
    src = ROLLEX_SRC / f"rollex_{comm}.parquet"
    dst = ROLLEX_DEST / f"rollex_{comm}.parquet"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Copied rollex_{comm}.parquet")
    else:
        print(f"  MISSING: rollex_{comm}.parquet (skipped)")
print("Rollex sync done.")
