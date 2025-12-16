import os
import argparse
import pandas as pd


ATTACK_ID_FOR_THIS_FILE = 7

ATTACK_TYPES = {
    0: "Genuine",
    1: "ConstPos",
    2: "ConstPosOffset",
    3: "RandomPos",
    4: "RandomPosOffset",
    5: "ConstSpeed",
    6: "ConstSpeedOffset",
    7: "RandomSpeed",
    8: "RandomSpeedOffset",
    9: "EventualStop",
    10: "Disruptive",
    11: "DataReplay",
    12: "StaleMessages",
    13: "DoS",
    14: "DoSRandom",
    15: "DoSDisruptive",
    16: "GridSybil",
    17: "DataReplaySybil",
    18: "DoSRandomSybil",
    19: "DoSDisruptiveSybil",
}


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned = []
    for c in df.columns:
        c2 = str(c).replace("\ufeff", "").strip()
        if len(c2) >= 2 and c2[0] == c2[-1] and c2[0] in ("'", '"'):
            c2 = c2[1:-1].strip()
        cleaned.append(c2)
    df.columns = cleaned
    return df


def insert_attack_id(df: pd.DataFrame, attack_id_for_ones: int) -> pd.DataFrame:
    if "label" not in df.columns:
        raise ValueError("Missing required column 'label'.")


    lab = df["label"]
    try:
        lab = lab.astype(int)
    except Exception:
        lab = pd.to_numeric(lab, errors="coerce").fillna(0).astype(int)


    attack_id = (lab != 0).astype(int) * int(attack_id_for_ones)


    if "attack_id" in df.columns:
        df = df.drop(columns=["attack_id"])


    cols = list(df.columns)
    idx = cols.index("label") + 1
    df = df.copy()
    df.insert(idx, "attack_id", attack_id)


    if "mb_version" in df.columns:
        cols = list(df.columns)
        if cols.index("attack_id") > cols.index("mb_version"):
            cols.remove("attack_id")
            mv_idx = cols.index("mb_version")
            cols = cols[:mv_idx] + ["attack_id"] + cols[mv_idx:]
            df = df[cols]

    return df


def main():
    ap = argparse.ArgumentParser(description="Add attack_id column after label.")
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument(
        "--output",
        help="Path to output CSV (if omitted and --inplace not set, <name>_with_attackid.csv is used)",
    )
    ap.add_argument("--inplace", action="store_true", help="Overwrite the input file")
    ap.add_argument(
        "--attack-id",
        type=int,
        dest="attack_id_cli",
        help="Attack type ID (0-19) for rows with label==1. Overrides ATTACK_ID_FOR_THIS_FILE.",
    )
    args = ap.parse_args()

    in_path = args.input
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")


    if args.inplace:
        out_path = in_path
    else:
        out_path = args.output
        if not out_path:
            root, ext = os.path.splitext(in_path)
            out_path = f"{root}_with_attackid{ext or '.csv'}"

    attack_id_val = (
        args.attack_id_cli
        if args.attack_id_cli is not None
        else ATTACK_ID_FOR_THIS_FILE
    )
    if attack_id_val < 0:
        raise ValueError("attack_id must be >= 0.")


    try:
        df = pd.read_csv(in_path, encoding="utf-8-sig", engine="python")
    except Exception:
        df = pd.read_csv(in_path, engine="python")

    df = sanitize_columns(df)
    df_out = insert_attack_id(df, attack_id_val)
    df_out.to_csv(out_path, index=False, encoding="utf-8")

    total = len(df_out)
    n_label1 = int((df_out["label"].astype(int) == 1).sum())
    n_label0 = total - n_label1
    attack_name = ATTACK_TYPES.get(int(attack_id_val), "unknown")
    print(f"✅ Done. Saved: {out_path}")
    print(
        f"Rows: {total} | label=0: {n_label0} -> attack_id=0 | label=1: {n_label1} -> attack_id={attack_id_val} ({attack_name})"
    )


if __name__ == "__main__":
    main()
