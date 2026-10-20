# src/sp500.py
import re, argparse, sys, unicodedata

# Known class/format quirks for Yahoo Finance
MANUAL_FIX = {
    # common class-dot → dash
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    # mistakes / brand names that aren't tickers in your list
    "XYZ": "SQ",        # Block, Inc. (formerly Square)
    # Optional: legacy/alt mappings if needed in the future
    # "PARA": "PARA",   # (Paramount Global, if your list used older name)
}

def normalize_ticker(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).strip().upper()
    # Replace “.” with “-” for Yahoo (class shares)
    t = t.replace(".", "-")
    return MANUAL_FIX.get(t, t)

def parse_lines(path_in: str, path_out: str):
    tickers = []
    patt = re.compile(r"\(([^)]+)\)\s*$")  # last (...) on the line
    with open(path_in, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line: 
                continue
            m = patt.search(line)
            if not m:
                # if a bare ticker line slips in, accept words without spaces
                parts = line.split()
                guess = parts[-1].upper().replace(".", "-")
                if guess.isalpha() or "-" in guess:
                    tickers.append(normalize_ticker(guess))
                continue
            t = m.group(1).strip()
            tickers.append(normalize_ticker(t))
    # de-dup while preserving order
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    with open(path_out, "w", encoding="utf-8") as f:
        for t in uniq:
            f.write(t + "\n")
    print(f"Wrote {len(uniq)} tickers → {path_out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path_in", required=True, help="Path to file containing your long company list")
    ap.add_argument("--out", dest="path_out", required=True, help="Output ticker file (one per line)")
    args = ap.parse_args()
    parse_lines(args.path_in, args.path_out)

if __name__ == "__main__":
    main()