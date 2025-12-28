#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jeff/Codes/Robots"
MANIFEST="$ROOT_DIR/output/offline_package_manifest.txt"
OUTDIR="$ROOT_DIR/output/offline_pkgs"
ARCHIVE="$ROOT_DIR/output/offline_pkgs_archive.tar.gz"
LOG="$ROOT_DIR/output/offline_download.log"

mkdir -p "$OUTDIR"
rm -f "$LOG"

echo "Starting offline package download" | tee -a "$LOG"

# Read manifest lines: url\tfn\tsize\tmd5\tsha256
# Skip header if present
while IFS=$'\t' read -r url fn size md5 sha256; do
    # skip comment/header lines
    if [[ "$url" == "#url" ]] || [[ "$url" == "" ]] || [[ "$url" =~ ^# ]]; then
        continue
    fi

    target="$OUTDIR/$fn"
    if [[ -f "$target" ]]; then
        echo "Already exists: $fn, verifying checksum..." | tee -a "$LOG"
        if [[ -n "$sha256" ]] && command -v sha256sum >/dev/null 2>&1; then
            echo "$sha256  $target" > "$OUTDIR/$fn.sha256"
            if sha256sum -c "$OUTDIR/$fn.sha256" >/dev/null 2>&1; then
                echo "Checksum OK for $fn" | tee -a "$LOG"
                rm -f "$OUTDIR/$fn.sha256"
                continue
            else
                echo "Checksum mismatch for existing $fn, will re-download" | tee -a "$LOG"
                rm -f "$target"
            fi
        else
            echo "No checksum available or sha256sum not found; skipping verification for $fn" | tee -a "$LOG"
            continue
        fi
    fi

    echo "Downloading: $fn from $url" | tee -a "$LOG"
    # Use curl with resume and retries
    curl -L --retry 5 --retry-delay 5 -C - -o "$target" "$url" 2>>"$LOG"

    # Verify sha256 if provided
    if [[ -n "$sha256" ]] && command -v sha256sum >/dev/null 2>&1; then
        echo "$sha256  $target" > "$OUTDIR/$fn.sha256"
        if ! sha256sum -c "$OUTDIR/$fn.sha256" >/dev/null 2>>"$LOG"; then
            echo "Checksum failed for $fn after download" | tee -a "$LOG"
            exit 2
        fi
        rm -f "$OUTDIR/$fn.sha256"
    else
        echo "No sha256 to verify for $fn" | tee -a "$LOG"
    fi

done < "$MANIFEST"

# Create archive
echo "Creating archive $ARCHIVE" | tee -a "$LOG"
cd "$OUTDIR"
tar -czf "$ARCHIVE" .

# Report size
du -h "$ARCHIVE" | tee -a "$LOG"

echo "Done. Archive at: $ARCHIVE" | tee -a "$LOG"
