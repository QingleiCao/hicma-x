#!/bin/bash
# Run testing_potrf_ir cases and collect their IR_summary lines in one TSV.
#
# Usage: scripts/ir_sweep.sh <build dir> <out dir> <cases file> [launcher...]
#   cases file : one case per line, "<label> <testing_potrf_ir options...>";
#                blank lines and lines starting with # are skipped
#   launcher   : optional command put in front of the binary
#                (e.g. numactl --cpunodebind=0 --membind=0)
# Output: <out dir>/ir_results.tsv, one row per run and trial: the label, the
# exit code, the kernel parameters the driver printed (sigma, beta, nu, noise,
# adddiag) and every key of the IR_summary line; one log per case in
# <out dir>/logs. A case whose printed --sigma/--beta/--nu/--noise differs
# from the requested value is reported and marked param_mismatch=1.
set -u
B=${1:?build dir}; OUT=${2:?out dir}; CASES=${3:?cases file}; shift 3
EXE=$B/tests/testing_potrf_ir
[ -x "$EXE" ] || { echo "$EXE not found" >&2; exit 1; }
mkdir -p "$OUT/logs"
TSV=$OUT/ir_results.tsv
clean() { tr -d '\0' | sed 's/\x1b\[[0-9;]*m//g'; }

# value of --name in an option list ("" when absent)
optval() { local name=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "--$name" ] && { echo "${2:-}"; return; }; shift; done; }

while read -r label args; do
    case "$label" in ''|'#'*) continue ;; esac
    log=$OUT/logs/$label.log
    # shellcheck disable=SC2086
    "$@" "$EXE" $args > "$log" 2>&1
    rc=$?
    o=$(clean < "$log")
    banner=$(echo "$o" | grep -m1 -oE 'sigma=[^ ]+ beta=[^ ]+ nu=[^ ]+')
    sigma=$(echo "$banner" | sed -E 's/.*sigma=([^ ]+).*/\1/'); beta=$(echo "$banner" | sed -E 's/.*beta=([^ ]+).*/\1/')
    nu=$(echo "$banner" | sed -E 's/.*nu=([^ ]+).*/\1/')
    noise=$(echo "$o" | grep -m1 -oE 'noise= [^ ]+' | awk '{print $2}')
    adddiag=$(echo "$o" | grep -m1 -oE 'add_diag=[^ ]+' | cut -d= -f2)
    mismatch=0
    # shellcheck disable=SC2086
    for k in sigma beta nu noise; do
        want=$(optval $k $args); [ -z "$want" ] && continue
        got=${!k}
        # the driver prints these with %f (6 decimals): compare to that precision,
        # then keep the requested value, which has all its digits
        if [ -z "$got" ] || ! awk -v a="$want" -v b="$got" 'BEGIN { d = a - b; if (d < 0) d = -d; s = (a < 0 ? -a : a); exit !(d <= 5e-7 + 1e-6 * s) }'; then
            echo "WARNING $label: --$k $want requested, $got used" >&2; mismatch=1
        else
            printf -v "$k" '%s' "$want"
        fi
    done
    lines=$(echo "$o" | grep '^IR_summary ')
    if [ -z "$lines" ]; then
        echo "WARNING $label: no IR_summary (exit code $rc), see $log" >&2
        lines="IR_summary status= no_output"
    fi
    while read -r line; do
        keys=$(echo "$line" | sed 's/^IR_summary //' | grep -oE '[A-Za-z0-9_]+=' | tr -d '=' | tr '\n' '\t' | sed 's/\t$//')
        vals=$(echo "$line" | sed 's/^IR_summary //' | sed -E 's/[A-Za-z0-9_]+= ?//g' | tr -s ' ' '\t')
        [ -s "$TSV" ] || printf "label\trc\tsigma\tbeta\tnu\tnoise\tadddiag\tparam_mismatch\t%s\n" "$keys" > "$TSV"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$label" "$rc" "$sigma" "$beta" "$nu" "$noise" "$adddiag" "$mismatch" "$vals" >> "$TSV"
    done <<< "$lines"
    echo "$label rc=$rc $(echo "$lines" | head -1 | grep -oE 'status= [a-z_]+ steps= [0-9]+|berr= [^ ]+' | tr '\n' ' ')"
done < "$CASES"
