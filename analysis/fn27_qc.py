#!/usr/bin/env python3
"""FN27: per-second QC index over 10 working days (same rules as Field Note #25), from fn27_pass2 outputs + IMU.
Exposure timing per second comes from that second's keyframe SEI (first frame of each second).
usage: python3 fn27_qc.py <out_dir>"""
import numpy as np, glob, os, csv, sys, re
ROOT = os.path.expanduser('~/mnt/ODL_work/fn27_qc_all'); SEG = os.path.expanduser('~/work/fn27/seg'); O = sys.argv[1]
os.makedirs(f'{O}/tables', exist_ok=True)
dt = None  # device-specific layout withheld
def imu(p):
    """Gyro/accel samples from the rig's .imu file (cached by fn27_imu.py)."""
    raise NotImplementedError('Origin Data Lab device-specific reader withheld; contact founder@origindatalab.io')
F = 482.3
rows = []
for day in sorted(os.listdir(ROOT)):
    sess = sorted(set(os.path.basename(p)[:15] for p in glob.glob(f'{ROOT}/{day}/*_left_raw.mp4')))
    for s in sess:
        ip = os.path.expanduser(f'~/work/fn27/imu/{s}.npz')
        if not os.path.exists(ip): continue
        zi = np.load(ip); gt, gw, at, A = zi['gt'], zi['gw'].astype(float), zi['at'], zi['A'].astype(float)
        if len(gt) == 0: continue
        for seg in sorted(glob.glob(f'{SEG}/{s}_*.npz')):
            z = np.load(seg); m, h, S = z['m'], z['h'], z['sei']; chunk = os.path.basename(seg)[16:21]
            for i in range(min(len(m), len(S))):
                t0, ex = int(S[i, 1]), int(S[i, 2])
                if t0 <= 0: continue
                t1 = t0 + 1_000_000
                i0, i1 = np.searchsorted(gt, [t0, t1]); w = gw[i0:i1]; blur = F * w * ex / 1e6
                j0, j1 = np.searchsorted(at, [t0, t1])
                if j1 > j0:
                    av = A[j0:j1].mean(0); elev = float(np.degrees(np.arcsin(np.clip(av[2] / np.linalg.norm(av), -1, 1))))
                else: elev = float('nan')
                rows.append(dict(date=day, session=s, chunk=chunk, second=i, brightness=round(float(m[i, 0]), 1), sat_frac=round(float(m[i, 1]), 4), dark_frac=round(float(m[i, 2]), 4),
                    exposure_us=ex, head_speed_dps_median=round(float(np.degrees(np.median(w))), 1) if len(w) else '',
                    blur_px_median=round(float(np.median(blur)), 2) if len(w) else '', blur_px_p90=round(float(np.percentile(blur, 90)), 2) if len(w) else '',
                    view_elevation_deg=round(elev, 1), hands_full=int(h[i, 0]), hands_lower=int(h[i, 1])))
with open(f'{O}/tables/qc_per_second_10days.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print(len(rows), 'seconds', round(len(rows) / 3600, 2), 'h')
