#!/usr/bin/env python3
"""FN27 tables and figures from qc_per_second_10days.csv + manual review labels."""
import os, sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, cv2, glob
O = sys.argv[1]; T = f'{O}/tables'; FG = f'{O}/figures'; os.makedirs(FG, exist_ok=True)
d = pd.read_csv(f'{T}/qc_per_second_10days.csv')
d['hands'] = d[['hands_full', 'hands_lower']].max(1)
d['exposure_ok'] = d.brightness.between(40, 200) & (d.sat_frac < 0.10) & (d.dark_frac < 0.20)
d['blur_ok'] = d.blur_px_p90 <= 8; d['view_ok'] = d.view_elevation_deg <= -30; d['hand_found'] = d.hands >= 1
d['image_usable'] = d.exposure_ok & d.blur_ok; d['usable'] = d.image_usable & d.view_ok & d.hand_found
d['hands_on'] = np.where(d.date < '2026-11-08', 'work gloves', 'bare hands')
agg = dict(seconds=('second', 'size'), exposure_ok=('exposure_ok', 'mean'), blur_ok=('blur_ok', 'mean'), view_ok=('view_ok', 'mean'),
           hand_found=('hand_found', 'mean'), image_usable=('image_usable', 'mean'), usable=('usable', 'mean'),
           blur_px_median=('blur_px_median', 'median'), head_speed_dps=('head_speed_dps_median', 'median'), view_elev_deg=('view_elevation_deg', 'median'))
D = d.groupby(['date', 'hands_on']).agg(**agg).reset_index(); D['recorded_h'] = D.seconds / 3600; D['usable_h'] = D.recorded_h * D.usable
G = d.groupby('hands_on').agg(**agg).reset_index(); G['recorded_h'] = G.seconds / 3600; G['usable_h'] = G.recorded_h * G.usable
A = d.agg({k: v[1] for k, v in agg.items() if v[1] == 'mean'});
for X in (D, G):
    for c in ['exposure_ok', 'blur_ok', 'view_ok', 'hand_found', 'image_usable', 'usable']: X[c] = (100 * X[c]).round(1)
    for c in ['recorded_h', 'usable_h']: X[c] = X[c].round(2)
D.to_csv(f'{T}/qc_by_day.csv', index=False); G.to_csv(f'{T}/qc_gloves_vs_bare.csv', index=False)
# manual review: 30 random image-OK, looking-down seconds per group; visible = worker's own hand at least partly visible (labelled by eye)
VIS = {'work gloves': [1,1,1,0,1,0,0,1,0,1,1,1,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,1,1],
       'bare hands':  [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,0,1,1,1,1]}
rows = []
for g, f in (('work gloves', 'glove'), ('bare hands', 'bare')):
    r = pd.read_csv(f'{O}/review_{f}.csv', index_col=0); r['visible'] = VIS[g]; r['detected'] = (r.hands > 0).astype(int); r['hands_on'] = g; rows.append(r)
R = pd.concat(rows); R.to_csv(f'{T}/manual_review_60_seconds.csv', index=False)
RV = R.groupby('hands_on').apply(lambda x: pd.Series(dict(reviewed=len(x), hand_visible=int(x.visible.sum()), detected=int(x.detected.sum()),
        detected_when_visible=int((x.visible & x.detected).sum()), false_detections=int(((1 - x.visible) & x.detected).sum())))).reset_index()
RV['recall_pct'] = (100 * RV.detected_when_visible / RV.hand_visible).round(0); RV.to_csv(f'{T}/manual_review_summary.csv', index=False)
print(G.to_string()); print(RV.to_string()); print('ALL', (100 * A).round(1).to_dict(), round(d.usable.sum() / 3600, 2), 'h of', round(len(d) / 3600, 2))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
C = {'work gloves': '#f59e0b', 'bare hands': '#2563eb'}
# fig 1: recorded vs usable hours per day
fig, ax = plt.subplots(figsize=(11, 4.6)); x = np.arange(len(D))
ax.bar(x, D.recorded_h, color='#e2e8f0', label='recorded'); ax.bar(x, D.usable_h, color=[C[h] for h in D.hands_on])
for i, r in D.iterrows(): ax.text(i, r.recorded_h + .04, f'{r.usable:.0f}%', ha='center', fontsize=10)
ax.set_xticks(x, [s[5:] for s in D.date]); ax.set_ylabel('hours'); ax.set_xlabel('recording date (2026-11)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#e2e8f0', label='recorded'), Patch(color=C['work gloves'], label='usable, work gloves (11-01 to 11-05)'), Patch(color=C['bare hands'], label='usable, bare hands (11-08 to 11-12)')], frameon=False, fontsize=9, loc='upper left')
ax.set_title(f'10 working days, {len(d)/3600:.1f} h recorded, every second checked: usable for manipulation each day'); ax.set_ylim(0, 3.1); ax.grid(axis='y', alpha=.3)
fig.tight_layout(); fig.savefig(f'{FG}/fn27_usable_by_day.png', dpi=150); plt.close(fig)
# fig 2: check pass rates per group
fig, ax = plt.subplots(figsize=(9, 4.4)); labs = ['exposure OK', 'camera blur ≤ 8 px', 'looking down at work', 'hand found by detector', 'all checks (usable)']; cols = ['exposure_ok', 'blur_ok', 'view_ok', 'hand_found', 'usable']
w = .38; xx = np.arange(len(cols))
for k, g in enumerate(['work gloves', 'bare hands']):
    v = G[G.hands_on == g][cols].values[0]; b = ax.bar(xx + (k - .5) * w, v, w, color=C[g], label=g)
    for xi, vi in zip(xx + (k - .5) * w, v): ax.text(xi, vi + 1, f'{vi:.0f}', ha='center', fontsize=9)
ax.set_xticks(xx, labs, fontsize=9); ax.set_ylabel('% of recorded seconds'); ax.set_ylim(0, 110); ax.legend(frameon=False); ax.grid(axis='y', alpha=.3)
ax.set_title('Same site, rig and detector: only the hand check moves'); fig.tight_layout(); fig.savefig(f'{FG}/fn27_checks_gloves_vs_bare.png', dpi=150); plt.close(fig)
# fig 3: manual review bars
fig, ax = plt.subplots(figsize=(7.5, 4))
for k, g in enumerate(['work gloves', 'bare hands']):
    r = RV[RV.hands_on == g].iloc[0]
    ax.bar(k * 2.2, 100 * r.hand_visible / r.reviewed, .9, color='#cbd5e1'); ax.text(k * 2.2, 100 * r.hand_visible / r.reviewed + 2, f'{r.hand_visible}/{r.reviewed}', ha='center')
    ax.bar(k * 2.2 + 1, r.recall_pct, .9, color=C[g]); ax.text(k * 2.2 + 1, r.recall_pct + 2, f'{r.detected_when_visible}/{r.hand_visible}', ha='center')
ax.set_xticks([0, 1, 2.2, 3.2], ['hand visible\n(gloves)', 'detector found it\n(gloves)', 'hand visible\n(bare)', 'detector found it\n(bare)'], fontsize=9)
ax.set_ylabel('%'); ax.set_ylim(0, 110); ax.set_title('60 seconds checked by eye: hands were there, the detector missed gloves'); ax.grid(axis='y', alpha=.3)
fig.tight_layout(); fig.savefig(f'{FG}/fn27_manual_review.png', dpi=150); plt.close(fig)
print('figs ok')
