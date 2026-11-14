#!/usr/bin/env python3
"""FN27 per-file pass, keyframes only: reads the MP4 sample tables, fetches only the 1-per-second keyframes
(about 1/8 of the bytes), takes exposure timing from each keyframe's SEI, computes image metrics and hands
(full frame + lower work region, as in Field Notes #23/#25). Resumable.
usage: python3 fn27_pass2.py <budget_s> <worker> <nworkers>"""
import subprocess, sys, os, glob, time, re, struct, numpy as np, cv2, mediapipe as mp
ROOT = os.path.expanduser('~/mnt/ODL_work/fn27_qc_all'); O = os.path.expanduser('~/work/fn27/seg'); os.makedirs(O, exist_ok=True)
budget, wk, nw = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); t0 = time.time()
files = sorted(glob.glob(f'{ROOT}/*/*_left_raw.mp4'))
ALL = len(files); files = [f for f in files if not os.path.exists(f'{O}/{os.path.basename(f)[:21]}.npz')]

def boxes(b, off=0, end=None):
    end = len(b) if end is None else end
    while off + 8 <= end:
        sz, typ = struct.unpack('>I4s', b[off:off + 8]); h = 8
        if sz == 1: sz = struct.unpack('>Q', b[off + 8:off + 16])[0]; h = 16
        if sz == 0: sz = end - off
        yield typ.decode('latin1'), off + h, off + sz; off += sz
def find(b, path, off=0, end=None):
    for t, s, e in boxes(b, off, end):
        if t == path[0]:
            return (s, e) if len(path) == 1 else find(b, path[1:], s, e)
    return None
def moov_of(fh):
    off = 0; size = os.fstat(fh.fileno()).st_size
    while off < size:
        fh.seek(off); hd = fh.read(16); sz, typ = struct.unpack('>I4s', hd[:8])
        if sz == 1: sz = struct.unpack('>Q', hd[8:16])[0]
        if typ == b'moov': fh.seek(off); return fh.read(sz)
        off += sz
def keyframes(path):
    fh = open(path, 'rb'); mv = moov_of(fh)
    for t, s, e in boxes(mv, 8):
        if t != 'trak': continue
        stbl = find(mv, ['mdia', 'minf', 'stbl'], s, e)
        if not stbl: continue
        tab = {t2: (s2, e2) for t2, s2, e2 in boxes(mv, *stbl)}
        if 'stss' not in tab: continue
        s2, _ = tab['stsd']; avcc = mv.find(b'avcC', s2)
        cfg = mv[avcc + 4:]; nsps = cfg[5] & 31; p = 6; ps = []
        for _ in range(nsps): l = struct.unpack('>H', cfg[p:p + 2])[0]; ps.append(cfg[p + 2:p + 2 + l]); p += 2 + l
        npps = cfg[p]; p += 1
        for _ in range(npps): l = struct.unpack('>H', cfg[p:p + 2])[0]; ps.append(cfg[p + 2:p + 2 + l]); p += 2 + l
        s3, _ = tab['stsz']; ss, cnt = struct.unpack('>II', mv[s3 + 4:s3 + 12])
        sizes = np.full(cnt, ss, np.int64) if ss else np.frombuffer(mv[s3 + 12:s3 + 12 + 4 * cnt], '>u4').astype(np.int64)
        if 'stco' in tab:
            s4, _ = tab['stco']; nc = struct.unpack('>I', mv[s4 + 4:s4 + 8])[0]; co = np.frombuffer(mv[s4 + 8:s4 + 8 + 4 * nc], '>u4').astype(np.int64)
        else:
            s4, _ = tab['co64']; nc = struct.unpack('>I', mv[s4 + 4:s4 + 8])[0]; co = np.frombuffer(mv[s4 + 8:s4 + 8 + 8 * nc], '>u8').astype(np.int64)
        s5, _ = tab['stsc']; ne = struct.unpack('>I', mv[s5 + 4:s5 + 8])[0]; sc = np.frombuffer(mv[s5 + 8:s5 + 8 + 12 * ne], '>u4').reshape(-1, 3)
        offs = np.zeros(cnt, np.int64); si = 0
        for c in range(nc):
            k = np.searchsorted(sc[:, 0], c + 1, side='right') - 1; spc = int(sc[k, 1]); o = co[c]
            for _ in range(spc):
                if si >= cnt: break
                offs[si] = o; o += sizes[si]; si += 1
        s6, _ = tab['stss']; nk = struct.unpack('>I', mv[s6 + 4:s6 + 8])[0]; keys = np.frombuffer(mv[s6 + 8:s6 + 8 + 4 * nk], '>u4').astype(np.int64) - 1
        out = []
        for k in keys:
            fh.seek(offs[k]); data = fh.read(sizes[k]); p = 0; nal = b''
            while p + 4 <= len(data):
                l = struct.unpack('>I', data[p:p + 4])[0]; nal += b'\x00\x00\x00\x01' + data[p + 4:p + 4 + l]; p += 4 + l
            out.append((int(k), nal))
        hdr = b''.join(b'\x00\x00\x00\x01' + x for x in ps)
        return hdr, out

def sei_of(nal):
    """(pts, exposure_start_us, exposure_us) from the rig's per-frame SEI payload."""
    raise NotImplementedError('Origin Data Lab device-specific reader withheld; contact founder@origindatalab.io')

HF = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, model_complexity=1)
HL = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3, model_complexity=1)
done = 0
for k, src in enumerate(files):
    if k % nw != wk: continue
    sid = os.path.basename(src)[:21]; out = f'{O}/{sid}.npz'
    if os.path.exists(out): continue
    if time.time() - t0 > budget - 60: break
    r = keyframes(src) if os.path.getsize(src) > 1_000_000 else None
    if not r or not r[1]:
        np.savez_compressed(out, m=np.zeros((0, 4), np.float32), h=np.zeros((0, 2), np.int8), sei=np.zeros((0, 3), np.int64), kidx=np.zeros(0, np.int64), nkey=0); print(sid, 'skipped (empty/aborted file)'); continue
    hdr, kf = r
    S = np.array([sei_of(n) for _, n in kf], np.int64); idx = np.array([i for i, _ in kf], np.int64)
    tmp = f'/tmp/fn27k_{wk}.h264'; open(tmp, 'wb').write(b''.join(hdr + n for _, n in kf))
    W, H = 480, 360
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', tmp, '-vf', f'scale={W}:{H}:flags=area', '-vsync', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1'], capture_output=True)
    f = np.frombuffer(p.stdout, np.uint8); n = len(f) // (W * H * 3); f = f[:n * W * H * 3].reshape(n, H, W, 3)
    M = np.zeros((n, 4), np.float32); HC = np.zeros((n, 2), np.int8)
    for i in range(n):
        g = cv2.cvtColor(f[i], cv2.COLOR_BGR2GRAY); c = g[45:315, 60:420]
        M[i] = (g.mean(), (g >= 250).mean(), (g <= 12).mean(), cv2.Laplacian(c, cv2.CV_32F).var())
        r1 = HF.process(cv2.cvtColor(f[i], cv2.COLOR_BGR2RGB)); HC[i, 0] = len(r1.multi_hand_landmarks or [])
        lo = cv2.resize(f[i][135:360, 60:420], (800, 500), interpolation=cv2.INTER_CUBIC)
        r2 = HL.process(cv2.cvtColor(lo, cv2.COLOR_BGR2RGB)); HC[i, 1] = len(r2.multi_hand_landmarks or [])
    np.savez_compressed(out, m=M, h=HC, sei=S, kidx=idx, nkey=n)
    done += 1; print(sid, n, len(kf), round(time.time() - t0), flush=True)
print('worker', wk, 'done', done, 'total', len(glob.glob(O + '/*.npz')), '/', ALL)
