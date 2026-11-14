import sys,glob,os,numpy as np,subprocess,cv2,pandas as pd
G=sys.argv[1]; sys.argv=['x','1','0','1']
exec(open('fn27_pass2.py').read().split("HF = mp.solutions")[0])
d=pd.read_csv(os.path.expanduser('~/mnt/ODL_FieldNote_Releases/_wip_20261108/fn27/tables/qc_per_second_10days.csv'))
d['hands']=d[['hands_full','hands_lower']].max(1)
ok=d.brightness.between(40,200)&(d.blur_px_p90<=8)&(d.view_elevation_deg<=-30)
gl=(d.date<'2026-11-08')
x=d[ok&(gl if G=='glove' else ~gl)]
# sample 30 seconds from at most 10 files to keep reads small
rng=np.random.default_rng(7); files=x.groupby(['date','session','chunk']).size().index.tolist()
pick=[files[i] for i in rng.choice(len(files),10,replace=False)]
S=pd.concat([x[(x.date==a)&(x.session==b)&(x.chunk==c)].sample(3,random_state=1) for a,b,c in pick]).reset_index(drop=True)
ims=[]
for (a,b,c),grp in S.groupby(['date','session','chunk'],sort=False):
    hdr,kf=keyframes(os.path.expanduser(f"~/mnt/ODL_work/fn27_qc_all/{a}/{b}_{int(c):05d}_left_raw.mp4"))
    for k,r in grp.iterrows():
        open('/tmp/one.h264','wb').write(hdr+kf[int(r.second)][1])
        p=subprocess.run(['ffmpeg','-v','error','-i','/tmp/one.h264','-vf','scale=480:360','-frames:v','1','-f','rawvideo','-pix_fmt','bgr24','pipe:1'],capture_output=True)
        im=np.frombuffer(p.stdout,np.uint8).reshape(360,480,3)[120:360].copy(); cv2.putText(im,f'{k:02d}',(6,26),0,0.9,(0,255,255),2); ims.append((k,im))
ims=[im for k,im in sorted(ims,key=lambda t:t[0])]
g=np.vstack([np.hstack(ims[i:i+6]) for i in range(0,30,6)])
cv2.imwrite(os.path.expanduser(f'~/mnt/ODL_FieldNote_Releases/_wip_20261108/fn27/review_{G}.jpg'),cv2.resize(g,(1920,int(1920*g.shape[0]/g.shape[1]))),[cv2.IMWRITE_JPEG_QUALITY,80])
S[['date','session','chunk','second','hands_full','hands_lower','hands']].to_csv(os.path.expanduser(f'~/mnt/ODL_FieldNote_Releases/_wip_20261108/fn27/review_{G}.csv'))
print(G,'detected',(S.hands>0).sum(),'/30', ''.join('1' if h>0 else '0' for h in S.hands))
