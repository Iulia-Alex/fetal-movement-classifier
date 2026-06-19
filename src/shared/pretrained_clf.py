"""
Pre-trained fetal movement classifiers (AttUNet 1D), collaborator contribution.
Re-implemented from model documentation + state dicts.
  Model 14: 5 feat, per-window norm, BINARY (cout=1), no self-attn
  Model 15: 5 feat, per-window norm, MULTICLASS (cout=4), no self-attn
  Model 18: 3 feat, signal-level norm, BINARY (cout=1), self-attn at bottleneck
"""
import numpy as np
import torch, torch.nn as nn
from scipy.signal import find_peaks, hilbert
from scipy.ndimage import uniform_filter1d

from config import CLF_MODELS_DIR as MODELS_DIR
FS = 500; WIN = 3840; MIN_DIST = 100; EPS = 1e-8
PATHS = {14:'14_all_Sig_AttUNet_5feat_step_best.pth',
         15:'15_AttUNet_multiclass_best.pth',
         18:'18_AttUNet_3feat_selfattn_best.pth'}
CFG = {14:dict(cin=5,cout=1,self_attn=False,binary=True, feat='5pw'),
       15:dict(cin=5,cout=4,self_attn=False,binary=False,feat='5pw'),
       18:dict(cin=3,cout=1,self_attn=True, binary=True, feat='3sl')}

# ── architecture ──────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(cin,cout,15,padding=7), nn.GroupNorm(4,cout), nn.ReLU(),
            nn.Conv1d(cout,cout,15,padding=7), nn.GroupNorm(4,cout))
        self.shortcut = nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
        self.act = nn.ReLU()
    def forward(self,x): return self.act(self.conv(x)+self.shortcut(x))

class AttGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter):
        super().__init__()
        self.W_g=nn.Conv1d(g_ch,inter,1); self.W_x=nn.Conv1d(x_ch,inter,1)
        self.psi=nn.Sequential(nn.Conv1d(inter,1,1),nn.Sigmoid()); self.relu=nn.ReLU()
    def forward(self,g,x): return x*self.psi(self.relu(self.W_g(g)+self.W_x(x)))

class BottleneckSelfAttention(nn.Module):
    def __init__(self, ch=256, heads=4):
        super().__init__()
        self.attn=nn.MultiheadAttention(ch,heads,batch_first=True); self.norm=nn.LayerNorm(ch)
    def forward(self,x):           # x: (B, C, L)
        h=x.transpose(1,2); a,_=self.attn(h,h,h); h=self.norm(h+a)
        return h.transpose(1,2)

class AttUNet1D(nn.Module):
    def __init__(self, cin, cout, self_attn=False):
        super().__init__()
        self.enc1=ResBlock(cin,32); self.enc2=ResBlock(32,64)
        self.enc3=ResBlock(64,128); self.enc4=ResBlock(128,256)
        self.bottleneck=ResBlock(256,256)
        self.self_attn=BottleneckSelfAttention(256) if self_attn else None
        self.pool=nn.MaxPool1d(2); self.up=nn.Upsample(scale_factor=2,mode='nearest')
        self.att4=AttGate(256,256,128); self.dec4=ResBlock(512,128)
        self.att3=AttGate(128,128,64);  self.dec3=ResBlock(256,64)
        self.att2=AttGate(64,64,32);    self.dec2=ResBlock(128,32)
        self.att1=AttGate(32,32,16);    self.dec1=ResBlock(64,32)
        self.final=nn.Conv1d(32,cout,1)
    def forward(self,x):
        s1=self.enc1(x); s2=self.enc2(self.pool(s1))
        s3=self.enc3(self.pool(s2)); s4=self.enc4(self.pool(s3))
        b=self.bottleneck(self.pool(s4))
        if self.self_attn is not None: b=self.self_attn(b)
        g=self.up(b);  d4=self.dec4(torch.cat([g,self.att4(g,s4)],1))
        g=self.up(d4); d3=self.dec3(torch.cat([g,self.att3(g,s3)],1))
        g=self.up(d3); d2=self.dec2(torch.cat([g,self.att2(g,s2)],1))
        g=self.up(d2); d1=self.dec1(torch.cat([g,self.att1(g,s1)],1))
        return self.final(d1)

def load_model(mid, device='cpu'):
    c=CFG[mid]; m=AttUNet1D(c['cin'],c['cout'],c['self_attn'])
    m.load_state_dict(torch.load(f'{MODELS_DIR}/{PATHS[mid]}',map_location=device),strict=True)
    return m.to(device).eval()

# ── features ─────────────────────────────────────────────────────────────────
def _z(x): return (x-x.mean())/(x.std()+EPS)
def _local_rms(s): return np.sqrt(uniform_filter1d(s**2,size=50,mode='nearest'))

def features_5(window, fs=FS):
    """5 features, PER-WINDOW normalisation (models 14, 15). window:(3840,)->(5,3840)"""
    L=len(window); s=(window-window.mean())/(window.std()+EPS)
    peaks,_=find_peaks(s,distance=MIN_DIST); lr=_local_rms(s); idx=np.arange(L)
    if len(peaks)<2:
        return np.stack([s,np.zeros(L),s,lr,np.zeros(L)]).astype(np.float32)
    le=np.interp(idx,peaks,s[peaks])
    rate=np.interp(idx,peaks[1:],fs/np.diff(peaks))
    return np.stack([le,_z(np.abs(hilbert(s))),s-le,lr,_z(rate)]).astype(np.float32)

def features_3_global(sig, fs=FS):
    """3 features (residual,local_rms,qrs_rate), SIGNAL-LEVEL normalisation (model 18). ->(3,N)"""
    N=len(sig); s=(sig-sig.mean())/(sig.std()+EPS)
    peaks,_=find_peaks(s,distance=MIN_DIST); lr=_local_rms(s); idx=np.arange(N)
    if len(peaks)<2:
        return np.stack([s,lr,np.zeros(N)]).astype(np.float32)
    le=np.interp(idx,peaks,s[peaks])
    rate=np.interp(idx,peaks[1:],fs/np.diff(peaks))
    return np.stack([s-le,lr,_z(rate)]).astype(np.float32)

# ── dense inference: sliding-window + overlap-add ────────────────────────────
def predict_dense(model, length, feat_provider, n_out, binary,
                  win=WIN, stride=256, batch=64, device='cpu'):
    L=max(length,win)
    starts=list(range(0,L-win+1,stride))
    if not starts or starts[-1]!=L-win: starts.append(L-win)
    psum=np.zeros((max(n_out,1),L)); cnt=np.zeros(L); buf=[]; bs=[]
    def flush():
        nonlocal buf,bs
        if not buf: return
        x=torch.from_numpy(np.stack(buf)).to(device)
        with torch.no_grad():
            o=model(x)
            p=(torch.sigmoid(o) if binary else torch.softmax(o,dim=1)).cpu().numpy()
        for bi,s0 in enumerate(bs):
            psum[:,s0:s0+win]+=p[bi]; cnt[s0:s0+win]+=1
        buf=[]; bs=[]
    for s0 in starts:
        buf.append(feat_provider(s0)); bs.append(s0)
        if len(buf)>=batch: flush()
    flush()
    cnt[cnt==0]=1; prob=psum/cnt
    if binary: return (prob[0]>0.5).astype(int)[:length], prob[0][:length]
    return np.argmax(prob,axis=0)[:length], prob[:,:length]

def run_model(mid, sig1d, model, stride=512):
    """Run pre-trained classifier `mid` on a 1D signal. Returns (pred, prob)."""
    c=CFG[mid]; N=len(sig1d)
    if c['feat']=='5pw':
        fp=lambda s0: features_5(sig1d[s0:s0+WIN])
    else:
        F3=features_3_global(sig1d); fp=lambda s0: F3[:,s0:s0+WIN]
    return predict_dense(model,N,fp,c['cout'],c['binary'],stride=stride)


if __name__=='__main__':
    import os
    from sklearn.metrics import accuracy_score, f1_score
    from config import NPY
    def load_ch(sub,name,ch):
        d=os.path.join(NPY,sub,name); f=[x for x in os.listdir(d) if f'_ch{ch+1}.npy' in x][0]
        return np.load(os.path.join(d,f)).astype(np.float32)
    def load_mask(sub,name):
        return np.load([os.path.join(NPY,sub,f) for f in os.listdir(os.path.join(NPY,sub)) if name in f][0])
    models={mid:load_model(mid) for mid in [14,15,18]}
    print("All 3 models loaded strict=True OK.\n")
    for name in ['Test_db_43_SNRmn=5dB_SNRfm=-5dB_SNRfn=0dB','Test_db_87_SNRmn=15dB_SNRfm=0dB_SNRfn=15dB']:
        mc=load_mask('mc_masks',name).astype(int); mb=load_mask('masks',name).astype(int)
        print(f"=== {name} (validation on clean GT) ===")
        for ch in [0,2]:
            gt=load_ch('signals',name,ch); N=min(len(gt),len(mc))
            # M15 multiclass
            p15,_=run_model(15,gt[:N],models[15]); acc15=accuracy_score(mc[:N],p15)
            f1_15b=f1_score((mc[:N]>0).astype(int),(p15>0).astype(int))
            # M14 binary
            p14,_=run_model(14,gt[:N],models[14]); f1_14=f1_score(mb[:N],p14); acc14=accuracy_score(mb[:N],p14)
            # M18 binary
            p18,_=run_model(18,gt[:N],models[18]); f1_18=f1_score(mb[:N],p18); acc18=accuracy_score(mb[:N],p18)
            print(f"  ch{ch+1}: M15 acc={acc15:.3f}(binF1={f1_15b:.3f}) | M14 binF1={f1_14:.3f}(acc{acc14:.3f}) | M18 binF1={f1_18:.3f}(acc{acc18:.3f})")
        print()
