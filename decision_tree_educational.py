"""Decision Tree Educacional"""
import numpy as np
from collections import Counter
import math

class Node:
    def __init__(self, depth=0, nid=0):
        self.depth, self.nid = depth, nid
        self.feat, self.thresh, self.left, self.right, self.value = None, None, None, None, None
        self.samples, self.imp, self.dist, self.gain = 0, None, {}, None
    def is_leaf(self): return self.value is not None

class EducationalDecisionTree:
    def __init__(self, criterion='gini', max_depth=3, min_samples_split=2, simple_mode=True):
        self.crit, self.max_d, self.min_s = criterion, max_depth, min_samples_split
        self.simple = simple_mode  # ✅ Nome simples, valor padrão claro
        self.root, self._nid, self.logs = None, 0, []
    
    def _log(self, msg, indent=0, explain=None, lvl='info'):
        e = {'msg': "  "*indent + msg, 'type': lvl}
        if explain and self.simple: e['explain'] = explain
        self.logs.append(e)
    
    def _gini(self, y):
        if not len(y): return 0, []
        cnt, n, g = Counter(y), len(y), 1.0
        steps = [f"📊 {n} amostras: {dict(cnt)}"]
        if self.simple: steps.append("💡 Gini mede 'mistura' das classes")
        for c, v in cnt.items():
            p = v/n; g -= p**2
            if self.simple: steps.append(f"  • Classe {c}: {p:.1%} → {p**2:.3f}")
        steps.append(f"✅ Gini = {g:.3f}" + (" | 0=puro" if self.simple else ""))
        return g, steps
    
    def _ent(self, y):
        if not len(y): return 0, []
        cnt, n, e = Counter(y), len(y), 0.0
        steps = [f"📊 {n} amostras: {dict(cnt)}"]
        if self.simple: steps.append("💡 Entropia: 0=certo, alto=incerto")
        for c, v in cnt.items():
            p = v/n
            if p>0: t = -p*math.log2(p); e += t
            if self.simple: steps.append(f"  • Classe {c}: {p:.1%} → {-p*math.log2(p) if p>0 else 0:.3f}")
        steps.append(f"✅ Entropia = {e:.3f} bits" + (" | 0 bits=certo" if self.simple else ""))
        return e, steps
    
    def _imp(self, y): return (self._gini if self.crit=='gini' else self._ent)(y)
    
    def _gain(self, y, yl, yr):
        pi, ps = self._imp(y); n, nl, nr = len(y), len(yl), len(yr)
        if nl==0 or nr==0: return -1, ps+["⚠ Split inválido"]
        il, _ = self._imp(yl); ir, _ = self._imp(yr)
        w = (nl/n)*il + (nr/n)*ir; g = pi - w
        if self.simple:
            steps = ps + [f"\n🔀 Split: Esq={nl}(imp={il:.3f}), Dir={nr}(imp={ir:.3f})",
                         f"✅ Ganho = {pi:.3f} - {w:.3f} = {g:.3f}", "💡 Maior ganho = melhor split!"]
        else:
            steps = ps + [f"→ Gain = {pi:.4f} - ({nl}/{n})×{il:.4f} - ({nr}/{n})×{ir:.4f} = {g:.4f}"]
        return g, steps
    
    def _best_split(self, X, y):
        bg, bf, bt, bs = -1, None, None, []
        for f in range(X.shape[1]):
            for t in np.unique(X[:,f]):
                l, r = X[:,f]<=t, ~(X[:,f]<=t)
                if np.sum(l)==0 or np.sum(r)==0: continue
                g, s = self._gain(y, y[l], y[r])
                if g > bg: bg, bf, bt, bs = g, f, t, s
        return bf, bt, bg, bs
    
    def _build(self, X, y, d=0):
        n = Node(depth=d, nid=self._nid); self._nid += 1
        n.samples, n.dist = len(y), dict(Counter(y)); n.imp, st = self._imp(y)
        self._log(f"\n🌲 Nó #{n.nid} (nível {d})", d, "Grupo de amostras sendo analisado" if self.simple else None)
        self._log(f"📦 {n.samples} amostras: {n.dist}", d, "Distribuição das classes" if self.simple else None)
        for s in st: self._log(s, d)
        if d>=self.max_d or len(np.unique(y))==1 or n.samples<self.min_s:
            n.value = Counter(y).most_common(1)[0][0]
            self._log(f"✅ FOLHA: classe {n.value}", d, "Decisão final!" if self.simple else None); return n
        f, t, g, st = self._best_split(X, y)
        if f is None or g<=1e-7:
            n.value = Counter(y).most_common(1)[0][0]
            self._log(f"⚠ Sem split válido → FOLHA: {n.value}", d); return n
        if self.simple:
            self._log(f"🔍 Melhor: Feature[{f}] ≤ {t:.3f} | Ganho={g:.3f}", d, "Pergunta que melhor separa as classes")
        else:
            self._log(f"🔍 Split: X[{f}]≤{t:.3f} | Gain={g:.4f}", d)
        for s in st: self._log(s, d)
        n.feat, n.thresh, n.gain = f, t, g
        lm, rm = X[:,f]<=t, ~(X[:,f]<=t)
        self._log(f"➡️ Esquerda (X[{f}]≤{t:.3f}):", d); n.left = self._build(X[lm], y[lm], d+1)
        self._log(f"➡️ Direita (X[{f}]>{t:.3f}):", d); n.right = self._build(X[rm], y[rm], d+1)
        return n
    
    def fit(self, X, y):
        self.logs = []; self._nid = 0
        self._log("🚀 Treinando Decision Tree..."); self._log(f"⚙️ {self.crit}, max_depth={self.max_d}")
        self.root = self._build(X, y); self._log("\n✅ Concluído!", 'success', "Árvore pronta para classificar!" if self.simple else None)
        return self.logs
    
    def predict(self, X): return np.array([self._pred(x, self.root) for x in X])
    def _pred(self, x, n):
        if n.is_leaf(): return n.value
        return self._pred(x, n.left if x[n.feat]<=n.thresh else n.right)
    
    def get_metrics(self, X, y):
        if not self.root: return {'accuracy':0,'max_depth':0,'total_nodes':0,'leaf_count':0}
        acc = np.mean(self.predict(X)==y)
        def cnt(n):
            if not n: return 0,0
            if n.is_leaf(): return 1,1
            tl,ll = cnt(n.left); tr,lr = cnt(n.right)
            return 1+tl+tr, ll+lr
        tot, lv = cnt(self.root); md = max(nd.depth for nd in self._nodes())
        return {'accuracy':float(acc),'max_depth':md,'total_nodes':tot,'leaf_count':lv}
    
    def _nodes(self, n=None):
        if n is None: n = self.root
        if not n: return []
        r = [n]
        if not n.is_leaf(): r.extend(self._nodes(n.left)); r.extend(self._nodes(n.right))
        return r

def plot_boundary(tree, X, y, res=150):
    import matplotlib.pyplot as plt, io, base64
    if X.shape[1]!=2:
        fig,ax = plt.subplots(figsize=(4,3)); ax.text(0.5,0.5,"Apenas 2D",ha='center'); ax.axis('off')
        buf=io.BytesIO(); fig.savefig(buf,format='png'); plt.close(fig); return base64.b64encode(buf.getvalue()).decode('utf-8')
    xm,xM = X[:,0].min()-0.5, X[:,0].max()+0.5; ym,yM = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx,yy = np.meshgrid(np.linspace(xm,xM,res), np.linspace(ym,yM,res))
    Z = tree.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    fig,ax = plt.subplots(figsize=(4.5,3.5),dpi=100); ax.contourf(xx,yy,Z,alpha=0.25,cmap='coolwarm',levels=15)
    for c in np.unique(y): ax.scatter(X[y==c,0],X[y==c,1],label=f'Classe {int(c)}',edgecolors='white',s=30)
    _splits(ax,tree.root,xm,xM,ym,yM); ax.set_xlabel('X₁'); ax.set_ylabel('X₂'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    buf=io.BytesIO(); fig.savefig(buf,format='png',bbox_inches='tight'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def _splits(ax,n,xm,xM,ym,yM,d=0):
    if not n or n.is_leaf() or d>10: return
    if n.feat==0: ax.axvline(n.thresh,color='gray',ls='--',alpha=0.6); _splits(ax,n.left,xm,n.thresh,ym,yM,d+1); _splits(ax,n.right,n.thresh,xM,ym,yM,d+1)
    else: ax.axhline(n.thresh,color='gray',ls='--',alpha=0.6); _splits(ax,n.left,xm,xM,ym,n.thresh,d+1); _splits(ax,n.right,xm,xM,n.thresh,yM,d+1)

def tree_to_text(tree, simple=True):
    if not tree.root: return "Árvore não treinada."
    lines = ["🌳 Árvore de Decisão:\n"]
    def trav(n,pfx="",last=True):
        if not n: return
        cn = "└── " if last else "├── "; npfx = pfx + ("    " if last else "│   ")
        if n.is_leaf():
            if simple: lines.append(f"{pfx}{cn}🍃 DECISÃO: Classe {n.value}\n{pfx}{'    'if last else'│   '}💡 {n.samples} amostras → classe {n.value}")
            else: lines.append(f"{pfx}{cn}🍃 Leaf: Class {n.value} (n={n.samples})")
        else:
            gs = f" | 💰 {n.gain:.3f}" if n.gain else ""
            if simple: lines.append(f"{pfx}{cn}🔀 Pergunta: X[{n.feat}] ≤ {n.thresh:.3f}?{gs}")
            else: lines.append(f"{pfx}{cn}🔀 Split: X[{n.feat}]≤{n.thresh:.3f}{gs}")
            if n.left: trav(n.left,npfx,n.right is None)
            if n.right: trav(n.right,npfx,True)
    trav(tree.root)
    if simple: lines.append("\n🔑 🔀=pergunta | 🍃=decisão final | 💰=ganho (maior=melhor)")
    return "\n".join(lines)

def generate_dataset(tp='moon', n=100, noise=0.15):
    np.random.seed(42)
    if tp=='moon':
        nh=n//2; ang=np.linspace(0,np.pi,nh)
        x1=np.column_stack([np.cos(ang)+np.random.normal(0,noise,nh), np.sin(ang)+np.random.normal(0,noise,nh)])
        x2=np.column_stack([np.cos(ang)+1+np.random.normal(0,noise,nh), -np.sin(ang)+0.5+np.random.normal(0,noise,nh)])
        return np.vstack([x1,x2]).astype(float), np.array([0]*nh+[1]*nh).astype(int)
    elif tp=='circle':
        nh=n//2; ang=np.random.uniform(0,2*np.pi,nh)
        x1=np.column_stack([np.cos(ang)+np.random.normal(0,noise,nh), np.sin(ang)+np.random.normal(0,noise,nh)])
        x2=np.column_stack([2*np.cos(ang)+np.random.normal(0,noise,nh), 2*np.sin(ang)+np.random.normal(0,noise,nh)])
        return np.vstack([x1,x2]).astype(float), np.array([0]*nh+[1]*nh).astype(int)
    else:
        X=np.random.randn(n,2); return X.astype(float), (X[:,0]+X[:,1]>0).astype(int)
