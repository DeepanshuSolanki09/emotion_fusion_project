import numpy as np

MODALITIES = ["EEG", "Image", "Speech", "Video"]
CLASSES = ["Negative", "Neutral", "Positive"]
EEG, IMG, SPE, VID = range(4)

IN_MF = {"L": (0.0, 0.0, 0.5), "M": (0.35, 0.55, 0.75), "H": (0.6, 1.0, 1.0)}
OUT_MF = {"Low": (0.0, 0.0, 0.35), "Medium": (0.2, 0.45, 0.7),
          "High": (0.5, 0.75, 1.0), "VeryHigh": (0.75, 1.0, 1.0)}
RULE_OUT = {"R1": "VeryHigh", "R2": "High", "R3": "Medium", "R4": "High", "R5": "Low",
            "R6": "Medium", "R7": "VeryHigh", "R8": "High", "R9": "Medium", "R10": "Low",
            "R11": "Medium", "R12": "High", "R13": "High", "R14": "High", "R15": "Medium"}


def tri(x, a, b, c):
    x = np.asarray(x, float)
    left = np.ones_like(x) if b <= a else (x - a) / (b - a)
    right = np.ones_like(x) if c <= b else (c - x) / (c - b)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def reliability_from_validation(val_macro_f1):
    q = np.maximum((np.asarray(val_macro_f1, float) - 1 / 3) / (2 / 3), 1e-6)
    return q / q.max()


class MamdaniFusion:
    def __init__(self, reliability, grid=201, tie_eps=1e-3):
        self.r = np.asarray(reliability, float)
        self.s = np.linspace(0, 1, grid)
        self.out = {k: tri(self.s, *v) for k, v in OUT_MF.items()}
        self.tie_eps = tie_eps

    def _alphas(self, P, A):
        P, A = np.asarray(P, float), np.asarray(A, bool)
        N, M, C = P.shape
        av = A[:, :, None]
        r = self.r[None, :, None]
        L = np.where(av, tri(P, *IN_MF["L"]), 0.0)
        Mm = np.where(av, r * tri(P, *IN_MF["M"]), 0.0)
        H = np.where(av, r * tri(P, *IN_MF["H"]), 0.0)
        MH = np.maximum(Mm, H)
        n = A.sum(1)[:, None]
        n_oth = A[:, 1:].sum(1)[:, None]

        def kth(X, k):
            return np.zeros((N, C)) if X.shape[1] < k else -np.sort(-X, axis=1)[:, k - 1, :]

        def amin(X, avm):
            return np.where(avm, X, 1.0).min(axis=1)

        eeg_ok = A[:, EEG][:, None]
        a = {}
        a["R1"] = np.where(n >= 3, amin(H, av), 0)
        a["R2"] = np.minimum(H[:, EEG], kth(MH[:, 1:], 2))
        a["R3"] = np.where(n_oth >= 1, np.minimum(H[:, EEG], amin(L[:, 1:], av[:, 1:])), 0)
        a["R4"] = np.minimum(Mm[:, EEG], kth(H[:, 1:], 2))
        a["R5"] = np.where(n >= 1, amin(L, av), 0)
        a["R6"] = np.where(n >= 2, amin(Mm, av), 0)
        a["R7"] = kth(H, 3)
        a["R8"] = kth(H, 2)
        r9 = np.zeros((N, C))
        for j in range(M):
            Hj = H.copy()
            Hj[:, j] = 0
            r9 = np.maximum(r9, np.minimum(L[:, j], kth(Hj, 2)))
        a["R9"] = r9
        a["R10"] = np.minimum(kth(H, 2), kth(L, 2))
        a["R11"] = np.where(~eeg_ok, H.max(1), 0)
        g = lambda m: (~A[:, m])[:, None].astype(float)
        a["R12"] = g(IMG) * np.minimum(H[:, EEG], np.maximum(MH[:, SPE], MH[:, VID]))
        a["R13"] = g(SPE) * np.minimum(H[:, EEG], np.maximum(MH[:, IMG], MH[:, VID]))
        a["R14"] = g(VID) * np.minimum(H[:, EEG], np.maximum(MH[:, IMG], MH[:, SPE]))
        a["R15"] = np.where(n == 1, H.max(1), 0)
        return a

    def predict(self, P, A):
        P, A = np.asarray(P, float), np.asarray(A, bool)
        al = self._alphas(P, A)
        N, _, C = P.shape
        agg = {k: np.zeros((N, C)) for k in OUT_MF}
        for rule, cons in RULE_OUT.items():
            agg[cons] = np.maximum(agg[cons], al[rule])
        B = np.max([np.minimum(agg[k][..., None], self.out[k]) for k in OUT_MF], axis=0)
        area = B.sum(-1)
        S = np.where(area > 0, (B * self.s).sum(-1) / np.maximum(area, 1e-12), 0.0)
        fallback = area.max(1) <= 1e-9
        w = self.r[None, :] * A
        wm = (w[:, :, None] * P).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-12)
        S = np.where(fallback[:, None], wm, S)
        cand = S >= S.max(1, keepdims=True) - self.tie_eps
        pred = np.where(cand, area, -1).argmax(1)
        pred = np.where(fallback, wm.argmax(1), pred)
        support = S / np.maximum(S.sum(1, keepdims=True), 1e-12)
        return pred, support, S, fallback

    def explain(self, p, a):
        al = self._alphas(np.asarray(p)[None], np.asarray(a)[None])
        return {CLASSES[c]: {k: round(float(v[0, c]), 3) for k, v in al.items() if v[0, c] > 1e-6}
                for c in range(3)}


def ds_fuse(P, A, r=None):
    P, A = np.asarray(P, float), np.asarray(A, bool)
    N, M, C = P.shape
    r = np.ones(M) if r is None else np.asarray(r, float)
    c, th = np.zeros((N, C)), np.ones((N, 1))
    for m in range(M):
        s, t = r[m] * P[:, m], 1 - r[m]
        nc, nth = c * s + c * t + th * s, th * t
        z = nc.sum(1, keepdims=True) + nth
        use = A[:, m][:, None]
        c = np.where(use, nc / z, c)
        th = np.where(use, nth / z, th)
    return c, th
