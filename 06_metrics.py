#!/usr/bin/env python3
"""
06_metrics.py
PCG Q1 Manuscript — CirCor 2022 resmi + tamamlayici metrikler.

Hasta-duzeyi degerlendirme. Sinif sirasi: Present=0, Unknown=1, Absent=2.

Metrikler:
  - weighted_accuracy : RESMI challenge metrigi (Present=5, Unknown=3, Absent=1)
  - macro_f1          : sinif-dengeli F1
  - uar               : Unweighted Average Recall (= macro recall)
  - per_class_recall, confusion_matrix, sensitivity/specificity yardimcilari

Resmi W.acc formulu (Reyna et al. 2022; arxiv 2509.18424 Eq.7):
  W.acc = (5*tp + 3*tu + 1*ta) / (5*cp + 3*cu + 1*ca)
  tp/tu/ta : ilgili sinifta DOGRU tahmin sayisi (confusion diagonali)
  cp/cu/ca : ilgili sinifin GERCEK toplam sayisi (confusion satir toplami)

Not: Resmi koddaki confusion matrix yonelimi [gercek, tahmin]; diagonal dogru
tahminler. Payda gercek-sinif toplamlari uzerinden agirliklanir.
"""
from __future__ import annotations
import numpy as np

# resmi sinif sirasi ve agirliklar
CLASSES = ["Present", "Unknown", "Absent"]
WEIGHTS = np.array([5.0, 3.0, 1.0])   # Present, Unknown, Absent
N_CLASSES = 3


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """[gercek, tahmin] yonelimli 3x3 confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RESMI CirCor 2022 weighted accuracy.
    Pay  : agirlikli dogru tahminler (diagonal).
    Payda: agirlikli gercek-sinif toplamlari (satir toplamlari)."""
    cm = confusion_matrix(y_true, y_pred)
    correct = np.diag(cm).astype(float)          # [tp, tu, ta]
    class_totals = cm.sum(axis=1).astype(float)  # [cp, cu, ca]
    num = float((WEIGHTS * correct).sum())
    den = float((WEIGHTS * class_totals).sum())
    return num / den if den > 0 else 0.0


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    correct = np.diag(cm).astype(float)
    totals = cm.sum(axis=1).astype(float)
    return np.divide(correct, totals, out=np.zeros_like(correct),
                     where=totals > 0)


def uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted Average Recall = macro recall (siniflarin esit agirlikli recall'u)."""
    return float(per_class_recall(y_true, y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    f1s = []
    for c in range(N_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tum metrikleri tek sozlukte dondur."""
    rec = per_class_recall(y_true, y_pred)
    return {
        "weighted_accuracy": weighted_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "uar": uar(y_true, y_pred),
        "recall_present": float(rec[0]),
        "recall_unknown": float(rec[1]),
        "recall_absent": float(rec[2]),
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
    }


def format_metrics(m: dict) -> str:
    return (f"W.Acc={m['weighted_accuracy']:.4f} | "
            f"Macro-F1={m['macro_f1']:.4f} | UAR={m['uar']:.4f} | "
            f"Acc={m['accuracy']:.4f}  "
            f"[recall P/U/A = {m['recall_present']:.3f}/"
            f"{m['recall_unknown']:.3f}/{m['recall_absent']:.3f}]")


# -----------------------------------------------------------------------------
# IKILI (binary) metrikler — dis validasyon (CinC: normal/abnormal) icin.
# Sinif sirasi ikili gorevde: 0=normal, 1=abnormal (pozitif sinif=abnormal).
# -----------------------------------------------------------------------------
def binary_metrics(y_true, p_abnormal):
    """y_true: 0/1 (1=abnormal), p_abnormal: abnormal olasiligi [0,1].
    AUROC + Youden-optimal esikte sens/spec/F1/acc + sabit 0.5 esikte de."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p_abnormal).astype(float)

    # AUROC (rank-tabanli, sklearn'siz)
    auroc = _auroc(y_true, p)

    # 0.5 esikte
    pred05 = (p >= 0.5).astype(int)
    m05 = _binary_at_threshold(y_true, pred05)

    # Youden-optimal esik (sens+spec-1 maksimum)
    thr = _youden_threshold(y_true, p)
    predY = (p >= thr).astype(int)
    mY = _binary_at_threshold(y_true, predY)

    return {
        "auroc": auroc,
        "sensitivity": m05["sens"], "specificity": m05["spec"],
        "f1": m05["f1"], "bacc": m05["bacc"], "accuracy": m05["acc"],
        "sens_youden": mY["sens"], "spec_youden": mY["spec"],
        "bacc_youden": mY["bacc"], "youden_thr": thr,
    }


def _binary_at_threshold(y_true, pred):
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall pozitif
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    bacc = 0.5 * (sens + spec)                         # balanced accuracy
    return {"sens": sens, "spec": spec, "f1": f1, "acc": acc, "bacc": bacc}


def _auroc(y_true, scores):
    """Mann-Whitney U tabanli AUROC (sklearn'siz)."""
    pos = scores[y_true == 1]; neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # tum (pos,neg) ciftlerinde pos>neg orani; beraberlik 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # beraberlik duzeltmesi
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_rank = np.zeros(len(scores))
    cum = 0
    avg_ranks = {}
    sorted_scores = np.sort(scores)
    # basit yaklasim: ortalama rank ata
    from collections import defaultdict
    idxs = defaultdict(list)
    for i, s in enumerate(scores):
        idxs[s].append(i)
    rank_pos = np.argsort(np.argsort(scores)) + 1.0
    # beraberlikler icin ortalama
    for s, ii in idxs.items():
        if len(ii) > 1:
            mean_r = np.mean([rank_pos[i] for i in ii])
            for i in ii:
                rank_pos[i] = mean_r
    sum_ranks_pos = rank_pos[y_true == 1].sum()
    n_pos = len(pos); n_neg = len(neg)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _youden_threshold(y_true, scores):
    """Youden J (sens+spec-1) maksimize eden esik."""
    thrs = np.unique(scores)
    best_j, best_t = -1.0, 0.5
    for t in thrs:
        pred = (scores >= t).astype(int)
        m = _binary_at_threshold(y_true, pred)
        j = m["sens"] + m["spec"] - 1
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def format_binary(m):
    return (f"AUROC={m['auroc']:.4f} | Sens={m['sensitivity']:.3f} | "
            f"Spec={m['specificity']:.3f} | F1={m['f1']:.3f} | "
            f"BAcc={m['bacc']:.3f}")


# -----------------------------------------------------------------------------
# Birim testler — formulu sabit ornekle dogrula
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*70); print("Metrik birim testleri"); print("="*70)

    # Test 1: mukemmel tahmin -> tum metrikler 1.0
    yt = np.array([0,0,1,1,2,2,2,2])
    yp = yt.copy()
    m = all_metrics(yt, yp)
    assert abs(m["weighted_accuracy"]-1.0) < 1e-9
    assert abs(m["macro_f1"]-1.0) < 1e-9
    assert abs(m["uar"]-1.0) < 1e-9
    print("[OK] Test1 mukemmel tahmin -> hepsi 1.0")

    # Test 2: el ile W.acc dogrulama
    # gercek: 2 Present, 2 Unknown, 4 Absent
    # tahmin: 1 Present dogru, 1 Present->Absent;  2 Unknown dogru; 4 Absent dogru
    yt = np.array([0,0, 1,1, 2,2,2,2])
    yp = np.array([0,2, 1,1, 2,2,2,2])
    # tp=1, tu=2, ta=4 ; cp=2, cu=2, ca=4
    # pay = 5*1 + 3*2 + 1*4 = 5+6+4 = 15
    # payda = 5*2 + 3*2 + 1*4 = 10+6+4 = 20
    # W.acc = 15/20 = 0.75
    wa = weighted_accuracy(yt, yp)
    print(f"   Hesaplanan W.acc={wa:.4f}, beklenen 0.7500")
    assert abs(wa-0.75) < 1e-9
    print("[OK] Test2 W.acc el-hesabi dogru (0.75)")

    # Test 3: hepsini Absent tahmin et (cogunluk sinifi tuzagi)
    # bu durumda duz accuracy yuksek ama W.acc DUSUK olmali (Present/Unknown kacirilir)
    yt = np.array([0]*10 + [1]*10 + [2]*80)   # gercekci dengesizlik
    yp = np.array([2]*100)                      # hepsi Absent
    m = all_metrics(yt, yp)
    print(f"   Hep-Absent: Acc={m['accuracy']:.3f} ama W.acc={m['weighted_accuracy']:.3f}")
    # Acc=0.80 (80 absent dogru) ama W.acc cok dusuk olmali
    assert m["accuracy"] == 0.80
    # W.acc = (5*0+3*0+1*80)/(5*10+3*10+1*80) = 80/160 = 0.5
    assert abs(m["weighted_accuracy"]-0.5) < 1e-9
    print(f"[OK] Test3 cogunluk-sinifi tuzagi yakalandi: duz Acc={m['accuracy']:.2f} "
          f"yaniltici, W.acc={m['weighted_accuracy']:.2f} dusuk (dogru ceza)")

    # Test 4: UAR = macro recall kontrolu
    rec = per_class_recall(yt, yp)  # [0, 0, 1.0]
    assert abs(uar(yt,yp) - rec.mean()) < 1e-9
    print(f"[OK] Test4 UAR=macro recall = {uar(yt,yp):.4f}")

    print("\nOrnek format:")
    print("  ", format_metrics(all_metrics(
        np.array([0,0,1,1,2,2,2,2]), np.array([0,2,1,1,2,2,2,0]))))

    # -------- IKILI metrik testleri --------
    print("\n" + "="*70); print("Ikili metrik testleri"); print("="*70)
    # mukemmel ayrim: abnormal'lar yuksek skor
    yt = np.array([0,0,0,0,1,1,1,1])
    p  = np.array([0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9])
    bm = binary_metrics(yt, p)
    print(f"  Mukemmel ayrim: AUROC={bm['auroc']:.3f} (beklenen 1.0)")
    assert abs(bm['auroc']-1.0) < 1e-9, "AUROC mukemmel ayrimda 1.0 olmali"
    assert abs(bm['sensitivity']-1.0) < 1e-9 and abs(bm['specificity']-1.0) < 1e-9
    print("  [OK] mukemmel ayrim AUROC=1.0, sens=spec=1.0")

    # rastgele ayrim ~0.5
    rng = np.random.default_rng(0)
    yt2 = rng.integers(0,2,1000); p2 = rng.random(1000)
    bm2 = binary_metrics(yt2, p2)
    print(f"  Rastgele: AUROC={bm2['auroc']:.3f} (beklenen ~0.5)")
    assert 0.42 < bm2['auroc'] < 0.58, "rastgele AUROC ~0.5 olmali"
    print("  [OK] rastgele AUROC ~0.5")

    # ters ayrim -> AUROC < 0.5
    bm3 = binary_metrics(yt, 1-p)
    print(f"  Ters ayrim: AUROC={bm3['auroc']:.3f} (beklenen 0.0)")
    assert abs(bm3['auroc']-0.0) < 1e-9
    print("  [OK] ters ayrim AUROC=0.0")

    # beraberlik testi: tum skorlar ayni -> AUROC=0.5
    bm4 = binary_metrics(np.array([0,1,0,1]), np.array([0.5,0.5,0.5,0.5]))
    print(f"  Beraberlik: AUROC={bm4['auroc']:.3f} (beklenen 0.5)")
    assert abs(bm4['auroc']-0.5) < 1e-9
    print("  [OK] beraberlik AUROC=0.5 (tie duzeltmesi calisiyor)")
    print("  Ornek:", format_binary(bm))

    print("\n=== Tum metrik testleri gecti ===")
