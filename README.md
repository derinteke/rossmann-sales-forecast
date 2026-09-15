# 🏪 Rossmann Store Sales — Forecasting

**EN —** I built this to forecast daily sales for 1,115 Rossmann drugstores six weeks
ahead — the same setup as the Kaggle competition. What I cared about most wasn't a flashy
score, but doing it *honestly*: no data leakage, and a real baseline to prove the model
actually learns something.

**TR —** Bunu 1.115 Rossmann mağazasının günlük satışını 6 hafta ileriye tahmin etmek için
yaptım — Kaggle yarışmasıyla aynı kurulum. En çok önemsediğim şey gösterişli bir skor değil,
işi *dürüst* yapmaktı: veri sızıntısı olmadan ve modelin gerçekten bir şey öğrendiğini
kanıtlayan gerçek bir baseline'la.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-tuned-green)
![Optuna](https://img.shields.io/badge/Optuna-HPO-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📊 Results / Sonuçlar

**EN —** I evaluated everything on a time-based hold-out (the last 6 weeks), using RMSPE —
the competition metric. Here's where I landed:

**TR —** Her şeyi zaman-bazlı bir hold-out'ta (son 6 hafta) ve yarışma metriği olan RMSPE
ile ölçtüm. Vardığım nokta şu:

| Model | RMSPE | RMSE | MAE |
|-------|:-----:|:----:|:---:|
| Baseline (store × day-of-week median) | 0.245 | 1,696 | 1,259 |
| LightGBM (hand-set params) | 0.128 | 898 | 609 |
| **LightGBM + Optuna** | **0.124** | **871** | **591** |

**EN —** My tuned model roughly halves the baseline error, and the tuning itself shaved a
bit more off (0.128 → 0.124). Because it clearly beats a strong naive baseline, I trust that
it's learning real structure rather than echoing the average. For reference, good single
models in the Kaggle competition scored around 0.11–0.13.

**TR —** Tuned modelim baseline hatasını neredeyse yarıya indiriyor, tuning de biraz daha
düşürdü (0.128 → 0.124). Naif baseline'ı açık ara geçtiği için, ortalamayı tekrarlamak yerine
gerçek yapı öğrendiğine güveniyorum. Referans olarak, yarışmadaki iyi tek modeller ~0.11-0.13
alıyordu.

<p align="center">
  <img src="reports/figures/forecast_store_262.png" width="760">
</p>
<p align="center">
  <img src="reports/figures/shap_importance.png" width="430">
  <img src="reports/figures/eda_weekly_pattern.png" width="430">
</p>

## 🧠 The two decisions I'm most proud of / En çok gurur duyduğum iki karar

### 1. No leakage / Sızıntı yok

**EN —** The real task predicts six weeks with *no recent sales available*, so I couldn't
use yesterday's or last week's sales as features — that would be leakage. Instead I leaned on
things I'd actually know in advance (calendar, promos, holidays, store metadata) plus
historical aggregates that I fit **only on the training split** (a store's typical sales by
weekday, and under promo vs not). It's the honest way to give the model store history without
cheating.

**TR —** Gerçek görev, *elimde yakın tarihli satış olmadan* 6 hafta tahmin etmek. Bu yüzden
dünün ya da geçen haftanın satışını feature olarak kullanamazdım — bu sızıntı olurdu. Onun
yerine, tahmin anında zaten bileceğim şeylere (takvim, promosyon, tatil, mağaza metası) ve
**yalnızca eğitim verisinden** hesapladığım geçmiş ortalamalara (mağazanın gün bazında ve
promosyonlu/promosyonsuz tipik satışı) dayandım. Bu, modele geçmişi hile yapmadan vermenin
dürüst yolu.

### 2. Tuning without peeking / Peeking'siz tuning

**EN —** When I tuned with Optuna, I refused to look at the final validation window. I tuned
on the 6 weeks *before* it, and I even picked the number of trees using a separate inner
split — so the real hold-out stayed untouched until the single final measurement. That's why
I believe the 0.124 is a fair number and not a lucky fit.

**TR —** Optuna ile tuning yaparken final validasyon penceresine bakmayı reddettim. Ondan
*önceki* 6 haftada tune ettim, ağaç sayısını bile ayrı bir iç bölmeyle seçtim — böylece
gerçek hold-out, tek seferlik final ölçüme kadar el değmemiş kaldı. 0.124'ün şanslı bir uyum
değil, adil bir sayı olduğuna bu yüzden inanıyorum.

## 🔑 What the data told me / Verinin bana söyledikleri

**EN / TR:**
- **Promotions really move sales — about +39%.** / **Promosyon satışı gerçekten hareketlendiriyor — yaklaşık +%39.**
- **There's a strong weekly rhythm** (Monday & Sunday high, Saturday low). / **Güçlü bir haftalık ritim var** (Pazartesi & Pazar yüksek, Cumartesi düşük).
- **December peaks** ~24% above an average month. / **Aralık zirvesi**, ortalama bir aydan ~%24 yüksek.
- **A store's own history is the best predictor** — it dominates the SHAP importance. / **Mağazanın kendi geçmişi en iyi öngörücü** — SHAP önem grafiğine hakim.

## 🗂️ Project structure / Proje yapısı

```
rossmann-sales-forecast/
├── data/raw/{train.csv, store.csv}
├── notebooks/01_sales_forecast.ipynb   # the full story end to end
├── src/
│   ├── config.py        # paths, split, tuned params
│   ├── data.py          # load + merge + clean
│   ├── features.py      # calendar/store features + leakage-safe encodings
│   ├── model.py         # baseline + LightGBM (log-target)
│   ├── tune.py          # Optuna search (no-peeking, resumable)
│   ├── evaluate.py      # RMSPE / RMSE / MAE + plots
│   ├── eda.py           # EDA figures
│   └── explain.py       # SHAP
├── train.py             # end-to-end pipeline
├── requirements.txt
└── README.md
```

## 🚀 Getting started / Başlangıç

```bash
git clone <your-repo-url>
cd rossmann-sales-forecast

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

python train.py                     # train + evaluate
python -m src.tune --trials 15      # reproduce/extend the Optuna search
python -m src.eda                   # EDA figures
python -m src.explain               # SHAP figure
jupyter notebook notebooks/01_sales_forecast.ipynb
```

## 🔭 Where I'd take it next / Bundan sonrası

**EN —** I'd add per-store prediction intervals (quantile LightGBM) so the forecast comes
with a confidence band, and I'd wrap the model in a small FastAPI + Streamlit app so someone
could actually pull up a store and see its six-week outlook.

**TR —** Sıradaki adım olarak mağaza bazında tahmin aralıkları (quantile LightGBM) eklerdim ki
tahmin bir güven bandıyla gelsin; bir de modeli küçük bir FastAPI + Streamlit uygulamasına
sarardım ki biri gerçekten bir mağazayı açıp 6 haftalık görünümünü görebilsin.

## 📚 Data / Veri

**EN —** Rossmann Store Sales (Kaggle): daily sales for 1,115 German drugstores,
Jan 2013 – Jul 2015, with promo, holiday and competition metadata.

**TR —** Rossmann Store Sales (Kaggle): 1.115 Alman mağazasının günlük satışı,
Oca 2013 – Tem 2015; promosyon, tatil ve rakip bilgileriyle.

## 📄 License

MIT.
