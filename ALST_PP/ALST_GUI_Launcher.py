import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import subprocess
import sys
from pathlib import Path
import threading
import json
import numpy as np
from typing import List
from tkinter import StringVar, DoubleVar, IntVar, BooleanVar

# --- Locator for predictor script -------------------------------------------------

def _resolve_predictor_path() -> Path:
    """Return the first existing path for the predictor among common locations.
    Searches both `ALST_Ticker_Predictor.py` and `ALST_Ticker_Predictor_Final.py`.
    Raises FileNotFoundError with diagnostics if not found.
    """
    here = Path(__file__).resolve().parent          # .../FIN_PP/ALST_PP
    project = here.parent                           # .../FIN_PP

    candidates: List[Path] = [
        # same folder as GUI
        here / "ALST_Ticker_Predictor_Final.py",
        here / "ALST_Ticker_Predictor.py",
        # project root
        project / "ALST_Ticker_Predictor_Final.py",
        project / "ALST_Ticker_Predictor.py",
        # common subfolders
        project / "ALST_PP" / "ALST_Ticker_Predictor_Final.py",
        project / "ALST_PP" / "ALST_Ticker_Predictor.py",
        project / "src" / "ALST_Ticker_Predictor_Final.py",
        project / "src" / "ALST_Ticker_Predictor.py",
        project / "scripts" / "ALST_Ticker_Predictor_Final.py",
        project / "scripts" / "ALST_Ticker_Predictor.py",
    ]

    for p in candidates:
        if p.exists():
            return p

    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not locate predictor script. Tried the following paths:\n" + tried
    )

# --- Tkinter GUI Setup -----------------------------------------------------------
window = tk.Tk()
window.title("Alstom Stock Predictor")
window.geometry("460x360")

# Form variables
var_ticker = StringVar(value="ALO.PA")
var_years = DoubleVar(value=5.0)
var_horizon = IntVar(value=5)
var_threshold = DoubleVar(value=0.57)
var_deadband = DoubleVar(value=0.05)
var_costbps = DoubleVar(value=10.0)
var_model = StringVar(value="histgb")
var_calibrate = BooleanVar(value=False)

label_var = tk.StringVar(value="Configure parameters then run prediction")

frm = tk.Frame(window)
frm.pack(padx=10, pady=10, fill=tk.BOTH)

# Row 1
row = 0
tk.Label(frm, text="Ticker").grid(row=row, column=0, sticky="w")
tk.Entry(frm, textvariable=var_ticker, width=12).grid(row=row, column=1, sticky="w")

tk.Label(frm, text="Years").grid(row=row, column=2, sticky="w")
tk.Entry(frm, textvariable=var_years, width=8).grid(row=row, column=3, sticky="w")

# Row 2
row += 1
tk.Label(frm, text="Horizon (days)").grid(row=row, column=0, sticky="w")
tk.Entry(frm, textvariable=var_horizon, width=8).grid(row=row, column=1, sticky="w")

options = ["histgb", "rf"]
tk.Label(frm, text="Classifier").grid(row=row, column=2, sticky="w")
opt = ttk.Combobox(frm, textvariable=var_model, values=options, state="readonly", width=10)
opt.grid(row=row, column=3, sticky="w")

# Row 3
row += 1
tk.Label(frm, text="Threshold").grid(row=row, column=0, sticky="w")
tk.Entry(frm, textvariable=var_threshold, width=8).grid(row=row, column=1, sticky="w")

tk.Label(frm, text="Deadband").grid(row=row, column=2, sticky="w")
tk.Entry(frm, textvariable=var_deadband, width=8).grid(row=row, column=3, sticky="w")

# Row 4
row += 1
tk.Label(frm, text="Cost (bps)").grid(row=row, column=0, sticky="w")
tk.Entry(frm, textvariable=var_costbps, width=8).grid(row=row, column=1, sticky="w")

chk = tk.Checkbutton(frm, text="Calibrate (isotonic)", variable=var_calibrate)
chk.grid(row=row, column=2, columnspan=2, sticky="w")

# Message label
row += 1
label = tk.Label(window, textvariable=label_var, font=("Arial", 11))
label.pack(pady=(8, 2))

progress = ttk.Progressbar(window, mode="indeterminate")

btn = tk.Button(window, text="Run Prediction", command=lambda: start_prediction(), font=("Arial", 12), bg="lightblue")
btn.pack(pady=10)

# Helper to read and run

def start_prediction():
    btn.config(state="disabled")
    label_var.set("Running prediction…")
    progress.pack(pady=5)
    progress.start()
    threading.Thread(target=_run_predictor_worker, daemon=True).start()


def _run_predictor_worker():
    try:
        script_path = _resolve_predictor_path()
    except FileNotFoundError as fnf:
        out, err = "", str(fnf)
        window.after(0, lambda: _on_prediction_done(out, err))
        return
    try:
        args = [
            sys.executable,
            str(script_path),
            "--ticker", var_ticker.get(),
            "--years", str(var_years.get()),
            "--horizon", str(var_horizon.get()),
            "--threshold", str(var_threshold.get()),
            "--deadband", str(var_deadband.get()),
            "--cost_bps", str(var_costbps.get()),
            "--clf_model", var_model.get(),
        ]
        if var_calibrate.get():
            args.append("--calibrate")
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        try:
            data = json.loads(result.stdout)
            out, err = data, None
        except json.JSONDecodeError:
            out, err = result.stdout, "NON_JSON"
    except subprocess.CalledProcessError as e:
        out, err = e.stdout or "", e.stderr or str(e)
    window.after(0, lambda: _on_prediction_done(out, err))


def _on_prediction_done(out, err):
    progress.stop()
    progress.pack_forget()
    btn.config(state="normal")
    label_var.set("Configure parameters then run prediction")

    if isinstance(out, dict) and not err:
        y_pred = out.get("y_pred", "?")
        proba_up = out.get("proba_up")
        last_close = out.get("last_close")
        predicted_price = out.get("predicted_price")
        predicted_return = out.get("predicted_return")
        metrics = out.get("metrics", {})
        asof = out.get("asof")

        if isinstance(proba_up, (int, float)):
            proba_txt = f"{proba_up*100:.1f}%" if (0 <= proba_up <= 1) else f"{proba_up:.1f}%"
        else:
            proba_txt = "n/a"

        lines = []
        if last_close is not None:
            lines.append(f"Dernière clôture connue : €{last_close:.2f}")
        if predicted_price is not None:
            lines.append(f"Prix prédit (t+{var_horizon.get()}j) : €{predicted_price:.2f}")
        if predicted_return is not None:
            lines.append(f"Retour prédit (t+{var_horizon.get()}j) : {predicted_return*100:.2f}%")
        lines.append(f"Décision : {y_pred}")
        lines.append(f"P(Up) : {proba_txt}")

        # CV metrics
        if isinstance(metrics, dict) and metrics:
            mae = metrics.get('reg_mae_cv_mean')
            rmse = metrics.get('reg_rmse_cv_mean')
            acc = metrics.get('clf_accuracy_cv_mean')
            auc = metrics.get('clf_auc_cv_mean')
            if mae is not None:
                lines.append(f"MAE ret (CV) : {mae*100:.2f}%")
            if rmse is not None:
                lines.append(f"RMSE ret (CV) : {rmse*100:.2f}%")
            if acc is not None:
                lines.append(f"Accuracy dir (CV) : {acc*100:.2f}%" if acc<=1 else f"Accuracy dir (CV) : {acc:.2f}%")
            if auc is not None and not np.isnan(auc):
                lines.append(f"ROC-AUC dir (CV) : {auc:.3f}")

        # Backtest metrics
        bt = metrics.get('backtest', {}) if isinstance(metrics, dict) else {}
        if bt:
            cr = bt.get('cum_return'); sh = bt.get('sharpe_annual'); hr = bt.get('hit_ratio'); dd = bt.get('max_drawdown'); tr = bt.get('trades')
            if cr is not None:
                lines.append(f"Backtest cum. return : {cr*100:.2f}%")
            if sh is not None:
                lines.append(f"Backtest Sharpe (ann.) : {sh:.2f}")
            if hr is not None:
                lines.append(f"Backtest hit ratio : {hr*100:.1f}%")
            if dd is not None:
                lines.append(f"Backtest max DD : {dd*100:.1f}%")
            if tr is not None:
                lines.append(f"Backtest trades : {tr}")

        msg = (f"Prédiction au {asof}:\n\n" if asof else "Prédiction:\n\n") + "\n".join(lines)
        messagebox.showinfo("Prediction Result", msg)
        return

    if err == "NON_JSON":
        messagebox.showwarning(
            "Result (non‑JSON)",
            "Le script n'a pas renvoyé de JSON. Affichage de la sortie brute :\n\n" + (out or "")
        )
        return

    if err:
        messagebox.showerror("Error", f"Script failed:\n{err}")
    else:
        messagebox.showinfo("Prediction Result", str(out))

window.mainloop()
