import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import subprocess
import sys
from pathlib import Path
import threading
import json
from typing import List


# Expected stdout JSON from predictor:
# {
#   "y_pred": 0 | 1 | "UP" | "DOWN",
#   "proba_up": 0.0-1.0,
#   "metrics": {"accuracy": 0.87, "precision_up": 0.74, ...},
#   "asof": "YYYY-MM-DD"
# }


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
        here / "ALST_Ticker_Predictor.py",
        here / "ALST_Ticker_Predictor_Final.py",
        # project root
        project / "ALST_Ticker_Predictor.py",
        project / "ALST_Ticker_Predictor_Final.py",
        # common subfolders
        project / "ALST_PP" / "ALST_Ticker_Predictor.py",
        project / "ALST_PP" / "ALST_Ticker_Predictor_Final.py",
        project / "src" / "ALST_Ticker_Predictor.py",
        project / "src" / "ALST_Ticker_Predictor_Final.py",
        project / "scripts" / "ALST_Ticker_Predictor.py",
        project / "scripts" / "ALST_Ticker_Predictor_Final.py",
    ]

    for p in candidates:
        if p.exists():
            return p

    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not locate predictor script. Tried the following paths:\n" + tried
    )


def start_prediction():
    # Disable UI and show spinner, then run worker in background thread
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
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--ticker", "ALO.PA",
                "--years", "5",
                "--horizon", "1"
            ],
            capture_output=True, text=True, check=True
        )
        try:
            data = json.loads(result.stdout)
            out, err = data, None
        except json.JSONDecodeError:
            # Fallback: not JSON, pass through raw text so UI can warn and display
            out, err = result.stdout, "NON_JSON"
    except subprocess.CalledProcessError as e:
        out, err = e.stdout or "", e.stderr or str(e)
    # Marshal back to the Tk thread
    window.after(0, lambda: _on_prediction_done(out, err))


def _on_prediction_done(out, err):
    # Stop spinner, re-enable UI
    progress.stop()
    progress.pack_forget()
    btn.config(state="normal")
    label_var.set("Click below to run the prediction script")

    if isinstance(out, dict) and not err:
        # Expected JSON contract: {"y_pred": 0/1 or label, "proba_up": float 0-1, "metrics": {…}, "asof": "YYYY-MM-DD"}
        y_pred = out.get("y_pred", "?")
        proba_up = out.get("proba_up")
        metrics = out.get("metrics", {})
        asof = out.get("asof")

        # Normalize and format
        if isinstance(proba_up, (int, float)):
            proba_txt = f"{proba_up*100:.1f}%" if (0 <= proba_up <= 1) else f"{proba_up:.1f}%"
        else:
            proba_txt = "n/a"

        # Build metrics text block
        if isinstance(metrics, dict) and metrics:
            m_lines = [f"  - {k}: {v}" for k, v in metrics.items()]
            metrics_txt = "\n".join(m_lines)
        else:
            metrics_txt = "  - none"

        header = f"Prediction as of {asof}:" if asof else "Prediction:"
        msg = f"{header}\n\n  Direction (y_pred): {y_pred}\n  P(Up): {proba_txt}\n\nMetrics:\n{metrics_txt}"
        messagebox.showinfo("Prediction Result", msg)
        return

    if err == "NON_JSON":
        # Show raw output with a gentle nudge to switch to JSON contract
        messagebox.showwarning(
            "Result (non‑JSON)",
            "Le script n'a pas renvoyé de JSON. Affichage de la sortie brute :\n\n" + (out or "")
        )
        return

    if err:
        messagebox.showerror("Error", f"Script failed:\n{err}")
    else:
        messagebox.showinfo("Prediction Result", str(out))

# --- Tkinter GUI Setup ---
window = tk.Tk()
window.title("Alstom Stock Predictor")
window.geometry("350x180")

label_var = tk.StringVar(value="Click below to run the prediction script")
label = tk.Label(window, textvariable=label_var, font=("Arial", 12))
label.pack(pady=10)

progress = ttk.Progressbar(window, mode="indeterminate")

btn = tk.Button(window, text="Run Prediction", command=start_prediction, font=("Arial", 12), bg="lightblue")
btn.pack(pady=10)

window.mainloop()
