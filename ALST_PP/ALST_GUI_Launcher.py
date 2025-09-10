
import tkinter as tk
from tkinter import messagebox
import subprocess
import sys

def run_predictor():
    try:
        # Run the predictor script and capture output
        result = subprocess.run(
            [sys.executable, "/Users/alexandrebredillot/Documents/GitHub/FIN_PP/ALST_Ticker_Predictor_Final.py"],
            capture_output=True, text=True, check=True
        )
        messagebox.showinfo("Prediction Result", result.stdout)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Script failed:\n{e.stderr}")

# --- Tkinter GUI Setup ---
window = tk.Tk()
window.title("Alstom Stock Predictor")
window.geometry("350x180")

label = tk.Label(window, text="Click below to run the prediction script", font=("Arial", 12))
label.pack(pady=10)

btn = tk.Button(window, text="Run Prediction", command=run_predictor, font=("Arial", 12), bg="lightblue")
btn.pack(pady=10)

window.mainloop()
