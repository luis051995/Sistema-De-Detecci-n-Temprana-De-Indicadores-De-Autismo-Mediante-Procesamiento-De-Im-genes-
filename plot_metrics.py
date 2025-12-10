import pandas as pd
import matplotlib.pyplot as plt

# Cargar CSV generado por YOLO
df = pd.read_csv("runs/detect/train/results.csv")

# ---- accuracy (precision) ----
plt.plot(df['epoch'], df['metrics/precision(B)'])
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.title("Precisión por Época")
plt.grid()
plt.show()

# ---- error (loss) ----
plt.plot(df['epoch'], df['train/box_loss'], label="Box Loss")
plt.plot(df['epoch'], df['train/cls_loss'], label="Class Loss")
plt.xlabel("Época")
plt.ylabel("Error")
plt.title("Pérdida de Entrenamiento")
plt.grid()
plt.legend()
plt.show()
