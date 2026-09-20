from pathlib import Path
import ctypes
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

import attendance
import database
from face_capture import capture_faces
from trainer import train_model


ROOT = Path(__file__).resolve().parent
CASCADE_PATH = ROOT / "haarcascade_frontalface_default.xml"
if not CASCADE_PATH.exists():
    CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
CASCADE = cv2.CascadeClassifier(str(CASCADE_PATH))


class AttendanceApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        database.initialize()
        self.title("Face Recognition Attendance")
        self.geometry("620x430")
        self.resizable(False, False)
        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self, padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Face Recognition Attendance", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Register an employee, train the model, then record attendance.").pack(anchor="w", pady=(4, 20))
        form = ttk.LabelFrame(frame, text="Employee registration", padding=14)
        form.pack(fill="x")
        tk.Label(form, text="Employee ID").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=6)
        self.employee_id = tk.StringVar()
        tk.Entry(form, textvariable=self.employee_id, width=32).grid(row=0, column=1, sticky="w", pady=6)
        ttk.Label(form, text="Name").grid(row=1, column=0, sticky="w", padx=(0, 12), pady=6)
        self.name = tk.StringVar()
        tk.Entry(form, textvariable=self.name, width=32).grid(row=1, column=1, sticky="w", pady=6)

        actions = ttk.Frame(frame, padding=(0, 20, 0, 0))
        actions.pack(fill="x")
        ttk.Button(actions, text="1. Capture faces", command=self._capture).grid(row=0, column=0, padx=(0, 8), pady=4)
        ttk.Button(actions, text="2. Train model", command=self._train).grid(row=0, column=1, padx=(0, 8), pady=4)
        ttk.Button(actions, text="3. Take attendance", command=self._attendance).grid(row=0, column=2, padx=(0, 8), pady=4)
        ttk.Button(actions, text="Export Excel", command=self._export).grid(row=0, column=3, pady=4)
        self.status = ttk.Label(frame, text="Ready", font=("Segoe UI", 10, "italic"))
        self.status.pack(anchor="w", pady=(18, 0))

    def _capture(self) -> None:
        employee_id, name = self.employee_id.get().strip(), self.name.get().strip()
        if not employee_id or not name:
            messagebox.showerror("Error", "Enter employee ID and name.")
            return
        database.upsert_employee(employee_id, name)
        count = capture_faces(employee_id, name, CASCADE)
        self.status.config(text=f"Captured {count} face samples for {name}.")

    def _train(self) -> None:
        try:
            count = train_model()
            self.status.config(text=f"Trained model on {count} face samples.")
            messagebox.showinfo("Success", f"Trained model on {count} face samples.")
        except Exception as error:
            messagebox.showerror("Error", str(error))

    def _attendance(self) -> None:
        try:
            name = attendance.recognize_once(CASCADE)
            if name:
                self.status.config(text=f"Recorded: {name}")
            else:
                self.status.config(text="Attendance recording cancelled.")
        except Exception as error:
            messagebox.showerror("Error", str(error))

    def _export(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        attendance.export_excel(path)
        messagebox.showinfo("Success", f"Exported attendance to {path}")


if __name__ == "__main__":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    AttendanceApp().mainloop()
