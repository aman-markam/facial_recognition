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
        self.employee_name = tk.StringVar()
        tk.Entry(form, textvariable=self.employee_name, width=32).grid(row=1, column=1, sticky="w", pady=6)
        password_frame = ttk.Frame(form)
        password_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(password_frame, text="Password").pack(side="left", padx=(0, 12))
        self.password = tk.StringVar()
        self.password_entry = ttk.Entry(password_frame, textvariable=self.password, show="*", width=25)
        self.password_entry.pack(side="left")
        self.show_password = tk.BooleanVar()
        ttk.Checkbutton(password_frame, text="Show", variable=self.show_password, command=self._toggle_password).pack(side="left", padx=8)
        self.caps_status = ttk.Label(password_frame, text="")
        self.caps_status.pack(side="left")
        self.password_entry.bind("<KeyPress>", self._check_caps_lock)
        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=22)
        ttk.Button(actions, text="Register employee", command=self.register_employee).grid(row=0, column=0, padx=(0, 8), pady=5)
        ttk.Button(actions, text="Train recognizer", command=self.train).grid(row=0, column=1, padx=8, pady=5)
        ttk.Button(actions, text="Take attendance", command=self.take_attendance).grid(row=1, column=0, padx=(0, 8), pady=5)
        ttk.Button(actions, text="View attendance", command=self.view_attendance).grid(row=1, column=1, padx=8, pady=5)
        ttk.Button(actions, text="Export Excel", command=self.export_excel).grid(row=0, column=2, rowspan=2, padx=24, pady=5, ipady=8)
        self.status = ttk.Label(frame, text="Ready", foreground="#555555")
        self.status.pack(anchor="w", pady=(8, 0))

    def _toggle_password(self) -> None:
        self.password_entry.configure(show="" if self.show_password.get() else "*")

    def _check_caps_lock(self, _event: tk.Event) -> None:
        try:
            if self.tk.call("tk windowingsystem") == "win32":
                caps_on = bool(ctypes.WinDLL("User32.dll").GetKeyState(0x14) & 1)
            else:
                caps_on = bool(self.tk.call("::tk::mac::IsCapsLockOn"))
        except tk.TclError:
            caps_on = False
        self.caps_status.configure(text="Caps Lock is on" if caps_on else "")

    def register_employee(self) -> None:
        employee_id, name = self.employee_id.get().strip(), self.employee_name.get().strip()
        if not employee_id.isdigit() or not name:
            messagebox.showwarning("Missing information", "Enter a numeric employee ID and a name.")
            return
        try:
            captured = capture_faces(employee_id, name, CASCADE)
            if captured:
                database.save_employee(employee_id, name)
                self.status.configure(text=f"Captured {captured} face images for {name}.")
            else:
                self.status.configure(text="Registration cancelled.")
        except (RuntimeError, cv2.error) as error:
            messagebox.showerror("Registration failed", str(error))

    def train(self) -> None:
        try:
            count = train_model()
            self.status.configure(text=f"Recognizer trained with {count} images.")
        except (RuntimeError, cv2.error) as error:
            messagebox.showerror("Training failed", str(error))

    def take_attendance(self) -> None:
        try:
            result = attendance.recognize_once(CASCADE)
            self.status.configure(text=result or "Attendance cancelled.")
        except (RuntimeError, cv2.error) as error:
            messagebox.showerror("Attendance failed", str(error))

    def view_attendance(self) -> None:
        window = tk.Toplevel(self)
        window.title("Attendance records")
        window.geometry("560x320")
        tree = ttk.Treeview(window, columns=("id", "name", "date", "time"), show="headings")
        for column, heading in zip(tree["columns"], ("Employee ID", "Name", "Date", "Time")):
            tree.heading(column, text=heading)
            tree.column(column, width=125)
        for row in database.attendance_rows():
            tree.insert("", "end", values=(row["employee_id"], row["name"], row["date"], row["time"]))
        tree.pack(fill="both", expand=True, padx=12, pady=12)

    def export_excel(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=(("Excel workbook", "*.xlsx"),))
        if path:
            attendance.export_excel(path)
            self.status.configure(text=f"Exported attendance to {Path(path).name}.")


if __name__ == "__main__":
    AttendanceApp().mainloop()