import tkinter as tk
from tkinter import ttk
from constants import *

def create_card(parent, title, title_color, **pack_kw):
    f = tk.LabelFrame(parent, text=f"  {title.upper()}  ",
                      font=F_HEAD,
                      fg=title_color, bg=C_PANEL,
                      bd=2, relief="flat", highlightbackground=title_color, 
                      highlightthickness=1, padx=20, pady=12)
    f.pack(**pack_kw)
    return f

def create_smart_slider(parent, label, var, color, lo, hi, res=0.1):
    f = tk.LabelFrame(parent, text=f" {label} ", bg=C_PANEL,
                      font=F_HEAD, fg=color, padx=25, pady=12)
    f.pack(fill="x", pady=10)
    
    sl = tk.Scale(f, from_=lo, to=hi, resolution=res, orient="horizontal",
                  variable=var, bg=C_PANEL, fg=C_TEXT_BRT, troughcolor=C_BG,
                  highlightthickness=0, length=600, font=F_BODY)
    sl.pack(side="left", padx=20)
    
    ent = tk.Entry(f, width=10, bg=C_BG, fg=C_ACCENT,
                  insertbackground=C_TEXT_BRT, bd=0, font=F_MONO)
    ent.insert(0, f"{var.get():.1f}")
    ent.pack(side="left")
    
    def sync_entry_from_var(*_):
        try:
            if str(parent.winfo_toplevel().focus_get()).endswith(str(ent).split(".")[-1]): return
        except: pass
        ent.delete(0, tk.END)
        ent.insert(0, f"{var.get():.1f}")
        ent.config(fg=C_ACCENT)
    
    var.trace_add("write", sync_entry_from_var)
    
    def commit_change(*_):
        try:
            val = float(ent.get())
            val = max(lo, min(hi, val))
            var.set(val)
            ent.config(fg=C_ACCENT)
        except ValueError:
            sync_entry_from_var()
        parent.winfo_toplevel().focus()
    
    ent.bind("<FocusIn>", lambda _: (ent.config(fg=C_TEXT_BRT), ent.select_range(0, tk.END)))
    ent.bind("<Return>", commit_change)
    ent.bind("<FocusOut>", commit_change)
    return sl
