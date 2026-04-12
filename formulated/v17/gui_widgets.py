import tkinter as tk

class SmartEntry(tk.Entry):
    """
    A persistent text entry widget that syncs with a Tkinter variable.
    Only updates the variable on Return or FocusOut, and erases on FocusIn.
    """
    def __init__(self, parent, variable, **kwargs):
        """
        Initialize the SmartEntry.
        
        Args:
            parent: Parent widget.
            variable (tk.Variable): The variable to sync with.
            kwargs: Standard tk.Entry keyword arguments.
        """
        super().__init__(parent, **kwargs)
        self.variable = variable
        self.insert(0, str(variable.get()))
        
        # Color state from kwargs or defaults
        self.muted_fg = "#94a3b8" if "fg" not in kwargs else kwargs["fg"]
        self.active_fg = "#f8fafc"
        
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<Return>", self._update_var)
        self.bind("<FocusOut>", self._update_var)
        
        self.variable.trace_add("write", self._update_entry)

    def _on_focus_in(self, event):
        """Handle focus in by erasing and highlighting."""
        self.config(fg=self.active_fg)
        self.delete(0, tk.END)

    def _update_var(self, event=None):
        """Update the variable from the entry text."""
        try:
            val = float(self.get())
            self.variable.set(val)
            self.selection_clear()
            self.winfo_toplevel().focus_set()
        except ValueError:
            self._update_entry() # Restore old value
        self.config(fg=self.muted_fg)

    def _update_entry(self, *args):
        """Update the entry text from the variable if not focused."""
        if self.focus_get() != self:
            val = self.variable.get()
            self.delete(0, tk.END)
            # Match formatting of v16 desync fix
            if isinstance(val, float):
                self.insert(0, f"{val:.1f}")
            else:
                self.insert(0, str(val))
            self.config(fg=self.muted_fg)
