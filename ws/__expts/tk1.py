import tkinter as tk


class TopFrame(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid(row=0, column=0, sticky="nsew")

        self.BottomFrame = tk.Frame(master=master)
        self.BottomFrame.grid(row=1, column=0, sticky="nsew")

        self.f1 = tk.Frame(master=self.BottomFrame)
        self.f2 = tk.Frame(master=self.BottomFrame)
        self.f3 = tk.Frame(master=self.BottomFrame)

        for f in (self.f1, self.f2, self.f3):
            f.grid(row=0, column=0, sticky="nsew")

        self.b1 = tk.Button(master=self, text="Add Words")
        self.b2 = tk.Button(master=self, text="Add From File")
        self.b3 = tk.Button(master=self, text="Change Words")

        self.add_button = tk.Button(master=self.f1, text="Add")
        self.open_button = tk.Button(master=self.f2, text="Open File")
        self.change_button = tk.Button(master=self.f3, text="Change")

        self.l1 = tk.Label(master=self.f1, text="English")
        self.l2 = tk.Label(master=self.f1, text="Turkish")
        self.l3 = tk.Label(master=self.f3, text="Old word")
        self.l4 = tk.Label(master=self.f3, text="New word")

        self.e1 = tk.Entry(master=self.f1)
        self.e2 = tk.Entry(master=self.f1)
        self.e3 = tk.Entry(master=self.f3)
        self.e4 = tk.Entry(master=self.f3)

        self.configure_buttons()
        self.configure_labels()
        self.configure_entries()

    def configure_buttons(self):
        self.b1.grid(row=0, column=0)
        self.b1.configure(command=lambda: self.f1.tkraise())
        self.b2.grid(row=0, column=1)
        self.b2.configure(command=lambda: self.f2.tkraise())
        self.b3.grid(row=0, column=2)
        self.b3.configure(command=lambda: self.f3.tkraise())

        self.add_button.grid(row=2, columnspan=2)
        #self.add_button.configure(command=self.add_word)
        self.open_button.pack(side="top")
        #self.open_button.configure(command=self.add_from_file)
        self.change_button.grid(row=2, columnspan=2)

    def configure_labels(self):
        self.l1.grid(row=0, column=0)
        self.l2.grid(row=0, column=1)
        self.l3.grid(row=0, column=0)
        self.l4.grid(row=0, column=1)

    def configure_entries(self):
        self.e1.grid(row=1, column=0)
        self.e2.grid(row=1, column=1)
        self.e3.grid(row=1, column=0)
        self.e4.grid(row=1, column=1)


if __name__ == "__main__":
    root = tk.Tk()
    example = TopFrame(master=root)
    example.mainloop()