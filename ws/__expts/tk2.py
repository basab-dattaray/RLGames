from tkinter import *
root1 = Tk()
for fm in ['violet', 'indigo', 'blue', 'green', 'yellow','orange','red']:
    Frame(height = 25,width = 740,bg = fm).pack()
Frame(height = 25,width = 740,bg = 'green').pack()
root1.mainloop()