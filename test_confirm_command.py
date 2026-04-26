from tkinter import *
from tkinter import ttk

#Additional Window for user input
def test_confirm_command():
    root = Tk()
    root.geometry("720x500")
    
    Label(root, text="R-Gesture").grid(row = 0, column = 2)
    Label(root, text="Open").grid(row = 1, column = 2)
    Label(root, text="Open").grid(row = 2, column = 2)
    Label(root, text="Open").grid(row = 3, column = 2)
    Label(root, text="Close").grid(row = 4, column = 2)
    Label(root, text="Close").grid(row = 5, column = 2)
    Label(root, text="Close").grid(row = 6, column = 2)
    Label(root, text="Pointer").grid(row = 7, column = 2)
    Label(root, text="Pointer").grid(row = 8, column = 2)
    Label(root, text="Pointer").grid(row = 9, column = 2)

    Label(root, text="L-Gesture").grid(row = 0, column = 1)
    Label(root, text="Open").grid(row = 1, column = 1)
    Label(root, text="Close").grid(row = 2, column = 1)
    Label(root, text="Pointer").grid(row = 3, column = 1)
    Label(root, text="Open").grid(row = 4, column = 1)
    Label(root, text="Close").grid(row = 5, column = 1)
    Label(root, text="Pointer").grid(row = 6, column = 1)
    Label(root, text="Open").grid(row = 7, column = 1)
    Label(root, text="Close").grid(row = 8, column = 1)
    Label(root, text="Pointer").grid(row = 9, column = 1)

    #Dropdown menu options
    options = ["Select-All", "Copy", "Paste", "Close-Window", "Backspace",
               "Cut", "Mouse-Movement", "Left-Click", "Right-Click","(None)"]

    #using combobox 
    OOMenu = ttk.Combobox(root, values = options)
    OOMenu.set("(None)")
    OOMenu.grid(row = 1, column = 3)

    OCMenu = ttk.Combobox(root, values = options)
    OCMenu.set("Copy")
    OCMenu.grid(row = 2, column = 3)

    OPMenu = ttk.Combobox(root, values = options)
    OPMenu.set("Paste")
    OPMenu.grid(row = 3, column = 3)

    COMenu = ttk.Combobox(root, values = options)
    COMenu.set("Backspace")
    COMenu.grid(row = 4, column = 3)

    CCMenu = ttk.Combobox(root, values = options)
    CCMenu.set("(None)")
    CCMenu.grid(row = 5, column = 3)
    
    CPMenu = ttk.Combobox(root, values = options)
    CPMenu.set("Cut")
    CPMenu.grid(row = 6, column = 3)

    POMenu = ttk.Combobox(root, values = options)
    POMenu.set("Mouse")
    POMenu.grid(row = 7, column = 3)

    PCMenu = ttk.Combobox(root, values = options)
    PCMenu.set("Left-Click")
    PCMenu.grid(row = 8, column = 3)

    PPMenu = ttk.Combobox(root, values = options)
    PPMenu.set("Right-Click")
    PPMenu.grid(row = 9, column = 3)

    #create and populate array of chosen command options
    selectedOptions = []
    selectedOptions.append(OOMenu.get())
    selectedOptions.append(OCMenu.get())
    selectedOptions.append(OPMenu.get())
    selectedOptions.append(COMenu.get()) 
    selectedOptions.append(CCMenu.get()) 
    selectedOptions.append(CPMenu.get()) 
    selectedOptions.append(POMenu.get())
    selectedOptions.append(PCMenu.get()) 
    selectedOptions.append(PPMenu.get())

    assert selectedOptions[0] == "(None)"
    assert selectedOptions[1] == "Copy"
    assert selectedOptions[2] == "Paste"
    assert selectedOptions[3] == "Backspace"
    assert selectedOptions[4] == "(None)", "Closed/Closed Command should be (None) by default"
    assert selectedOptions[5] == "Cut"
    assert selectedOptions[6] == "Mouse"
    assert selectedOptions[7] == "Left-Click"
    assert selectedOptions[8] == "Right-Click"
        