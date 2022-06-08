import imp
import os
import tkinter as tk
from tkinter import Label, image_names
from tkinter import filedialog
from pydicom import dcmread
import numpy as np
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from pre_processing import preprocess
from predict_result import feed_model


def select_image():
    image_loc = filedialog.askopenfilename(filetypes=[("Image File", '.dcm')])
    ds = dcmread(image_loc)
    filename = os.path.basename(image_loc)
    a = preprocess(ds.pixel_array, ds.PixelSpacing, 30, 70, ds)
    b = preprocess(ds.pixel_array, ds.PixelSpacing, 600, 2000, ds)
    c = preprocess(ds.pixel_array, ds.PixelSpacing, 80, 180, ds)
    j = preprocess(ds.pixel_array, ds.PixelSpacing, 50, 120, ds)

    path_image = ttk.Label(win, text=filename)
    path_image.grid(column=1, row=1, padx=5, pady=5)

    SOURCE = "data\\image\\"
    plt.imsave(SOURCE + 'brain' + '.png', a, cmap='gray')
    plt.imsave(SOURCE + 'bone' + '.png', b, cmap='gray')
    plt.imsave(SOURCE + 'blood' + '.png', c, cmap='gray')
    plt.imsave(SOURCE + 'default' + '.png', j, cmap='gray')

    im1 = Image.fromarray(j)
    new_image = im1.resize((200, 200))
    tkimage1 = ImageTk.PhotoImage(new_image)
    myvar1 = Label(
        win,
        image=tkimage1,
    )
    myvar1.image = tkimage1
    myvar1.grid(row=2, column=0, pady=2)
    original_label = ttk.Label(win, text="Original")
    original_label.grid(column=0, row=3, padx=5, pady=5)

    im2 = Image.fromarray(a)
    new_image = im2.resize((200, 200))

    tkimage2 = ImageTk.PhotoImage(new_image)
    myvar2 = Label(
        win,
        image=tkimage2,
    )
    myvar2.image = tkimage2
    myvar2.grid(row=2, column=1, pady=2)
    brain_label = ttk.Label(win, text="Brain")
    brain_label.grid(column=1, row=3, padx=5, pady=5)

    im3 = Image.open(SOURCE + 'bone' + '.png')
    new_image = im3.resize((200, 200))
    tkimage3 = ImageTk.PhotoImage(new_image)
    myvar3 = Label(
        win,
        image=tkimage3,
    )
    myvar3.image = tkimage3
    myvar3.grid(row=4, column=0, pady=2)
    bone_label = ttk.Label(win, text="Bone")
    bone_label.grid(column=0, row=5, padx=5, pady=5)

    im4 = Image.fromarray(c)
    new_image = im4.resize((200, 200))

    tkimage4 = ImageTk.PhotoImage(new_image)
    myvar4 = Label(
        win,
        image=tkimage4,
    )
    myvar4.image = tkimage4
    myvar4.grid(row=4, column=1, pady=2)
    blood_label = ttk.Label(win, text="Blood")
    blood_label.grid(column=1, row=5, padx=5, pady=5)
    diagnosis = feed_model()

    blood_label = tk.Label(win,
                           text=str(diagnosis),
                           fg="white",
                           bg="black",
                           justify='center',
                           width=25,
                           height=3,
                           font=("Arial", 10))
    blood_label.grid(row=6, padx=5, pady=5)


win = tk.Tk()
win.title('ICH Deep Learning Application')
win.geometry("430x700")
#You want the size of the app to be 500x500
# win.rowconfigure(0, weight=1)
# win.grid_columnconfigure(0, weight=1)

greeting = tk.Label(text="Intracranial Hemorrhage Detection",
                    fg="white",
                    bg="black",
                    height=3,
                    justify='center')
button = tk.Button(text="Select Image",
                   width=25,
                   height=3,
                   bg="blue",
                   fg="yellow",
                   relief=tk.RAISED,
                   justify='center',
                   command=select_image)
button.rowconfigure(0, weight=1)
button.columnconfigure(1, weight=1)

button.grid_rowconfigure(
    1,
    weight=1,
)

greeting.grid(
    row=0,
    pady=2,
)
button.grid(row=1, column=0, pady=2)
win.mainloop()
