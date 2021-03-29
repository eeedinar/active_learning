from tkinter import *
from PIL import ImageTk, Image
import os, shutil

height = 300
width = 225
seek_str = "jpeg"


#### Tkinter Initialization/Started text and attributes
root = Tk()
root.title("Human Annotator")
root.geometry("800x600")
#root.iconbitmap('Linux.ico')

frame_radio  = Frame(root)
frame_status = Frame(root)
frame_radio.grid(row=2, column=3, columnspan=1,)               # Frame Location in ROOT
frame_status.grid(row=4,column=3, columnspan=1,)               # Frame Location in ROOT


#### Text Label
myLabel  = Label(root,  text="Please Enter Folder Address ", fg="red").grid(row=0,column=0)    # label widget height=300, width=225

#### Text Entry
e = Entry(root, width=40, borderwidth=5,)
e.insert(0,"/Users/bashit.a/Documents/EECE7370/project/images")                                 # 0 because there is only on e box 
e.grid(row=0,column=1, columnspan=2, padx=10, pady=10)

#### Button
img_files = []  # files to annotate  ['unt-1.jpeg', 'unt-2.jpeg']
status_label = Label(frame_status, fg='black')         # show status label
status_label.grid(row=0, column=0)

def myClick():
    global folder, img_files, img_objs , status_label
    folder = e.get()   # folder = "/Users/bashit.a/Documents/EECE7370/project/images"
    for file in [each for each in os.listdir(folder) if each.endswith(seek_str)]:
        img_files.append(file)     
    img_files = sorted(img_files)

    if len(img_files)!=0:
        img_objs = [ImageTk.PhotoImage(Image.open(folder+"/"+item).resize((height, width), Image.ANTIALIAS)) for item in img_files]   # list of resized PhotoImage objects
        status_label.grid_forget()
        annotation()
    else:
        status_label = Label(frame_status, text = "No image to Annonate", fg='black')
        status_label.grid(row=0, column=0)

myButton = Button(root, text="GO!", padx=50, pady=5, fg="blue", bg="#ffffff",command=myClick)    # state=DISABLED
myButton.grid(row=0,column=3,columnspan=1,)

#### image read and resize, Image Label
# my_img = Image.open("Linux.ico")                               # open image
# my_img = my_img.resize((height, width), Image.ANTIALIAS)       # resize image
# my_img = ImageTk.PhotoImage(my_img)         # ImageTk.PhotoImage(Image.open("Linux.ico")) or ImageTk.PhotoImage(file="Linux.ico")
# my_img_label = Label(root, image=my_img)                           # put image to a label
# my_img_label.grid(row=6,column=0,columnspan=3,)                    # display in grid



#### Forward and Backward Button to scroll around throught pictures in a folder

class image_show:

    def __init__(self, master):

        self.master = master
        self.frame_im   = Frame(master)
        self.frame_tool = Frame(master)

    def initialize(self, img_idx=0):
        # Global variables initialization
        self.height, self.width = height, width     
        self.img_files = img_files      

        self.img_idx = img_idx if img_idx < len(self.img_files) else len(self.img_files)-1   # if last image is poped then go to the new last index

        if len(self.img_files)!=0: 

            self.img_objs = img_objs 
            self.my_img_label = Label(self.frame_im, image=self.img_objs[self.img_idx])  # put first image idx to a label
            self.my_img_label.grid(row=0,column=0,columnspan=3,)                         # display in grid
            
            self.button_back    = Button(self.frame_tool, text="<<", fg='black', state=DISABLED) if self.img_idx ==0 \
                             else Button(self.frame_tool, text="<<", fg='black', command=lambda: self.back(self.img_idx-1));         
            self.button_back.grid(row=0,column=0,columnspan=1,)
            self.button_forward = Button(self.frame_tool, text=">>", fg='black', command=lambda: self.forward(self.img_idx+1));      
            self.button_forward.grid(row=0,column=2,columnspan=1,)   # calling forward function
        
        else:
            pass
            self.my_img = Image.open("Linux.ico")                               # open image
            self.my_img = self.my_img.resize((self.height, self.width), Image.ANTIALIAS)       # resize image
            self.my_img = ImageTk.PhotoImage(self.my_img)                      
            self.my_img_label = Label(self.frame_im, image=self.my_img)   # WILL NOT WORK without self --> ImageTk.PhotoImage(Image.open("Linux.ico")) or ImageTk.PhotoImage(file="Linux.ico")
            self.my_img_label.grid(row=0,column=0,columnspan=3,)                    # display in grid
            self.button_back    = Button(self.frame_tool, text="<<", fg='black', state=DISABLED).grid(row=0,column=0,columnspan=1,)
            self.button_forward = Button(self.frame_tool, text=">>", fg='black', state=DISABLED).grid(row=0,column=2,columnspan=1,) 

        Button(self.frame_tool, text="Exit Program", fg='black', command=self.master.quit).grid(row=0,column=1,columnspan=1,)                   # one line Exit Button

    def back(self, img_num):
        """ img_num can't go negetive """

        ### Same as forward
        #global img_idx, my_img_label, button_back, button_forward               # manipulation of these variable will be used globally
        self.img_idx = img_num;
        self.my_img_label.grid_forget()                                     # forget current image
        self.my_img_label = Label(self.frame_im, image = self.img_objs[img_num]);                                         
        self.my_img_label.grid(row=0,column=0,columnspan=3,)

        # forward button update
        self.button_forward = Button(self.frame_tool, text=">>", fg='black', command=lambda: self.forward(img_num+1))
        self.button_forward.grid(row=0,column=2,columnspan=1,)

        # DISABLED Backward Button when last image otherwise call back
        self.button_back = Button(self.frame_tool, text="<<", fg='black', state=DISABLED) if img_num == 0 \
                      else Button(self.frame_tool, text="<<", fg='black', command=lambda: self.back(img_num-1));      
        self.button_back.grid(row=0,column=0,columnspan=1,)

    def forward(self, img_num):
        
        ### Same as back
        #global img_idx, my_img_label, button_back, button_forward               # manipulation of these variable will be used globally
        self.img_idx = img_num;
        self.my_img_label.grid_forget()                                     # forget current image
        self.my_img_label = Label(self.frame_im, image = self.img_objs[img_num]);                                         
        self.my_img_label.grid(row=0,column=0,columnspan=3,) 
        
        # DISABLED Forward Button when last image otherwise call forward
        self.button_forward = Button(self.frame_tool, text=">>", fg='black', state=DISABLED) if img_num == len(self.img_objs)-1 \
                    else Button(self.frame_tool, text=">>", fg='black', command=lambda: self.forward(img_num+1));      
        self.button_forward.grid(row=0,column=2,columnspan=1,)

        # back button update
        self.button_back = Button(self.frame_tool, text="<<", fg='black', command=lambda: self.back(img_num-1));            
        self.button_back.grid(row=0,column=0,columnspan=1,)



im = image_show(root)
im.initialize(img_idx=0)
im.frame_im.grid(row=2,column=0,columnspan=3,)               # Frame Location in ROOT
im.frame_tool.grid(row=3,column=0,columnspan=3,)             # Frame Location in ROOT

#### Radio Button Executes when Button Clicked   # Images Folder operation starts now

def annotation():

    im = image_show(root)
    im.initialize(img_idx=0)
    im.frame_im.grid(row=2,column=0,columnspan=3,)               # Frame Location in ROOT
    im.frame_tool.grid(row=3,column=0,columnspan=3,)             # Frame Location in ROOT

    RADIO_MODES = next(os.walk(folder))[1]                       # Folders in the images directory ['Cat', 'Dog']
    RADIO_MODES = list(zip(RADIO_MODES, RADIO_MODES))            # [("Cat", "cat"), ("Dog","dog")]    # Option, Value pair

    r = StringVar();
    r.set("String Var")

    def myRadio(value):
        img_idx = im.img_idx
        
        Label(frame_status, text = img_files[img_idx] + " was annotated as " + value , fg='black').grid(row=0, column=0)             # change row, column number later

        shutil.move(folder+"/"+img_files[img_idx], folder+f'/{value}/'+img_files[img_idx])                         # move file into folder shutil.move(original,target)

        img_files.pop(img_idx)
        img_objs.pop(img_idx)
        im.initialize(img_idx=img_idx)
        if len(img_files) ==0:
            myRadioButtonClicked = Button(frame_radio, text="Annotate", fg='black', state=DISABLED).grid(row=idx+1,column=0,columnspan=1,)

    for idx, (text, value) in enumerate(RADIO_MODES):
        my_rad = Radiobutton(frame_radio, text = text, fg='black', variable = r, value = value, )  # command=lambda: myRadio(r.get())
        my_rad.grid(row=idx, column=0)
    myRadioButtonClicked = Button(frame_radio, text="Annotate", fg='black', command=lambda: myRadio(r.get())).grid(row=idx+1,column=0,columnspan=1,)



# pack
# pack(padx = 10, anchor=W)

#### Tkinter eventloop (Mandatory at the bottom)    
root.mainloop()                                               # tkinter mainloop
