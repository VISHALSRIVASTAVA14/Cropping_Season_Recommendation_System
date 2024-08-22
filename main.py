from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import mysql.connector
import numpy as np
import pandas as pd

class cropping:
    def __init__(self,root):
        self.root=root
        self.root.title("CROPPING RECOMMENDATION SYSTEM")
        self.root.geometry("1360x700+0+0")
        self.root.resizable(False,False)

        #heading label
        label1=Label(self.root,text="CROPPING SEASON RECOMMENDATION SYSTEM",font=("lucida sans",20,"bold"),bg="#81CAD6",fg="white",relief="solid")
        label1.place(x=2,y=2,width=1354,height=100)

        #background label
        label2=Label(self.root,bg="#EDCD44",relief="solid")
        label2.place(x=2,y=104,width=1354,height=590)

        #label frame1
        labelframe1=LabelFrame(label2,text="CROPPING DETAILS",relief="solid",bg="white")
        labelframe1.place(x=30,y=30,width=650,height=510)

        #region label
        label3=Label(labelframe1,text="REGION",font=("lucida sans",15,"bold"),bg="#DB504A",fg="white")
        label3.place(x=30,y=30,width=250,height=40)

        combobox1=ttk.Combobox(labelframe1,font=("lucida sans",15,"bold"),state="readonly")
        combobox1["values"]=["SELECT REGION"]
        combobox1.current(0)
        combobox1.place(x=330,y=30,width=250,height=40)

        #crop type label
        label4=Label(labelframe1,text="CROP",font=("lucida sans",15,"bold"),bg="#DB504A",fg="white")
        label4.place(x=30,y=100,width=250,height=40)

        combobox2=ttk.Combobox(labelframe1,font=("lucida sans",15,"bold"),state="readonly")
        combobox2["values"]=["SELECT CROP"]
        combobox2.current(0)
        combobox2.place(x=330,y=100,width=250,height=40)

        #soil type
        label5=Label(labelframe1,text="SOIL",font=("lucida sans",15,"bold"),bg="#DB504A",fg="white")
        label5.place(x=30,y=170,width=250,height=40)

        combobox3=ttk.Combobox(labelframe1,font=("lucida sans",15,"bold"),state="readonly")
        combobox3["values"]=["SELECT SOIL"]
        combobox3.current(0)
        combobox3.place(x=330,y=170,width=250,height=40)

        #name label
        label6=Label(labelframe1,text="NAME",font=("lucida sans",15,"bold"),bg="#DB504A",fg="white")
        label6.place(x=30,y=240,width=250,height=40)

        entry1=ttk.Entry(labelframe1,font=("lucida sans",15,"bold"))
        entry1.place(x=330,y=240,width=250,height=40)

        #cropping season button
        b1=Button(labelframe1,text="PRESS TO GENERATE CROPPING SEASON",font=("lucida sans",15,"bold"),cursor="hand2",bg="#084C61",relief="solid",fg="White")
        b1.place(x=30,y=350,width=600,height=80)

if __name__=="__main__":
    root=Tk()
    obj=cropping(root)
    root.mainloop()