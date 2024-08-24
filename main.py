from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import mysql.connector
import numpy as np
import pandas as pd
import requests

class cropping:
    def __init__(self,root):
        self.root=root
        self.root.title("CROPPING RECOMMENDATION SYSTEM")
        self.root.geometry("1360x700+0+0")
        self.root.resizable(False,False)

        self.region=StringVar()
        self.crop=StringVar()
        self.soil=StringVar()
        self.name=StringVar()

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

        combobox1=ttk.Combobox(labelframe1,font=("lucida sans",15,"bold"),state="readonly",textvariable=self.region)
        combobox1["values"]=["SELECT REGION"]
        combobox1.current(0)
        combobox1.place(x=330,y=30,width=280,height=40)

        #crop type label
        label4=Label(labelframe1,text="CROP",font=("lucida sans",15,"bold"),bg="#DB504A",fg="white")
        label4.place(x=30,y=100,width=250,height=40)

        combobox2=ttk.Combobox(labelframe1,font=("lucida sans",15,"bold"),state="readonly",textvariable=self.crop)
        combobox2["values"]=["SELECT CROP"]
        combobox2.current(0)
        combobox2.place(x=330,y=100,width=280,height=40)

        #soil type
        label5=Label(labelframe1,text="SOIL_TYPE",font=("lucida sans",15,"bold"),bg="#DB504A",fg="white")
        label5.place(x=30,y=170,width=250,height=40)

        combobox3=ttk.Combobox(labelframe1,font=("lucida sans",15,"bold"),state="readonly",textvariable=self.soil)
        combobox3["values"]=["SELECT SOIL"]
        combobox3.current(0)
        combobox3.place(x=330,y=170,width=280,height=40)

        #name label
        label6=Label(labelframe1,text="NAME",font=("lucida sans",15,"bold"),bg="#DB504A",fg="white")
        label6.place(x=30,y=240,width=250,height=40)

        entry1=ttk.Entry(labelframe1,font=("lucida sans",15,"bold"),textvariable=self.name)
        entry1.place(x=330,y=240,width=280,height=40)

        #cropping season button
        b1=Button(labelframe1,text="PRESS TO GENERATE CROPPING SEASON",font=("lucida sans",15,"bold"),cursor="hand2",bg="#084C61",relief="solid",fg="White",command=self.season_generation_button)
        b1.place(x=30,y=350,width=580,height=80)

        #frame
        f1=Frame(label2,bg="white")
        f1.place(x=700,y=30,width=620,height=510)

        #scrollbar
        scroll_x=ttk.Scrollbar(f1,orient="horizontal")
        scroll_y=ttk.Scrollbar(f1,orient="vertical")

        #Treeview
        self.cropping_table=ttk.Treeview(f1,columns=("S.NO.","FARMER_NAME","REGION","CROP","SOIL_TYPE","TEMPERATURE","RAINFALL","HUMIDITY","SEASON"),xscrollcommand=scroll_x.set,yscrollcommand=scroll_y.set)
        scroll_x.pack(side="bottom",fill="x")
        scroll_y.pack(side="right",fill="y")
        scroll_x.config(command=self.cropping_table.xview)
        scroll_y.config(command=self.cropping_table.yview)

        self.cropping_table.heading("S.NO.",text="S.NO.")
        self.cropping_table.heading("FARMER_NAME",text="FARMER_NAME")
        self.cropping_table.heading("REGION",text="REGION")
        self.cropping_table.heading("CROP",text="CROP")
        self.cropping_table.heading("SOIL_TYPE",text="SOIL_TYPE")
        self.cropping_table.heading("TEMPERATURE",text="TEMPERATURE")
        self.cropping_table.heading("RAINFALL",text="RAINFALL")
        self.cropping_table.heading("HUMIDITY",text="HUMIDITY")
        self.cropping_table.heading("SEASON",text="SEASON")

        self.cropping_table["show"]="headings"
        
        self.cropping_table.column("S.NO.",width=150)
        self.cropping_table.column("FARMER_NAME",width=150)
        self.cropping_table.column("REGION",width=150)
        self.cropping_table.column("CROP",width=150)
        self.cropping_table.column("SOIL_TYPE",width=150)
        self.cropping_table.column("TEMPERATURE",width=150)
        self.cropping_table.column("RAINFALL",width=150)
        self.cropping_table.column("HUMIDITY",width=150)
        self.cropping_table.column("SEASON",width=150)

        self.cropping_table.pack(fill="both",expand=1)

        self.load_model_and_data()

    def season_generation_button(self):
        if self.region.get() == "SELECT REGION" or self.crop.get() == "SELECT CROP" or self.soil.get() == "SELECT SOIL" or self.name.get() == "":
            messagebox.showerror("ERROR", "All fields should be filled!!!", parent=self.root)
        else:
            # Fetch weather data
            temperature, rainfall, humidity = self.get_weather_data()
            x_input = [self.region.get(), self.crop.get(), self.soil.get(), temperature, humidity, rainfall]
            season = self.predict_season(self.model, x_input)
            messagebox.showinfo("Prediction", f"The recommended cropping season is: {season}")

    def get_weather_data(self):
        api_key = "YOUR_WEATHER_API_KEY"
        url = F"http://api.weatherapi.com/v1/current.json?key={api_key}&q={self.region.get()}"
        response = requests.get(url)
        data = response.json()
        
        temperature = data["current"]["temp_c"]
        humidity = data["current"]["humidity"]
        rainfall = data["current"]["precip_mm"]

        return temperature, rainfall, humidity
    
    def fetch_data(self):
        conn = mysql.connector.connect(host="localhost", user="root", password="An@nd3009", database="cropping_recommendation_system")
        my_cursor = conn.cursor()

        my_cursor.execute("SELECT * FROM recommendation")
        data = my_cursor.fetchall()

        if len(data) != 0:
            self.cropping_table.delete(*self.cropping_table.get_children())
            for i in data:
                self.cropping_table.insert("", END, values=i)
            conn.commit()
        conn.close()
    
    def load_data(self):
        conn = mysql.connector.connect(host="localhost", user="root", password="An@nd3009", database="cropping_recommendation_system")
        my_cursor = conn.cursor()

        my_cursor.execute("SELECT REGION, CROP, SOIL_TYPE, TEMPERATURE, HUMIDITY, RAINFALL, SEASON FROM recommendation")
        data = my_cursor.fetchall()

        df = pd.DataFrame(data, columns=["REGION", "CROP", "SOIL_TYPE", "TEMPERATURE", "HUMIDITY", "RAINFALL", "SEASON"])
        
        conn.commit()
        conn.close()

        x = df[["REGION", "CROP", "SOIL_TYPE", "TEMPERATURE", "HUMIDITY", "RAINFALL"]]
        y = df["SEASON"]
        
        return x, y

    def preprocess_data(self, x):
        le = LabelEncoder()
        scalar = StandardScaler()

        x["REGION"] = le.fit_transform(x["REGION"])
        x["CROP"] = le.fit_transform(x["CROP"])
        x["SOIL_TYPE"] = le.fit_transform(x["SOIL_TYPE"])

        x[["TEMPERATURE", "HUMIDITY", "RAINFALL"]] = scalar.fit_transform(x[["TEMPERATURE", "HUMIDITY", "RAINFALL"]])
        return x
    
    def train_model(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        y_prediction = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_prediction)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        return model
    
    def predict_season(self, model, x_input):
        x_input_df = pd.DataFrame([x_input], columns=["REGION", "CROP", "SOIL_TYPE", "TEMPERATURE", "HUMIDITY", "RAINFALL"])
        x_input_processed = self.preprocess_data(x_input_df)
        predicted_season = model.predict(x_input_processed)
        return predicted_season[0]

    def load_model_and_data(self):
        x, y = self.load_data()
        x = self.preprocess_data(x)
        self.model = self.train_model(x, y)

if __name__ == "__main__":
    root = Tk()
    obj = cropping(root)
    root.mainloop()
