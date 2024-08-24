from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import mysql.connector
import numpy as np
import pandas as pd
import requests

class Cropping:
    def __init__(self, root):
        self.root = root
        self.root.title("CROPPING RECOMMENDATION SYSTEM")
        self.root.geometry("1360x700+0+0")
        self.root.resizable(False, False)

        self.region = StringVar()
        self.crop = StringVar()
        self.soil = StringVar()
        self.name = StringVar()
        self.temperature = StringVar()
        self.humidity = StringVar()
        self.rainfall = StringVar()
        self.id = 0
        self.season = StringVar()

        #Heading label
        label1 = Label(self.root, text="CROPPING SEASON RECOMMENDATION SYSTEM", font=("lucida sans", 20, "bold"), bg="#81CAD6", fg="white", relief="solid")
        label1.place(x=2, y=2, width=1354, height=100)

        #Background label
        label2 = Label(self.root, bg="#EDCD44", relief="solid")
        label2.place(x=2, y=104, width=1354, height=590)

        #Label frame1
        labelframe1 = LabelFrame(label2, text="CROPPING DETAILS", relief="solid", bg="white")
        labelframe1.place(x=30, y=30, width=650, height=510)

        #Region label
        label3 = Label(labelframe1, text="REGION", font=("lucida sans", 15, "bold"), bg="#DB504A", fg="white")
        label3.place(x=30, y=30, width=250, height=40)

        combobox1 = ttk.Combobox(labelframe1, font=("lucida sans", 15, "bold"), state="readonly", textvariable=self.region)
        combobox1["values"]=["SELECT REGION"]
        combobox1.current(0)
        combobox1.place(x=330, y=30, width=280, height=40)

        #Crop type label
        label4 = Label(labelframe1, text="CROP", font=("lucida sans", 15, "bold"), bg="#DB504A", fg="white")
        label4.place(x=30, y=100, width=250, height=40)

        combobox2 = ttk.Combobox(labelframe1, font=("lucida sans", 15, "bold"), state="readonly", textvariable=self.crop)
        combobox2["values"]=["SELECT CROP"]
        combobox2.current(0)
        combobox2.place(x=330, y=100, width=280, height=40)

        #Soil type
        label5 = Label(labelframe1, text="SOIL_TYPE", font=("lucida sans", 15, "bold"), bg="#DB504A", fg="white")
        label5.place(x=30, y=170, width=250, height=40)

        combobox3 = ttk.Combobox(labelframe1, font=("lucida sans", 15, "bold"), state="readonly", textvariable=self.soil)
        combobox3["values"]=["SELECT SOIL"]
        combobox3.current(0)
        combobox3.place(x=330, y=170, width=280, height=40)

        #Name label
        label6 = Label(labelframe1, text="NAME", font=("lucida sans", 15, "bold"), bg="#DB504A", fg="white")
        label6.place(x=30, y=240, width=250, height=40)

        entry1 = ttk.Entry(labelframe1, font=("lucida sans", 15, "bold"), textvariable=self.name)
        entry1.place(x=330, y=240, width=280, height=40)

        #Cropping season button
        b1 = Button(labelframe1, text="PRESS TO GENERATE CROPPING SEASON", font=("lucida sans", 15, "bold"), cursor="hand2", bg="#084C61", relief="solid", fg="White", command=self.season_generation_button)
        b1.place(x=30, y=350, width=580, height=80)

        #Frame
        f1 = Frame(label2, bg="white")
        f1.place(x=700, y=30, width=620, height=510)

        # Scrollbar
        scroll_x = ttk.Scrollbar(f1, orient="horizontal")
        scroll_y = ttk.Scrollbar(f1, orient="vertical")

        #Treeview
        self.cropping_table = ttk.Treeview(f1, columns=("S.NO.", "FARMER_NAME", "REGION", "CROP", "SOIL_TYPE", "TEMPERATURE", "RAINFALL", "HUMIDITY", "SEASON"), xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        scroll_x.pack(side="bottom", fill="x")
        scroll_y.pack(side="right", fill="y")
        scroll_x.config(command=self.cropping_table.xview)
        scroll_y.config(command=self.cropping_table.yview)

        self.cropping_table.heading("S.NO.", text="S.NO.")
        self.cropping_table.heading("FARMER_NAME", text="FARMER_NAME")
        self.cropping_table.heading("REGION", text="REGION")
        self.cropping_table.heading("CROP", text="CROP")
        self.cropping_table.heading("SOIL_TYPE", text="SOIL_TYPE")
        self.cropping_table.heading("TEMPERATURE", text="TEMPERATURE")
        self.cropping_table.heading("RAINFALL", text="RAINFALL")
        self.cropping_table.heading("HUMIDITY", text="HUMIDITY")
        self.cropping_table.heading("SEASON", text="SEASON")

        self.cropping_table["show"] = "headings"
        
        self.cropping_table.column("S.NO.", width=150)
        self.cropping_table.column("FARMER_NAME", width=150)
        self.cropping_table.column("REGION", width=150)
        self.cropping_table.column("CROP", width=150)
        self.cropping_table.column("SOIL_TYPE", width=150)
        self.cropping_table.column("TEMPERATURE", width=150)
        self.cropping_table.column("RAINFALL", width=150)
        self.cropping_table.column("HUMIDITY", width=150)
        self.cropping_table.column("SEASON", width=150)

        self.cropping_table.pack(fill="both", expand=1)

    def season_generation_button(self):
        if self.region.get() == "SELECT REGION" or self.crop.get() == "SELECT CROP" or self.soil.get() == "SELECT SOIL" or self.name.get() == "":
            messagebox.showerror("ERROR", "All fields should be filled!!!", parent=self.root)
        else:
            self.temperature, self.rainfall, self.humidity = self.get_weather_data()
            
            x_input = [self.region.get(), self.crop.get(), self.soil.get(), self.temperature, self.humidity, self.rainfall]
            
            x, y = self.load_data_and_train(x_input)
            
            self.season = self.predict_season(self.model, x_input)
            
            conn = mysql.connector.connect(host="localhost", user="root", password="An@nd3009", database="cropping_recommendation_system")
            my_cursor = conn.cursor()
            my_cursor.execute("INSERT INTO recommendation (FARMER_NAME, REGION, CROP, SOIL_TYPE, TEMPERATURE, RAINFALL, HUMIDITY, SEASON) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                              (self.name.get(), self.region.get(), self.crop.get(), self.soil.get(), self.temperature, self.rainfall, self.humidity, self.season))
            conn.commit()
            conn.close()

            self.fetch_data()

            messagebox.showinfo("Prediction", f"The recommended cropping season is: {self.season}")

    def get_weather_data(self):
        api_key="your api key"
        region = self.region.get()
        url = f"http://apiname/data/2.5/weather?q={region}&appid={api_key}&units=metric"
        
        try:
            response = requests.get(url)
            response.raise_for_status()  #Check for HTTP errors
            data = response.json()

            temperature = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            
            #Rainfall is part of the 'rain' dictionary, which might not be present if there is no rain
            rainfall = data.get("rain", {}).get("1h", 0)  #Get the rainfall in the last hour, default to 0 if not available

            return temperature, rainfall, humidity
        
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Failed to retrieve weather data: {e}")
            return None, None, None
    
    def load_data_and_train(self, x_input):
        conn = mysql.connector.connect(host="localhost", user="root", password="An@nd3009", database="cropping_recommendation_system")
        my_cursor = conn.cursor()

        my_cursor.execute("SELECT * FROM recommendation")
        records = my_cursor.fetchall()
        conn.close()

        if records:
            df = pd.DataFrame(records, columns=["ID", "FARMER_NAME", "REGION", "CROP", "SOIL_TYPE", "TEMPERATURE", "RAINFALL", "HUMIDITY", "SEASON"])
            x = df[["REGION", "CROP", "SOIL_TYPE", "TEMPERATURE", "HUMIDITY", "RAINFALL"]]
            y = df["SEASON"]

            #Append new input data
            new_data = pd.DataFrame([x_input], columns=x.columns)
            x = pd.concat([x, new_data], ignore_index=True)

            #Label encode categorical variables
            le_region = LabelEncoder()
            le_crop = LabelEncoder()
            le_soil = LabelEncoder()
            x["REGION"] = le_region.fit_transform(x["REGION"])
            x["CROP"] = le_crop.fit_transform(x["CROP"])
            x["SOIL_TYPE"] = le_soil.fit_transform(x["SOIL_TYPE"])

            #Standardize the data
            scaler = StandardScaler()
            x = scaler.fit_transform(x)

            #Train the model
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            self.model = RandomForestClassifier()
            self.model.fit(x_train, y_train)

            #Evaluate the model
            y_pred = self.model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model accuracy: {accuracy * 100:.2f}%")

            return scaler.transform(new_data), y

        else:
            df = pd.DataFrame([x_input], columns=["REGION", "CROP", "SOIL_TYPE", "TEMPERATURE", "HUMIDITY", "RAINFALL"])

            #Label encode categorical variables
            le_region = LabelEncoder()
            le_crop = LabelEncoder()
            le_soil = LabelEncoder()
            df["REGION"] = le_region.fit_transform(df["REGION"])
            df["CROP"] = le_crop.fit_transform(df["CROP"])
            df["SOIL_TYPE"] = le_soil.fit_transform(df["SOIL_TYPE"])

            #Standardize the data
            scaler = StandardScaler()
            x_input = scaler.fit_transform(df)

            #Initialize model with dummy prediction
            self.model = RandomForestClassifier()
            self.model.fit(x_input, ["Unknown"])  #Placeholder training with dummy label

            return x_input, ["Unknown"]

    def predict_season(self, model, x_input):
        prediction = model.predict(x_input)
        return prediction[0]

    def fetch_data(self):
        conn = mysql.connector.connect(host="localhost", user="root", password="An@nd3009", database="cropping_recommendation_system")
        my_cursor = conn.cursor()
        my_cursor.execute("SELECT * FROM recommendation")
        rows = my_cursor.fetchall()
        conn.close()
        
        if len(rows) != 0:
            self.cropping_table.delete(*self.cropping_table.get_children())
            for row in rows:
                self.cropping_table.insert('', END, values=row)

if __name__ == "__main__":
    root = Tk()
    obj = Cropping(root)
    root.mainloop()
