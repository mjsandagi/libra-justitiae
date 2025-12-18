import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

read = pd.read_csv("criminal_records.csv")

read_sentences = read[(read["Main Source of Conviction"] != "") & (read["Sentences Served"] != 0)]
read_fines = read[(read["Main Source of Conviction"] != "") & (read["Amount Fined"] != 0)]

"""print(read_fines["Main Source of Conviction"])
print(read_sentences["Main Source of Conviction"])"""

reads_pre = read_sentences.copy()
readf_pre = read_fines.copy()

label_enc = LabelEncoder()
read_sentences["Main Source of Conviction"] = label_enc.fit_transform(read_sentences["Main Source of Conviction"]) 
read_fines["Main Source of Conviction"] = label_enc.transform(read_fines["Main Source of Conviction"])

scalar_for_sentences = MinMaxScaler()
scaler_fines = MinMaxScaler()

read_sentences["Past Convictions"] = scalar_for_sentences.fit_transform(read_sentences[["Past Convictions"]])

read_fines["Past Convictions"] = scaler_fines.fit_transform(read_fines[["Past Convictions"]])
read_fines["Amount Fined"] = scaler_fines.fit_transform(read_fines[["Amount Fined"]])

X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(read_sentences[["Main Source of Conviction", "Past Convictions"]], read_sentences[["Sentences Served"]], test_size=0.2, random_state=40)
X_train_fine, X_test_fine, y_train_fine, y_test_fine = train_test_split(read_fines[["Main Source of Conviction", "Past Convictions"]], read_fines[["Amount Fined"]], test_size=0.2, random_state=40)

model_sentences = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(2,)),
    layers.Dense(24, activation='elu'),
    layers.Dense(16, activation='elu'),
    layers.Dense(1, name="sentences_output")  
])

model_fines = model_sentences
''' keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(2,)),
    layers.Dense(24, activation='elu'),
    layers.Dense(16, activation='elu'),
    layers.Dense(1, name="fine_output")  
])'''

model_sentences.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate = 0.01),
                         loss='mae',
                           metrics=['mae'])
model_fines.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate = 0.01),
                     loss='mae',
                       metrics=['mae'])

def predict(conviction_source, past_convictions):
    if conviction_source == "" or past_convictions == 0:
        print("Invalid input: Fields cannot be zero.")
        return
    
    model_sentences.fit(X_train_sent, y_train_sent, epochs=69, batch_size=12, validation_data=(X_test_sent, y_test_sent))
    model_fines.fit(X_train_fine, y_train_fine, epochs=69, batch_size=12, validation_data=(X_test_fine, y_test_fine))

    conviction_encoded = label_enc.transform([conviction_source])[0]
    past_convictions_scaled_sent = scalar_for_sentences.transform([[past_convictions]])[0][0]
    past_convictions_scaled_fine = scaler_fines.transform([[past_convictions]])[0][0]
    
    input_data_sent = np.array([[conviction_encoded, past_convictions_scaled_sent]])
    input_data_fine = np.array([[conviction_encoded, past_convictions_scaled_fine]])

    pred_sentences = abs(model_sentences.predict(input_data_sent)[0][0])
    pred_fine = abs(model_fines.predict(input_data_fine)[0][0])
    


    sentences_served_pred = scalar_for_sentences.inverse_transform([[pred_sentences]])[0][0]
    amount_fined_pred = scaler_fines.inverse_transform([[pred_fine]])[0][0]

    sentences_served_pred = round(sentences_served_pred, 1)
    amount_fined_pred = round(amount_fined_pred, 2)

    preference = "Imprisonment" if determine_pref(conviction_source) else "Fine"
    print(f"Predicted Sentences Served: {sentences_served_pred:.2f} years")
    print(f"Predicted Amount Fined: ${amount_fined_pred:.2f}")
    

    output_dict = {
        "Predicted Sentence to be Served: ": sentences_served_pred,
        "Predicted Amount to Fine: " : amount_fined_pred,
        "Statistically Preferred Punishment" : preference
    }

    with open("output.json", "w") as f:
        json.dump(output_dict, f)
    return json.dumps(output_dict)

def determine_pref(by: str) -> bool:
    sentences = 0
    fines = 0
    try:
        for row in reads_pre["Main Source of Conviction"]:
            if row == by:
                sentences += 1
        for row in readf_pre["Main Source of Conviction"]:
            if row == by:
                fines += 1
    except(ValueError):
        pass
    return sentences >= fines
