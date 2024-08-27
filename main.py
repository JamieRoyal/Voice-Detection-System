import tkinter as tk
from tkinter import filedialog, ttk
import librosa
import numpy as np
from tensorflow.keras.models import load_model


class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Detection App")
        self.master.geometry("1280x800")
        self.master.configure(bg="#222831")

        self.import_button = ttk.Button(self.master, text="Import Audio Clip", command=self.browse_file,
                                        style='Flat.TButton')
        self.import_button.place(relx=0.5, rely=0.6, anchor="center")

        self.percentage_label = tk.Label(self.master, text="...", font=("Arial", 100), fg="#00adb5", bg="#222831")
        self.percentage_label.place(relx=0.5, rely=0.425, anchor="center")

        self.subtitle_label = tk.Label(self.master, text="Waiting for Audio Clip...", font=("Arial", 20), fg="#eeeeee",
                                       bg="#222831")
        self.subtitle_label.place(relx=0.5, rely=0.525, anchor="center")

        # Load the saved model
        self.model = load_model("ai_detection_system.h5")

    def extract_features(self, file_path):
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_flattened = np.mean(mfccs.T, axis=0)
        return mfccs_flattened.reshape(1, 13, 1)

    def classify_audio(self, file_path):
        features = self.extract_features(file_path)
        prediction = self.model.predict(features)
        return prediction[0][0]  # return the probability

    def browse_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            probability = self.classify_audio(file_path)
            if probability <= 0.24:
                prediction = "Human"
            elif probability <= 0.49:
                prediction = "Likely to be Human"
            elif probability == 0.5:
                prediction = "Equally Likely Human and AI"
            elif probability <= 0.75:
                prediction = "Likely to be AI"
            else:
                prediction = "AI"

            percentage = f"{probability * 100:.2f}%"
            print(prediction)
            print(percentage)

            self.percentage_label.config(text=percentage)
            self.subtitle_label.config(text=prediction)


def main():
    root = tk.Tk()
    app = App(root)
    style = ttk.Style(root)
    style.configure('Flat.TButton', foreground='#000000', background='#393e46', borderwidth=0, highlightthickness=0)
    root.mainloop()


if __name__ == "__main__":
    main()