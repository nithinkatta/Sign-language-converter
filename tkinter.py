import tkinter as tk
import cv2
from PIL import Image, ImageTk

class OpenCVTkinterApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.cap = cv2.VideoCapture(0)  # Change to the appropriate video source if needed
        
        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        
        # Button to close the application
        self.close_button = tk.Button(window, text="Close", width=20, command=self.close)
        self.close_button.pack(pady=10)
        
        self.update()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.cap.read()
        
        if ret:
            # Convert the frame from OpenCV to PIL format
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            self.photo = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.window.after(10, self.update)

    def close(self):
        self.window.destroy()
        self.cap.release()

# Create a Tkinter window
root = tk.Tk()
app = OpenCVTkinterApp(root, "OpenCV and Tkinter")
root.mainloop()
