# -------------------- IMPORT LIBRARIES --------------------
import cv2                      # OpenCV for image processing
import numpy as np              # NumPy for mathematical operations
import customtkinter as ctk     # Modern Tkinter library for GUI
from tkinter import filedialog, messagebox   # For file dialogs and alerts
from PIL import Image, ImageTk  # To display OpenCV images in Tkinter
import matplotlib.pyplot as plt  # For histogram and 3D surface plots

# -------------------- WINDOW SETUP --------------------
ctk.set_appearance_mode("dark")             # Dark theme
ctk.set_default_color_theme("blue")         # Blue accent theme

root = ctk.CTk()                            # Create main window
root.title("Digital Image Processing Toolkit")
root.geometry("1000x700")

# Global variables for images
img = None
processed_img = None


# -------------------- HELPER FUNCTION: SHOW IMAGE --------------------
def show_image(image, original=False):
    """
    Display image on GUI in correct panel.
    Converts OpenCV BGR format to RGB and displays in Tkinter.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert color space
    im_pil = Image.fromarray(image_rgb)                 # Convert to PIL image
    im_pil = im_pil.resize((400, 400))                  # Resize for display
    imgtk = ImageTk.PhotoImage(im_pil)                  # Convert for Tkinter

    # Assign to correct image label
    if original:
        lbl_original.configure(image=imgtk)
        lbl_original.image = imgtk
    else:
        lbl_processed.configure(image=imgtk)
        lbl_processed.image = imgtk


# -------------------- IMAGE LOADING --------------------
def load_image():
    """Open file dialog to load an image."""
    global img
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        img = cv2.imread(file_path)
        show_image(img, original=True)


# -------------------- IMAGE PROCESSING FUNCTIONS --------------------
def convert_gray():
    """Convert image to grayscale."""
    global img, processed_img
    if img is None:
        messagebox.showerror("Error", "Load an image first!")
        return
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    show_image(processed_img)


def negative_image():
    """Create negative version of the image."""
    global img, processed_img
    if img is None:
        messagebox.showerror("Error", "Load an image first!")
        return
    processed_img = 255 - img
    show_image(processed_img)


def histogram_equalization():
    """Enhance image contrast using histogram equalization."""
    global img, processed_img
    if img is None:
        messagebox.showerror("Error", "Load an image first!")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    processed_img = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    show_image(processed_img)


def show_histogram():
    """Display histogram of image intensity."""
    global img
    if img is None:
        messagebox.showerror("Error", "Load an image first!")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure("Image Histogram")
    plt.title("Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.hist(gray.ravel(), 256, [0, 256], color='gray')
    plt.show()


def apply_filter(filter_type):
    """Apply filters: Gaussian, Median, or Sharpen."""
    global img, processed_img
    if img is None:
        messagebox.showerror("Error", "Load an image first!")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Select and apply filter type
    if filter_type == "gaussian":
        filtered = cv2.GaussianBlur(gray, (5, 5), 0)
    elif filter_type == "median":
        filtered = cv2.medianBlur(gray, 5)
    elif filter_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        filtered = cv2.filter2D(gray, -1, kernel)
    else:
        filtered = gray

    processed_img = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    show_image(processed_img)


def surface_plot():
    """Generate 3D surface plot of grayscale intensity."""
    global img
    if img is None:
        messagebox.showerror("Error", "Load an image first!")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    from mpl_toolkits.mplot3d import Axes3D

    # Create meshgrid
    X = np.arange(0, gray.shape[1], 1)
    Y = np.arange(0, gray.shape[0], 1)
    X, Y = np.meshgrid(X, Y)

    # Plot the surface
    fig = plt.figure("3D Surface Plot")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, gray, cmap='gray')
    ax.set_title("3D Surface Plot of Image")
    plt.show()


def save_image():
    """Save the processed image to disk."""
    global processed_img
    if processed_img is None:
        messagebox.showerror("Error", "No processed image to save!")
        return
    path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp")]
    )
    if path:
        cv2.imwrite(path, processed_img)
        messagebox.showinfo("Success", f"Image saved successfully at:\n{path}")


# -------------------- HEADER SECTION --------------------
header = ctk.CTkLabel(
    root,
    text="Digital Image Processing Toolkit",
    font=("Helvetica", 24, "bold")
)
header.pack(pady=15)

sub_header = ctk.CTkLabel(
    root,
    text="Developed by Krish Singh [23U03001] and Aakarsh Singh [23U03011] (IT Dept.)",
    font=("Helvetica", 13),
)
sub_header.pack(pady=2)


# -------------------- IMAGE DISPLAY AREA --------------------
image_frame = ctk.CTkFrame(root, corner_radius=15)
image_frame.pack(pady=10)

# Left (Original) and Right (Processed) display labels
lbl_original = ctk.CTkLabel(image_frame, text="", width=400, height=400, fg_color="gray30")
lbl_original.grid(row=0, column=0, padx=15, pady=10)

lbl_processed = ctk.CTkLabel(image_frame, text="", width=400, height=400, fg_color="gray30")
lbl_processed.grid(row=0, column=1, padx=15, pady=10)

# Add text BELOW the images
ctk.CTkLabel(image_frame, text="Original Image", font=("Helvetica", 13, "bold")).grid(row=1, column=0, pady=5)
ctk.CTkLabel(image_frame, text="Processed Image", font=("Helvetica", 13, "bold")).grid(row=1, column=1, pady=5)


# -------------------- BUTTONS FOR FUNCTIONS --------------------
button_frame = ctk.CTkFrame(root, corner_radius=15)
button_frame.pack(pady=15)

# Row 1 - Basic transformations
ctk.CTkButton(button_frame, text="Grayscale", width=150, command=convert_gray).grid(row=0, column=0, padx=10, pady=10)
ctk.CTkButton(button_frame, text="Negative", width=150, command=negative_image).grid(row=0, column=1, padx=10, pady=10)
ctk.CTkButton(button_frame, text="Histogram Equalization", width=200, command=histogram_equalization).grid(row=0, column=2, padx=10, pady=10)

# Row 2 - Filters and visualization
ctk.CTkButton(button_frame, text="Show Histogram", width=150, command=show_histogram).grid(row=1, column=0, padx=10, pady=10)
ctk.CTkButton(button_frame, text="Gaussian Filter", width=150, command=lambda: apply_filter("gaussian")).grid(row=1, column=1, padx=10, pady=10)
ctk.CTkButton(button_frame, text="Median Filter", width=150, command=lambda: apply_filter("median")).grid(row=1, column=2, padx=10, pady=10)
ctk.CTkButton(button_frame, text="Sharpen Filter", width=150, command=lambda: apply_filter("sharpen")).grid(row=1, column=3, padx=10, pady=10)
ctk.CTkButton(button_frame, text="3D Surface Plot", width=180, command=surface_plot).grid(row=2, column=1, columnspan=2, padx=10, pady=10)


# -------------------- SEPARATE LOAD & SAVE BUTTONS --------------------
bottom_frame = ctk.CTkFrame(root, fg_color="transparent")
bottom_frame.pack(pady=15)

ctk.CTkButton(bottom_frame, text="Load Image", width=180, height=40, fg_color="#2E8BFF", command=load_image).grid(row=0, column=0, padx=30)
ctk.CTkButton(bottom_frame, text="Save Image", width=180, height=40, fg_color="#00C851", command=save_image).grid(row=0, column=1, padx=30)


# -------------------- RUN THE MAIN LOOP --------------------
root.mainloop()
