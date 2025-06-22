import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import KMeans
import os

class CartoonifyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Cartoonify App")
        self.root.geometry("1000x700")
        
        # Variables
        self.original_image = None
        self.processed_image = None
        self.display_size = (400, 300)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Upload button
        ttk.Button(control_frame, text="Upload Image", 
                  command=self.upload_image).grid(row=0, column=0, pady=5, sticky=tk.W)
        
        # Style selection
        ttk.Label(control_frame, text="Cartoon Style:").grid(row=1, column=0, pady=(20, 5), sticky=tk.W)
        
        self.style_var = tk.StringVar()
        style_combo = ttk.Combobox(control_frame, textvariable=self.style_var, 
                                  state="readonly", width=25)
        style_combo['values'] = (
            "Enhanced Current",
            "Advanced Filter", 
            "Anime Style",
            "Oil Painting",
            "Pencil + Color",
            "Edge Preserving"
        )
        style_combo.grid(row=2, column=0, pady=5, sticky=tk.W)
        style_combo.bind('<<ComboboxSelected>>', self.on_style_change)
        
        # Process button
        ttk.Button(control_frame, text="Apply Style", 
                  command=self.process_image).grid(row=3, column=0, pady=20, sticky=tk.W)
        
        # Save button
        ttk.Button(control_frame, text="Save Result", 
                  command=self.save_image).grid(row=4, column=0, pady=5, sticky=tk.W)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(control_frame, text="Parameters", padding="5")
        params_frame.grid(row=5, column=0, pady=20, sticky=(tk.W, tk.E))
        
        # Number of colors parameter
        ttk.Label(params_frame, text="Colors:").grid(row=0, column=0, sticky=tk.W)
        self.colors_var = tk.IntVar(value=8)
        colors_spin = ttk.Spinbox(params_frame, from_=3, to=20, width=10, 
                                 textvariable=self.colors_var)
        colors_spin.grid(row=0, column=1, padx=(5, 0))
        
        # Blur parameter
        ttk.Label(params_frame, text="Blur:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.blur_var = tk.IntVar(value=7)
        blur_spin = ttk.Spinbox(params_frame, from_=3, to=21, width=10, 
                               textvariable=self.blur_var)
        blur_spin.grid(row=1, column=1, padx=(5, 0), pady=(5, 0))
        
        # Line size parameter
        ttk.Label(params_frame, text="Line Size:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.line_var = tk.IntVar(value=7)
        line_spin = ttk.Spinbox(params_frame, from_=3, to=21, width=10, 
                               textvariable=self.line_var)
        line_spin.grid(row=2, column=1, padx=(5, 0), pady=(5, 0))
        
        # Image display area
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original image
        ttk.Label(image_frame, text="Original Image").grid(row=0, column=0, pady=(0, 5))
        self.original_label = ttk.Label(image_frame, text="No image loaded", 
                                       background="lightgray", width=50)
        self.original_label.grid(row=1, column=0, padx=(0, 10))
        
        # Processed image
        ttk.Label(image_frame, text="Processed Image").grid(row=0, column=1, pady=(0, 5))
        self.processed_label = ttk.Label(image_frame, text="No processing done", 
                                        background="lightgray", width=50)
        self.processed_label.grid(row=1, column=1)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Upload an image to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def upload_image(self):
        file_types = [
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff *.tif'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select an image",
            filetypes=file_types
        )
        
        if filename:
            try:
                # Load image with OpenCV
                img = cv2.imread(filename)
                if img is None:
                    raise ValueError("Could not load image")
                    
                self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.display_image(self.original_image, self.original_label)
                self.status_var.set(f"Loaded: {os.path.basename(filename)}")
                
                # Clear processed image
                self.processed_image = None
                self.processed_label.configure(image='', text="No processing done")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")
                
    def display_image(self, cv_image, label_widget):
        # Resize image for display
        h, w = cv_image.shape[:2]
        max_w, max_h = self.display_size
        
        # Calculate scaling factor
        scale = min(max_w/w, max_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        # Resize image
        resized = cv2.resize(cv_image, (new_w, new_h))
        
        # Convert to PIL format
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Display in label
        label_widget.configure(image=photo, text="")
        label_widget.image = photo  # Keep a reference
        
    def on_style_change(self, event=None):
        if self.original_image is not None:
            self.process_image()
            
    def process_image(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please upload an image first")
            return
            
        if not self.style_var.get():
            messagebox.showwarning("Warning", "Please select a style")
            return
            
        try:
            self.status_var.set("Processing...")
            self.root.update()
            
            style = self.style_var.get()
            img = self.original_image.copy()
            
            # Get parameters
            num_colors = self.colors_var.get()
            blur_value = self.blur_var.get()
            line_size = self.line_var.get()
            
            # Apply selected style
            if style == "Enhanced Current":
                self.processed_image = self.cartoonify_method1(img, num_colors, blur_value, line_size)
            elif style == "Advanced Filter":
                self.processed_image = self.cartoonify_method2(img, num_colors, line_size)
            elif style == "Anime Style":
                self.processed_image = self.cartoonify_method3(img, num_colors, blur_value)
            elif style == "Oil Painting":
                self.processed_image = self.oil_painting_effect(img)
            elif style == "Pencil + Color":
                self.processed_image = self.pencil_sketch_cartoon(img)
            elif style == "Edge Preserving":
                self.processed_image = self.enhanced_cartoon(img, k=num_colors)
                
            self.display_image(self.processed_image, self.processed_label)
            self.status_var.set(f"Applied {style} style")
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set("Processing failed")
            
    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return
            
        file_types = [
            ('PNG files', '*.png'),
            ('JPEG files', '*.jpg'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            filetypes=file_types
        )
        
        if filename:
            try:
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, img_bgr)
                self.status_var.set(f"Saved: {os.path.basename(filename)}")
                messagebox.showinfo("Success", "Image saved successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not save image: {str(e)}")

    # Cartoon processing methods
    def cartoonify_method1(self, img, num_colors=8, blur_value=7, line_size=7):
        bilateral = cv2.bilateralFilter(img, d=15, sigmaColor=80, sigmaSpace=80)
        data = bilateral.reshape((-1, 3))
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(data)
        quantized_data = kmeans.cluster_centers_[kmeans.labels_]
        quantized_img = quantized_data.reshape(bilateral.shape).astype(np.uint8)
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.medianBlur(gray, blur_value)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, line_size, blur_value)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        cartoon = cv2.bitwise_and(quantized_img, edges_colored)
        return cartoon

    def cartoonify_method2(self, img, num_colors=12, ksize=7):
        img_smooth = cv2.bilateralFilter(img, d=15, sigmaColor=200, sigmaSpace=200)
        data = img_smooth.reshape((-1, 3))
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(data)
        quantized_data = kmeans.cluster_centers_[kmeans.labels_]
        quantized_img = quantized_data.reshape(img_smooth.shape).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, ksize, 10)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
        edges = cv2.erode(edges, np.ones((2,2), np.uint8), iterations=1)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges_colored = 255 - edges_colored

        cartoon = cv2.multiply(quantized_img.astype(np.float32), 
                              edges_colored.astype(np.float32) / 255.0)
        return cartoon.astype(np.uint8)

    def cartoonify_method3(self, img, num_colors=8, blur_value=7):
        """Clean anime/manga style cartoonification"""
        smooth = img.copy()
        for _ in range(3):
            smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=200, sigmaSpace=200)

        data = smooth.reshape((-1, 3))
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(data)
        quantized_data = kmeans.cluster_centers_[kmeans.labels_]
        quantized = quantized_data.reshape(smooth.shape).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, blur_value, blur_value)

        edges = cv2.medianBlur(edges, 5)
        edges_3d = np.stack([edges, edges, edges], axis=2)
        cartoon = cv2.bitwise_and(quantized, edges_3d)
        return cartoon

    def oil_painting_effect(self, img, size=7, dynRatio=1):
        try:
            return cv2.xphoto.oilPainting(img, size, dynRatio)
        except:
            # Fallback if xphoto module not available
            return self.cartoonify_method1(img)

    def pencil_sketch_cartoon(self, img):
        # Create pencil sketch
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        inv_gray = 255 - gray
        blur_inv = cv2.GaussianBlur(inv_gray, (25, 25), 0, 0)
        sketch = cv2.divide(gray, 255 - blur_inv, scale=256)
        
        # Create very simplified color regions
        # More aggressive color quantization
        cartoon = cv2.bilateralFilter(img, d=20, sigmaColor=200, sigmaSpace=200)
        
        # Reduce to fewer colors using k-means
        data = cartoon.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        cartoon_quantized = centers[labels.flatten()].reshape(img.shape)
        
        # Create strong edge lines
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 7, 7)
        
        # Combine everything
        sketch_3d = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        edges_3d = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Use edges as mask to darken lines on cartoon
        edges_mask = edges_3d.astype(np.float32) / 255.0
        cartoon_float = cartoon_quantized.astype(np.float32) / 255.0
        sketch_float = sketch_3d.astype(np.float32) / 255.0
        
        # Combine: cartoon colors where edges are white, darker where edges are black
        result = cartoon_float * edges_mask * sketch_float
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)

    def enhanced_cartoon(self, img, flags=2, sigma_s=50, sigma_r=0.4, k=7):
        smooth = cv2.edgePreservingFilter(img, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)
        data = smooth.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        quantized_data = kmeans.cluster_centers_[kmeans.labels_]
        quantized = quantized_data.reshape(smooth.shape).astype(np.uint8)
        return quantized

def main():
    root = tk.Tk()
    app = CartoonifyApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()