import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import threading
import io
from image_generator import LowResourceImageGenerator, ImageOrientation
import logging

class ImageGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Generator")
        self.root.geometry("800x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize the image generator
        self.generator = LowResourceImageGenerator()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image display area
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.grid(row=0, column=0, columnspan=2, pady=10)
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0)
        
        # Default image display (gray background)
        default_img = Image.new('RGB', (512, 512), '#e0e0e0')
        self.display_image(default_img)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.main_frame,
            variable=self.progress_var,
            maximum=100,
            length=780,
            mode='determinate'
        )
        self.progress_bar.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to generate")
        self.status_label = ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            font=('Arial', 10)
        )
        self.status_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Prompt input area
        self.prompt_frame = ttk.LabelFrame(self.main_frame, text="Enter your prompt", padding="5")
        self.prompt_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.prompt_entry = scrolledtext.ScrolledText(
            self.prompt_frame,
            wrap=tk.WORD,
            width=70,
            height=3
        )
        self.prompt_entry.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Orientation selection
        self.orientation_frame = ttk.Frame(self.main_frame)
        self.orientation_frame.grid(row=4, column=0, columnspan=2, pady=5)
        
        self.orientation_var = tk.StringVar(value="landscape")
        
        ttk.Label(self.orientation_frame, text="Image Orientation:").grid(row=0, column=0, padx=5)
        
        orientations = [("Landscape", "landscape"), ("Portrait", "portrait"), ("Square", "square")]
        for i, (text, value) in enumerate(orientations):
            ttk.Radiobutton(
                self.orientation_frame,
                text=text,
                value=value,
                variable=self.orientation_var
            ).grid(row=0, column=i+1, padx=10)
        
        # Creature type selection
        self.creature_frame = ttk.Frame(self.main_frame)
        self.creature_frame.grid(row=5, column=0, columnspan=2, pady=5)
        
        self.creature_var = tk.StringVar(value="animal")
        
        ttk.Label(self.creature_frame, text="Creature Type:").grid(row=0, column=0, padx=5)
        
        creatures = [
            ("Animal", "animal"), 
            ("Bird", "bird"),
            ("Dragon", "dragon"),
            ("Unicorn", "unicorn"),
            ("Phoenix", "phoenix"),
            ("Mermaid", "mermaid"),
            ("Griffin", "griffin"),
            ("Werewolf", "werewolf")
        ]

        creature_frame_inner = ttk.Frame(self.creature_frame)
        creature_frame_inner.grid(row=0, column=1, columnspan=4)
        
        for i, (text, value) in enumerate(creatures):
            row = i // 4
            col = i % 4
            ttk.Radiobutton(
                creature_frame_inner,
                text=text,
                value=value,
                variable=self.creature_var
            ).grid(row=row, column=col, padx=5)
        
        # Generate button
        self.generate_button = ttk.Button(
            self.main_frame,
            text="Generate Image",
            command=self.generate_image_thread,
            style='Accent.TButton'
        )
        self.generate_button.grid(row=6, column=0, columnspan=2, pady=10)
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('Accent.TButton', font=('Arial', 12))
        
        # Make the window resizable
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        # Bind enter key to generate button
        self.root.bind('<Return>', lambda e: self.generate_image_thread())
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Custom logging handler to update GUI
        class GUIHandler(logging.Handler):
            def __init__(self, callback):
                super().__init__()
                self.callback = callback
            
            def emit(self, record):
                self.callback(self.format(record))
        
        # Add GUI handler
        self.logger.addHandler(GUIHandler(self.update_status))
    
    def update_progress(self, value):
        self.progress_var.set(value)
        self.root.update_idletasks()
    
    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def display_image(self, image):
        # Resize image to fit display area while maintaining aspect ratio
        display_size = (512, 512)
        image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image)
        
        # Update image label
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference
    
    def generate_image_thread(self):
        # Disable generate button while processing
        self.generate_button.configure(state='disabled')
        
        # Start generation in a separate thread
        thread = threading.Thread(target=self.generate_image)
        thread.daemon = True
        thread.start()
    
    def generate_image(self):
        try:
            # Get prompt, orientation and creature type
            prompt = self.prompt_entry.get("1.0", tk.END).strip()
            orientation_str = self.orientation_var.get()
            creature_type = self.creature_var.get()
            
            # Map orientation string to enum
            orientation_map = {
                "landscape": ImageOrientation.LANDSCAPE,
                "portrait": ImageOrientation.PORTRAIT,
                "square": ImageOrientation.SQUARE
            }
            orientation = orientation_map[orientation_str]
            
            # Update status
            self.update_status("Generating image...")
            self.update_progress(0)
            
            # Generate image with creature type
            image = self.generator.generate_image(
                prompt=prompt,
                subject_type=creature_type,
                orientation=orientation,
                num_inference_steps=50,
                guidance_scale=10.0
            )
            
            # Update progress and status
            self.update_progress(100)
            self.update_status("Image generated successfully!")
            
            # Display the generated image
            self.display_image(image)
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.logger.error(f"Error generating image: {str(e)}")
        
        finally:
            # Re-enable generate button
            self.root.after(0, lambda: self.generate_button.configure(state='normal'))
            self.update_progress(0)

def main():
    root = tk.Tk()
    app = ImageGeneratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 