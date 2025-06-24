import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import logging
import os
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
import gc
from enum import Enum
from datasets import load_dataset
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageOrientation(Enum):
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"
    SQUARE = "square"

class AnimalDataset(Dataset):
    def __init__(self, tokenizer):
        self.dataset = load_dataset("Abirate/animals_10_10", split="train")
        self.tokenizer = tokenizer
        self.max_length = 77

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = f"a photo of a {item['labels']}, {item['text']}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding.input_ids[0],
            "attention_mask": encoding.attention_mask[0]
        }

class LowResourceImageGenerator:
    # Quality enhancement presets with more detailed modifiers
    REALISM_MODIFIERS = {
        "animal": "highly detailed, professional wildlife photography, national geographic style, natural lighting, detailed fur texture, anatomically correct, hyperrealistic, 8k uhd, sharp focus, high resolution, professional camera, perfect composition, award winning photo, masterpiece",
        "bird": "highly detailed, professional bird photography, audubon style, natural lighting, detailed feathers, anatomically correct, hyperrealistic, 8k uhd, sharp focus, high resolution, professional camera, perfect composition, award winning photo, masterpiece",
        "nature": "highly detailed, professional landscape photography, natural lighting, detailed textures, photorealistic, 8k uhd, sharp focus, high resolution, professional camera, perfect composition, award winning photo, masterpiece",
    }

    # Enhanced negative prompts for better quality
    BASE_NEGATIVE_PROMPT = """
        deformed, blurry, ugly, cartoon, lowres, watermark, disfigured, glitch, pixelated, grainy, distorted, 
        creepy, draft, badhands, bad anatomy, bad proportions, duplicate limbs, extra limbs, missing limbs, 
        poorly drawn face, poorly drawn hands, poorly drawn feet, mutation, mutated, extra fingers, missing fingers, 
        floating limbs, disconnected limbs, malformed limbs, missing arms, missing legs, extra arms, extra legs, 
        fused fingers, too many fingers, long neck, cross-eyed, mutated hands, cropped, out of frame, jpeg artifacts, 
        signature, text, logo, wordmark, duplicate, error, cropped, worst quality, low quality, normal quality, 
        artificial, fake looking, unnatural pose, poor lighting, stiff pose
    """.replace('\n', ' ').replace('    ', ' ').strip()

    def __init__(self, model_id: str = "CompVis/stable-diffusion-v1-4"):
        """Initialize the image generator with optimizations for low resource systems"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Memory optimization settings
        torch.backends.cudnn.benchmark = True
        
        try:
            # Load model with optimizations
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Use DPMSolver++ scheduler for better quality
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                solver_order=2
            )
            
            # Enable memory efficient settings
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                self.pipe.enable_model_cpu_offload()
                torch.cuda.empty_cache()
            
            logger.info("Model loaded successfully with optimizations")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def train_on_animals(self, num_epochs: int = 3, batch_size: int = 8, learning_rate: float = 1e-5):
        """Train the model on animal dataset"""
        try:
            logger.info("Starting training on animal dataset...")
            
            # Initialize dataset and dataloader
            dataset = AnimalDataset(self.pipe.tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Get text encoder for training
            text_encoder = self.pipe.text_encoder
            
            # Prepare optimizer
            optimizer = torch.optim.AdamW(text_encoder.parameters(), lr=learning_rate)
            
            # Training loop
            text_encoder.train()
            for epoch in range(num_epochs):
                total_loss = 0
                start_time = time.time()
                
                for batch in dataloader:
                    # Move batch to device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    
                    # Forward pass
                    outputs = text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    
                    # Calculate loss (using mean pooled outputs)
                    loss = outputs.loss if hasattr(outputs, "loss") else torch.mean(outputs.last_hidden_state)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    total_loss += loss.item()
                
                # Log epoch results
                avg_loss = total_loss / len(dataloader)
                epoch_time = time.time() - start_time
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")
            
            # Save the fine-tuned model
            text_encoder.save_pretrained("./fine_tuned_model")
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def get_dimensions(self, orientation: ImageOrientation) -> Tuple[int, int]:
        """Get dimensions based on orientation while maintaining aspect ratio"""
        if orientation == ImageOrientation.LANDSCAPE:
            return (896, 512)  # 16:9 aspect ratio
        elif orientation == ImageOrientation.PORTRAIT:
            return (512, 896)  # 9:16 aspect ratio
        else:
            return (768, 768)  # Square format
    
    def clear_memory(self):
        """Clear CUDA memory cache"""
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            logger.warning(f"Error clearing memory: {str(e)}")

    def enhance_prompt(self, prompt: str, subject_type: str = "animal") -> str:
        """Enhance prompt with realism modifiers based on subject type"""
        modifiers = self.REALISM_MODIFIERS.get(subject_type, self.REALISM_MODIFIERS["animal"])
        enhanced_prompt = f"{prompt}, {modifiers}"
        return enhanced_prompt

    def get_negative_prompt(self, subject_type: str = "animal") -> str:
        """Get specialized negative prompt based on subject type"""
        subject_specific = ""
        if subject_type == "animal":
            subject_specific = "anthropomorphic, humanoid, human features, clothes, dressed, cartoon animal, stuffed animal, toy, "
        elif subject_type == "bird":
            subject_specific = "anthropomorphic, humanoid, human features, clothes, dressed, cartoon bird, stuffed bird, toy, wrong beak, "
        
        return f"{subject_specific}{self.BASE_NEGATIVE_PROMPT}"
    
    def generate_image(
        self,
        prompt: str,
        subject_type: str = "animal",
        orientation: ImageOrientation = ImageOrientation.LANDSCAPE,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 40,  # Increased for better quality
        guidance_scale: float = 9.5,
        output_dir: Optional[str] = "outputs",
        filename: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image]]:
        """Generate image with memory-efficient settings and enhanced quality"""
        try:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Enhance prompt with realism modifiers
            enhanced_prompt = self.enhance_prompt(prompt, subject_type)
            
            # Get specialized negative prompt if none provided
            if negative_prompt is None:
                negative_prompt = self.get_negative_prompt(subject_type)
            
            # Get dimensions based on orientation
            width, height = self.get_dimensions(orientation)
            
            logger.info(f"Generating {orientation.value} image ({width}x{height})")
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
            
            with torch.inference_mode():
                image = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height
                ).images[0]
            
            if output_dir:
                if filename is None:
                    orientation_suffix = f"_{orientation.value}"
                    filename = f"generated_{len(os.listdir(output_dir))}{orientation_suffix}.png"
                save_path = os.path.join(output_dir, filename)
                # Save with maximum quality
                image.save(save_path, "PNG", quality=100, optimize=False)
                logger.info(f"Image saved to {save_path}")
            
            self.clear_memory()
            return image
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise
    
    def __del__(self):
        """Cleanup when object is deleted"""
        try:
            self.clear_memory()
        except:
            pass

if __name__ == "__main__":
    # Example usage with user input
    generator = LowResourceImageGenerator()
    
    # First train on animal dataset
    print("Training model on animal dataset...")
    generator.train_on_animals()
    
    while True:
        try:
            # Get user input
            prompt = input("\nEnter your prompt (or 'quit' to exit): ")
            if prompt.lower() == 'quit':
                break
                
            # Get orientation preference
            print("\nSelect orientation:")
            print("1. Landscape")
            print("2. Portrait")
            print("3. Square")
            orientation_choice = input("Enter choice (1-3): ")
            
            orientation_map = {
                "1": ImageOrientation.LANDSCAPE,
                "2": ImageOrientation.PORTRAIT,
                "3": ImageOrientation.SQUARE
            }
            
            orientation = orientation_map.get(orientation_choice, ImageOrientation.LANDSCAPE)
            
            # Generate image
            print("\nGenerating image...")
            image = generator.generate_image(
                prompt=prompt,
                orientation=orientation
            )
            
            print("Image generated successfully!")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            continue 