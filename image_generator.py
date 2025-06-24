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
        "dragon": "highly detailed, mythical creature, scales texture, powerful pose, majestic wings, fantasy art, professional digital art, hyperrealistic, 8k uhd, sharp focus, high resolution, cinematic lighting, perfect composition, award winning artwork, masterpiece",
        "unicorn": "highly detailed, mythical creature, iridescent coat, flowing mane, magical aura, fantasy art, professional digital art, hyperrealistic, 8k uhd, sharp focus, high resolution, ethereal lighting, perfect composition, award winning artwork, masterpiece",
        "phoenix": "highly detailed, mythical bird, flame feathers, glowing embers, rising from ashes, fantasy art, professional digital art, hyperrealistic, 8k uhd, sharp focus, high resolution, dramatic lighting, perfect composition, award winning artwork, masterpiece",
        "mermaid": "highly detailed, mythical being, iridescent scales, flowing hair, underwater scene, fantasy art, professional digital art, hyperrealistic, 8k uhd, sharp focus, high resolution, underwater lighting, perfect composition, award winning artwork, masterpiece",
        "griffin": "highly detailed, mythical creature, eagle head, lion body, majestic wings, fantasy art, professional digital art, hyperrealistic, 8k uhd, sharp focus, high resolution, dramatic lighting, perfect composition, award winning artwork, masterpiece",
        "werewolf": "highly detailed, mythical creature, detailed fur, muscular form, transformation, fantasy art, professional digital art, hyperrealistic, 8k uhd, sharp focus, high resolution, moonlit scene, perfect composition, award winning artwork, masterpiece"
    }

    # Enhanced negative prompts for better quality and specific creature types
    NEGATIVE_PROMPTS = {
        "animal": "deformed, blurry, ugly, cartoon, lowres, watermark, disfigured, glitch, pixelated, grainy, distorted, poorly drawn face, poorly drawn feet, mutation, mutated, extra limbs, missing limbs, floating limbs, disconnected limbs, malformed limbs, extra fingers, missing fingers, human features, anthropomorphic",
        "bird": "deformed, blurry, ugly, cartoon, lowres, watermark, disfigured, glitch, pixelated, grainy, distorted, poorly drawn wings, poorly drawn beak, mutation, mutated, extra wings, missing wings, floating feathers, disconnected parts, malformed beak, human features, anthropomorphic",
        "dragon": "cute, chibi, cartoon, anime, poorly drawn scales, poorly drawn wings, deformed, blurry, ugly, lowres, watermark, disfigured, glitch, pixelated, grainy, distorted, childish, toy-like, plastic",
        "unicorn": "cute, chibi, cartoon, anime, poorly drawn horn, poorly drawn mane, deformed, blurry, ugly, lowres, watermark, disfigured, glitch, pixelated, grainy, distorted, childish, toy-like, plastic",
        "phoenix": "cute, chibi, cartoon, anime, poorly drawn flames, poorly drawn wings, deformed, blurry, ugly, lowres, watermark, disfigured, glitch, pixelated, grainy, distorted, childish, toy-like",
        "mermaid": "cute, chibi, cartoon, anime, poorly drawn scales, poorly drawn tail, deformed, blurry, ugly, lowres, watermark, disfigured, glitch, pixelated, grainy, distorted, childish, toy-like",
        "griffin": "cute, chibi, cartoon, anime, poorly drawn beak, poorly drawn wings, deformed, blurry, ugly, lowres, watermark, disfigured, glitch, pixelated, grainy, distorted, childish, toy-like",
        "werewolf": "cute, chibi, cartoon, anime, poorly drawn fur, poorly drawn claws, deformed, blurry, ugly, lowres, watermark, disfigured, glitch, pixelated, grainy, distorted, childish, toy-like"
    }

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
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
        return self.NEGATIVE_PROMPTS.get(subject_type, self.NEGATIVE_PROMPTS["animal"])

    def generate_image(
        self,
        prompt: str,
        subject_type: str = "animal",
        orientation: ImageOrientation = ImageOrientation.LANDSCAPE,
        num_inference_steps: int = 50,  # Increased for better quality
        guidance_scale: float = 10.0,   # Increased for more prompt adherence
        seed: Optional[int] = None,     # Added seed parameter for reproducibility
        strength: float = 1.0           # Added strength parameter for prompt influence
    ) -> Union[Image.Image, List[Image.Image]]:
        """Generate an image with enhanced settings for realism"""
        try:
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed) if self.device == "cuda" else None

            # Get image dimensions
            width, height = self.get_dimensions(orientation)
            
            # Enhance prompt and get negative prompt
            enhanced_prompt = self.enhance_prompt(prompt, subject_type)
            negative_prompt = self.get_negative_prompt(subject_type)
            
            # Generate image with enhanced settings
            with torch.no_grad():
                image = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength
                ).images[0]
            
            # Clear CUDA cache
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