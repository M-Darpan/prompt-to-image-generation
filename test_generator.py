from image_generator import LowResourceImageGenerator, ImageOrientation
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize generator
    generator = LowResourceImageGenerator()
    
    try:
        # Load prompt from file (snow leopard)
        prompt_data = generator.load_prompt_from_file("prompts.txt", 0)
        
        # Generate landscape image (wide shot of habitat)
        logger.info(f"Generating landscape image for: {prompt_data['prompt']}")
        landscape_image = generator.generate_image(
            prompt=prompt_data['prompt'],
            subject_type=prompt_data['subject_type'],
            orientation=ImageOrientation.LANDSCAPE,
            num_inference_steps=35,  # Increased steps for realism
            guidance_scale=9.5,      # Higher guidance for better details
            output_dir="outputs"
        )
        logger.info("Landscape image generated successfully!")
        
        # Load different prompt (tree frog for close-up)
        prompt_data = generator.load_prompt_from_file("prompts.txt", 1)
        
        # Generate portrait image (close-up shot)
        logger.info(f"Generating portrait image for: {prompt_data['prompt']}")
        portrait_image = generator.generate_image(
            prompt=prompt_data['prompt'],
            subject_type=prompt_data['subject_type'],
            orientation=ImageOrientation.PORTRAIT,
            num_inference_steps=35,
            guidance_scale=9.5,
            output_dir="outputs"
        )
        logger.info("Portrait image generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating images: {str(e)}")
        raise

if __name__ == "__main__":
    main() 