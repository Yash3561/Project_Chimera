import os
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

class VisionTool:
    def __init__(self):
        print("Initializing Vision Tool (Florence-2)...")
        self.model_id = 'microsoft/Florence-2-large'
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        print("Vision Tool ready.")

    def _run_and_save(self, image_path: str, task_prompt: str, output_filename: str) -> str:
        """Helper to run the model and directly save the output to a file."""
        full_image_path = os.path.join("sandbox", image_path.lstrip('/\\'))
        if not os.path.exists(full_image_path):
            return f"Error: Image file not found at '{image_path}'."
        
        full_output_path = os.path.join("sandbox", output_filename.lstrip('/\\'))
        if not os.path.abspath(full_output_path).startswith(os.path.abspath("sandbox")):
            return f"Error: Access denied. Output path is outside the sandbox."

        try:
            image = Image.open(full_image_path).convert("RGB")
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to("cuda", torch.bfloat16)
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=2048, num_beams=3
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text, task=task_prompt, image_size=(image.width, image.height)
            )
            clean_answer = re.sub(r'<loc_\d+>', '', str(parsed_answer)).strip()
            
            with open(full_output_path, "w", encoding="utf-8") as f:
                f.write(clean_answer)

            return f"Successfully performed '{task_prompt}' on '{image_path}' and saved the output to '{output_filename}'."
        except Exception as e:
            return f"Error processing image: {e}"

    def ocr(self, image_path: str, output_filename: str) -> str:
        """Performs OCR on an image and saves the full text to a file."""
        return self._run_and_save(image_path, "<OCR>", output_filename)

    def caption(self, image_path: str, output_filename: str) -> str:
        """Generates a caption for an image and saves it to a file."""
        return self._run_and_save(image_path, "<MORE_DETAILED_CAPTION>", output_filename)