import os
import time
import base64
import logging
import threading
from typing import Any, Dict
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import snapshot_download

from reachy_mini_teacher_app.config import config


logger = logging.getLogger(__name__)


@dataclass
class VisionConfig:
    """Configuration for vision processing."""

    model_path: str = config.LOCAL_VISION_MODEL
    vision_interval: float = 5.0
    max_new_tokens: int = 64
    jpeg_quality: int = 85
    max_retries: int = 3
    retry_delay: float = 1.0
    device_preference: str = "auto"  # "auto", "cuda", "cpu"


class VisionProcessor:
    """
    The 'Visual Brain' of the robot. This class manages the SmolVLM2 model, 
    handling everything from loading the weights onto the GPU to performing 
    complex image-to-text inference.

    WHAT IT DOES:
    - Automatically detects and utilizes the best hardware (CUDA, MPS, or CPU).
    - Encodes raw camera frames into a format the AI can 'see'.
    - Executes 'Zero-Shot' image description, meaning it can describe things 
      it hasn't been specifically trained for.
    - Manages memory carefully to prevent the robot's brain from crashing 
      during long conversations.

    WHY IT IS NEEDED:
    Traditional computer vision is great at finding boxes (like faces), but it 
    cannot tell you *what* is happening. This class allows the robot to say, 
    "I see you are wearing a blue hat" or "The room looks a bit messy," 
    making it a much more interactive and aware companion.
    """

    def __init__(self, vision_config: VisionConfig | None = None):
        """
        Sets up the basic configuration and hardware paths.
        
        WHY: We don't load the model here yet because loading takes a long time. 
        We just prepare the 'paperwork' so the robot knows where its brain is stored.
        """
        self.vision_config = vision_config or VisionConfig()
        self.model_path = self.vision_config.model_path
        # Step 1: Figure out what kind of hardware we are running on
        self.device = self._determine_device()
        self.processor = None
        self.model = None
        self._initialized = False

    def _determine_device(self) -> str:
        """
        Hardware Detection Logic.
        
        WHAT: Checks for NVIDIA GPUs (CUDA), Apple Silicon (MPS), or fallback CPUs.
        WHY: Running vision on a CPU is slow (10-20 seconds per frame). 
             Running on a GPU is fast (sub-second). This ensures the robot 
             uses the fastest 'muscle' available.
        """
        pref = self.vision_config.device_preference
        if pref == "cpu": return "cpu"
        if pref == "cuda": return "cuda" if torch.cuda.is_available() else "cpu"
        if pref == "mps": return "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Auto-preference: Apple Silicon (MPS) > NVIDIA (CUDA) > CPU
        if torch.backends.mps.is_available(): return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def initialize(self) -> bool:
        """
        Loads the massive AI model weights from storage into RAM/VRAM.
        
        WHAT: Loads the AutoProcessor (for inputs) and AutoModel (the brain).
        WHY: This is the 'heavy lifting' phase. It sets the data types (dtype) 
             based on the hardware to maximize speed without losing accuracy.
        """
        try:
            logger.info(f"Loading SmolVLM2 model on {self.device}")
            # Step 1: Load the 'eye' (Processor)
            self.processor = AutoProcessor.from_pretrained(self.model_path)

            # Step 2: Set bit-depth. BFloat16 for CUDA saves memory; Float32 for Mac is more stable.
            if self.device == "cuda":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

            model_kwargs: Dict[str, Any] = {"dtype": dtype}

            # Step 3: Enable 'Flash Attention' if on NVIDIA to speed up processing significantly
            if self.device == "cuda":
                model_kwargs["_attn_implementation"] = "flash_attention_2"

            # Step 4: Physically move the model onto the graphics card
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path, **model_kwargs
            ).to(self.device)

            if self.model is not None:
                self.model.eval() # Set to evaluation mode (turns off training features)
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False

    def process_image(
        self,
        cv2_image: NDArray[np.uint8],
        prompt: str = "Briefly describe what you see in one sentence.",
    ) -> str:
        """
        The main inference loop. Converts a picture into a sentence.
        
        WHAT: Encodes image -> Creates Chat Template -> Generates Text -> Cleans Output.
        WHY: This includes 'Retry Logic'. If the GPU is busy or out of memory, 
             it doesn't crash the robot; it waits and tries again.
        """
        if not self._initialized or self.processor is None or self.model is None:
            return "Vision model not initialized"

        for attempt in range(self.vision_config.max_retries):
            try:
                # Step 1: Compress the high-res camera frame into a JPEG to save space
                success, jpeg_buffer = cv2.imencode(
                    ".jpg", cv2_image, [cv2.IMWRITE_JPEG_QUALITY, self.vision_config.jpeg_quality]
                )
                if not success: return "Failed to encode image"

                # Step 2: Convert to Base64 (The AI needs a text-string version of the image)
                image_base64 = base64.b64encode(jpeg_buffer.tobytes()).decode("utf-8")

                # Step 3: Format the request like a chat conversation
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "url": f"data:image/jpeg;base64,{image_base64}"},
                        {"type": "text", "text": prompt},
                    ],
                }]

                # Step 4: Convert text and image into 'Tensors' (the numbers the AI eats)
                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, 
                    return_dict=True, return_tensors="pt"
                )

                # Move numbers to the GPU
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

                # Step 5: The Actual Inference (The AI 'thinks' here)
                with torch.no_grad(): # Disable gradients to save massive amounts of RAM
                    generated_ids = self.model.generate(
                        **inputs,
                        do_sample=False, # Stay deterministic (no random guessing)
                        max_new_tokens=self.vision_config.max_new_tokens,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )

                # Step 6: Decode the AI's binary thoughts back into human English
                generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                response = self._extract_response(generated_texts[0])

                # Step 7: Memory Cleanup (Very important for robots running for hours)
                if self.device == "cuda": torch.cuda.empty_cache()
                elif self.device == "mps": torch.mps.empty_cache()

                return response.strip()

            except torch.cuda.OutOfMemoryError:
                # If the brain is full, take a break and try again
                torch.cuda.empty_cache()
                time.sleep(self.vision_config.retry_delay * (attempt + 1))
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(self.vision_config.retry_delay)

        return "Vision processing error"

    def _extract_response(self, full_text: str) -> str:
        """
        Parses the model output to find the actual answer.
        
        WHY: VLMs often repeat the user's prompt or add 'Assistant:' before their answer. 
             This function cuts through the noise to get the raw description.
        """
        markers = ["assistant\n", "Assistant:", "Response:", "\n\n"]
        for marker in markers:
            if marker in full_text:
                response = full_text.split(marker)[-1].strip()
                if response: return response
        return full_text.strip()

    def get_model_info(self) -> Dict[str, Any]:
        """Returns diagnostic data about the 'Visual Brain'."""
        return {
            "initialized": self._initialized,
            "device": self.device,
            "model_path": self.model_path,
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory // (1024**3) 
                          if torch.cuda.is_available() else "N/A",
        }

class VisionManager:
    """
    The 'Project Manager' for the robot's vision system. 
    
    WHAT IT DOES:
    - Sets up a background heartbeat (thread) that triggers vision checks.
    - Manages the timing (e.g., look every 5 seconds).
    - Coordinates between the Camera hardware and the AI Vision Processor.
    - Handles 'Loop Safety'—if the vision system crashes, it waits and restarts.

    WHY IT IS NEEDED:
    AI vision processing is 'expensive' in terms of time and battery. If the 
    robot tried to describe the room 30 times a second, it would overheat. 
    The VisionManager paces the robot, ensuring it stays 'aware' of its 
    surroundings at a sustainable rate.
    """

    def __init__(self, camera: Any, vision_config: VisionConfig | None = None):
        """
        Connects the camera to the AI brain and prepares the background thread.
        """
        # Step 1: Link the camera hardware
        self.camera = camera
        # Step 2: Set the timing rules (e.g., look every X seconds)
        self.vision_config = vision_config or VisionConfig()
        self.vision_interval = self.vision_config.vision_interval
        
        # Step 3: Instantiate the actual AI engine
        self.processor = VisionProcessor(self.vision_config)

        # Step 4: Setup synchronization primitives for threading
        self._last_processed_time = 0.0
        self._stop_event = threading.Event() # Used to signal the thread to shut down
        self._thread: threading.Thread | None = None

        # Step 5: Boot up the AI model (Loads weights into GPU)
        if not self.processor.initialize():
            logger.error("Failed to initialize vision processor")
            raise RuntimeError("Vision processor initialization failed")

    def start(self) -> None:
        """
        Launches the background vision thread.
        
        WHY: We use a 'Thread' so the robot can walk and talk at the same 
        time while the vision system is 'thinking' in the background.
        """
        self._stop_event.clear()
        # Create a 'Daemon' thread—this means it dies automatically if the main app stops
        self._thread = threading.Thread(target=self._working_loop, daemon=True) # This tells the thread exactly which function it is responsible for running.
        self._thread.start()
        logger.info("Local vision processing started")

    def stop(self) -> None:
        """
        Safely shuts down the vision thread.
        """
        self._stop_event.set() # Signal the loop to stop
        if self._thread is not None:
            self._thread.join() # Wait for the thread to finish its current task
        logger.info("Local vision processing stopped")

    def _working_loop(self) -> None:
        """
        The continuous 'Heartbeat' of the vision system.
        
        WHAT: 
        1. Checks if enough time has passed since the last look.
        2. Grabs a fresh frame from the camera.
        3. Asks the processor: 'What do you see?'
        4. Logs the result.
        """
        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Step 1: Pacing check (Is it time to look again?)
                if current_time - self._last_processed_time >= self.vision_interval:
                    
                    # Step 2: Grab the most recent picture from the camera
                    frame = self.camera.get_latest_frame()
                    
                    if frame is not None:
                        # Step 3: Run the AI inference
                        description = self.processor.process_image(
                            frame,
                            "Briefly describe what you see in one sentence.",
                        )

                        # Step 4: Validation—ensure we didn't get an error message back
                        if description and not description.startswith(("Vision", "Failed", "Error")):
                            self._last_processed_time = current_time
                            logger.debug(f"Vision update: {description}")
                        else:
                            logger.warning(f"Invalid vision response: {description}")

                # Step 5: Short sleep to prevent the CPU from spinning at 100%
                time.sleep(1.0) 

            except Exception:
                # Step 6: Error Handling—if something breaks, wait 5 seconds and try again
                logger.exception("Vision processing loop error")
                time.sleep(5.0) 

        logger.info("Vision loop finished")

    def get_status(self) -> Dict[str, Any]:
        """
        Returns a snapshot of the vision system's health and last known scene.
        """
        return {
            "last_processed": self._last_processed_time,
            "processor_info": self.processor.get_model_info(),
            "config": {
                "interval": self.vision_interval,
            },
        }
        
def initialize_vision_manager(camera_worker: Any) -> VisionManager | None:
    """
    Bootstrap function to prepare, download, and launch the Vision system.

    WHAT IT DOES:
    - Prepares the local file system (HF_HOME) to store large AI model files.
    - Downloads the specific Vision-Language Model (SmolVLM2) from HuggingFace.
    - Creates a VisionConfig object with specific performance tunables.
    - Instantiates the VisionManager which boots the AI onto the GPU/CPU.
    - Returns a fully operational VisionManager or None if something fails.

    WHY IT IS NEEDED:
    AI models are massive (gigabytes of data) and hardware-sensitive. You cannot 
    just 'run' the code; you must first ensure the weights are downloaded, 
    the cache is mapped, and the specific device (CUDA/MPS) is ready. This 
    function encapsulates all that 'boring' infrastructure work into one call.
    """
    try:
        # Step 1: Identify the model and where to store it
        # We pull these from a central config file so we can change models easily
        model_id = config.LOCAL_VISION_MODEL
        cache_dir = os.path.expanduser(config.HF_HOME)

        # Step 2: Prepare the 'HuggingFace Home' directory
        # This is where the AI's 'brain files' are stored on the hard drive
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        logger.info("HF_HOME set to %s", cache_dir)

        # Step 3: Download the Model (The 'Snapshot')
        # This checks if you already have the model. If not, it downloads it.
        # This can take several minutes on the first run.
        logger.info(f"Downloading vision model {model_id} to cache...")
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            cache_dir=cache_dir,
        )
        logger.info(f"Model {model_id} downloaded to {cache_dir}")

        # Step 4: Define the 'Personality' and Performance of the vision system
        # Here we decide how often to look (5s) and how much to 'talk' (64 tokens)
        vision_config = VisionConfig(
            model_path=model_id,
            vision_interval=5.0,
            max_new_tokens=64,
            jpeg_quality=85,
            max_retries=3,
            retry_delay=1.0,
            device_preference="auto", # Let the system pick GPU vs CPU automatically
        )

        # Step 5: Boot up the Manager
        # This physically loads the model weights into the GPU VRAM
        vision_manager = VisionManager(camera_worker, vision_config)

        # Step 6: Diagnostic Logging
        # We confirm exactly which hardware (e.g., 'cuda' or 'mps') was chosen
        device_info = vision_manager.processor.get_model_info()
        logger.info(
            f"Vision processing enabled: {device_info.get('model_path')} on {device_info.get('device')}",
        )

        return vision_manager

    except Exception as e:
        # If the internet is down or the GPU is full, we catch the error here
        # so the entire robot app doesn't crash—just the vision part fails.
        logger.error(f"Failed to initialize vision manager: {e}")
        return None