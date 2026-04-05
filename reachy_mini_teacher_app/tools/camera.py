import base64
import asyncio
import logging
from typing import Any, Dict

import cv2

from reachy_mini_teacher_app.config import config
from reachy_mini_teacher_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


async def _gemini_describe_image(b64_jpeg: str, question: str) -> str:
    """Use a regular Gemini API call to describe an image.

    Falls back gracefully if the API key is missing or the call fails.
    """
    api_key = config.GEMINI_API_KEY
    if not api_key:
        return "No vision processor and no Gemini API key configured."

    try:
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(api_key=api_key)
        vision_model = "gemini-2.5-flash"

        response = await client.aio.models.generate_content(
            model=vision_model,
            contents=[
                gtypes.Content(parts=[
                    gtypes.Part.from_bytes(
                        data=base64.b64decode(b64_jpeg),
                        mime_type="image/jpeg",
                    ),
                    gtypes.Part(text=question),
                ]),
            ],
        )
        text = response.text
        if text:
            logger.info("Gemini vision response (%d chars): %s", len(text), text[:200])
            return text.strip()
        return "Gemini vision returned no text."
    except Exception as e:
        logger.error("Gemini vision API call failed: %s", e)
        return f"Vision analysis failed: {e}"


async def _openai_describe_image(b64_jpeg: str, question: str) -> str:
    """Use a regular OpenAI API call to describe an image.

    Falls back gracefully if the API key is missing or the call fails.
    """
    api_key = config.OPENAI_API_KEY
    if not api_key:
        return "No vision processor and no OpenAI API key configured."

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"},
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
            max_tokens=300,
        )
        text = response.choices[0].message.content
        if text:
            logger.info("OpenAI vision response (%d chars): %s", len(text), text[:200])
            return text.strip()
        return "OpenAI vision returned no text."
    except Exception as e:
        logger.error("OpenAI vision API call failed: %s", e)
        return f"Vision analysis failed: {e}"


async def _describe_image(b64_jpeg: str, question: str) -> str:
    """Route vision analysis to available API (OpenAI or Gemini)."""
    mode = config.APP_MODE
    if mode == "openai" and config.OPENAI_API_KEY:
        return await _openai_describe_image(b64_jpeg, question)
    elif config.GEMINI_API_KEY:
        return await _gemini_describe_image(b64_jpeg, question)
    elif config.OPENAI_API_KEY:
        return await _openai_describe_image(b64_jpeg, question)
    return "No vision API key configured (set GEMINI_API_KEY or OPENAI_API_KEY)."


class Camera(Tool):
    """Take a picture with the camera and ask a question about it."""

    name = "camera"
    description = "Take a picture with the camera and ask a question about it."
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask about the picture",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and ask a question about it."""
        image_query = (kwargs.get("question") or "").strip()
        if not image_query:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", image_query[:120])

        # Get frame from camera worker buffer
        if deps.camera_worker is not None:
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.error("No frame available from camera worker")
                return {"error": "No frame available"}
        else:
            logger.error("Camera worker not available")
            return {"error": "Camera worker not available"}

        # Use local vision manager for processing if available
        if deps.vision_manager is not None:
            vision_result = await asyncio.to_thread(
                deps.vision_manager.processor.process_image, frame, image_query,
            )
            if isinstance(vision_result, dict) and "error" in vision_result:
                return vision_result
            return (
                {"image_description": vision_result}
                if isinstance(vision_result, str)
                else {"error": "vision returned non-string"}
            )

        # No local vision — use Gemini API to analyze the image
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")

        b64_jpeg = base64.b64encode(buffer.tobytes()).decode("utf-8")
        description = await _describe_image(b64_jpeg, image_query)
        return {"image_description": description}
