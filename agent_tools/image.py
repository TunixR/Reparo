import asyncio
import base64
from io import BytesIO

from fastapi import WebSocketDisconnect
from PIL import Image
from strands import ToolContext, tool

from observability.instrumentation.gui_agent import GuiTaskTelemetry
from robot.channel import RobotIO

# Seconds to wait for the robot to send back a screenshot.
SCREENSHOT_TIMEOUT_SECONDS = 15.0


@tool(description="Convert an image file to a base64-encoded string.")
def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to a base64-encoded string.

    Args:
        image_path (str): Path to the image file to convert to base64

    Returns:
        base64 string or error message

        Success: Returns the base64-encoded JPEG image as text.
        Error: Returns information about what went wrong.
    """
    if not image_path:
        return "image_path is required"

    try:
        with open(image_path, "rb") as image_file:
            # convert to jpeg format
            pil_image = Image.open(image_file)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Save the image to a bytes buffer
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")

            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:  # noqa: BLE001
        return str(e)


@tool(
    description="Take a screenshot and return it as a base64-encoded string.",
    context=True,
)
async def take_screenshot(
    tool_context: ToolContext,
) -> list[dict[str, dict[str, str | dict[str, bytes]]]]:
    """
    Take a screenshot and return it as a base64-encoded string.

    Args:
        (no inputs required) - tool input can be empty

    Returns:
        Dictionary containing Image:
        {"image": {"format":"JPEG","source":{"bytes": b"..."}}}

        Success: Returns the screenshot bytes in the content as an image object.
        Error: Returns information about what went wrong.
    """

    if "websocket" not in tool_context.invocation_state:
        raise ValueError(
            "WebSocket connection is required in tool context for taking screenshot."
        )
    websocket = tool_context.invocation_state["websocket"]

    try:
        return [
            {
                "image": {
                    "format": "JPEG",
                    "source": {"bytes": await screenshot_bytes(websocket)},
                }
            }
        ]
    except WebSocketDisconnect as _:
        raise
    except Exception as e:
        raise Exception(f"Error taking screenshot: {e!s}") from e  # noqa: TRY002


async def request_remote_screenshot(websocket: RobotIO) -> bytes:
    """
    Send a request to the robot and wait for it to send back a screenshot.

    The client may respond in one of the following ways:
    - send bytes directly (use `await websocket.receive_bytes()`)
    - send a text message containing a base64-encoded image
    - send a JSON message containing an image under keys like `image` or `image_base64`

    Args:
        websocket (RobotIO): The robot, normally its recovery connection.

    Returns:
        bytes: Raw image bytes (JPEG/PNG) as received or decoded from base64.

    Raises:
        TimeoutError: If no response is received within
            `SCREENSHOT_TIMEOUT_SECONDS`.
        RuntimeError: For unexpected message formats or disconnections.
    """

    await websocket.send_json({"type": "screenshot", "content": ""})

    try:
        async with asyncio.timeout(SCREENSHOT_TIMEOUT_SECONDS):
            return await websocket.receive_bytes()
    except WebSocketDisconnect as _:
        raise
    except TimeoutError as e:
        raise TimeoutError("Timed out waiting for screenshot from client") from e
    except Exception as e:
        raise RuntimeError(f"Unable to interpret client response as image: {e}") from e


async def screenshot_bytes(websocket: RobotIO) -> bytes:
    """
    Take a screenshot and return it as bytes.

    Args:
        websocket (RobotIO): The robot, normally its recovery connection.

    Returns:
        bytes: Screenshot image data in bytes.

    Usage:
        screenshot_data = await screenshot_bytes(websocket)
    """
    return await request_remote_screenshot(websocket)


async def capture_screenshot(
    websocket: RobotIO, telemetry: GuiTaskTelemetry
) -> tuple[bytes, tuple[int, int]]:
    """Fetch a client screenshot in a ``gui.screenshot`` span.

    Returns:
        The image bytes and its (width, height) in pixels.
    """
    with telemetry.screenshot() as shot:
        image = await screenshot_bytes(websocket)
        width, height = Image.open(BytesIO(image)).size
        shot.taken(image, width, height)
    return image, (width, height)
