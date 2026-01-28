from typing import Optional, Literal, List
from pydantic import BaseModel, Field, conint, conlist


class StyleBible(BaseModel):
    visual_style: str
    camera_rules: str
    lighting_rules: str
    continuity_rules: str


Lens = Literal["24mm", "35mm", "50mm", "85mm"]


class Shot(BaseModel):
    # Exactly 3 shots per clip in trial
    shot_id: conint(ge=1, le=3)

    # Keep time ranges human-readable (WAN prompt is textual anyway)
    time_range: str = Field(..., description='e.g., "0–3s", "3–7s", "7–10s"')

    framing: str
    lens: Lens
    camera_movement: str = Field(
        ...,
        description='Use: static|slow dolly|pan|tilt|handheld subtle|crane|static explanatory hold (no camera motion)'
    )
    subject_action: str
    on_screen_text_overlay: str = Field(..., description='Exact words in quotes, or "none"')
    continuity_note: str
    environment_details: conlist(str, min_length=2, max_length=3)
    lighting: str
    cinematic_look: str


class Clip(BaseModel):
    # Trial: exactly 2 clips
    clip_id: conint(ge=1, le=2)

    # Trial: fixed 10s per clip
    duration_s: Literal[10] = Field(10, description="Clip duration in seconds (fixed)")

    # 10-second script derived from the 20-second script
    clip_script: str

    # Trial: exactly 3 shots per clip
    shots: conlist(Shot, min_length=3, max_length=3)

    negative_prompt: str = (
        "logos, watermarks, brand marks, extra limbs, distorted faces, flicker, glitch, "
        "people, human, face, hands, cockpit, unreadable tiny text"
    )

    camera: Optional[str] = None


class ImageFrame(BaseModel):
    """
    Single image-based beat in the explainer.
    Each frame will be turned into:
      1) a still image via text-to-image
      2) a short animated segment from that still (e.g. hold, pan, Ken Burns)
    """

    frame_id: conint(ge=1)

    # Duration of this frame's animated segment in seconds
    duration_s: conint(ge=1, le=20) = Field(
        ...,
        description="Duration in seconds this frame should be visible as an animated segment.",
    )

    # High-level visual description to feed into the text-to-image model
    image_prompt: str = Field(
        ...,
        description="Natural language description of what the image should look like.",
    )

    # How this still should move over time when turned into video
    animation_type: Literal[
        "hold",
        "ken_burns_zoom_in",
        "ken_burns_zoom_out",
        "pan_left",
        "pan_right",
        "pan_up",
        "pan_down",
    ] = Field(
        "hold",
        description="How to animate this still when creating the video segment.",
    )

    # Optional on-screen text overlay for this frame
    on_screen_text_overlay: str = Field(
        "none",
        description='Exact words in quotes to overlay on screen, or "none".',
    )

    # Describes how this frame connects visually to the next one
    continuity_note: str = Field(
        ...,
        description="Describe what carries over into the next frame for smooth continuity.",
    )

    # Additional environment details to guide the image model
    environment_details: conlist(str, min_length=1, max_length=4) = Field(
        ...,
        description="Concrete details about the surrounding environment or context.",
    )

    lighting: Optional[str] = None
    cinematic_look: Optional[str] = None


class ProjectState(BaseModel):
    project_id: str
    user_prompt: str

    # Trial: total must be 20s
    target_duration_s: Literal[20]

    style_bible: StyleBible

    # Legacy WAN-based clip structure (kept for backwards compatibility, but unused
    # in the new image-based pipeline). New projects may leave this empty.
    clips: List[Clip] = Field(default_factory=list)

    # New image-based structure: ordered frames that will each become an animated segment.
    frames: conlist(ImageFrame, min_length=1) = Field(
        ...,
        description="Ordered list of frames that drive image generation and animation.",
    )

    # 20-second master teaching script (no narration)
    full_script_20s: str
