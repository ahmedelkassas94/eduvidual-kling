from typing import Optional, Literal, List
from pydantic import BaseModel, Field, conint, conlist


# ---------------------------------------------------------
# NEW ARCHITECTURE: 15s shot-by-shot script + named ingredients
# ---------------------------------------------------------


class Ingredient(BaseModel):
    """One visual element (e.g. airplane, wing, arrows, environment). Has a name and a structured T2I prompt (5-part)."""
    name: str = Field(..., description="Short identifier, e.g. 'airplane', 'airplane_wing', 'environment'")
    t2i_prompt: str = Field(
        ...,
        description="Text-to-image prompt in 5 parts: [Scientific Subject], [Action/State], [Style/Medium], [Lighting & Texture], [Framing]. Omit if matplotlib_code is set.",
    )
    matplotlib_code: Optional[str] = Field(
        None,
        description="If this ingredient is a chart/data plot, Python Matplotlib code to render it precisely. When set, T2I is not used for this ingredient.",
    )


class ScriptShot(BaseModel):
    """One shot in the main 15s script. Very detailed description; references ingredients by name."""
    shot_id: conint(ge=1)
    time_range: str = Field(..., description='e.g. "0-2s", "2-5s"')
    duration_s: conint(ge=1, le=10)
    detailed_description: str = Field(
        ...,
        description="Very detailed description of what should happen in this shot (action, camera, elements, continuity).",
    )
    movement_prompt: str = Field(
        ...,
        description="Prompt for I2V: how to move/animate from start to end of this shot (camera, objects, continuity).",
    )
    ingredient_names: conlist(str, min_length=1) = Field(
        ...,
        description="Names of ingredients that appear in this shot (must match Ingredient.name).",
    )
    new_ingredient_names: conlist(str, min_length=0) = Field(
        default_factory=list,
        description="Ingredient names introduced in THIS shot (not in previous shots). Empty for shot 1.",
    )
    narration_text: str = Field("", description="Narration spoken during this shot.")
    on_screen_text_overlay: str = Field("none", description='Short on-screen text or "none".')
    i2i_spatial_prompt: str = Field(
        "",
        description="EXTREMELY DETAILED prompt for I2I generation of shot 1's first frame. Only used for shot_id=1. Must describe: exact spatial relationships between all ingredients (positions relative to each other and camera), camera position/angle (e.g., 'camera facing directly at the scene', 'mirror placed above the table'), precise placement of each object, lighting, shadows, and how objects interact spatially. Leave empty for shots 2+.",
    )


class StyleBible(BaseModel):
    visual_style: str
    camera_rules: str
    lighting_rules: str
    continuity_rules: str
    style_suffix: str = Field(
        "",
        description="Appended to every t2i_prompt for consistency (e.g. 'muted professional teal and grey palette, 4k, cinematic').",
    )


Lens = Literal["24mm", "35mm", "50mm", "85mm"]


class Shot(BaseModel):
    # Exactly 3 shots per clip in trial
    shot_id: conint(ge=1, le=3)

    # Keep time ranges human-readable (WAN prompt is textual anyway)
    time_range: str = Field(..., description='e.g., "0ÔÇô3s", "3ÔÇô7s", "7ÔÇô10s"')

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
        "",
        description="Describe what carries over into the next frame for smooth continuity. (Legacy field)",
    )
    
    # NEW: Describes how this frame should transition TO the next frame
    transition_to_next_frame: str = Field(
        ...,
        description="Describe how this frame transitions to the next frame: camera movement, object movements, and final state that should match the next frame's description.",
    )

    # Additional environment details to guide the image model
    environment_details: conlist(str, min_length=1, max_length=4) = Field(
        ...,
        description="Concrete details about the surrounding environment or context.",
    )

    lighting: Optional[str] = None
    cinematic_look: Optional[str] = None
    
    # Narration text for this frame segment (will be converted to speech)
    narration_text: str = Field(
        "",
        description="Narration text to be spoken during this frame segment. Should be concise and match the duration.",
    )


class ProjectState(BaseModel):
    project_id: str
    user_prompt: str

    # 20s (legacy) or 15s (new ingredients pipeline)
    target_duration_s: Literal[15, 20] = 20

    style_bible: StyleBible

    # Legacy WAN-based clip structure (kept for backwards compatibility)
    clips: List[Clip] = Field(default_factory=list)

    # Legacy image-based: ordered frames (used when ingredients/shots not present)
    frames: List[ImageFrame] = Field(default_factory=list)

    # 20-second master teaching script (legacy)
    full_script_20s: str = ""

    # ---- NEW: 15s shot-by-shot + ingredients pipeline ----
    # Full shot-by-shot video description (15s). Reference for all generation.
    main_script_15s: str = Field("", description="Full 15s script: shot-by-shot detailed description of the whole video.")
    # Named ingredients: one T2I image per ingredient (lengthy prompt each).
    ingredients: List[Ingredient] = Field(default_factory=list)
    # Shots: each references ingredients by name; movement_prompt drives I2V.
    shots: List[ScriptShot] = Field(default_factory=list)
