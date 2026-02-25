from typing import Optional, Literal, List
from pydantic import BaseModel, Field, conint, conlist


# ---------------------------------------------------------
# NEW ARCHITECTURE: 15s shot-by-shot script + named ingredients
# ---------------------------------------------------------


class Ingredient(BaseModel):
    """Visual element reference (name + optional short description for I2V). When first_frame_t2i_prompt is used, no per-ingredient T2I."""
    name: str = Field(..., description="Short identifier, e.g. 'airplane', 'lab_environment'")
    description: str = Field("", description="Short description for I2V when this ingredient appears or is newly introduced.")
    t2i_prompt: Optional[str] = Field(
        None,
        description="Optional per-ingredient T2I (used only in legacy mode). When first_frame_t2i_prompt is set, this is unused.",
    )
    matplotlib_code: Optional[str] = Field(
        None,
        description="If this ingredient is a chart/data plot, Python Matplotlib code to render it. When set, T2I is not used for this ingredient.",
    )


class ScriptShot(BaseModel):
    """
    One shot in the main 15s script.
    Architecture: first frame and last frame of each shot are generated (T2I or I2I).
    I2V animates from first frame to last frame. Last frame of shot N = first frame of shot N+1.
    """
    shot_id: conint(ge=1)
    time_range: str = Field(..., description='e.g. "0-2s", "2-5s"')
    duration_s: conint(ge=1, le=10)
    detailed_description: str = Field(
        ...,
        description="Very detailed description of what should happen in this shot (action, camera, elements, continuity).",
    )
    movement_prompt: str = Field(
        ...,
        description="Prompt for I2V: how the scene moves from first frame to last frame (camera, objects, continuity).",
    )
    last_frame_t2i_prompt: str = Field(
        "",
        description="T2I prompt for the END state of this shot. Used to generate the last frame (or as intent for I2I reviser). Same detail level as first_frame_t2i_prompt: environment, every object, spatial relationships.",
    )
    ingredient_names: conlist(str, min_length=0) = Field(
        default_factory=list,
        description="Names of ingredients that appear in this shot (must match Ingredient.name).",
    )
    new_ingredient_names: conlist(str, min_length=0) = Field(
        default_factory=list,
        description="Ingredient names introduced in THIS shot (not in previous shots). Empty for shot 1.",
    )
    narration_text: str = Field("", description="Narration spoken during this shot (explaining the content).")
    on_screen_text_overlay: str = Field("none", description='Short on-screen text or "none".')
    assisting_visual_aids: str = Field(
        "",
        description="Assisting visual aids for this shot when they serve the video: arrows, highlights, text boxes, text labels. Empty or 'none' when not needed.",
    )
    i2i_spatial_prompt: str = Field(
        "",
        description="Legacy: I2I spatial prompt for shot 1 first frame. Unused when first_frame_t2i_prompt is set.",
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

    # Video duration in seconds: 20 (legacy), 15 (legacy ingredients), or actual from narration-first pipeline (e.g. up to 24)
    target_duration_s: int = 24

    style_bible: StyleBible

    # Legacy WAN-based clip structure (kept for backwards compatibility)
    clips: List[Clip] = Field(default_factory=list)

    # Legacy image-based: ordered frames (used when ingredients/shots not present)
    frames: List[ImageFrame] = Field(default_factory=list)

    # 20-second master teaching script (legacy)
    full_script_20s: str = ""

    # ---- NEW: 15s shot-by-shot + ingredients pipeline ----
    main_script_15s: str = Field("", description="Full 15s script: shot-by-shot detailed description of the whole video.")
    # Single T2I prompt for shot 1 first frame (environment + all ingredients + spatial relationships). When set, no per-ingredient T2I.
    first_frame_t2i_prompt: str = Field("", description="One detailed T2I prompt for the first frame: environment, every ingredient in detail, spatial relationships.")
    ingredients: List[Ingredient] = Field(default_factory=list)
    shots: List[ScriptShot] = Field(default_factory=list)
