import json
from pathlib import Path
import base64
import dash
from dash import html, dcc, Input, Output, State, ALL, ctx


dash.register_page(__name__, path="/gallery")

REPORT_PATH = Path("data/processed/reports.json")
IMG_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


def load_reports():
    if REPORT_PATH.exists():
        try:
            with open(REPORT_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def list_blurred_images():
    """Return list of processed image Paths sorted by modified time (newest first)."""
    if not IMG_DIR.exists():
        return []
    imgs = list(IMG_DIR.glob("*_blurred.*"))
    # Sort by name ascending
    imgs.sort(key=lambda p: p.name)
    return imgs


def list_raw_images():
    """Return list of raw images (jpg/png)."""
    if not RAW_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png"}
    imgs = [p for p in RAW_DIR.iterdir() if p.suffix.lower() in exts]
    # Sort by name ascending
    imgs.sort(key=lambda p: p.name)
    return imgs


def get_current_images(unlocked: bool):
    return list_raw_images() if unlocked else list_blurred_images()


def img_to_data_uri(p: Path):
    try:
        mime = "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
        b = p.read_bytes()
        return f"data:{mime};base64,{base64.b64encode(b).decode()}"
    except Exception:
        return ""


layout = html.Div([
    html.Div(className="gallery-header", children=[
        html.Div(className="gallery-header-left", children=[
            html.Img(src="/assets/App Logo.png", className="gallery-logo"),
            html.Div(className="gallery-header-text", children=[
                html.H1("Gallery", className="gallery-title"),
            ])
        ]),
        html.Div(className="gallery-actions", children=[
            html.Button("Unlock", id="unlock-btn", className="nav-button", n_clicks=0)
        ])
    ]),
    dcc.Store(id="gallery-state", data={"open": False, "index": 0}),
    dcc.Store(id="gallery-mode", data={"unlocked": False}),
    html.Div(id="gallery-grid", className="gallery-grid"),
    # Lightbox Modal
    html.Div(id="lightbox-modal", className="lightbox hidden", children=[
        html.Div(className="lightbox-inner", children=[
            html.Button("×", id="close-lightbox", className="lightbox-close", n_clicks=0),
            html.Button("‹", id="lightbox-prev", className="lightbox-nav prev", n_clicks=0, title="Previous"),
            html.Button("›", id="lightbox-next", className="lightbox-nav next", n_clicks=0, title="Next"),
            html.Img(id="lightbox-image", className="lightbox-image", alt="preview"),
            html.Div(id="lightbox-tags", className="lightbox-tags")
        ])
    ]),
    dcc.Interval(id="refresh-int", interval=5000, n_intervals=0)
], className="gallery-wrapper")


@dash.callback(
    Output("gallery-mode", "data"),
    Input("unlock-btn", "n_clicks"),
    State("gallery-mode", "data"),
    prevent_initial_call=True
)
def toggle_unlock(n, data):
    data = data or {"unlocked": False}
    if not n:
        return data
    # toggle each click
    return {"unlocked": not data.get("unlocked", False)}


@dash.callback(
    Output("unlock-btn", "children"),
    Input("gallery-mode", "data")
)
def set_unlock_label(mode):
    unlocked = (mode or {}).get("unlocked", False)
    return "Lock" if unlocked else "Unlock"


@dash.callback(
    Output("gallery-grid", "children"),
    Input("refresh-int", "n_intervals"),
    Input("gallery-mode", "data")
)
def refresh_gallery(_, mode):
    unlocked = (mode or {}).get("unlocked", False)
    reports = load_reports()
    report_map = {Path(rep.get("image_path", "")).stem: rep.get("regions", []) for rep in reports}
    cards = []
    images = get_current_images(unlocked)
    for i, img_path in enumerate(images):
        stem = img_path.stem
        regions = []
        if not unlocked:  # only attempt regions for processed images
            base_stem = img_path.stem.replace("_blurred", "")
            regions = report_map.get(base_stem, [])
        if regions:
            badges = [html.Span(r.get('tag', '?'), className="region-badge", title=r.get('tag','')) for r in regions if r]
        else:
            badges = [html.Span("No Tags", className="region-badge empty")]
        overlay = html.Div(badges, className="img-overlay")
        cards.append(
            html.Div([
                html.Div([
                    html.Img(
                        src=img_to_data_uri(img_path),
                        alt=stem,
                        className="gallery-image"
                    ),
                    overlay
                ], className="img-wrapper"),
            ], className="gallery-item", id={"type": "gallery-item", "index": i}, n_clicks=0)
        )
    if not cards:
        msg = "No raw images yet." if unlocked else "No processed images yet. Run the pipeline to populate the gallery."
        return html.Div(msg, className="empty-gallery")
    return cards


@dash.callback(
    Output("gallery-state", "data"),
    Input({"type": "gallery-item", "index": ALL}, "n_clicks"),
    Input("lightbox-prev", "n_clicks"),
    Input("lightbox-next", "n_clicks"),
    Input("close-lightbox", "n_clicks"),
    Input("gallery-mode", "data"),
    State("gallery-state", "data"),
    prevent_initial_call=True
)
def update_gallery_state(item_clicks, prev_clicks, next_clicks, close_clicks, mode, state):
    state = state or {"open": False, "index": 0}
    unlocked = (mode or {}).get("unlocked", False)

    triggered = ctx.triggered
    if not triggered:
        return state
    trig = triggered[0]
    prop_id = trig.get("prop_id", "")
    value = trig.get("value")

    if not value and not prop_id.startswith("gallery-mode"):
        return state

    images = get_current_images(unlocked)

    if prop_id.endswith('.n_clicks') and 'gallery-item' in prop_id:
        id_part = prop_id[:-len('.n_clicks')]
        try:
            id_obj = json.loads(id_part)
        except Exception:
            id_obj = None
        if isinstance(id_obj, dict) and id_obj.get("type") == "gallery-item":
            idx = id_obj.get("index", 0)
            if 0 <= idx < len(images):
                return {"open": True, "index": idx}

    total = len(images)
    if prop_id.startswith("lightbox-prev") and state.get("open") and total:
        return {"open": True, "index": (state.get("index", 0) - 1) % total}
    if prop_id.startswith("lightbox-next") and state.get("open") and total:
        return {"open": True, "index": (state.get("index", 0) + 1) % total}
    if prop_id.startswith("close-lightbox"):
        return {"open": False, "index": state.get("index", 0)}
    if prop_id.startswith("gallery-mode"):
        # Closing lightbox when switching modes
        return {"open": False, "index": 0}

    return state


@dash.callback(
    Output("lightbox-modal", "className"),
    Output("lightbox-image", "src"),
    Output("lightbox-tags", "children"),
    Input("gallery-state", "data"),
    Input("gallery-mode", "data")
)
def render_lightbox(state, mode):
    unlocked = (mode or {}).get("unlocked", False)
    images = get_current_images(unlocked)
    if not images or not state or not state.get("open"):
        return "lightbox hidden", "", []
    idx = min(max(state.get("index", 0), 0), len(images)-1)
    img_path = images[idx]

    reports = load_reports()
    report_map = {Path(rep.get("image_path", "")).stem: rep.get("regions", []) for rep in reports}
    tag_nodes = []
    if not unlocked:
        base_stem = img_path.stem.replace("_blurred", "")
        regions = report_map.get(base_stem, [])
        if regions:
            tag_nodes = [html.Span(r.get('tag','?'), className='lightbox-tag') for r in regions if r]
        else:
            tag_nodes = [html.Span("No Tags", className='lightbox-tag empty')]
    return "lightbox", img_to_data_uri(img_path), tag_nodes