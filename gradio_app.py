import os
import sys

# Ensure local analysis modules are found
ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
sys.path.insert(0, ANALYSIS_DIR)

import gradio as gr
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
from app import app as flask_app

# Mount existing Flask application inside FastAPI
fastapi_app = FastAPI()
fastapi_app.mount("/app", WSGIMiddleware(flask_app))

# Gradio Interface embedding the interactive GIS planner
with gr.Blocks(title="Village Pond Planning GIS | IIT Bhilai", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 💧 Automated Village Pond Planning & Rainwater Harvesting System
        **Phase 3 Implementation — IIT Bhilai**
        *Click and explore the interactive GIS interface below. Select land parcels, tune rainfall, and identify optimal catchment basins.*
        """
    )
    gr.HTML(
        '<iframe src="/app/planner" width="100%" height="900px" '
        'style="border: 1px solid #243B55; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.3);">'
        '</iframe>'
    )

app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
