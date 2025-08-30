import os
import json
from pathlib import Path
import dash
from dash import html, dcc, dash_table, Input, Output, State
from dotenv import load_dotenv

load_dotenv(override=False)

app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True, title="Privacy Gallery")
server = app.server

# Only show the current page content
app.layout = dash.page_container

if __name__ == "__main__":
    app.run_server(debug=True)