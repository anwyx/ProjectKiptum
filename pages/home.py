from dash import dcc, html, callback, Input, Output, no_update
import dash
from pathlib import Path

# Register page
_dash_page = dash.register_page(__name__, path="/")

home_layout = html.Div(
    style={
        'minHeight': '100vh',
        'width': '100%',
        'margin': '0',
        'padding': '0',
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': 'var(--background-color)',
        'display': 'flex',
        'flexDirection': 'column',
        'position': 'relative'
    },
    children=[
        dcc.Location(id='redirect-home', refresh=True),
        html.Div(
            style={'width': '240px','height': '240px','overflow': 'hidden','margin': '10px auto 0 auto'},
            children=[html.Img(src="/assets/App Logo.png", style={'width':'100%','height':'100%','objectFit':'contain'})]
        ),
        html.Div(
            style={'flex': '0.8','display':'flex','flexDirection':'column','justifyContent':'flex-start','alignItems':'center','marginTop':'-10px'},
            children=[
                html.H1("Privacy-Aware Photo Gallery", className='shine-text', style={'fontSize':'54px','textAlign':'center','marginBottom':'12px'}),
                html.H2("On-device AI to detect & protect sensitive visuals", className='shine-text', style={'fontSize':'24px','textAlign':'center','marginBottom':'28px'}),
                html.Button("Get Started", id="start-btn", className="shine-button", style={'marginTop':'10px','fontWeight':'bold','border':'none','padding':'14px 34px','borderRadius':'8px','cursor':'pointer','fontSize':'18px','boxShadow':'0 4px 10px rgba(0,0,0,0.15)'}),
                html.Div(id="start-msg", style={'marginTop':'12px'})
            ]
        ),
        html.Div(
            style={'display':'flex','justifyContent':'space-between','alignItems':'center','flexWrap':'wrap','maxWidth':'1200px','margin':'40px auto','padding':'0 40px','width':'100%'},
            children=[
                html.Div(style={'flex':'1','margin':'20px','minWidth':'300px','textAlign':'center','maxWidth':'350px','background':'#ffffff22','padding':'18px','borderRadius':'10px','backdropFilter':'blur(4px)'},children=[
                    html.H3("AI Detection", className='shine-subtext', style={'marginBottom':'10px','fontSize':'24px'}),
                    html.P("Faces & sensitive objects are located locally – nothing leaves your device.", className='shine-subtext', style={'fontSize':'16px','lineHeight':'1.5'})
                ]),
                html.Div(style={'flex':'1','margin':'20px','minWidth':'300px','textAlign':'center','maxWidth':'350px','background':'#ffffff22','padding':'18px','borderRadius':'10px','backdropFilter':'blur(4px)'},children=[
                    html.H3("Automatic Blurring", className='shine-subtext', style={'marginBottom':'10px','fontSize':'24px'}),
                    html.P("Sensitive regions are blurred until you unlock them.", className='shine-subtext', style={'fontSize':'16px','lineHeight':'1.5'})
                ]),
                html.Div(style={'flex':'1','margin':'20px','minWidth':'300px','textAlign':'center','maxWidth':'350px','background':'#ffffff22','padding':'18px','borderRadius':'10px','backdropFilter':'blur(4px)'},children=[
                    html.H3("Local Privacy", className='shine-subtext', style={'marginBottom':'10px','fontSize':'24px'}),
                    html.P("All processing stays on your device – ensuring privacy by design.", className='shine-subtext', style={'fontSize':'16px','lineHeight':'1.5'})
                ])
            ]
        ),
    ]
)

layout = home_layout

@callback(
    Output('redirect-home', 'href'),
    Output('start-msg','children'),
    Input('start-btn','n_clicks'),
    prevent_initial_call=True
)
def go_gallery(n):
    if not n:
        return no_update, no_update
    return '/gallery', "Loading gallery..."
