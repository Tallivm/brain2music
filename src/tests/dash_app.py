from torch.multiprocessing import Queue

import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output


def start_dash(parameter_queue: Queue) -> None:
    app = dash.Dash(__name__)

    NUM_CHANNELS = 8

    channel_checklist_options = [
        {'label': f'Channel {i}', 'value': i} for i in range(1, NUM_CHANNELS+1)
    ]

    app.layout = html.Div([
        html.H1('Hello, Dash!'),
        html.Div('This is a simple example of a Dash app.'),

        html.H1('Checkbox Example'),
        dcc.Checklist(
            id='my-checkbox',
            options=channel_checklist_options,
            value=list(range(1, NUM_CHANNELS+1))
        ),
        html.Div(id='output')
    ])

    @app.callback(
        Output('output', 'children'),
        Input('my-checkbox', 'value'),
    )
    def update_output(value: list[int]):
        print("Value: ", ", ".join(str(x) for x in value))
        parameter_queue.put(value)

    app.run(debug=True, port=4560)  # Debug=True seems to start main a second time
