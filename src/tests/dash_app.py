from torch.multiprocessing import Queue

import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output


def start_dash(parameter_queue: Queue) -> None:
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1('Hello, Dash!'),
        html.Div('This is a simple example of a Dash app.'),

        html.H1('Checkbox Example'),
        dcc.Checklist(
            id='my-checkbox',
            options=[
                {'label': 'Option 1', 'value': 'option1'},
                {'label': 'Option 2', 'value': 'option2'}
            ],
            value=[]
        ),
        html.Div(id='output')
    ])

    @app.callback(
        Output('output', 'children'),
        Input('my-checkbox', 'value'),
    )
    def update_output(value):
        if 'option1' in value:
            print('You have selected Option 1')
        elif 'option2' in value:
            print('You have selected Option 2')
        else:
            print('Please select an option')

        print("Value: ", ", ".join(value))

    app.run(debug=True, port=4560)  # Debug=True seems to start main a second time
