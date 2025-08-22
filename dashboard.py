import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from kmeans_algo import KMeansClustering
from em_algo import EMClustering

# Load datasets
faithful_df = pd.read_csv("faithful.csv")
if faithful_df.columns[0] == "":
    faithful_df = faithful_df.drop(columns=[faithful_df.columns[0]])

dataset_df = pd.read_csv("dataset.csv")
if dataset_df.columns[0] == "":
    dataset_df = dataset_df.drop(columns=[dataset_df.columns[0]])

app = dash.Dash(__name__)

app.layout = html.Div(
    style={"display": "flex", "flexDirection": "column", "height": "100vh", "padding": "20px", "gap": "20px"},
    children=[
        html.H1("Dataset Selector & Clustering", style={"textAlign": "center"}),

        html.Div(
            style={"display": "flex", "flex": "1", "gap": "20px"},
            children=[
                # Left panel (styled like first layout)
                html.Div(
                    style={
                        "flex": "0 0 250px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "15px",
                        "backgroundColor": "#f8f9fa",
                        "padding": "15px",
                        "borderRadius": "10px",
                        "boxShadow": "0px 0px 8px rgba(0,0,0,0.1)"
                    },
                    children=[
                        html.H3("Controls", style={"marginBottom": "10px"}),

                        html.Label("Dataset:"),
                        dcc.Dropdown(
                            id="dataset-dropdown",
                            options=[
                                {"label": "Old Faithful Geyser", "value": "faithful"},
                                {"label": "Dataset.csv", "value": "dataset"}
                            ],
                            value="faithful",
                            clearable=False
                        ),

                        html.Label("Algorithm:"),
                        dcc.Dropdown(
                            id="algo-dropdown",
                            options=[
                                {"label": "K-Means", "value": "kmeans"},
                                {"label": "Expectation Maximization (GMM)", "value": "em"}
                            ],
                            value="kmeans",
                            clearable=False
                        ),

                        html.Label("Number of Clusters:"),
                        dcc.Input(id="num-clusters", type="number", value=2, min=1, step=1),

                        html.Button("Run Algorithm", id="run-button", n_clicks=0, style={"padding": "10px"}),
                        html.Button("Reset Results", id="reset-button", n_clicks=0, style={"padding": "10px"})
                    ]
                ),

                # Right panel (plot)
                html.Div(
                    children=[
                        dcc.Graph(
                            id="scatter-plot",
                            style={
                                "width": "600px",
                                "height": "500px",
                                "margin": "auto"
                            }
                        )
                    ]
                )
            ]
        ),

        # Hidden storage for clustering results
        dcc.Store(id="stored-figures", data={})
    ]
)

# Function to generate plain scatter
def generate_plain_scatter(dataset_name):
    if dataset_name == "faithful":
        df, x_col, y_col = faithful_df, "eruptions", "waiting"
        title = "Old Faithful Geyser Eruptions"
    else:
        df, x_col, y_col = dataset_df, "F1", "F2"
        title = "Custom Dataset Scatter Plot"

    fig = px.scatter(df, x=x_col, y=y_col, title=title, template="plotly_white")
    fig.update_layout(width=700, height=500)
    return fig

# Callback 1: Update plot when dataset changes or stored results reset
@app.callback(
    Output("scatter-plot", "figure"),
    Input("dataset-dropdown", "value"),
    Input("stored-figures", "data")
)
def display_dataset(selected_dataset, stored_figures):
    if stored_figures and selected_dataset in stored_figures:
        return go.Figure(stored_figures[selected_dataset])
    return generate_plain_scatter(selected_dataset)

# Callback 2: Run clustering and store results
@app.callback(
    Output("stored-figures", "data"),
    Input("run-button", "n_clicks"),
    State("dataset-dropdown", "value"),
    State("algo-dropdown", "value"),
    State("num-clusters", "value"),
    State("stored-figures", "data"),
    prevent_initial_call=True
)
def run_clustering(n_clicks, selected_dataset, algo_choice, n_clusters, stored_figures):
    stored_figures = stored_figures or {}
    if selected_dataset == "faithful":
        df, x_col, y_col = faithful_df.copy(), "eruptions", "waiting"
    else:
        df, x_col, y_col = dataset_df.copy(), "F1", "F2"

    if algo_choice == "kmeans":
        algo = KMeansClustering(n_clusters=n_clusters)
    elif algo_choice == "em":
        algo = EMClustering(n_clusters=n_clusters)
    else:
        raise ValueError("Unknown algorithm")

    labels, centers = algo.fit(df[[x_col, y_col]].values)
    df["Cluster"] = labels

    fig = px.scatter(
        df, x=x_col, y=y_col, color="Cluster",
        title=f"{algo_choice.upper()} with {n_clusters} Clusters",
        template="plotly_white"
    )
    fig.add_scatter(
        x=centers[:, 0],
        y=centers[:, 1],
        mode="markers",
        marker=dict(color="black", size=12, symbol="x"),
        name="Centers"
    )
    fig.update_layout(width=700, height=500)

    stored_figures[selected_dataset] = fig.to_dict()
    return stored_figures

@app.callback(
    Output("stored-figures", "data", allow_duplicate=True),
    Input("reset-button", "n_clicks"),
    State("dataset-dropdown", "value"),  # get current dataset
    State("stored-figures", "data"),
    prevent_initial_call=True
)
def reset_stored_results(n_clicks, selected_dataset, stored_data):
    if stored_data is None:
        stored_data = {}

    # Remove only the selected dataset from stored results
    stored_data.pop(selected_dataset, None)

    return stored_data

if __name__ == "__main__":
    app.run(debug=True)
