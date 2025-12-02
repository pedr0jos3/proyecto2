
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px

# 1. Cargar datos y construir df igual que en el notebook
data = pd.read_csv("listings_cleaned.csv")
data["review_scores"] = data["review_scores"].astype(float)

df = data.copy()
df["recommended"] = (
    (df["occupancy_rate"] >= 0.65) &
    (df["review_scores"] >= 4.6)
).astype(int)

num_features = [
    "accommodates", "bedrooms", "beds", "latitude", "longitude",
    "occupancy_rate", "price_per_person", "minimum_nights",
    "maximum_nights", "number_of_reviews", "host_total_listings_count",
    "review_scores", "reviews_per_month"
]

df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(0)
df["beds"] = pd.to_numeric(df["beds"], errors="coerce").fillna(0)

cat_features = ["neighbourhood_cleansed", "property_type_simple", "host_is_superhost"]
df = pd.get_dummies(df, columns=cat_features, drop_first=True)

feature_cols = num_features + [c for c in df.columns if any(feat in c for feat in cat_features)]

# 2. Cargar modelos y scaler
reg_model = load_model("models/regression_model.h5")
clf_model = load_model("models/classification_model.h5")
scaler = joblib.load("models/scaler.pkl")

# 3. Crear app Dash
app = Dash(__name__)

neigh_options = sorted(data["neighbourhood_cleansed"].dropna().unique())
ptype_options = sorted(data["property_type_simple"].dropna().unique())

app.layout = html.Div([
    html.H2("Tablero Airbnb – Predicción de precio y recomendación"),

    html.Div([
        # PANEL IZQUIERDO – Inputs
        html.Div([
            html.H4("Describe tu alojamiento"),

            html.Label("Barrio"),
            dcc.Dropdown(
                id="inp-neighbourhood",
                options=[{"label": n, "value": n} for n in neigh_options],
                value=neigh_options[0]
            ),

            html.Label("Tipo de propiedad"),
            dcc.Dropdown(
                id="inp-property-type",
                options=[{"label": p, "value": p} for p in ptype_options],
                value=ptype_options[0]
            ),

            html.Label("Superhost"),
            dcc.Dropdown(
                id="inp-superhost",
                options=[
                    {"label": "Sí", "value": "t"},
                    {"label": "No", "value": "f"},
                ],
                value="f"
            ),

            html.Label("Huéspedes (accommodates)"),
            dcc.Input(id="inp-accommodates", type="number", value=2),

            html.Label("Habitaciones (bedrooms)"),
            dcc.Input(id="inp-bedrooms", type="number", value=1),

            html.Label("Camas (beds)"),
            dcc.Input(id="inp-beds", type="number", value=1),

            html.Label("Noche mínima"),
            dcc.Input(id="inp-min-nights", type="number", value=1),

            html.Label("Noche máxima"),
            dcc.Input(id="inp-max-nights", type="number", value=30),

            html.Label("Precio por persona"),
            dcc.Input(id="inp-price-pp", type="number", value=100000),

            html.Label("Tasa de ocupación (0–1)"),
            dcc.Slider(id="inp-occ", min=0, max=1, step=0.01, value=0.7),

            html.Label("Número de reseñas"),
            dcc.Input(id="inp-nreviews", type="number", value=10),

            html.Label("Calificación promedio (0–5)"),
            dcc.Slider(id="inp-score", min=0, max=5, step=0.1, value=4.7),

            html.Label("Reseñas por mes"),
            dcc.Input(id="inp-rpm", type="number", value=0.5),

            html.Label("Número de listings del host"),
            dcc.Input(id="inp-host-listings", type="number", value=1),

            html.Br(),
            html.Button("Calcular", id="btn-calc", n_clicks=0),

        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),

        # PANEL DERECHO – Resultados
        html.Div([
            html.H4("Resultados del modelo"),
            html.Div(id="predicted-price", style={"fontSize": "20px", "marginBottom": "10px"}),
            html.Div(id="predicted-recommended", style={"fontSize": "18px"}),
            html.Hr(),
            html.Div(id="interpretacion"),
        ], style={"width": "68%", "display": "inline-block", "padding": "10px"}),
    ]),

    html.Hr(),

    # VISUALIZACIONES
    html.Div([
        html.H3("Visualizaciones de contexto"),

        dcc.Dropdown(
            id="filtro-barrio",
            options=[{"label": "Todos", "value": "ALL"}] +
                    [{"label": n, "value": n} for n in neigh_options],
            value="ALL",
            style={"width": "300px"}
        ),

        dcc.Graph(id="mapa-listings"),
        dcc.Graph(id="precio-por-barrio"),
        dcc.Graph(id="scatter-score-price")
    ])
])

# 4. Callback para predecir
@app.callback(
    [Output("predicted-price", "children"),
     Output("predicted-recommended", "children"),
     Output("interpretacion", "children")],
    Input("btn-calc", "n_clicks"),
    state=[
        State("inp-neighbourhood", "value"),
        State("inp-property-type", "value"),
        State("inp-superhost", "value"),
        State("inp-accommodates", "value"),
        State("inp-bedrooms", "value"),
        State("inp-beds", "value"),
        State("inp-min-nights", "value"),
        State("inp-max-nights", "value"),
        State("inp-price-pp", "value"),
        State("inp-occ", "value"),
        State("inp-nreviews", "value"),
        State("inp-score", "value"),
        State("inp-rpm", "value"),
        State("inp-host-listings", "value"),
    ]
)
def hacer_prediccion(n_clicks, neigh, ptype, superhost, acc, bed, beds,
                     min_n, max_n, price_pp, occ, nrev, score, rpm, host_list):
    if n_clicks == 0:
        return "", "", ""

    # Crear df con una fila y todas las columnas feature_cols
    x_input = pd.DataFrame(columns=feature_cols)
    x_input.loc[0] = 0  # inicializar en cero

    # Numéricas
    x_input.loc[0, "accommodates"] = acc
    x_input.loc[0, "bedrooms"] = bed
    x_input.loc[0, "beds"] = beds
    x_input.loc[0, "latitude"] = df["latitude"].mean()    
    x_input.loc[0, "longitude"] = df["longitude"].mean()
    x_input.loc[0, "occupancy_rate"] = occ
    x_input.loc[0, "price_per_person"] = price_pp
    x_input.loc[0, "minimum_nights"] = min_n
    x_input.loc[0, "maximum_nights"] = max_n
    x_input.loc[0, "number_of_reviews"] = nrev
    x_input.loc[0, "host_total_listings_count"] = host_list
    x_input.loc[0, "review_scores"] = score
    x_input.loc[0, "reviews_per_month"] = rpm

    # Dummies categóricas
    col_neigh = f"neighbourhood_cleansed_{neigh}"
    col_ptype = f"property_type_simple_{ptype}"
    col_super = f"host_is_superhost_{superhost}"

    for col in [col_neigh, col_ptype, col_super]:
        if col in x_input.columns:
            x_input.loc[0, col] = 1
        

    # Escalar con el scaler que entrenaste
    X_scaled = scaler.transform(x_input[feature_cols].values)

    # Predicciones
    price_pred = reg_model.predict(X_scaled)[0][0]
    prob_rec = float(clf_model.predict(X_scaled)[0][0])

    texto_price = f"Precio recomendado: COP ${price_pred:,.0f}"
    texto_prob = f"Probabilidad de ser 'recommended': {prob_rec*100:,.1f}%"

    if prob_rec > 0.7:
        mensaje = "Alta probabilidad de estar bien posicionado."
    elif prob_rec > 0.4:
        mensaje = "Desempeño medio, con oportunidades de mejora."
    else:
        mensaje = "Baja probabilidad de recomendación con las condiciones actuales."

    interpretacion = (
        f"Para un alojamiento en {neigh} de tipo {ptype} con capacidad para {acc} huésped(es), "
        f"el modelo sugiere un precio alrededor de COP ${price_pred:,.0f} y estima una "
        f"probabilidad de {prob_rec*100:,.1f}% de que el anuncio sea recomendado."
    )

    return texto_price, texto_prob + " – " + mensaje, interpretacion

# 5. Callbacks de visualizaciones
@app.callback(
    [Output("mapa-listings", "figure"),
     Output("precio-por-barrio", "figure"),
     Output("scatter-score-price", "figure")],
    Input("filtro-barrio", "value")
)
def actualizar_graficas(filtro_barrio):
    dff = data.copy()
    if filtro_barrio != "ALL":
        dff = dff[dff["neighbourhood_cleansed"] == filtro_barrio]

    fig_map = px.scatter_mapbox(
        dff,
        lat="latitude",
        lon="longitude",
        color="occupancy_rate",
        zoom=10,
        height=400
    )
    fig_map.update_layout(mapbox_style="open-street-map")

    barrio_stats = data.groupby("neighbourhood_cleansed").agg(
        avg_price=("price", "mean"),
        avg_occ=("occupancy_rate", "mean")
    ).reset_index()

    fig_bar = px.bar(
        barrio_stats,
        x="neighbourhood_cleansed",
        y="avg_price",
        title="Precio promedio por barrio"
    )

    fig_scatter = px.scatter(
        data,
        x="review_scores",
        y="price_per_person",
        color="recommended",
        title="Relación entre calificación y precio por persona"
    )

    return fig_map, fig_bar, fig_scatter


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)
