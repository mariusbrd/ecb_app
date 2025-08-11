
import streamlit as st
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ecbdata import ecbdata
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, acf
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import combinations
import joblib
import warnings
import io
import base64
import json
from datetime import datetime, timedelta
from scipy import stats
warnings.filterwarnings("ignore")

# Streamlit Konfiguration
st.set_page_config(
    page_title="SARIMAX ECB Prognose Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für besseres Design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Alle verfügbaren ECB-Indikatoren definieren
ALL_AVAILABLE_INDICATORS = {
    # Haushalts- und Einlagen-Indikatoren
    "QSA.Q.N.DE.W0.S128.S1.N.L.LE.F6._Z._Z.XDC._T.S.V.N._T": "Overnight-Einlagen Haushalte",
    "QSA.Q.N.DE.W0.S1M.S1.N.A.LE.F62._Z._Z.XDC._T.S.V.N._T": "Termineinlagen Haushalte",
    "QSA.Q.N.DE.W0.S128.S1.N.L.LE.F2._Z._Z.XDC._T.S.V.N._T": "Bargeld und Einlagen Haushalte",
    "ICO.A.DE.S128.L.W0.A.EUR": "Einlagenzinsen Haushalte",
    
    # Makroökonomische Indikatoren
    "QSA.Q.N.DE.W0.S1M.S1._Z.B.B6G._Z._Z._Z.XDC._T.S.V.N._T": "Verfügbares Einkommen",
    "QSA.Q.N.DE.W0.S1M.S1._Z.B.B1GH._Z._Z._Z.XDC._T.S.V.N._T": "BIP (Ausgabenansatz)",
    "QSA.Q.N.DE.W0.S1M.S1._Z.B.B8G._Z._Z._Z.XDC._T.S.V.N._T": "Sparen der Haushalte",
    "GFS.A.N.DE.W0.S13.S1._Z.B.B8G._Z._Z._Z.XDC._Z.S.V.GY._T": "Haushaltssaldo Staat",
    "GFS.A.N.DE.W0.S13.S1._Z.B.B9._Z._Z._Z.XDC._Z.S.V.GY._T": "Finanzierungssaldo Staat",
    
    # Arbeitsmarkt-Indikatoren
    "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T": "Arbeitslosenquote",
    "LFSI.M.DE.S.EMPRT.TOTAL0.15_74.T": "Beschäftigungsquote",
    "LFSI.M.DE.S.PARTRT.TOTAL0.15_74.T": "Erwerbsquote",
    
    # Zins- und Geldpolitik-Indikatoren
    "IRS.M.DE.L.L40.CI.0000.EUR.N.Z": "Zinssatz (Tagesgeld)",
    "IRS.M.DE.L.L40.CI.0010.EUR.N.Z": "Zinssatz (1 Monat)",
    "IRS.M.DE.L.L40.CI.0030.EUR.N.Z": "Zinssatz (3 Monate)",
    "IRS.M.DE.L.L40.CI.0060.EUR.N.Z": "Zinssatz (6 Monate)",
    "IRS.M.DE.L.L40.CI.0120.EUR.N.Z": "Zinssatz (1 Jahr)",
    "FM.B.DE.EUR.4F.BB.TOTRESNS.LEV": "Bankreserven",
    
    # Preis- und Inflations-Indikatoren
    "ICP.M.DE.N.000000.4.ANR": "Inflation (HVPI)",
    "ICP.M.DE.N.010000.4.ANR": "Lebensmittel-Inflation",
    "ICP.M.DE.N.070000.4.ANR": "Transport-Inflation",
    "ICP.M.DE.N.040000.4.ANR": "Wohnkosten-Inflation",
    
    # Konjunktur-Indikatoren
    "ESI.M.DE.INDI.NACE2.CI": "Vertrauensindikator",
    "ESI.M.DE.CONS.NACE2.CI": "Verbrauchervertrauen",
    "ESI.M.DE.BUIL.NACE2.CI": "Baugewerbe-Vertrauen",
    
    # Externe Faktoren
    "EXR.M.USD.EUR.SP00.A": "USD/EUR Wechselkurs",
    "YC.B.DE.EUR.4F.G_N_A.SV_C_YM.SR_1Y": "Staatsanleihe 1 Jahr",
    "YC.B.DE.EUR.4F.G_N_A.SV_C_YM.SR_10Y": "Staatsanleihe 10 Jahre"
}

# Kategorien für bessere Organisation
INDICATOR_CATEGORIES = {
    "💰 Einlagen & Sparverhalten": [
        "QSA.Q.N.DE.W0.S128.S1.N.L.LE.F6._Z._Z.XDC._T.S.V.N._T",
        "QSA.Q.N.DE.W0.S1M.S1.N.A.LE.F62._Z._Z.XDC._T.S.V.N._T",
        "QSA.Q.N.DE.W0.S128.S1.N.L.LE.F2._Z._Z.XDC._T.S.V.N._T",
        "ICO.A.DE.S128.L.W0.A.EUR",
        "QSA.Q.N.DE.W0.S1M.S1._Z.B.B8G._Z._Z._Z.XDC._T.S.V.N._T"
    ],
    "📊 Makroökonomie": [
        "QSA.Q.N.DE.W0.S1M.S1._Z.B.B6G._Z._Z._Z.XDC._T.S.V.N._T",
        "QSA.Q.N.DE.W0.S1M.S1._Z.B.B1GH._Z._Z._Z.XDC._T.S.V.N._T",
        "GFS.A.N.DE.W0.S13.S1._Z.B.B8G._Z._Z._Z.XDC._Z.S.V.GY._T",
        "GFS.A.N.DE.W0.S13.S1._Z.B.B9._Z._Z._Z.XDC._Z.S.V.GY._T"
    ],
    "👥 Arbeitsmarkt": [
        "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T",
        "LFSI.M.DE.S.EMPRT.TOTAL0.15_74.T",
        "LFSI.M.DE.S.PARTRT.TOTAL0.15_74.T"
    ],
    "💳 Zinsen & Geldpolitik": [
        "IRS.M.DE.L.L40.CI.0000.EUR.N.Z",
        "IRS.M.DE.L.L40.CI.0010.EUR.N.Z",
        "IRS.M.DE.L.L40.CI.0030.EUR.N.Z",
        "IRS.M.DE.L.L40.CI.0060.EUR.N.Z",
        "IRS.M.DE.L.L40.CI.0120.EUR.N.Z",
        "FM.B.DE.EUR.4F.BB.TOTRESNS.LEV"
    ],
    "📈 Preise & Inflation": [
        "ICP.M.DE.N.000000.4.ANR",
        "ICP.M.DE.N.010000.4.ANR",
        "ICP.M.DE.N.070000.4.ANR",
        "ICP.M.DE.N.040000.4.ANR"
    ],
    "🏭 Konjunktur & Vertrauen": [
        "ESI.M.DE.INDI.NACE2.CI",
        "ESI.M.DE.CONS.NACE2.CI",
        "ESI.M.DE.BUIL.NACE2.CI"
    ],
    "🌍 Externe Faktoren": [
        "EXR.M.USD.EUR.SP00.A",
        "YC.B.DE.EUR.4F.G_N_A.SV_C_YM.SR_1Y",
        "YC.B.DE.EUR.4F.G_N_A.SV_C_YM.SR_10Y"
    ]
}

# Variablen-Beschreibungen
VARIABLE_DESCRIPTIONS = {
    "Overnight-Einlagen Haushalte": "Tägliche Einlagen privater Haushalte bei deutschen Banken",
    "Termineinlagen Haushalte": "Befristete Einlagen privater Haushalte",
    "Bargeld und Einlagen Haushalte": "Gesamt-Liquidität der Haushalte",
    "Einlagenzinsen Haushalte": "Zinssätze für Haushaltseinlagen",
    "Verfügbares Einkommen": "Nach Steuern und Abgaben verfügbares Haushaltseinkommen",
    "BIP (Ausgabenansatz)": "Bruttoinlandsprodukt nach Ausgabenansatz",
    "Sparen der Haushalte": "Ersparnisse der privaten Haushalte",
    "Haushaltssaldo Staat": "Staatlicher Budgetsaldo als % des BIP",
    "Finanzierungssaldo Staat": "Staatlicher Finanzierungssaldo",
    "Arbeitslosenquote": "Anteil arbeitsloser Personen an der Erwerbsbevölkerung",
    "Beschäftigungsquote": "Anteil der Beschäftigten an der Bevölkerung",
    "Erwerbsquote": "Anteil der Erwerbspersonen an der Bevölkerung",
    "Zinssatz (Tagesgeld)": "Zinssatz für Tagesgeld-Einlagen",
    "Zinssatz (1 Monat)": "Zinssatz für 1-Monats-Einlagen",
    "Zinssatz (3 Monate)": "Zinssatz für 3-Monats-Einlagen", 
    "Zinssatz (6 Monate)": "Zinssatz für 6-Monats-Einlagen",
    "Zinssatz (1 Jahr)": "Zinssatz für 1-Jahres-Einlagen",
    "Bankreserven": "Reserven der Banken bei der Zentralbank",
    "Inflation (HVPI)": "Harmonisierter Verbraucherpreisindex (Jahresrate)",
    "Lebensmittel-Inflation": "Inflation für Lebensmittel und alkoholfreie Getränke",
    "Transport-Inflation": "Inflation im Transportsektor",
    "Wohnkosten-Inflation": "Inflation der Wohnkosten",
    "Vertrauensindikator": "Allgemeiner Wirtschaftsvertrauensindikator",
    "Verbrauchervertrauen": "Vertrauen der Verbraucher in die Wirtschaft",
    "Baugewerbe-Vertrauen": "Vertrauen im Baugewerbe",
    "USD/EUR Wechselkurs": "Wechselkurs US-Dollar zu Euro",
    "Staatsanleihe 1 Jahr": "Rendite deutscher 1-jähriger Staatsanleihen",
    "Staatsanleihe 10 Jahre": "Rendite deutscher 10-jähriger Staatsanleihen"
}

# Initialisierung Session State
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

@st.cache_data(ttl=3600)
def get_default_variables():
    """Standard-Variablenkonfiguration"""
    dep_var = ["QSA.Q.N.DE.W0.S128.S1.N.L.LE.F6._Z._Z.XDC._T.S.V.N._T"]
    det_vars = [
        "QSA.Q.N.DE.W0.S1M.S1._Z.B.B6G._Z._Z._Z.XDC._T.S.V.N._T",
        "GFS.A.N.DE.W0.S13.S1._Z.B.B8G._Z._Z._Z.XDC._Z.S.V.GY._T",
        "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T",
        "IRS.M.DE.L.L40.CI.0000.EUR.N.Z",
        "ICP.M.DE.N.000000.4.ANR"
    ]
    return dep_var, det_vars

def validate_variable_selection(dep_var, det_vars):
    """Validiert die Variablenauswahl"""
    issues = []
    
    if not dep_var or len(dep_var) != 1:
        issues.append("❌ Genau eine Zielvariable erforderlich")
    
    if len(det_vars) < 2:
        issues.append("❌ Mindestens 2 erklärende Variablen erforderlich")
    elif len(det_vars) > 10:
        issues.append("⚠️ Mehr als 10 erklärende Variablen können zu Overfitting führen")
    
    if dep_var and dep_var[0] in det_vars:
        issues.append("❌ Zielvariable darf nicht als erklärende Variable verwendet werden")
    
    all_vars = dep_var + det_vars
    for var in all_vars:
        if not var or len(var) < 10:
            issues.append(f"⚠️ Möglicherweise ungültiger ECB-Code: {var}")
    
    return issues

def get_variable_category(ecb_code):
    """Bestimmt die Kategorie einer Variable basierend auf ECB-Code"""
    for category, codes in INDICATOR_CATEGORIES.items():
        if ecb_code in codes:
            return category
    return "🔍 Unbekannt"

def get_variable_info_table(dep_var, det_vars, all_indicators):
    """Erstellt eine Informationstabelle für ausgewählte Variablen"""
    var_info = []
    
    if dep_var:
        var_info.append({
            'Typ': "🎯 Zielvariable",
            'Variable': all_indicators.get(dep_var[0], dep_var[0]),
            'ECB Code': dep_var[0],
            'Kategorie': get_variable_category(dep_var[0])
        })
    
    for var in det_vars:
        var_info.append({
            'Typ': "📊 Erklärende Variable",
            'Variable': all_indicators.get(var, var),
            'ECB Code': var,
            'Kategorie': get_variable_category(var)
        })
    
    return pd.DataFrame(var_info)

def create_variable_selection_summary(dep_var, det_vars, all_indicators):
    """Erstellt eine Zusammenfassung der Variablenauswahl"""
    summary = {
        'Zielvariable': all_indicators.get(dep_var[0], dep_var[0]) if dep_var else "Nicht ausgewählt",
        'Anzahl_erklaerende_Variablen': len(det_vars),
        'Erklaerende_Variablen': [all_indicators.get(var, var) for var in det_vars],
        'Kategorien_verwendet': list(set([get_variable_category(var) for var in det_vars])),
        'Gesamt_Variablen': len(dep_var) + len(det_vars)
    }
    return summary

@st.cache_data(ttl=3600)
def process_series(var, start='2001-01', end='2024-12'):
    """Einzelne Zeitreihe verarbeiten"""
    try:
        df = ecbdata.get_series(var, start=start, end=end)
        if df.empty:
            st.error(f"Keine Daten für Variable {var} gefunden")
            return pd.DataFrame()
            
        key = df['KEY'].iloc[0]
        dates = df['TIME_PERIOD'].astype(str)

        if dates.str.len().max() == 10:
            df['Datum'] = pd.to_datetime(df['TIME_PERIOD']).dt.to_period('M').dt.to_timestamp()
        elif dates.str.contains('Q').any():
            df['quarter'] = pd.PeriodIndex(df['TIME_PERIOD'], freq='Q')
            rows = []
            for _, row in df.iterrows():
                for i in range(3):
                    rows.append({
                        'Datum': (row['quarter'].start_time + pd.DateOffset(months=i)).replace(day=1),
                        'OBS_VALUE': row['OBS_VALUE']
                    })
            df = pd.DataFrame(rows)
        else:
            df['Datum'] = pd.to_datetime(df['TIME_PERIOD']).dt.to_period('M').dt.to_timestamp()

        return df.rename(columns={'OBS_VALUE': key})[['Datum', key]]
    except Exception as e:
        st.error(f"Fehler beim Laden der Variable {var}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def merge_data(all_vars):
    """Alle Zeitreihen zusammenführen"""
    try:
        dfs = []
        failed_vars = []
        
        for var in all_vars:
            df = process_series(var)
            if not df.empty:
                dfs.append(df)
            else:
                failed_vars.append(var)
        
        if failed_vars:
            st.warning(f"Folgende Variablen konnten nicht geladen werden: {failed_vars}")
        
        if not dfs:
            raise ValueError("Keine Daten konnten geladen werden")
            
        result = reduce(lambda left, right: pd.merge(left, right, on='Datum', how='outer'), dfs)
        return result, failed_vars
    except Exception as e:
        st.error(f"Fehler beim Zusammenführen der Daten: {str(e)}")
        return pd.DataFrame(), all_vars

def prepare_data(df, dep_var):
    """Daten für Modellierung vorbereiten"""
    df = df.set_index("Datum")
    valid_dates = df[dep_var[0]].dropna().index
    df_filtered = df.loc[valid_dates].copy()
    df_filtered = df_filtered.interpolate(method='time').ffill().bfill()
    df_filtered.reset_index(inplace=True)
    return df_filtered

def perform_stationarity_test(series, var_name):
    """Augmented Dickey-Fuller Test für Stationarität"""
    try:
        result = adfuller(series.dropna())
        return {
            'variable': var_name,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    except:
        return None

def calculate_model_metrics(y_true, y_pred):
    """Modellgüte-Metriken berechnen"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def run_gridsearch_enhanced(y, X, max_combinations=15, variable_mapping=None):
    """Verbesserte Grid Search"""
    results = []
    combinations_tested = 0
    
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_display = st.empty()
    
    model_configs = [
        {'max_p': 2, 'max_q': 2, 'max_P': 1, 'max_Q': 1},
        {'max_p': 3, 'max_q': 3, 'max_P': 1, 'max_Q': 1}
    ]
    
    for config in model_configs:
        for k in range(3, min(len(X.columns) + 1, 6)):
            for subset in combinations(X.columns, k):
                if combinations_tested >= max_combinations:
                    break
                    
                progress = combinations_tested / max_combinations
                progress_bar.progress(progress)
                
                readable_vars = [variable_mapping.get(var, var) for var in subset] if variable_mapping else list(subset)
                status_text.text(f"Teste Kombination {combinations_tested + 1}/{max_combinations}")
                metrics_display.text(f"Variablen: {', '.join(readable_vars[:3])}...")
                
                X_subset = X[list(subset)]
                
                mask_valid = (
                    ~y.isna() &
                    ~X_subset.isna().any(axis=1) &
                    ~np.isinf(X_subset.to_numpy()).any(axis=1)
                )
                y_clean = y[mask_valid]
                X_clean = X_subset.loc[mask_valid]

                if len(y_clean) < 24:
                    continue

                try:
                    model = auto_arima(
                        y_clean,
                        exogenous=X_clean,
                        start_p=0, max_p=config['max_p'],
                        start_q=0, max_q=config['max_q'],
                        d=None,
                        test='adf',
                        seasonal=True,
                        m=12,
                        start_P=0, max_P=config['max_P'],
                        start_Q=0, max_Q=config['max_Q'],
                        D=None,
                        seasonal_test='ocsb',
                        stepwise=True,
                        random=False,
                        n_fits=30,
                        scoring='aic',
                        error_action='ignore',
                        suppress_warnings=True,
                        random_state=42
                    )

                    fitted_values = model.fittedvalues()
                    metrics = calculate_model_metrics(y_clean, fitted_values)

                    results.append({
                        'exog_vars': subset,
                        'exog_vars_readable': readable_vars,
                        'aic': model.aic(),
                        'bic': model.bic(),
                        'order': model.order,
                        'seasonal_order': model.seasonal_order,
                        'model': model,
                        'config': config,
                        'sample_size': len(y_clean),
                        **metrics
                    })

                except Exception as e:
                    st.warning(f"Fehler bei Kombination {readable_vars}: {str(e)}")
                
                combinations_tested += 1
                
            if combinations_tested >= max_combinations:
                break
        if combinations_tested >= max_combinations:
            break
    
    progress_bar.progress(1.0)
    status_text.text("Grid Search abgeschlossen!")
    metrics_display.empty()
    
    return results

def create_interactive_forecast_plot(y, forecast_index, forecast, conf_int, title):
    """Interaktiver Prognose-Plot mit Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y.index,
        y=y.values,
        mode='lines',
        name='Beobachtete Werte',
        line=dict(color='blue', width=2),
        hovertemplate='Datum: %{x}<br>Wert: %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast,
        mode='lines',
        name='Prognose',
        line=dict(color='orange', width=2, dash='dash'),
        hovertemplate='Datum: %{x}<br>Prognose: %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=list(forecast_index) + list(forecast_index[::-1]),
        y=list(conf_int[:, 1]) + list(conf_int[:, 0][::-1]),
        fill='toself',
        fillcolor='rgba(255,165,0,0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Konfidenzintervall',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Datum",
        yaxis_title="Werte",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def export_results_to_excel(forecast_df, model_summary, metrics):
    """Ergebnisse nach Excel exportieren"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        forecast_df.to_excel(writer, sheet_name='Prognose', index=False)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name='Metriken', index=False)
        summary_df = pd.DataFrame({'Modell_Zusammenfassung': [str(model_summary)]})
        summary_df.to_excel(writer, sheet_name='Modell_Details', index=False)
    
    return output.getvalue()

def main():
    # Header
    st.markdown('<div class="main-header">🏦 SARIMAX ECB Prognose Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Erweiterte Zeitreihenanalyse für ECB-Indikatoren mit individueller Variablenauswahl**")
    
    # Info Box
    st.markdown("""
    <div class="info-box">
    <strong>ℹ️ Über diese Anwendung:</strong><br>
    Dieses Dashboard verwendet SARIMAX-Modelle zur Prognose von ECB-Zeitreihen mit individueller Variablenauswahl.
    Die Daten stammen direkt von der Europäischen Zentralbank (ECB) und umfassen über 28 verschiedene Indikatoren.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar für Variablenauswahl
    st.sidebar.header("⚙️ Konfiguration")
    st.sidebar.markdown("---")
    
    selection_mode = st.sidebar.radio(
        "Auswahlmodus:",
        ["🎯 Standard (Vordefiniert)", "🔧 Individuell", "📚 Nach Kategorien"],
        help="Wählen Sie, wie Sie die Variablen auswählen möchten"
    )
    
    if selection_mode == "🎯 Standard (Vordefiniert)":
        dep_var, det_vars = get_default_variables()
        st.sidebar.success("✅ Standard-Variablen geladen")
        
    elif selection_mode == "🔧 Individuell":
        st.sidebar.markdown("**Zielvariable auswählen:**")
        
        target_options = {code: name for code, name in ALL_AVAILABLE_INDICATORS.items()}
        selected_target_name = st.sidebar.selectbox(
            "Abhängige Variable:",
            options=list(target_options.values()),
            index=list(target_options.values()).index("Overnight-Einlagen Haushalte"),
            help="Die Variable, die Sie prognostizieren möchten"
        )
        
        selected_target_code = None
        for code, name in target_options.items():
            if name == selected_target_name:
                selected_target_code = code
                break
        
        dep_var = [selected_target_code]
        
        st.sidebar.markdown("**Erklärende Variablen auswählen:**")
        
        explanatory_options = {code: name for code, name in ALL_AVAILABLE_INDICATORS.items() 
                             if code != selected_target_code}
        
        selected_explanatory_names = st.sidebar.multiselect(
            "Unabhängige Variablen:",
            options=list(explanatory_options.values()),
            default=["Verfügbares Einkommen", "Arbeitslosenquote", "Zinssatz (Tagesgeld)", "Inflation (HVPI)"],
            help="Variablen, die zur Prognose verwendet werden sollen"
        )
        
        det_vars = []
        for name in selected_explanatory_names:
            for code, var_name in explanatory_options.items():
                if var_name == name:
                    det_vars.append(code)
                    break
        
        if len(det_vars) < 2:
            st.sidebar.warning("⚠️ Mindestens 2 erklärende Variablen erforderlich")
        else:
            st.sidebar.success(f"✅ {len(det_vars)} erklärende Variablen ausgewählt")
    
    else:  # Kategorie-basierte Auswahl
        st.sidebar.markdown("**Zielvariable auswählen:**")
        
        target_category = st.sidebar.selectbox(
            "Kategorie für Zielvariable:",
            options=list(INDICATOR_CATEGORIES.keys()),
            index=0
        )
        
        target_indicators_in_category = INDICATOR_CATEGORIES[target_category]
        target_options_in_category = {code: ALL_AVAILABLE_INDICATORS[code] 
                                    for code in target_indicators_in_category 
                                    if code in ALL_AVAILABLE_INDICATORS}
        
        selected_target_name = st.sidebar.selectbox(
            "Abhängige Variable:",
            options=list(target_options_in_category.values()),
            help="Die Variable, die Sie prognostizieren möchten"
        )
        
        selected_target_code = None
        for code, name in target_options_in_category.items():
            if name == selected_target_name:
                selected_target_code = code
                break
        
        dep_var = [selected_target_code]
        
        st.sidebar.markdown("**Erklärende Variablen nach Kategorien:**")
        
        selected_categories = st.sidebar.multiselect(
            "Kategorien auswählen:",
            options=[cat for cat in INDICATOR_CATEGORIES.keys() if cat != target_category],
            default=["📊 Makroökonomie", "👥 Arbeitsmarkt", "💳 Zinsen & Geldpolitik", "📈 Preise & Inflation"],
            help="Wählen Sie Kategorien für erklärende Variablen"
        )
        
        det_vars = []
        for category in selected_categories:
            category_vars = INDICATOR_CATEGORIES[category]
            det_vars.extend([var for var in category_vars if var != selected_target_code])
        
        det_vars = list(set(det_vars))
        
        if len(det_vars) < 2:
            st.sidebar.warning("⚠️ Mindestens 2 erklärende Variablen erforderlich")
        else:
            st.sidebar.success(f"✅ {len(det_vars)} erklärende Variablen aus {len(selected_categories)} Kategorien")
        
        with st.sidebar.expander("🔍 Ausgewählte Variablen anzeigen"):
            st.write("**Zielvariable:**")
            st.write(f"• {ALL_AVAILABLE_INDICATORS.get(selected_target_code, selected_target_code)}")
            st.write("**Erklärende Variablen:**")
            for var in det_vars:
                st.write(f"• {ALL_AVAILABLE_INDICATORS.get(var, var)}")
    
    # Zusätzliche Optionen in Sidebar
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📊 Dateneinstellungen")
    start_date = st.sidebar.date_input("Startdatum", pd.to_datetime("2001-01-01"))
    end_date = st.sidebar.date_input("Enddatum", pd.to_datetime("2024-12-31"))
    
    st.sidebar.subheader("🔍 Modelleinstellungen")
    forecast_steps = st.sidebar.slider("Prognosehorizont (Monate)", 1, 48, 12)
    max_combinations = st.sidebar.slider("Max. Variablenkombinationen", 5, 25, 15)
    confidence_level = st.sidebar.selectbox("Konfidenzintervall", [90, 95, 99], index=1)
    
    with st.sidebar.expander("🔧 Erweiterte Optionen"):
        seasonal_period = st.selectbox("Saisonalität", [12, 4], index=0, help="12 für monatlich, 4 für quartalsweise")
        use_log_transform = st.checkbox("Log-Transformation anwenden", help="Kann bei exponentiellen Trends helfen")
        min_sample_size = st.slider("Minimale Stichprobengröße", 12, 60, 24)
        
        st.markdown("**🔬 Experten-Modus:**")
        if st.checkbox("Manuelle ECB-Code Eingabe"):
            custom_target = st.text_input("Zielvariable (ECB-Code):", value=dep_var[0] if dep_var else "")
            custom_explanatory = st.text_area(
                "Erklärende Variablen (ECB-Codes, eine pro Zeile):",
                value="\n".join(det_vars) if det_vars else ""
            )
            
            if custom_target and custom_explanatory:
                dep_var = [custom_target.strip()]
                det_vars = [var.strip() for var in custom_explanatory.split('\n') if var.strip()]
                st.success(f"✅ Benutzerdefinierte Codes: 1 Ziel, {len(det_vars)} erklärende Variablen")
    
    # Variable Mapping für aktuell ausgewählte Variablen aktualisieren
    current_variables = dep_var + det_vars
    VARIABLE_MAPPING = {var: ALL_AVAILABLE_INDICATORS.get(var, var) for var in current_variables}
    
    # Variablen-Validierung
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Aktuelle Variablenauswahl")
        
        validation_issues = validate_variable_selection(dep_var, det_vars)
        
        if validation_issues:
            for issue in validation_issues:
                if issue.startswith("❌"):
                    st.error(issue)
                else:
                    st.warning(issue)
        else:
            st.success("✅ Variablenauswahl ist valid")
        
        if dep_var and det_vars:
            summary = create_variable_selection_summary(dep_var, det_vars, ALL_AVAILABLE_INDICATORS)
            
            st.markdown(f"""
            **🎯 Zielvariable:** {summary['Zielvariable']}  
            **📊 Erklärende Variablen:** {summary['Anzahl_erklaerende_Variablen']}  
            **🏷️ Verwendete Kategorien:** {', '.join(summary['Kategorien_verwendet'])}
            """)
    
    with col2:
        st.subheader("📊 Statistiken")
        
        if dep_var and det_vars:
            st.metric("🎯 Zielvariablen", len(dep_var))
            st.metric("📈 Erklärende Variablen", len(det_vars))
            st.metric("🔢 Gesamt Variablen", len(dep_var) + len(det_vars))
            
            categories = [get_variable_category(var) for var in det_vars]
            unique_categories = len(set(categories))
            st.metric("🏷️ Kategorien", unique_categories)
    
    # Detaillierte Variablentabelle
    with st.expander("🔍 Detaillierte Variableninformationen", expanded=False):
        if dep_var and det_vars:
            var_info_df = get_variable_info_table(dep_var, det_vars, ALL_AVAILABLE_INDICATORS)
            st.dataframe(var_info_df, use_container_width=True)
            
            var_config = {
                'timestamp': datetime.now().isoformat(),
                'target_variable': {
                    'code': dep_var[0],
                    'name': ALL_AVAILABLE_INDICATORS.get(dep_var[0], dep_var[0])
                },
                'explanatory_variables': [
                    {
                        'code': var,
                        'name': ALL_AVAILABLE_INDICATORS.get(var, var),
                        'category': get_variable_category(var)
                    } for var in det_vars
                ]
            }
            
            config_json = json.dumps(var_config, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Variablenkonfiguration exportieren",
                data=config_json,
                file_name=f"variable_config_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
        else:
            st.info("Wählen Sie Variablen in der Sidebar aus, um Details anzuzeigen.")
    
    # Quick Actions
    st.markdown("---")
    st.subheader("⚡ Schnellkonfigurationen")
    
    quick_configs = {
        "🏠 Haushalts-Fokus": {
            "target": "QSA.Q.N.DE.W0.S128.S1.N.L.LE.F6._Z._Z.XDC._T.S.V.N._T",
            "explanatory": [
                "QSA.Q.N.DE.W0.S1M.S1._Z.B.B6G._Z._Z._Z.XDC._T.S.V.N._T",
                "QSA.Q.N.DE.W0.S1M.S1._Z.B.B8G._Z._Z._Z.XDC._T.S.V.N._T",
                "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T",
                "ICP.M.DE.N.000000.4.ANR"
            ]
        },
        "💰 Geldpolitik-Fokus": {
            "target": "QSA.Q.N.DE.W0.S128.S1.N.L.LE.F6._Z._Z.XDC._T.S.V.N._T",
            "explanatory": [
                "IRS.M.DE.L.L40.CI.0000.EUR.N.Z",
                "IRS.M.DE.L.L40.CI.0030.EUR.N.Z",
                "IRS.M.DE.L.L40.CI.0120.EUR.N.Z",
                "ICP.M.DE.N.000000.4.ANR"
            ]
        },
        "📊 Makroökonomie-Fokus": {
            "target": "QSA.Q.N.DE.W0.S128.S1.N.L.LE.F6._Z._Z.XDC._T.S.V.N._T",
            "explanatory": [
                "QSA.Q.N.DE.W0.S1M.S1._Z.B.B1GH._Z._Z._Z.XDC._T.S.V.N._T",
                "QSA.Q.N.DE.W0.S1M.S1._Z.B.B6G._Z._Z._Z.XDC._T.S.V.N._T",
                "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T",
                "GFS.A.N.DE.W0.S13.S1._Z.B.B8G._Z._Z._Z.XDC._Z.S.V.GY._T"
            ]
        }
    }
    
    config_cols = st.columns(len(quick_configs))
    
    for i, (config_name, config_data) in enumerate(quick_configs.items()):
        with config_cols[i]:
            if st.button(config_name, key=f"quick_config_{i}"):
                dep_var = [config_data["target"]]
                det_vars = config_data["explanatory"]
                VARIABLE_MAPPING = {var: ALL_AVAILABLE_INDICATORS.get(var, var) for var in dep_var + det_vars}
                st.success(f"✅ {config_name} Konfiguration geladen!")
                st.rerun()
    
    # Abschließende Validierung
    if not validation_issues:
        all_vars = dep_var + det_vars
        VARIABLE_MAPPING = {var: ALL_AVAILABLE_INDICATORS.get(var, var) for var in all_vars}
    else:
        st.error("⚠️ Bitte beheben Sie die Validierungsfehler bevor Sie fortfahren.")
        st.stop()
    
    # Tabs für verschiedene Funktionen
    tab1, tab2, tab3 = st.tabs([
        "📈 Datenexploration", 
        "🔍 Modellauswahl", 
        "📊 Prognose"
    ])
    
    with tab1:
        st.header("📈 Datenexploration")
        
        if st.button("🔄 Daten laden/aktualisieren", type="primary"):
            with st.spinner("Lade aktuelle ECB-Daten..."):
                df_merged, failed_vars = merge_data(all_vars)
                
                if not df_merged.empty:
                    st.session_state.df_merged = df_merged
                    st.session_state.failed_vars = failed_vars
                    st.session_state.data_loaded = True
                    st.success("✅ Daten erfolgreich geladen!")
                else:
                    st.error("❌ Fehler beim Laden der Daten")
                    return
        
        if st.session_state.data_loaded:
            df_merged = st.session_state.df_merged
            df_filtered = prepare_data(df_merged, dep_var)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📅 Zeitraum", 
                         f"{df_filtered['Datum'].min().strftime('%Y-%m')} bis {df_filtered['Datum'].max().strftime('%Y-%m')}")
            
            with col2:
                st.metric("📊 Beobachtungen", f"{len(df_filtered):,}")
            
            with col3:
                missing_pct = (df_filtered.isnull().sum().sum() / (len(df_filtered) * len(df_filtered.columns))) * 100
                st.metric("❓ Fehlende Werte", f"{missing_pct:.1f}%")
            
            with col4:
                st.metric("🔢 Variablen", len(all_vars))
            
            # Variablenübersicht
            st.subheader("📋 Variablenübersicht")
            var_info = []
            for var in all_vars:
                readable_name = VARIABLE_MAPPING.get(var, var)
                description = VARIABLE_DESCRIPTIONS.get(readable_name, "Keine Beschreibung verfügbar")
                var_type = "🎯 Zielvariable" if var in dep_var else "📊 Erklärende Variable"
                
                var_info.append({
                    'Typ': var_type,
                    'Variable': readable_name,
                    'Beschreibung': description,
                    'ECB Code': var
                })
            
            st.dataframe(pd.DataFrame(var_info), use_container_width=True)
            
            # Zeitreihenvisualisierung
            st.subheader("📈 Interaktive Zeitreihenanalyse")
            
            selected_vars = st.multiselect(
                "Variablen für Visualisierung auswählen:",
                options=[VARIABLE_MAPPING.get(var, var) for var in all_vars],
                default=[VARIABLE_MAPPING.get(dep_var[0], dep_var[0])]
            )
            
            if selected_vars:
                df_plot = df_filtered.set_index('Datum')
                
                fig = go.Figure()
                
                for var_name in selected_vars:
                    ecb_code = None
                    for code, name in VARIABLE_MAPPING.items():
                        if name == var_name:
                            ecb_code = code
                            break
                    
                    if ecb_code and ecb_code in df_plot.columns:
                        fig.add_trace(go.Scatter(
                            x=df_plot.index,
                            y=df_plot[ecb_code],
                            mode='lines',
                            name=var_name,
                            hovertemplate=f'{var_name}: %{{y:,.2f}}<br>Datum: %{{x}}<extra></extra>'
                        ))
                
                fig.update_layout(
                    title="Zeitreihenverläufe",
                    xaxis_title="Datum",
                    yaxis_title="Werte",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Stationaritätstests
            st.subheader("🧪 Stationaritätstests (Augmented Dickey-Fuller)")
            
            stationarity_results = []
            df_plot = df_filtered.set_index('Datum')
            
            for var in all_vars:
                if var in df_plot.columns:
                    readable_name = VARIABLE_MAPPING.get(var, var)
                    result = perform_stationarity_test(df_plot[var], readable_name)
                    if result:
                        stationarity_results.append(result)
            
            if stationarity_results:
                stat_df = pd.DataFrame(stationarity_results)
                stat_df['Stationarität'] = stat_df['is_stationary'].apply(lambda x: "✅ Stationär" if x else "❌ Nicht stationär")
                stat_df['p-Wert'] = stat_df['p_value'].apply(lambda x: f"{x:.4f}")
                
                display_cols = ['variable', 'Stationarität', 'p-Wert', 'adf_statistic']
                st.dataframe(stat_df[display_cols].rename(columns={
                    'variable': 'Variable',
                    'adf_statistic': 'ADF-Statistik'
                }), use_container_width=True)
            
            # Korrelationsanalyse
            st.subheader("🔗 Korrelationsanalyse")
            
            corr_matrix = df_plot.corr()
            
            readable_corr = corr_matrix.copy()
            readable_corr.index = [VARIABLE_MAPPING.get(var, var) for var in readable_corr.index]
            readable_corr.columns = [VARIABLE_MAPPING.get(var, var) for var in readable_corr.columns]
            
            fig = px.imshow(
                readable_corr,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Korrelationsmatrix der Variablen"
            )
            fig.update_layout(height=600)
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Klicken Sie auf 'Daten laden/aktualisieren', um zu beginnen.")
    
    with tab2:
        st.header("🔍 Modellauswahl und Grid Search")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Bitte laden Sie zuerst die Daten im Tab 'Datenexploration'.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Grid Search Einstellungen")
            st.write(f"**Maximale Kombinationen:** {max_combinations}")
            st.write(f"**Minimale Stichprobengröße:** {min_sample_size}")
            st.write(f"**Saisonperiode:** {seasonal_period}")
        
        with col2:
            st.subheader("📊 Zu testende Variablen")
            for var in det_vars:
                readable_name = VARIABLE_MAPPING.get(var, var)
                st.write(f"• {readable_name}")
        
        if st.button("🚀 Grid Search starten", type="primary"):
            with st.spinner("Führe Grid Search durch..."):
                try:
                    df_merged = st.session_state.df_merged
                    df_filtered = prepare_data(df_merged, dep_var)
                    df_filtered = df_filtered.set_index("Datum")
                    
                    y = df_filtered[dep_var[0]]
                    X = df_filtered[det_vars]
                    
                    if use_log_transform and (y > 0).all():
                        y = np.log(y)
                        st.info("📊 Log-Transformation auf Zielvariable angewendet")
                    
                    results = run_gridsearch_enhanced(y, X, max_combinations, VARIABLE_MAPPING)
                    
                    if results:
                        results_df = pd.DataFrame(results)
                        results_df_sorted = results_df.sort_values(by='aic').reset_index(drop=True)
                        
                        st.success("✅ Grid Search erfolgreich abgeschlossen!")
                        
                        st.session_state.search_results = results_df_sorted
                        st.session_state.y_data = y
                        st.session_state.X_data = X
                        st.session_state.analysis_complete = True
                        
                        st.subheader("🏆 Top 10 Modelle (sortiert nach AIC)")
                        
                        display_cols = ['exog_vars_readable', 'aic', 'bic', 'order', 'seasonal_order', 'MAPE', 'sample_size']
                        display_df = results_df_sorted[display_cols].head(10).copy()
                        display_df.columns = ['Exogene Variablen', 'AIC', 'BIC', 'ARIMA Order', 'Saisonal Order', 'MAPE (%)', 'Stichprobe']
                        display_df['AIC'] = display_df['AIC'].round(2)
                        display_df['BIC'] = display_df['BIC'].round(2)
                        display_df['MAPE (%)'] = display_df['MAPE (%)'].round(2)
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        best_model = results_df_sorted.iloc[0]
                        
                        st.markdown("""
                        <div class="success-box">
                        <h3>🎯 Bestes Modell Details</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🏅 AIC", f"{best_model['aic']:.2f}")
                            st.metric("📊 BIC", f"{best_model['bic']:.2f}")
                        
                        with col2:
                            st.metric("📈 MAPE", f"{best_model['MAPE']:.2f}%")
                            st.metric("📏 RMSE", f"{best_model['RMSE']:.0f}")
                        
                        with col3:
                            st.metric("📋 Variablen", len(best_model['exog_vars']))
                            st.metric("🔢 Stichprobe", best_model['sample_size'])
                        
                        st.write("**📊 Verwendete exogene Variablen:**")
                        for var in best_model['exog_vars_readable']:
                            description = VARIABLE_DESCRIPTIONS.get(var, "Keine Beschreibung")
                            st.write(f"• **{var}**: {description}")
                        
                        st.write(f"**🔧 ARIMA Konfiguration:** {best_model['order']}")
                        st.write(f"**🌊 Saisonale Konfiguration:** {best_model['seasonal_order']}")
                        
                    else:
                        st.error("❌ Keine gültigen Modelle gefunden!")
                        
                except Exception as e:
                    st.error(f"❌ Fehler bei Grid Search: {str(e)}")
        
        if st.session_state.analysis_complete:
            st.markdown("---")
            st.subheader("💾 Gespeicherte Analyseergebnisse")
            st.info("Grid Search bereits durchgeführt. Ergebnisse sind verfügbar für Prognose.")
    
    with tab3:
        st.header("📊 Prognose")
        
        if not st.session_state.analysis_complete:
            st.warning("⚠️ Bitte führen Sie zuerst eine Modellsuche durch.")
            return
        
        try:
            results_df_sorted = st.session_state.search_results
            y = st.session_state.y_data
            X = st.session_state.X_data
            
            st.subheader("🎯 Modellauswahl für Prognose")
            
            model_options = []
            for i, row in results_df_sorted.head(5).iterrows():
                model_desc = f"Modell {i+1} (AIC: {row['aic']:.2f}, MAPE: {row['MAPE']:.2f}%)"
                model_options.append(model_desc)
            
            selected_model_idx = st.selectbox(
                "Modell für Prognose auswählen:",
                range(len(model_options)),
                format_func=lambda x: model_options[x]
            )
            
            selected_model_row = results_df_sorted.iloc[selected_model_idx]
            best_model = selected_model_row['model']
            best_exog_vars = list(selected_model_row['exog_vars'])
            
            with st.expander("🔍 Details des ausgewählten Modells"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**AIC:** {selected_model_row['aic']:.2f}")
                    st.write(f"**BIC:** {selected_model_row['bic']:.2f}")
                    st.write(f"**MAPE:** {selected_model_row['MAPE']:.2f}%")
                    st.write(f"**RMSE:** {selected_model_row['RMSE']:.0f}")
                
                with col2:
                    st.write(f"**ARIMA Order:** {selected_model_row['order']}")
                    st.write(f"**Saisonal Order:** {selected_model_row['seasonal_order']}")
                    st.write(f"**Stichprobengröße:** {selected_model_row['sample_size']}")
            
            # Einfache Trend-Extrapolation
            st.subheader("📈 Prognose erstellen")
            
            if st.button("🚀 Prognose erstellen", type="primary"):
                with st.spinner("Erstelle Prognose..."):
                    forecast_index = pd.date_range(
                        start=y.index[-1] + pd.DateOffset(months=1),
                        periods=forecast_steps,
                        freq='MS'
                    )
                    
                    X_future = pd.DataFrame(index=forecast_index, columns=X[best_exog_vars].columns)
                    
                    for col in X[best_exog_vars].columns:
                        if not X[col].isnull().any():
                            model_lr = LinearRegression()
                            time_index = np.arange(len(X)).reshape(-1, 1)
                            model_lr.fit(time_index, X[col].values)
                            future_index = np.arange(len(X), len(X) + forecast_steps).reshape(-1, 1)
                            X_future[col] = model_lr.predict(future_index)
                    
                    forecast, conf_int = best_model.predict(
                        n_periods=forecast_steps,
                        exogenous=X_future,
                        return_conf_int=True,
                        alpha=1-confidence_level/100
                    )
                    
                    # Rück-Transformation falls Log-Transform verwendet wurde
                    if use_log_transform:
                        forecast = np.exp(forecast)
                        conf_int = np.exp(conf_int)
                        y_plot = np.exp(y)
                    else:
                        y_plot = y
                    
                    # Interaktive Prognose-Visualisierung
                    st.subheader("📈 Prognoseergebnisse")
                    
                    title = "Prognose der ausgewählten Zielvariable"
                    fig = create_interactive_forecast_plot(y_plot, forecast_index, forecast, conf_int, title)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prognosemetriken
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_forecast = np.mean(forecast)
                        st.metric("📊 Ø Prognose", f"{avg_forecast:,.0f}")
                    
                    with col2:
                        last_actual = y_plot.iloc[-1]
                        change_pct = ((avg_forecast - last_actual) / last_actual) * 100
                        st.metric("📈 Veränderung", f"{change_pct:+.1f}%")
                    
                    with col3:
                        forecast_std = np.std(forecast)
                        st.metric("📏 Volatilität", f"{forecast_std:,.0f}")
                    
                    with col4:
                        ci_width = np.mean(conf_int[:, 1] - conf_int[:, 0])
                        st.metric(f"🎯 Ø {confidence_level}% KI", f"±{ci_width:,.0f}")
                    
                    # Prognosetabelle
                    st.subheader("📋 Detaillierte Prognosewerte")
                    
                    forecast_df = pd.DataFrame({
                        'Datum': forecast_index,
                        'Prognose': forecast,
                        f'Unteres {confidence_level}% KI': conf_int[:, 0],
                        f'Oberes {confidence_level}% KI': conf_int[:, 1],
                        'KI Breite': conf_int[:, 1] - conf_int[:, 0]
                    })
                    
                    # Formatierung
                    for col in ['Prognose', f'Unteres {confidence_level}% KI', f'Oberes {confidence_level}% KI', 'KI Breite']:
                        forecast_df[col] = forecast_df[col].apply(lambda x: f"{x:,.0f}")
                    
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    # Speichere Prognose für Export
                    st.session_state.forecast_results = {
                        'forecast_df': forecast_df,
                        'model_summary': str(best_model.summary()),
                        'model_metrics': {
                            'AIC': selected_model_row['aic'],
                            'BIC': selected_model_row['bic'],
                            'MAPE': selected_model_row['MAPE'],
                            'RMSE': selected_model_row['RMSE']
                        },
                        'scenario_type': 'Trend-Extrapolation'
                    }
                    
                    # CSV-Export für Prognose
                    csv_forecast = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Prognose als CSV herunterladen",
                        data=csv_forecast,
                        file_name=f"prognose_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Fehler bei Prognose: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>🏦 <strong>SARIMAX ECB Prognose Dashboard v2.1</strong></p>
    <p>Entwickelt für die flexible Analyse von ECB-Zeitreihen mit individueller Variablenauswahl</p>
    <p>📊 Letzte Aktualisierung: {}</p>
</div>
""".format(datetime.now().strftime('%d.%m.%Y %H:%M')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()