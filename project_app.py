import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import os
from datetime import datetime, timedelta

# ML imports
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Page configuration with Mario theme
st.set_page_config(
    page_title="Super Finance Bros - Data & Insights",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inline Mario theme CSS for robust loading on all platforms
MARIO_THEME_CSS = '''
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

/* Main Theme Variables */
:root {
    --main-color: var(--mario-red); /* Default to Mario */
    --mario-red: #E52521;
    --luigi-green: #4CAF50;
    --peach-pink: #FF69B4;
    --yoshi-dark-green: #228B22;
    --mario-yellow: #F8D210;
    --mario-blue: #0066CC;
    --bg-color: #fffbe6;
    --text-color: #262730;
    --border-color: #8B4513;
    --panel-color: #FFE4B5;
}

body {
    font-family: Arial, sans-serif;
    background-color: var(--bg-color);
    color: var(--text-color);
}

.mario-header {
    color: var(--main-color);
    font-family: 'Press Start 2P', cursive;
    text-shadow: 3px 3px 0px var(--border-color);
    font-size: 2rem;
    margin-bottom: 2rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.mario-section {
    color: var(--main-color);
    font-family: 'Press Start 2P', cursive;
    text-shadow: 1px 1px 0px var(--border-color);
    font-size: 1.3rem;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.main .block-container {
    background-color: var(--panel-color);
    border: 4px solid var(--border-color);
    border-radius: 15px;
    padding: 2.5rem;
    margin: 1.5rem;
    box-shadow: 5px 5px 0px var(--border-color);
}

.stButton>button {
    background-color: var(--main-color);
    color: white;
    border-radius: 20px;
    padding: 10px 20px;
    border: 3px solid var(--border-color);
    font-family: 'Press Start 2P', cursive;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 3px 3px 0px var(--border-color);
    transition: all 0.2s;
}

.stButton>button:hover {
    transform: translate(2px, 2px);
    box-shadow: 1px 1px 0px var(--border-color);
}

.stSelectbox, .stNumberInput, .stTextInput, .stFileUploader {
    background-color: var(--panel-color);
    color: var(--text-color);
    border: 2px solid var(--border-color);
    border-radius: 10px;
    font-family: 'Press Start 2P', cursive;
    box-shadow: 3px 3px 0px var(--border-color);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
}

.stTabs [data-baseweb="tab"] {
    background-color: var(--panel-color);
    border: 2px solid var(--border-color);
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-family: 'Press Start 2P', cursive;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stTabs [aria-selected="true"] {
    background-color: var(--main-color) !important;
    color: white !important;
}

.stDataFrame {
    background-color: var(--panel-color);
    border: 3px solid var(--border-color);
    border-radius: 15px;
    padding: 1rem;
}

.css-1d391kg {
    background-color: var(--panel-color) !important;
    border-right: 4px solid var(--border-color);
}

.stProgress > div > div > div {
    background-color: var(--main-color);
    border-radius: 10px;
}

.card-container {
    background-color: var(--panel-color);
    color: var(--text-color);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(230,57,70,0.10);
    border: 1.5px solid var(--main-color);
    margin-bottom: 20px;
}

.character-card {
    background-color: rgba(255, 251, 230, 0.8);
    border: 3px solid var(--border-color);
    border-radius: 15px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 3px 3px 0px var(--border-color);
    transition: transform 0.2s;
}

.character-card:hover {
    transform: scale(1.02);
}

.track-selector {
    background-color: rgba(229, 37, 33, 0.1);
    padding: 20px;
    border-radius: 15px;
    margin: 10px 0;
    border: 3px solid var(--border-color);
    box-shadow: 3px 3px 0px var(--border-color);
}
'''

# Inject the CSS at the top of the app
st.markdown(f"<style>{MARIO_THEME_CSS}</style>", unsafe_allow_html=True)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = []
if 'current_viz' not in st.session_state:
    st.session_state.current_viz = None
if 'budget_limits' not in st.session_state:
    st.session_state.budget_limits = {}
if 'savings_goal' not in st.session_state:
    st.session_state.savings_goal = 0
if 'game_score' not in st.session_state:
    st.session_state.game_score = 0
if 'selected_character' not in st.session_state:
    st.session_state.selected_character = 'Mario'

# Character configuration
CHARACTERS = {
    'Mario': {
        'color': '#E52521',
        'icon': '👨‍🔧',
        'description': 'The classic racer! Balanced stats for steady financial growth.',
        'image': 'https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Mario/3D/mario_3d.png',
        'fallback': '👨‍🔧'
    },
    'Luigi': {
        'color': '#4CAF50',
        'icon': '👨‍🌾',
        'description': 'The careful saver! Great at managing expenses and building wealth.',
        'image': 'https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Luigi/3D/luigi_3d.png',
        'fallback': '👨‍🌾'
    },
    'Peach': {
        'color': '#FF69B4',
        'icon': '👸',
        'description': 'The strategic investor! Excels at making smart financial decisions.',
        'image': 'https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Peach/3D/peach_3d.png',
        'fallback': '👸'
    },
    'Yoshi': {
        'color': '#228B22',
        'icon': '🦖',
        'description': 'The consistent performer! Great at maintaining steady financial habits.',
        'image': 'https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Yoshi/3D/yoshi_3d.png',
        'fallback': '🦖'
    }
}

# After character selection logic, inject dynamic CSS for --main-color
character_main_colors = {
    "Mario": "#E52521",         # Red
    "Luigi": "#4CAF50",         # Green
    "Peach": "#FF69B4",         # Pink
    "Yoshi": "#228B22",         # Dark Green
}
selected_color = character_main_colors.get(st.session_state.selected_character, "#E52521")

st.markdown(
    f"""
    <style>
    :root {{
        --main-color: {selected_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def load_data(uploaded_file):
    """
    Load data from an uploaded file into a pandas DataFrame.
    
    Args:
        uploaded_file: The uploaded file object from Streamlit's file_uploader
        
    Returns:
        tuple: (DataFrame, file_name) or (None, None) if loading fails
    """
    file_name = uploaded_file.name
    
    try:
        # Determine file type by extension
        file_extension = os.path.splitext(file_name)[1].lower()
        
        # Read into a bytes buffer first
        bytes_data = uploaded_file.getvalue()
        
        if file_extension in ['.csv']:
            # Try different encodings
            for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(io.BytesIO(bytes_data), encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    raise Exception(f"Error reading CSV file: {str(e)}")
            else:
                raise Exception("Could not decode the CSV file with any of the attempted encodings.")
                
        elif file_extension in ['.xlsx', '.xls']:
            try:
                df = pd.read_excel(io.BytesIO(bytes_data))
            except Exception as e:
                raise Exception(f"Error reading Excel file: {str(e)}")
        else:
            raise Exception(f"Unsupported file format: {file_extension}")
        
        # Basic data cleaning
        # Convert column names to strings and strip whitespace
        df.columns = [str(col).strip() for col in df.columns]
        
        # Replace empty strings with NaN
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        
        return df, file_name
    
    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")

def get_column_types(dataframe):
    """
    Determine column types of a pandas DataFrame.
    
    Args:
        dataframe: Pandas DataFrame
        
    Returns:
        dict: Mapping of column names to simplified type names
    """
    column_types = {}
    
    for column in dataframe.columns:
        dtype = dataframe[column].dtype
        
        if pd.api.types.is_numeric_dtype(dtype):
            if pd.api.types.is_integer_dtype(dtype):
                column_types[column] = "Integer"
            else:
                column_types[column] = "Decimal"
        elif pd.api.types.is_datetime64_dtype(dtype):
            column_types[column] = "Date/Time"
        elif pd.api.types.is_bool_dtype(dtype):
            column_types[column] = "Boolean"
        else:
            # Check if it could be a date
            try:
                if dataframe[column].dropna().iloc[0] and pd.to_datetime(dataframe[column], errors='coerce').notna().all():
                    column_types[column] = "Date/Time"
                else:
                    column_types[column] = "Text"
            except (IndexError, ValueError):
                column_types[column] = "Text"
    
    return column_types

def get_column_summary(dataframe):
    """
    Get a summary of each column in the dataframe.
    
    Args:
        dataframe: Pandas DataFrame
        
    Returns:
        dict: Dictionary with column summaries
    """
    summary = {}
    
    for column in dataframe.columns:
        col_data = dataframe[column]
        col_summary = {
            "name": column,
            "type": str(col_data.dtype),
            "missing_values": col_data.isna().sum(),
            "unique_values": col_data.nunique()
        }
        
        # Add numeric summaries if applicable
        if pd.api.types.is_numeric_dtype(col_data.dtype):
            col_summary.update({
                "min": col_data.min(),
                "max": col_data.max(),
                "mean": col_data.mean(),
                "median": col_data.median()
            })
        # Add categorical summaries if applicable
        elif col_data.nunique() < 10:  # Only for columns with few unique values
            value_counts = col_data.value_counts().head(5).to_dict()
            col_summary["top_values"] = value_counts
        
        summary[column] = col_summary
    
    return summary
# Style Plotly visualizations with Mario theme
def style_mario_visualization(fig, character_color='#E52521'):
    """Apply Mario-themed styling to Plotly figures"""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
        font=dict(
            family='Press Start 2P, cursive',
            size=10,
            color='#262730'
        ),
        title=dict(
            font=dict(
                family='Press Start 2P, cursive',
                size=14,
                color=character_color
            )
        ),
        legend=dict(
            font=dict(
                family='Press Start 2P, cursive',
                size=8
            )
        )
    )
    return fig

# Helper for card container
card_style = "background-color: #fffbe6; color: #111; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(230,57,70,0.10); border: 1.5px solid #e63946;"

# -------------- Sidebar Navigation (Mario Style) ----------------
with st.sidebar:
    st.markdown(f"<h2 class='mario-header' style='font-size: 1.5rem;'>🍄 Super Finance Bros</h2>", unsafe_allow_html=True)
    
    # Character selection
    st.markdown("<div class='mario-section' style='font-size: 1rem;'>Choose Your Character</div>", unsafe_allow_html=True)
    
    character_cols = st.columns(2)
    for i, (char_name, char_info) in enumerate(CHARACTERS.items()):
        col_idx = i % 2
        with character_cols[col_idx]:
            char_selected = st.session_state.selected_character == char_name
            st.markdown(f"""
            <div class='character-card' style='border-color: {char_info['color'] if char_selected else '#8B4513'};
                            background-color: {char_info['color'] + '20' if char_selected else 'rgba(255, 251, 230, 0.8)'};'>
                <div style='text-align: center;'>
                    <div style='font-size: 24px;'>{char_info['icon']}</div>
                    <div style='font-weight: bold; font-size: 12px;'>{char_name}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select {char_name}", key=f"select_{char_name}"):
                st.session_state.selected_character = char_name
                st.rerun()
    
    st.markdown("---")
    
    # Main navigation
    st.markdown("<div class='mario-section' style='font-size: 1rem;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", ["🏠 Welcome", "📊 Data Explorer", "📈 Visualizations", 
                        "💰 Budget Planner", "🧮 Credit Risk Analyzer", "🔮 Lifestyle Analyzer"])

# Character's color for styling
character_color = CHARACTERS[st.session_state.selected_character]['color']


def get_visualization_options():
    """
    Returns a dictionary of available visualization types.
    
    Returns:
        dict: Mapping of visualization names to descriptions
    """
    return {
        "Bar Chart": "Compare values across categories",
        "Line Chart": "Show trends over time or sequence",
        "Scatter Plot": "Show relationship between two variables",
        "Pie Chart": "Show proportion of a whole",
        "Histogram": "Display distribution of a numeric variable",
        "Area Chart": "Show cumulative values across categories",
        "Heatmap": "Visualize correlation between variables"
    }

def create_visualization(data, viz_type, params):
    """
    Create a visualization based on the specified type and parameters.
    
    Args:
        data: Pandas DataFrame
        viz_type: Type of visualization to create
        params: Parameters for the visualization
        
    Returns:
        plotly.graph_objects.Figure: The created visualization
    """
    # Default figure parameters
    height = params.get('height', 400)
    title = params.get('title', viz_type)
    theme = params.get('theme', 'plotly')
    
    # Create the visualization based on type
    if viz_type == "Bar Chart":
        x = params.get('x')
        y = params.get('y')
        color = params.get('color')
        
        if x is None or y is None:
            return None
        
        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            height=height,
            template=theme
        )
    
    elif viz_type == "Line Chart":
        x = params.get('x')
        y = params.get('y')
        
        if x is None or y is None:
            return None
        
        # Sort data by x if it's numeric
        if pd.api.types.is_numeric_dtype(data[x].dtype):
            sorted_data = data.sort_values(by=x)
        else:
            sorted_data = data
        
        fig = px.line(
            sorted_data,
            x=x,
            y=y,
            title=title,
            height=height,
            template=theme
        )
    
    elif viz_type == "Scatter Plot":
        x = params.get('x')
        y = params.get('y')
        color = params.get('color')
        size = params.get('size')
        
        if x is None or y is None:
            return None
        
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            size=size,
            title=title,
            height=height,
            template=theme
        )
    
    elif viz_type == "Pie Chart":
        names = params.get('names')
        values = params.get('values')
        
        if names is None or values is None:
            return None
        
        fig = px.pie(
            data,
            names=names,
            values=values,
            title=title,
            height=height,
            template=theme
        )
    
    elif viz_type == "Histogram":
        column = params.get('column')
        bins = params.get('bins', 20)
        
        if column is None:
            return None
        
        fig = px.histogram(
            data,
            x=column,
            nbins=bins,
            title=title,
            height=height,
            template=theme
        )
    
    elif viz_type == "Area Chart":
        x = params.get('x')
        y = params.get('y')
        
        if x is None or y is None:
            return None
        
        # Sort data by x if it's numeric
        if pd.api.types.is_numeric_dtype(data[x].dtype):
            sorted_data = data.sort_values(by=x)
        else:
            sorted_data = data
        
        fig = px.area(
            sorted_data,
            x=x,
            y=y,
            title=title,
            height=height,
            template=theme
        )
    
    elif viz_type == "Heatmap":
        heatmap_type = params.get('type')
        
        if heatmap_type == "Correlation Matrix":
            # Get numeric columns only
            numeric_data = data.select_dtypes(include=['number'])
            
            if numeric_data.shape[1] < 2:
                return None
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                title=title,
                height=height,
                template=theme
            )
    else:
        return None
    
    # Update layout for better appearance
    fig.update_layout(
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    return fig

def export_visualization_as_html(fig):
    """
    Export a Plotly figure as an HTML string.
    
    Args:
        fig: Plotly figure object
        
    Returns:
        str: HTML representation of the figure
    """
    return fig.to_html(include_plotlyjs="cdn")

def export_visualization_as_image(fig, format="png"):
    """
    Export a Plotly figure as an image.
    
    Args:
        fig: Plotly figure object
        format: Image format (png, jpg, svg)
        
    Returns:
        bytes: Image data
    """
    return fig.to_image(format=format)

# --- Helper functions for Lifestyle Analyzer (add above the Lifestyle Analyzer section) ---
def validate_lifestyle_csv(df):
    required_columns = ['Groceries', 'Dining', 'Shopping', 'Utilities', 'Entertainment', 'Savings']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns: " + ", ".join([col for col in required_columns if col not in df.columns]))
    if df.isnull().any().any():
        raise ValueError("Missing values detected. Please clean your data.")
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"'{col}' contains non-numeric values.")

def generate_lifestyle_mock_data():
    np.random.seed(42)
    data = {
        'Groceries': np.random.normal(500, 100, 30),
        'Dining': np.random.normal(300, 50, 30),
        'Shopping': np.random.normal(400, 150, 30),
        'Utilities': np.random.normal(200, 30, 30),
        'Entertainment': np.random.normal(250, 75, 30),
        'Savings': np.random.normal(1000, 200, 30)
    }
    return pd.DataFrame(data)

def get_lifestyle_description(cluster_label, n_clusters):
    if n_clusters == 3:
        classes = {
            0: {'label': '🛡️ Frugal', 'description': 'You tend to be more conservative with your spending, focusing on essentials and saving.'},
            1: {'label': '⚖️ Balanced', 'description': 'You maintain a balanced approach to spending and saving.'},
            2: {'label': '💎 Lavish', 'description': 'You spend more freely, with higher discretionary spending.'}
        }
        return classes.get(cluster_label, {'label': f'Class {cluster_label+1}', 'description': 'Unclassified pattern.'})
    else:
        return {'label': f'Class {cluster_label + 1}', 'description': 'A unique spending pattern.'}

def create_lifestyle_radar_chart(df, cluster_labels, user_index):
    categories = df.columns.drop('Cluster')
    user_values = df.iloc[user_index].drop('Cluster')
    cluster_avg = df[df['Cluster'] == cluster_labels[user_index]].mean().drop('Cluster')
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=cluster_avg,
        theta=categories,
        fill='toself',
        name='Cluster Average',
        line_color='#4CAF50',
        fillcolor='rgba(76,175,80,0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='Your Stats',
        line_color='#E52521',
        fillcolor='rgba(229,37,33,0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(user_values), max(cluster_avg)) * 1.2]
            )
        ),
        showlegend=True,
        title="Spending Pattern Comparison"
    )
    return fig


# -------------- Welcome Page ----------------
if page == "🏠 Welcome":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <h1 class='mario-header' style="text-align: center;">🍄 Super Finance Bros</h1>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="{card_style} text-align: center; margin-bottom: 20px;">
            <p style="font-size: 18px;">Welcome to your Mario-themed finance data toolkit!</p>
            <p>Use the sidebar to navigate to different features:</p>
            <ul style="list-style-type: none; padding-left: 0; font-size: 16px;">
                <li style="margin: 10px 0;">📊 <strong>Data Explorer</strong> - Upload and explore your data</li>
                <li style="margin: 10px 0;">📈 <strong>Visualizations</strong> - Create custom charts</li>
                <li style="margin: 10px 0;">💬 <strong>AI Assistant</strong> - Chat with your data using AI</li>
                <li style="margin: 10px 0;">💰 <strong>Budget Planner</strong> - Plan and optimize your budget</li>
                <li style="margin: 10px 0;">🧮 <strong>Credit Risk Analyzer</strong> - Assess your creditworthiness</li>
                <li style="margin: 10px 0;">🔮 <strong>Lifestyle Analyzer</strong> - Understand your spending patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        character_info = CHARACTERS[st.session_state.selected_character]
        st.markdown(f"""
        <div style="{card_style} text-align: center; margin-bottom: 20px;">
            <h3 class='mario-section'>Your Character: {st.session_state.selected_character}</h3>
            <div style="font-size: 48px; margin: 20px 0;">{character_info['icon']}</div>
            <p>{character_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="{card_style} text-align: center;">
            <h3 class='mario-section'>🔍 See Your Financial Future Clearly</h3>
            <p style="color: #457b9d;">Make informed decisions with data insights and AI-powered analysis</p>
        </div>
        """, unsafe_allow_html=True)

# -------------- Data Explorer ----------------
elif page == "📊 Data Explorer":
    st.markdown("<h1 class='mario-header'>📊 Data Explorer</h1>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="{card_style}">
        <h3 class='mario-section'>Upload Your Data</h3>
        <p>Upload your data file (CSV or Excel) to begin exploring and visualizing it.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            data, file_name = load_data(uploaded_file)
            
            if data is not None:
                st.session_state.data = data
                st.session_state.file_name = file_name
                
                st.success(f"Successfully loaded {file_name} with {len(data)} rows and {len(data.columns)} columns.")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(data.head(5), use_container_width=True)
                
                # Display column information
                st.subheader("Column Information")
                column_types = get_column_types(data)
                col_info = pd.DataFrame({
                    'Column': column_types.keys(),
                    'Type': column_types.values(),
                    'Sample Values': [', '.join(str(x) for x in data[col].dropna().head(3).tolist()) for col in column_types.keys()]
                })
                st.table(col_info)
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    st.dataframe(data[numeric_cols].describe())
                else:
                    st.info("No numeric columns found for summary statistics.")
                
                st.info("Navigate to other tabs to analyze, visualize, or chat with your data.")
            else:
                st.error("Failed to load the data. Please check if the file format is correct.")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    else:
        st.info("Please upload a data file to begin.")
        
        # Show sample data option
        if st.button("🎮 Use Sample Data"):
            # Generate sample financial data
            np.random.seed(42)
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            categories = ['Housing', 'Food', 'Transportation', 'Entertainment', 'Utilities', 'Shopping', 'Healthcare', 'Savings']
            
            # Create sample transactions
            data = []
            for date in dates:
                # Add 1-3 transactions per day
                for _ in range(np.random.randint(1, 4)):
                    category = np.random.choice(categories)
                    
                    # Different amount ranges based on category
                    if category == 'Housing':
                        amount = np.random.uniform(500, 1500)
                    elif category == 'Food':
                        amount = np.random.uniform(10, 100)
                    elif category == 'Transportation':
                        amount = np.random.uniform(5, 80)
                    elif category == 'Entertainment':
                        amount = np.random.uniform(20, 150)
                    elif category == 'Utilities':
                        amount = np.random.uniform(50, 200)
                    elif category == 'Shopping':
                        amount = np.random.uniform(20, 300)
                    elif category == 'Healthcare':
                        amount = np.random.uniform(50, 500)
                    elif category == 'Savings':
                        amount = np.random.uniform(100, 1000)
                    
                    # Determine if expense or income
                    if category == 'Savings':
                        trans_type = 'Income'
                    else:
                        trans_type = 'Expense'
                    
                    data.append({
                        'Date': date,
                        'Category': category,
                        'Description': f"{category} {'payment' if trans_type == 'Expense' else 'deposit'}",
                        'Amount': amount,
                        'Type': trans_type
                    })
            
            # Convert to DataFrame
            sample_df = pd.DataFrame(data)
            sample_df['Date'] = pd.to_datetime(sample_df['Date'])
            sample_df['Amount'] = sample_df['Amount'].round(2)
            
            # Store in session state
            st.session_state.data = sample_df
            st.session_state.file_name = "sample_financial_data.csv"
            
            st.success("Sample financial data loaded successfully!")
            st.dataframe(sample_df.head(10), use_container_width=True)

# -------------- Visualizations ----------------
elif page == "📈 Visualizations":
    st.markdown("<h1 class='mario-header'>📈 Data Visualizations</h1>", unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        st.markdown(f"""
        <div style="{card_style}">
            <h3 class='mario-section'>Create Custom Visualizations</h3>
            <p>Select visualization type and customize parameters to explore your data visually.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization options
        viz_options = get_visualization_options()
        viz_type = st.selectbox("Select Visualization Type", list(viz_options.keys()))
        
        # Create columns for options and preview
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Visualization Options")
            
            # Dynamic options based on visualization type and data columns
            viz_params = {}
            
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()
            all_cols = data.columns.tolist()
            
            if viz_type in ["Bar Chart", "Line Chart", "Area Chart"]:
                viz_params['x'] = st.selectbox("X-axis", all_cols)
                viz_params['y'] = st.selectbox("Y-axis", numeric_cols if numeric_cols else all_cols)
                if viz_type == "Bar Chart":
                    viz_params['color'] = st.selectbox("Color (Optional)", ["None"] + categorical_cols)
                    if viz_params['color'] == "None":
                        viz_params['color'] = None
            
            elif viz_type == "Scatter Plot":
                viz_params['x'] = st.selectbox("X-axis", numeric_cols if numeric_cols else all_cols)
                viz_params['y'] = st.selectbox("Y-axis", numeric_cols if numeric_cols else all_cols)
                viz_params['color'] = st.selectbox("Color (Optional)", ["None"] + categorical_cols)
                if viz_params['color'] == "None":
                    viz_params['color'] = None
                viz_params['size'] = st.selectbox("Size (Optional)", ["None"] + numeric_cols)
                if viz_params['size'] == "None":
                    viz_params['size'] = None
            
            elif viz_type == "Histogram":
                viz_params['column'] = st.selectbox("Column", numeric_cols if numeric_cols else all_cols)
                viz_params['bins'] = st.slider("Number of bins", 5, 100, 20)
            
            elif viz_type == "Pie Chart":
                viz_params['names'] = st.selectbox("Categories", categorical_cols if categorical_cols else all_cols)
                viz_params['values'] = st.selectbox("Values", numeric_cols if numeric_cols else all_cols)
            
            elif viz_type == "Heatmap":
                if len(numeric_cols) >= 2:
                    corr_options = ["Correlation Matrix"]
                    viz_params['type'] = st.selectbox("Heatmap Type", corr_options)
                else:
                    st.warning("Need at least 2 numeric columns for a heatmap.")
                    viz_params['type'] = "Correlation Matrix"
            
            # Common customization options
            st.subheader("Styling")
            viz_params['title'] = st.text_input("Chart Title", f"{viz_type} of {st.session_state.file_name}")
            viz_params['height'] = st.slider("Chart Height", 300, 800, 400)
            viz_params['theme'] = st.selectbox("Color Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2"])
            
            # Apply Mario character color
            viz_params['mario_color'] = character_color
            
            create_viz = st.button("🎮 Create Visualization")
            
        with col2:
            st.subheader("Visualization Preview")
            
            if create_viz:
                try:
                    figure = create_visualization(data, viz_type, viz_params)
                    # Apply Mario theme styling
                    figure = style_mario_visualization(figure, character_color)
                    
                    st.session_state.current_viz = {
                        'type': viz_type,
                        'params': viz_params,
                        'figure': figure
                    }
                    
                    # Store visualization in history
                    if figure is not None:
                        if viz_type not in [v['type'] for v in st.session_state.visualizations]:
                            st.session_state.visualizations.append({
                                'type': viz_type,
                                'params': viz_params,
                                'figure': figure
                            })
                    
                    st.plotly_chart(figure, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
            
            elif st.session_state.current_viz is not None:
                st.plotly_chart(st.session_state.current_viz['figure'], use_container_width=True)
            else:
                st.info("Configure your visualization and click 'Create Visualization' to see the preview.")
        
        # Export and sharing options
        if st.session_state.current_viz is not None:
            st.subheader("Export & Share")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔽 Export as HTML"):
                    buffer = io.StringIO()
                    fig = st.session_state.current_viz['figure']
                    html_str = fig.to_html(include_plotlyjs="cdn")
                    
                    b64 = base64.b64encode(html_str.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="visualization.html">Download HTML File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    st.success("HTML file ready for download")
            
            with col2:
                if st.button("🖼️ Export as Image (PNG)"):
                    fig = st.session_state.current_viz['figure']
                    img_bytes = fig.to_image(format="png")
                    
                    b64 = base64.b64encode(img_bytes).decode()
                    href = f'<a href="data:image/png;base64,{b64}" download="visualization.png">Download PNG Image</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    st.success("PNG image ready for download")
    else:
        st.info("Please upload data in the 'Data Explorer' tab to start creating visualizations.")

# -------------- Budget Planner ----------------
elif page == "💰 Budget Planner":
    st.markdown("<h1 class='mario-header'>💰 Budget Planner</h1>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="{card_style}">
        <h3 class='mario-section'>Plan Your Budget</h3>
        <p>Set budget limits for each spending category and track your progress.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add dedicated data upload for Budget Planner
    st.subheader("Budget Data")
    
    budget_data_source = st.radio("Choose data source:", ["📁 Upload Budget Dataset", "🎮 Use Sample Financial Data"])
    
    data = None
    if budget_data_source == "📁 Upload Budget Dataset":
        uploaded_file = st.file_uploader("Upload budget data (CSV or Excel with Category, Amount, Type columns)", type=["csv", "xlsx", "xls"], key="budget_upload")
        
        if uploaded_file is not None:
            try:
                data, file_name = load_data(uploaded_file)
                st.success(f"Successfully loaded {file_name} with {len(data)} rows and {len(data.columns)} columns.")
                
                # Check if data has necessary columns for budget planning
                required_cols = ['Category', 'Amount', 'Type']
                missing_cols = [col for col in required_cols if col not in data.columns]
                
                if missing_cols:
                    st.error(f"Your data is missing required columns for budget planning: {', '.join(missing_cols)}")
                    st.info("Your data should have columns for Category, Amount, and Type (Income/Expense).")
                    data = None
            except Exception as e:
                st.error(f"Error loading budget data: {str(e)}")
                data = None
    else:
        # Generate sample financial data
        st.info("Using simulated budget data for demonstration.")
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        categories = ['Housing', 'Food', 'Transportation', 'Entertainment', 'Utilities', 'Shopping', 'Healthcare', 'Savings']
        
        # Create sample transactions
        sample_data = []
        for date in dates:
            # Add 1-3 transactions per day
            for _ in range(np.random.randint(1, 4)):
                category = np.random.choice(categories)
                
                # Different amount ranges based on category
                if category == 'Housing':
                    amount = np.random.uniform(50000, 150000)
                elif category == 'Food':
                    amount = np.random.uniform(10000, 40000)
                elif category == 'Transportation':
                    amount = np.random.uniform(5000, 15000)
                elif category == 'Entertainment':
                    amount = np.random.uniform(10000, 50000)
                elif category == 'Utilities':
                    amount = np.random.uniform(8000, 20000)
                elif category == 'Shopping':
                    amount = np.random.uniform(10000, 30000)
                elif category == 'Healthcare':
                    amount = np.random.uniform(5000, 25000)
                elif category == 'Savings':
                    amount = np.random.uniform(20000, 100000)
                
                # Determine if expense or income
                if category == 'Savings':
                    trans_type = 'Income'
                else:
                    trans_type = 'Expense'
                
                sample_data.append({
                    'Date': date,
                    'Category': category,
                    'Description': f"{category} {'payment' if trans_type == 'Expense' else 'deposit'}",
                    'Amount': amount,
                    'Type': trans_type
                })
        
        # Convert to DataFrame
        data = pd.DataFrame(sample_data)
        data['Amount'] = data['Amount'].round(2)
        
        # Show preview of sample data
        with st.expander("Preview Sample Budget Data"):
            st.dataframe(data.head(10), use_container_width=True)
    
    # Continue with budget planning if data is available
    if data is not None:
        # Display data overview
        st.subheader("Budget Overview")
        
        # Check if data has necessary columns for budget planning
        required_cols = ['Category', 'Amount', 'Type']
        if all(col in data.columns for col in required_cols):
            # Calculate spending by category
            expenses = data[data['Type'] == 'Expense']
            expense_by_category = expenses.groupby('Category')['Amount'].sum().reset_index()
            
            # Calculate total income vs expenses
            total_income = data[data['Type'] == 'Income']['Amount'].sum()
            total_expenses = data[data['Type'] == 'Expense']['Amount'].sum()
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Income", f"PKR {total_income:.2f}")
            with col2:
                st.metric("Total Expenses", f"PKR {total_expenses:.2f}")
            with col3:
                balance = total_income - total_expenses
                st.metric("Balance", f"PKR {balance:.2f}", delta=f"{(balance/total_income*100):.1f}%" if total_income > 0 else "0%")
            
            # Display category breakdown
            with st.expander("Expense Categories Breakdown"):
                st.dataframe(expense_by_category.sort_values('Amount', ascending=False), use_container_width=True)
            
            # Set budget limits for categories
            st.subheader("Set Budget Limits")
            
            expense_categories = sorted(expenses['Category'].unique())
            budget_limits = {}
            
            # Store category-wise expenses for reference
            category_expenses = {}
            for category in expense_categories:
                category_expenses[category] = expenses[expenses['Category'] == category]['Amount'].sum()
            
            # Create budget limit sliders in 2 columns for better spacing
            col1, col2 = st.columns(2)
            
            for i, category in enumerate(expense_categories):
                current_spend = category_expenses[category]
                current_limit = st.session_state.budget_limits.get(category, current_spend * 1.1)
                
                with col1 if i % 2 == 0 else col2:
                    new_limit = st.slider(
                        f"Budget for {category}",
                        min_value=0.0,
                        max_value=float(current_spend * 2),
                        value=float(current_limit),
                        step=10.0,
                        format="PKR %.2f"
                    )
                    budget_limits[category] = new_limit
            
            # Update session state with new limits
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Budget Limits", key="save_budget"):
                    st.session_state.budget_limits = budget_limits
                    st.success("Budget limits saved!")
            
            # Display spending vs budget visualization
            st.subheader("Spending vs Budget")
            
            # Create data for the budget comparison chart
            budget_data = []
            for category in expense_categories:
                current_spend = category_expenses[category]
                budget = st.session_state.budget_limits.get(category, current_spend * 1.1)
                
                # Add current spending data
                budget_data.append({
                    'Category': category,
                    'Amount': current_spend,
                    'Type': 'Current Spending'
                })
                
                # Add budget limit data
                budget_data.append({
                    'Category': category,
                    'Amount': budget,
                    'Type': 'Budget Limit'
                })
                
                # Progress bar grid
                col1, col2 = st.columns([3, 1])
                with col1:
                    percentage = min(100, (current_spend / budget) * 100) if budget > 0 else 0
                    
                    # Color code based on percentage
                    if percentage < 70:
                        bar_color = "green"
                    elif percentage < 90:
                        bar_color = "orange"
                    else:
                        bar_color = "red"
                    
                    st.progress(percentage / 100)
                
                with col2:
                    st.markdown(f"**{category}**: {percentage:.1f}%")
            
            # Convert to DataFrame
            budget_df = pd.DataFrame(budget_data)
            
            # Create grouped bar chart
            fig = px.bar(
                budget_df, 
                x='Category', 
                y='Amount', 
                color='Type',
                barmode='group',
                title='Current Spending vs. Budget Limits',
                color_discrete_map={
                    'Current Spending': '#E52521',
                    'Budget Limit': '#4CAF50'
                }
            )
            
            # Apply Mario theme
            fig = style_mario_visualization(fig, character_color)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Budget recommendations
            st.subheader("Budget Recommendations")
            
            # Calculate % of budget used for each category
            recommendations = []
            for category in expense_categories:
                current_spend = category_expenses[category]
                budget = st.session_state.budget_limits.get(category, current_spend * 1.1)
                
                percentage = (current_spend / budget) * 100 if budget > 0 else 0
                
                if percentage > 100:
                    recommendations.append(f"⚠️ You've exceeded your budget for {category} by {percentage - 100:.1f}%. Consider increasing your budget or reducing spending.")
                elif percentage > 90:
                    recommendations.append(f"🔶 You're close to your budget limit for {category} ({percentage:.1f}%). Monitor your spending closely.")
                elif percentage < 50 and budget > 0:
                    recommendations.append(f"💫 You've only used {percentage:.1f}% of your {category} budget. You could allocate some of this budget to other categories if needed.")
            
            if recommendations:
                for recommendation in recommendations:
                    st.markdown(f"<div style='{card_style}; margin-bottom: 10px;'>{recommendation}</div>", unsafe_allow_html=True)
            else:
                st.info("You're doing well with your budget! All categories are within reasonable limits.")
        else:
            st.error(f"Your data is missing required columns for budget planning: {', '.join(missing_cols)}")
            st.info("Your data should have columns for Category, Amount, and Type (Income/Expense).")

# -------------- Credit Risk Analyzer ----------------
elif page == "🧮 Credit Risk Analyzer":
    st.markdown("<h1 class='mario-header'>🧮 Credit Risk Analyzer</h1>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="{card_style}">
        <h3 class='mario-section'>Understanding Your Credit Risk</h3>
        <p>This module helps you assess your credit risk based on various financial and personal factors. 
        The analysis considers multiple aspects of your financial health to provide a comprehensive risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not SKLEARN_AVAILABLE:
        st.warning("The scikit-learn library is required for credit risk analysis. Please install scikit-learn to use this feature.")
        st.markdown(f"""
        <div style="{card_style}">
            <p>This feature requires the scikit-learn package to perform machine learning analysis of credit risk factors.</p>
            <p>However, you can still explore the other features of the application:</p>
            <ul>
                <li>Data Explorer - Upload and analyze your financial data</li>
                <li>Visualizations - Create custom charts and graphs</li>
                <li>AI Assistant - Chat with your data using natural language</li>
                <li>Budget Planner - Set budget limits and track spending</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Use full width layout with tabs instead of columns
        st.subheader("Credit Analysis")
        
        # Create tabs for better organization and layout
        credit_tabs = st.tabs(["💾 Credit Data", "👤 Credit Profile"])
        
        with credit_tabs[0]:  # DATA TAB
            st.markdown("### Credit Data Source")
            
            if st.session_state.data is not None and 'Income' in st.session_state.data.columns:
                st.success("Using your uploaded financial data for credit risk analysis.")
                df = st.session_state.data
            else:
                st.info("Using simulated data for credit risk demonstration.")
                # Simulate credit risk data
                np.random.seed(42)
                # Simulate base features
                base_df = pd.DataFrame({
                    'Income': np.random.normal(50000, 15000, 100).astype(int),
                    'Age': np.random.normal(35, 10, 100).astype(int),
                    'Debt': np.random.normal(15000, 5000, 100).astype(int),
                    'Employment_Length': np.random.normal(5, 3, 100).astype(int),
                    'Married': np.random.choice([0, 1], 100, p=[0.4, 0.6]),
                    'Dependents': np.random.poisson(1, 100),
                    'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100, p=[0.2, 0.5, 0.2, 0.1]),
                    'Home_Ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], 100, p=[0.3, 0.2, 0.5])
                })
                # Map education and home ownership to numeric
                education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
                home_map = {'Rent': 0, 'Mortgage': 1, 'Own': 2}
                base_df['Education_num'] = base_df['Education'].map(education_map)
                base_df['Home_num'] = base_df['Home_Ownership'].map(home_map)
                # Simulate expenses and savings
                base_df['Expenses'] = (base_df['Income'] * (0.4 + 0.1 * base_df['Dependents']) + np.random.normal(0, 2000, 100)).clip(5000, None)
                base_df['Savings'] = (base_df['Income'] - base_df['Expenses'] - 0.1 * base_df['Debt']).clip(0, None)
                base_df['Debt_to_Income'] = base_df['Debt'] / (base_df['Income'] + 1)
                # Improved credit score simulation
                base_df['Credit_Score'] = (
                    600
                    + (base_df['Income'] - 50000) / 1000
                    - (base_df['Debt'] - 15000) / 2000
                    + base_df['Employment_Length'] * 5
                    + base_df['Education_num'] * 20
                    + base_df['Home_num'] * 15
                    + np.random.normal(0, 30, 100)
                ).clip(300, 850).astype(int)
                # Improved default probability
                prob_default = (
                    0.15
                    + 0.25 * base_df['Debt_to_Income']
                    + 0.03 * base_df['Dependents']
                    - 0.02 * base_df['Employment_Length']
                    - 0.03 * base_df['Home_num']
                    - 0.01 * base_df['Education_num']
                    + 0.01 * np.abs(base_df['Age'] - 40) / 10
                    + 0.02 * (base_df['Married'] == 1)
                    - 0.05 * (base_df['Savings'] > 10000)
                    + np.random.normal(0, 0.03, 100)
                )
                base_df['Default'] = (np.random.rand(100) < prob_default.clip(0.05, 0.7)).astype(int)
                df = base_df
            
            st.markdown("### Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Train models section
            st.markdown("### Train Prediction Models")
            
            if st.button("🛠️ Train Model"):
                with st.spinner("Training model..."):
                    try:
                        # Train both models
                        if 'Credit_Score' in df.columns and 'Income' in df.columns:
                            mini_X = df[['Income', 'Age', 'Debt', 'Employment_Length', 'Married', 'Dependents', 'Education_num', 'Home_num', 'Expenses', 'Savings', 'Debt_to_Income']]
                            mini_y = df['Credit_Score']
                            mini_model = RandomForestRegressor(n_estimators=100, random_state=42)
                            mini_model.fit(mini_X, mini_y)
                            st.session_state['mini_model'] = mini_model
                            st.session_state['credit_features'] = mini_X.columns.tolist()
                            st.success("Credit score prediction model trained successfully!")

                        if 'Default' in df.columns:
                            df_processed = df.copy()
                            if 'Education' in df.columns:
                                df_processed = pd.get_dummies(df_processed, columns=['Education'])
                            if 'Home_Ownership' in df.columns:
                                df_processed = pd.get_dummies(df_processed, columns=['Home_Ownership'])
                            
                            # Get all columns except Default
                            feature_cols = [col for col in df_processed.columns if col != 'Default']
                            
                            X = df_processed[feature_cols]
                            y = df_processed['Default']
                            
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            model = LogisticRegression()
                            model.fit(X_train, y_train)
                            
                            st.session_state['default_model'] = model
                            st.session_state['X_test'] = X_test
                            st.session_state['y_test'] = y_test
                            st.session_state['X'] = X
                            st.session_state['feature_names'] = X.columns
                            
                            st.success("Default risk model trained successfully!")
                    except Exception as e:
                        st.error(f"Error training models: {str(e)}")
        
        with credit_tabs[1]:  # PROFILE TAB
            st.markdown("### Your Credit Profile")
            
            # Use a grid layout for better spacing and organization
            # First row: personal info in 4 columns
            st.markdown("#### Personal Information")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                age = st.number_input("Age", min_value=18, max_value=100, value=30, key="age_input")
            with c2:
                education = st.selectbox("Education Level", ['High School', 'Bachelor', 'Master', 'PhD'], key="education_select")
            with c3:
                married = st.selectbox("Marital Status", ["Single", "Married"], key="marital_status")
            with c4:
                dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0, key="dependents_input")
            
            # Second row: Financial info in 4 columns 
            st.markdown("#### Financial Information")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                income = st.number_input("Monthly Income (PKR)", min_value=0, value=50000, key="income_input", 
                                       step=5000)
            with c2:
                debt = st.number_input("Current Debt (PKR)", min_value=0, value=10000, key="debt_input",
                                      step=5000)
            with c3:
                employment_length = st.number_input("Years Employed", min_value=0, max_value=50, value=5, 
                                                  key="employment_input")
            with c4:
                home_ownership = st.selectbox("Home Ownership", ['Rent', 'Own', 'Mortgage'], key="ownership_select")
            
            # Encode categorical variables
            married_encoded = 1 if married == "Married" else 0
            education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
            home_map = {'Rent': 0, 'Mortgage': 1, 'Own': 2}
            education_num = education_map[education]
            home_num = home_map[home_ownership]
            
            # Calculate derived metrics
            user_expenses = (income * (0.4 + 0.1 * dependents))
            user_savings = max(income - user_expenses - 0.1 * debt, 0)
            user_debt_to_income = debt / (income + 1)
            
            # Show calculated metrics
            st.markdown("#### Calculated Financial Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Est. Expenses", f"PKR {user_expenses:,.2f}")
            with col2:
                st.metric("Est. Savings", f"PKR {user_savings:,.2f}")
            with col3:
                st.metric("Debt-to-Income", f"{user_debt_to_income*100:.1f}%")
            
            # Divider
            st.markdown("---")
            
            # Predict credit score using mini model
            if 'mini_model' in st.session_state and 'credit_features' in st.session_state:
                # Create input dataframe based on available features
                input_data = {}
                for feature in st.session_state['credit_features']:
                    if feature == 'Income':
                        input_data[feature] = [income]
                    elif feature == 'Age':
                        input_data[feature] = [age]
                    elif feature == 'Debt':
                        input_data[feature] = [debt]
                    elif feature == 'Employment_Length':
                        input_data[feature] = [employment_length]
                    elif feature == 'Married':
                        input_data[feature] = [married_encoded]
                    elif feature == 'Dependents':
                        input_data[feature] = [dependents]
                    elif feature == 'Education_num':
                        input_data[feature] = [education_num]
                    elif feature == 'Home_num':
                        input_data[feature] = [home_num]
                    elif feature == 'Expenses':
                        input_data[feature] = [user_expenses]
                    elif feature == 'Savings':
                        input_data[feature] = [user_savings]
                    elif feature == 'Debt_to_Income':
                        input_data[feature] = [user_debt_to_income]
                
                mini_input = pd.DataFrame(input_data)
                predicted_credit_score = int(st.session_state['mini_model'].predict(mini_input)[0])
            else:
                # Fallback calculation if no model
                predicted_credit_score = int(
                    600
                    + (income - 50000) / 1000
                    - (debt - 15000) / 2000
                    + employment_length * 5
                    + education_num * 20
                    + home_num * 15
                )
            
            # Display predicted credit score
            st.markdown(f"<div style='{card_style} margin-bottom:10px; text-align:center;'><b>Predicted Credit Score:</b> <span style='font-size:1.5em;'>{predicted_credit_score}</span></div>", unsafe_allow_html=True)
            
            # Credit score gauge chart
            def create_gauge_chart(score):
                score_color = "#ff0000" if score < 580 else "#ffa500" if score < 670 else "#ffff00" if score < 740 else "#00ff00" if score < 800 else "#00ced1"
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Credit Score", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': score_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [300, 580], 'color': 'rgba(255, 0, 0, 0.3)'},
                            {'range': [580, 670], 'color': 'rgba(255, 165, 0, 0.3)'},
                            {'range': [670, 740], 'color': 'rgba(255, 255, 0, 0.3)'},
                            {'range': [740, 800], 'color': 'rgba(0, 255, 0, 0.3)'},
                            {'range': [800, 850], 'color': 'rgba(0, 206, 209, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': score
                        }
                    }
                ))
                
                # Style the gauge
                fig = style_mario_visualization(fig, character_color)
                fig.update_layout(height=250)
                
                return fig
            
            st.plotly_chart(create_gauge_chart(predicted_credit_score), use_container_width=True)
            
            # Credit score interpretation
            if predicted_credit_score < 580:
                interpretation = "Poor credit score. You may have difficulty getting approved for loans or credit cards."
            elif predicted_credit_score < 670:
                interpretation = "Fair credit score. You might qualify for loans but at higher interest rates."
            elif predicted_credit_score < 740:
                interpretation = "Good credit score. You'll likely be approved for loans with competitive interest rates."
            elif predicted_credit_score < 800:
                interpretation = "Very good credit score. You'll receive favorable rates and terms for most financial products."
            else:
                interpretation = "Excellent credit score. You'll get the best rates and terms available."
            
            st.markdown(f"<div style='{card_style}'><b>Interpretation:</b> {interpretation}</div>", unsafe_allow_html=True)
            
            # Default risk prediction
            if 'default_model' in st.session_state:
                st.markdown("<div class='mario-section' style='font-size: 1.2rem; margin-top: 20px;'>Default Risk Prediction</div>", unsafe_allow_html=True)
                
                if st.button("🔮 Predict Default Risk", key="predict_button"):
                    try:
                        # Prepare user input for the model
                        user_input = {}
                        for feature in st.session_state['feature_names']:
                            if feature == 'Income':
                                user_input[feature] = [income]
                            elif feature == 'Age':
                                user_input[feature] = [age]
                            elif feature == 'Debt':
                                user_input[feature] = [debt]
                            elif feature == 'Employment_Length':
                                user_input[feature] = [employment_length]
                            elif feature == 'Married':
                                user_input[feature] = [married_encoded]
                            elif feature == 'Dependents':
                                user_input[feature] = [dependents]
                            elif feature == 'Education_num':
                                user_input[feature] = [education_num]
                            elif feature == 'Home_num':
                                user_input[feature] = [home_num]
                            elif feature == 'Expenses':
                                user_input[feature] = [user_expenses]
                            elif feature == 'Savings':
                                user_input[feature] = [user_savings]
                            elif feature == 'Debt_to_Income':
                                user_input[feature] = [user_debt_to_income]
                            elif 'Education_' in feature:
                                user_input[feature] = [1 if education in feature else 0]
                            elif 'Home_Ownership_' in feature:
                                user_input[feature] = [1 if home_ownership in feature else 0]
                            else:
                                user_input[feature] = [0]  # Default value for unknown features
                        
                        # Fill missing columns with 0
                        for col in st.session_state['feature_names']:
                            if col not in user_input:
                                user_input[col] = [0]
                        
                        user_df = pd.DataFrame(user_input)
                        
                        # Ensure correct feature order
                        user_df = user_df[st.session_state['feature_names']]
                        
                        # Make prediction
                        default_prob = st.session_state['default_model'].predict_proba(user_df)[0][1]
                        
                        # Display result
                        risk_level = "High" if default_prob > 0.5 else "Moderate" if default_prob > 0.3 else "Low"
                        risk_color = "#ff0000" if default_prob > 0.5 else "#ffa500" if default_prob > 0.3 else "#00ff00"
                        
                        st.markdown(f"""
                        <div style='{card_style} margin-top:15px;'>
                            <p style='text-align:center;'><b>Default Risk:</b> <span style='color:{risk_color};'>{risk_level}</span></p>
                            <p style='text-align:center;'>Probability: {default_prob:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Recommendations based on risk level
                        st.markdown("<div class='mario-section' style='font-size: 1rem; margin-top: 15px;'>Recommendations</div>", unsafe_allow_html=True)
                        
                        if risk_level == "High":
                            recommendations = [
                                "Reduce your debt-to-income ratio by paying down existing debt",
                                "Increase your savings to build a financial buffer",
                                "Consider debt consolidation to simplify payments",
                                "Avoid taking on new debt until your situation improves"
                            ]
                        elif risk_level == "Moderate":
                            recommendations = [
                                "Continue making on-time payments to all creditors",
                                "Work on reducing your highest-interest debt first",
                                "Build up emergency savings to at least 3 months of expenses",
                                "Be cautious about taking on additional debt"
                            ]
                        else:
                            recommendations = [
                                "Maintain your good financial habits",
                                "Consider opportunities to invest excess savings",
                                "Review your credit report regularly to ensure accuracy",
                                "You're in a good position to consider favorable loan terms if needed"
                            ]
                        
                        for rec in recommendations:
                            st.markdown(f"<div style='{card_style} margin:5px 0; padding:10px;'>🍄 {rec}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error predicting default risk: {str(e)}")

# -------------- Lifestyle Analyzer ----------------
elif page == "🔮 Lifestyle Analyzer":
    st.markdown("<h1 class='mario-header'>🔮 Lifestyle Analyzer</h1>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="{card_style}">
        <h3 class='mario-section'>Understand Your Spending Patterns</h3>
        <p>Upload your lifestyle data or use sample data to analyze your spending habits and get a lifestyle class assignment.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your CSV (Groceries, Dining, Shopping, Utilities, Entertainment, Savings)", type=['csv'])
    use_sample = st.button("Use Sample Data")

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            validate_lifestyle_csv(df)
            st.success("File loaded and validated!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    elif use_sample:
        df = generate_lifestyle_mock_data()
        st.success("Sample data loaded!")

    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Preprocessing
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled, columns=df.columns)

        # Clustering
        n_clusters = 3 if len(df) >= 3 else len(df)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_df)
        df['Cluster'] = cluster_labels
        scaled_df['Cluster'] = cluster_labels

        # Show cluster assignment for the last (most recent) user
        user_index = -1
        lifestyle_info = get_lifestyle_description(cluster_labels[user_index], n_clusters)
        st.markdown(f"**Lifestyle Class:** {lifestyle_info['label']}")
        st.markdown(f"_{lifestyle_info['description']}_")

        # Radar chart
        fig = create_lifestyle_radar_chart(df, cluster_labels, user_index)
        fig = style_mario_visualization(fig, character_color)
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart of spending
        st.subheader("Your Spending Breakdown")
        st.bar_chart(df.drop('Cluster', axis=1).iloc[user_index])

        # Option to download results
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Results as CSV", csv, "lifestyle_analysis.csv", "text/csv")
    else:
        st.info("Upload a file or use sample data to begin lifestyle analysis.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; font-size: 0.8rem;">
    Super Finance Bros - A Mario-themed finance toolkit for data visualization and analysis
    <br>Character: {st.session_state.selected_character} | {CHARACTERS[st.session_state.selected_character]['icon']}
</div>
""", unsafe_allow_html=True)

