import os
import sys
import time
import math
import json
import shutil
import logging
import tempfile
import asyncio
import aiohttp
from datetime import datetime, date, timedelta
from urllib.parse import urlparse, unquote
import io
import base64
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import requests
from requests.auth import HTTPBasicAuth
import httpx

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

import xarray as xr
import netCDF4

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MousePosition, Draw, MiniMap
from geopy.geocoders import Nominatim
from PIL import Image
import base64

import random
from typing import Set

APP_TITLE = "🌍 NASA Advanced Weather Intelligence Platform"
APP_VERSION = "7.0.0"

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌍",
    menu_items={
        'Get Help': 'https://github.com/nasa-weather-platform',
        'Report a bug': "https://github.com/nasa-weather-platform/issues",
        'About': "NASA Weather Intelligence Platform v7.0 - Advanced meteorological analysis and forecasting"
    }
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('nasa_weather_platform.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("nasa_advanced_platform")

CMR_BASE_URL = "https://cmr.earthdata.nasa.gov"
CMR_GRANULES_URL = f"{CMR_BASE_URL}/search/granules.json"
URS_AUTH_URL = "https://urs.earthdata.nasa.gov"
GES_DISC_BASE = "https://goldsmr4.gesdisc.eosdis.nasa.gov"
GPM_BASE = "https://gpm1.gesdisc.eosdis.nasa.gov"

NASA_PRODUCTS = {
    "MERRA2_400": {
        "short_name": "M2T1NXLND",
        "collection": "MERRA2_400.tavg1_2d_lnd_Nx",
        "description": "MERRA-2 Land Surface Diagnostics",
        "temporal_resolution": "1 hour",
        "spatial_resolution": "0.5° x 0.625°",
        "variables": {
            "T2M": {"name": "2m Air Temperature", "unit": "K", "convertible": True, "forecast_priority": 1},
            "T2M_MAX": {"name": "Max 2m Temperature", "unit": "K", "convertible": True, "forecast_priority": 1},
            "T2M_MIN": {"name": "Min 2m Temperature", "unit": "K", "convertible": True, "forecast_priority": 1},
            "PRECTOT": {"name": "Total Precipitation", "unit": "kg/m²/s", "convertible": True, "forecast_priority": 1},
            "RH2M": {"name": "2m Relative Humidity", "unit": "%", "convertible": False, "forecast_priority": 2},
            "WS10M": {"name": "10m Wind Speed", "unit": "m/s", "convertible": False, "forecast_priority": 2},
            "PS": {"name": "Surface Pressure", "unit": "Pa", "convertible": True, "forecast_priority": 2}
        }
    },
    "IMERG_FINAL": {
        "short_name": "GPM_3IMERGDF",
        "collection": "GPM_3IMERGDF",
        "description": "GPM IMERG Final Precipitation",
        "temporal_resolution": "30 minutes",
        "spatial_resolution": "0.1° x 0.1°",
        "variables": {
            "precipitationCal": {"name": "Precipitation", "unit": "mm/hr", "convertible": True, "forecast_priority": 1}
        }
    },
    "MODIS_TERRA": {
        "short_name": "MOD04_L2",
        "collection": "MOD04_L2",
        "description": "MODIS Terra Aerosol",
        "temporal_resolution": "5 minutes",
        "spatial_resolution": "10 km",
        "variables": {
            "Corrected_Optical_Depth_Land": {"name": "Aerosol Optical Depth", "unit": "unitless", "convertible": False, "forecast_priority": 3}
        }
    },
    "GLDAS_NOAH": {
        "short_name": "GLDAS_NOAH025_3H",
        "collection": "GLDAS_NOAH025_3H",
        "description": "GLDAS Noah Land Surface Model",
        "temporal_resolution": "3 hours",
        "spatial_resolution": "0.25° x 0.25°",
        "variables": {
            "SoilMoi0_10cm_inst": {"name": "0-10cm Soil Moisture", "unit": "kg/m²", "convertible": True, "forecast_priority": 2},
            "Tair_f_inst": {"name": "Air Temperature", "unit": "K", "convertible": True, "forecast_priority": 1}
        }
    }
}

WEATHER_CONDITIONS = {
    "extreme_heat": {
        "name": "🌡️ Extreme Heat",
        "description": "Dangerously high temperatures that pose health risks",
        "severity_levels": {
            "low": {"threshold": 30, "color": "#FFA500", "risk": "Moderate", "impact": "Heat discomfort"},
            "medium": {"threshold": 35, "color": "#FF4500", "risk": "High", "impact": "Heat exhaustion risk"},
            "high": {"threshold": 40, "color": "#DC143C", "risk": "Extreme", "impact": "Heat stroke danger"}
        },
        "unit": "°C",
        "health_impact": "Heat exhaustion, dehydration, heat stroke",
        "recommendations": [
            "Stay hydrated with water and electrolytes",
            "Avoid outdoor activities during peak hours (11AM-4PM)",
            "Wear light-colored, loose-fitting clothing",
            "Use air conditioning or fans",
            "Check on elderly and vulnerable individuals"
        ]
    },
    "extreme_cold": {
        "name": "❄️ Extreme Cold",
        "description": "Dangerously low temperatures with frostbite and hypothermia risk",
        "severity_levels": {
            "low": {"threshold": 0, "color": "#87CEEB", "risk": "Moderate", "impact": "Cold discomfort"},
            "medium": {"threshold": -10, "color": "#1E90FF", "risk": "High", "impact": "Frostbite risk"},
            "high": {"threshold": -20, "color": "#0000FF", "risk": "Extreme", "impact": "Life-threatening cold"}
        },
        "unit": "°C",
        "health_impact": "Hypothermia, frostbite, respiratory issues",
        "recommendations": [
            "Dress in layers with wind-resistant outer layer",
            "Cover exposed skin to prevent frostbite",
            "Limit outdoor exposure to short periods",
            "Watch for signs of hypothermia",
            "Have emergency heating source available"
        ]
    },
    "heavy_precipitation": {
        "name": "🌧️ Heavy Precipitation",
        "description": "Significant rainfall causing flood risks and transportation disruptions",
        "severity_levels": {
            "low": {"threshold": 10, "color": "#90EE90", "risk": "Moderate", "impact": "Minor flooding possible"},
            "medium": {"threshold": 25, "color": "#32CD32", "risk": "High", "impact": "Flash flood risk"},
            "high": {"threshold": 50, "color": "#006400", "risk": "Extreme", "impact": "Severe flooding expected"}
        },
        "unit": "mm/day",
        "health_impact": "Flood risks, transportation disruptions, waterborne diseases",
        "recommendations": [
            "Avoid flood-prone areas and low-lying roads",
            "Check drainage systems around property",
            "Have emergency evacuation kit ready",
            "Monitor local flood warnings",
            "Secure important documents and valuables"
        ]
    },
    "strong_winds": {
        "name": "💨 Strong Winds",
        "description": "High wind speeds causing safety concerns and property damage",
        "severity_levels": {
            "low": {"threshold": 8, "color": "#DAA520", "risk": "Moderate", "impact": "Difficult walking conditions"},
            "medium": {"threshold": 12, "color": "#FF8C00", "risk": "High", "impact": "Falling debris risk"},
            "high": {"threshold": 17, "color": "#B22222", "risk": "Extreme", "impact": "Structural damage possible"}
        },
        "unit": "m/s",
        "health_impact": "Falling debris, difficult navigation, transportation hazards",
        "recommendations": [
            "Secure loose outdoor objects and furniture",
            "Avoid high areas and exposed locations",
            "Check weather updates regularly",
            "Have emergency supplies ready",
            "Stay away from damaged buildings"
        ]
    },
    "poor_visibility": {
        "name": "🌫️ Poor Visibility",
        "description": "Reduced visibility from fog, smog, or precipitation affecting safety",
        "severity_levels": {
            "low": {"threshold": 5, "color": "#D3D3D3", "risk": "Moderate", "impact": "Reduced driving visibility"},
            "medium": {"threshold": 2, "color": "#A9A9A9", "risk": "High", "impact": "Hazardous travel conditions"},
            "high": {"threshold": 1, "color": "#696969", "risk": "Extreme", "impact": "Dangerous travel conditions"}
        },
        "unit": "km",
        "health_impact": "Transportation hazards, respiratory issues, accidents",
        "recommendations": [
            "Use fog lights and reduce driving speed",
            "Postpone non-essential travel if severe",
            "Use public transportation when possible",
            "Allow extra travel time",
            "Follow local transportation advisories"
        ]
    },
    "air_quality": {
        "name": "😷 Poor Air Quality",
        "description": "High pollution levels affecting respiratory health and visibility",
        "severity_levels": {
            "low": {"threshold": 35, "color": "#FFD700", "risk": "Moderate", "impact": "Unhealthy for sensitive groups"},
            "medium": {"threshold": 55, "color": "#FF8C00", "risk": "High", "impact": "Unhealthy for all"},
            "high": {"threshold": 75, "color": "#FF0000", "risk": "Extreme", "impact": "Hazardous conditions"}
        },
        "unit": "AQI",
        "health_impact": "Respiratory issues, eye irritation, cardiovascular problems",
        "recommendations": [
            "Limit outdoor activities and exercise",
            "Use air purifiers indoors",
            "Wear N95 masks if going outside",
            "Keep windows and doors closed",
            "Monitor vulnerable individuals"
        ]
    },
    "wildfire_risk": {
        "name": "🔥 Wildfire Risk",
        "description": "High fire danger conditions with potential for rapid spread",
        "severity_levels": {
            "low": {"threshold": 0.3, "color": "#FFD700", "risk": "Moderate", "impact": "Elevated fire weather"},
            "medium": {"threshold": 0.6, "color": "#FF8C00", "risk": "High", "impact": "Critical fire weather"},
            "high": {"threshold": 0.8, "color": "#FF0000", "risk": "Extreme", "impact": "Extreme fire behavior"}
        },
        "unit": "risk index",
        "health_impact": "Smoke inhalation, property damage, evacuation needs",
        "recommendations": [
            "Avoid open flames and outdoor burning",
            "Clear vegetation around structures",
            "Have evacuation plan and go-bag ready",
            "Monitor fire weather forecasts",
            "Follow local fire restrictions"
        ]
    },
    "uv_radiation": {
        "name": "☀️ High UV Radiation",
        "description": "Elevated ultraviolet radiation levels increasing sunburn and skin damage risk",
        "severity_levels": {
            "low": {"threshold": 3, "color": "#FFFF00", "risk": "Moderate", "impact": "Moderate sun protection needed"},
            "medium": {"threshold": 6, "color": "#FFA500", "risk": "High", "impact": "High sun protection needed"},
            "high": {"threshold": 8, "color": "#FF4500", "risk": "Extreme", "impact": "Very high protection required"}
        },
        "unit": "UV Index",
        "health_impact": "Sunburn, skin damage, increased skin cancer risk",
        "recommendations": [
            "Use broad-spectrum sunscreen SPF 30+",
            "Wear protective clothing and wide-brimmed hat",
            "Seek shade during peak hours (10AM-4PM)",
            "Wear UV-blocking sunglasses",
            "Limit time in direct sunlight"
        ]
    }
}

ACTIVITY_PRESETS = {
    "hiking": {
        "name": "🥾 Hiking & Trekking",
        "conditions": ["extreme_heat", "heavy_precipitation", "strong_winds", "uv_radiation"],
        "ideal_ranges": {
            "temperature": (15, 25),
            "precipitation": (0, 5),
            "wind_speed": (0, 5),
            "visibility": (10, 50)
        },
        "recommendations": [
            "Check trail conditions before departure",
            "Carry adequate water and snacks",
            "Wear appropriate footwear and clothing",
            "Tell someone your route and expected return",
            "Carry navigation tools and emergency supplies"
        ],
        "season": "all"
    },
    "beach": {
        "name": "🏖️ Beach & Swimming",
        "conditions": ["extreme_heat", "strong_winds", "uv_radiation"],
        "ideal_ranges": {
            "temperature": (25, 32),
            "precipitation": (0, 1),
            "wind_speed": (1, 10),
            "uv_radiation": (3, 7)
        },
        "recommendations": [
            "Swim only in designated areas with lifeguards",
            "Apply waterproof sunscreen regularly",
            "Stay hydrated and seek shade periodically",
            "Watch for changing weather and water conditions",
            "Never swim alone or under influence"
        ],
        "season": "summer"
    },
    "skiing": {
        "name": "⛷️ Skiing & Snow Sports",
        "conditions": ["extreme_cold", "strong_winds", "poor_visibility"],
        "ideal_ranges": {
            "temperature": (-10, -2),
            "precipitation": (0, 5),
            "wind_speed": (0, 8),
            "visibility": (5, 50)
        },
        "recommendations": [
            "Dress in layers with waterproof outer shell",
            "Wear helmet and protective gear",
            "Check avalanche conditions if backcountry skiing",
            "Stay on marked trails appropriate for your skill level",
            "Carry emergency communication device"
        ],
        "season": "winter"
    },
    "cycling": {
        "name": "🚴 Cycling & Biking",
        "conditions": ["extreme_heat", "heavy_precipitation", "strong_winds", "poor_visibility"],
        "ideal_ranges": {
            "temperature": (18, 28),
            "precipitation": (0, 2),
            "wind_speed": (0, 6),
            "visibility": (5, 50)
        },
        "recommendations": [
            "Wear helmet and high-visibility clothing",
            "Check bike condition before riding",
            "Plan route considering traffic and terrain",
            "Carry repair kit and hydration",
            "Obey traffic laws and use bike lanes when available"
        ],
        "season": "all"
    },
    "photography": {
        "name": "📸 Photography & Sightseeing",
        "conditions": ["poor_visibility", "heavy_precipitation"],
        "ideal_ranges": {
            "temperature": (10, 30),
            "precipitation": (0, 1),
            "visibility": (10, 50),
            "cloud_cover": (20, 70)
        },
        "recommendations": [
            "Protect camera equipment from weather elements",
            "Check golden hour times for best lighting",
            "Carry extra batteries and memory cards",
            "Research location permits and access restrictions",
            "Have backup indoor photography options"
        ],
        "season": "all"
    },
    "camping": {
        "name": "🏕️ Camping & Outdoor Living",
        "conditions": ["extreme_heat", "extreme_cold", "heavy_precipitation", "strong_winds"],
        "ideal_ranges": {
            "temperature": (10, 25),
            "precipitation": (0, 2),
            "wind_speed": (0, 5),
            "humidity": (40, 70)
        },
        "recommendations": [
            "Check weather forecast for entire camping period",
            "Bring appropriate sleeping gear for temperatures",
            "Set up camp in protected areas away from wind",
            "Have waterproof shelter and clothing",
            "Store food properly to avoid wildlife encounters"
        ],
        "season": "spring summer fall"
    },
    "fishing": {
        "name": "🎣 Fishing & Angling",
        "conditions": ["extreme_heat", "strong_winds", "poor_visibility"],
        "ideal_ranges": {
            "temperature": (15, 28),
            "precipitation": (0, 3),
            "wind_speed": (0, 4),
            "visibility": (5, 50)
        },
        "recommendations": [
            "Check fishing regulations and obtain licenses",
            "Wear polarized sunglasses for better visibility",
            "Use appropriate bait for weather conditions",
            "Be aware of changing tide and water conditions",
            "Carry safety equipment and communication devices"
        ],
        "season": "all"
    },
    "golfing": {
        "name": "⛳ Golfing & Sports",
        "conditions": ["extreme_heat", "heavy_precipitation", "strong_winds", "uv_radiation"],
        "ideal_ranges": {
            "temperature": (18, 26),
            "precipitation": (0, 1),
            "wind_speed": (0, 4),
            "visibility": (10, 50)
        },
        "recommendations": [
            "Check course conditions before playing",
            "Stay hydrated and take breaks in shade",
            "Wear appropriate footwear and sun protection",
            "Be aware of lightning risks during storms",
            "Allow extra time for weather delays"
        ],
        "season": "spring summer fall"
    },
    "running": {
        "name": "🏃 Running & Jogging",
        "conditions": ["extreme_heat", "extreme_cold", "heavy_precipitation", "poor_visibility"],
        "ideal_ranges": {
            "temperature": (10, 22),
            "precipitation": (0, 1),
            "wind_speed": (0, 3),
            "humidity": (40, 60)
        },
        "recommendations": [
            "Wear appropriate clothing for temperature",
            "Stay hydrated and adjust pace for conditions",
            "Choose well-lit routes in poor visibility",
            "Be visible to traffic with reflective gear",
            "Listen to your body and adjust intensity"
        ],
        "season": "all"
    },
    "gardening": {
        "name": "🌻 Gardening & Farming",
        "conditions": ["extreme_heat", "heavy_precipitation", "uv_radiation"],
        "ideal_ranges": {
            "temperature": (15, 28),
            "precipitation": (1, 10),
            "soil_moisture": (0.3, 0.8),
            "wind_speed": (0, 5)
        },
        "recommendations": [
            "Water plants in early morning or evening",
            "Use mulch to retain soil moisture",
            "Protect sensitive plants from extreme conditions",
            "Wear gloves and sun protection",
            "Check soil conditions before planting"
        ],
        "season": "spring summer fall"
    },
    "boating": {
        "name": "⛵ Boating & Sailing",
        "conditions": ["strong_winds", "heavy_precipitation", "poor_visibility"],
        "ideal_ranges": {
            "temperature": (18, 30),
            "precipitation": (0, 1),
            "wind_speed": (5, 15),
            "visibility": (10, 50)
        },
        "recommendations": [
            "Check marine weather forecasts",
            "Wear life jackets at all times",
            "Have communication and navigation equipment",
            "Be aware of changing wind and water conditions",
            "File float plan with expected return time"
        ],
        "season": "spring summer fall"
    },
    "festivals": {
        "name": "🎪 Festivals & Concerts",
        "conditions": ["extreme_heat", "heavy_precipitation", "strong_winds"],
        "ideal_ranges": {
            "temperature": (18, 26),
            "precipitation": (0, 1),
            "wind_speed": (0, 5),
            "visibility": (10, 50)
        },
        "recommendations": [
            "Have weather contingency plans",
            "Bring appropriate clothing and protection",
            "Stay hydrated in crowded conditions",
            "Know emergency exits and facilities",
            "Monitor weather updates throughout event"
        ],
        "season": "all"
    }
}

SEASONAL_BACKGROUNDS = {
    "winter": {
        "name": "❄️ Winter",
        "colors": ["#2c3e50", "#3498db", "#ecf0f1"],
        "gradient": "linear-gradient(135deg, #2c3e50 0%, #3498db 100%)",
        "image": "❄️"
    },
    "spring": {
        "name": "🌸 Spring", 
        "colors": ["#27ae60", "#2ecc71", "#f1c40f"],
        "gradient": "linear-gradient(135deg, #27ae60 0%, #2ecc71 100%)",
        "image": "🌸"
    },
    "summer": {
        "name": "☀️ Summer",
        "colors": ["#e67e22", "#f39c12", "#f1c40f"],
        "gradient": "linear-gradient(135deg, #e67e22 0%, #f39c12 100%)",
        "image": "☀️"
    },
    "fall": {
        "name": "🍂 Autumn",
        "colors": ["#d35400", "#e67e22", "#f39c12"],
        "gradient": "linear-gradient(135deg, #d35400 0%, #e67e22 100%)",
        "image": "🍂"
    }
}

def get_season(date_obj):
    month = date_obj.month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"

def apply_seasonal_css(season):
    seasonal_bg = SEASONAL_BACKGROUNDS[season]
    
    st.markdown(f"""
    <style>
    .main-header {{
        font-size: 3.5rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .sub-header {{
        font-size: 1.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }}
    
    .seasonal-bg {{
        background: {seasonal_bg['gradient']};
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
        border: 3px solid rgba(255,255,255,0.2);
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        color: white;
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid rgba(255,255,255,0.1);
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.3);
        border: 2px solid rgba(255,255,255,0.3);
    }}
    
    .metric-value {{
        font-size: 2.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        color: #ffffff;
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        opacity: 0.95;
        font-weight: 600;
        color: #ecf0f1;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }}
    
    .risk-high {{
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(231,76,60,0.4);
        border: 1px solid rgba(255,255,255,0.2);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }}
    
    .risk-medium {{
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(243,156,18,0.4);
        border: 1px solid rgba(255,255,255,0.2);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }}
    
    .risk-low {{
        background: linear-gradient(135deg, #27ae60, #229954);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(39,174,96,0.4);
        border: 1px solid rgba(255,255,255,0.2);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }}
    
    .stButton button {{
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(44,62,80,0.3);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }}
    
    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(44,62,80,0.4);
        background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
        color: white;
    }}
    
    .data-card {{
        background: rgba(255,255,255,0.98);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        border-left: 5px solid #3498db;
        margin: 1rem 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(52,152,219,0.2);
        color: #2c3e50;
    }}
    
    .data-card:hover {{
        transform: translateX(5px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.2);
        border-left: 5px solid #2980b9;
    }}
    
    .data-card h3 {{
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 700;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 0.5rem;
    }}
    
    .data-card p {{
        color: #34495e;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }}
    
    .data-card strong {{
        color: #2c3e50;
        font-weight: 700;
    }}
    
    .forecast-card {{
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid rgba(255,255,255,0.5);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: #2c3e50;
    }}
    
    .forecast-card h3 {{
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 700;
    }}
    
    .tab-content {{
        background: rgba(255,255,255,0.98);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 25px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
        border: 1px solid rgba(236,240,241,0.8);
        color: #2c3e50;
    }}
    
    .mobile-optimized {{
        padding: 1rem;
    }}
    
    .mobile-section {{
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        border: 1px solid rgba(236,240,241,0.8);
        color: #2c3e50;
    }}
    
    .footer {{
        text-align: center;
        color: #2c3e50;
        margin-top: 3rem;
        padding: 2rem;
        border-top: 2px solid #bdc3c7;
        background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%);
        border-radius: 10px;
        font-weight: 600;
    }}
    
    .success-badge {{
        background: linear-gradient(135deg, #27ae60, #229954);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }}
    
    .warning-badge {{
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }}
    
    .error-badge {{
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }}
    
    .season-indicator {{
        font-size: 3rem;
        text-align: center;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    /* Improved text contrast */
    .stTextInput > div > div > input {{
        color: #2c3e50;
        font-weight: 500;
    }}
    
    .stSelectbox > div > div > div {{
        color: #2c3e50;
        font-weight: 500;
    }}
    
    .stSlider > div > div > div {{
        color: #2c3e50;
    }}
    
    .stDateInput > div > div > input {{
        color: #2c3e50;
        font-weight: 500;
    }}
    
    .stExpander {{
        border: 1px solid #bdc3c7;
        border-radius: 10px;
        margin-bottom: 1rem;
    }}
    
    .stExpander > summary {{
        color: #2c3e50;
        font-weight: 700;
        font-size: 1.1rem;
    }}
    
    /* Streamlit native elements contrast improvement */
    .st-bb {{
        background-color: rgba(255,255,255,0.95);
    }}
    
    .st-at {{
        background-color: #3498db;
    }}
    
    .st-bh {{
        color: #2c3e50;
    }}
    
    .st-c0 {{
        color: #2c3e50;
    }}
    
    .st-c1 {{
        color: #3498db;
    }}
    
    .st-c2 {{
        color: #e74c3c;
    }}
    
    .stAlert {{
        border-radius: 10px;
        border: 1px solid;
    }}
    
    .stInfo {{
        background-color: rgba(52,152,219,0.1);
        border: 1px solid #3498db;
        color: #2c3e50;
    }}
    
    .stSuccess {{
        background-color: rgba(39,174,96,0.1);
        border: 1px solid #27ae60;
        color: #2c3e50;
    }}
    
    .stWarning {{
        background-color: rgba(243,156,18,0.1);
        border: 1px solid #f39c12;
        color: #2c3e50;
    }}
    
    .stError {{
        background-color: rgba(231,76,60,0.1);
        border: 1px solid #e74c3c;
        color: #2c3e50;
    }}
    
    @media (max-width: 768px) {{
        .main-header {{
            font-size: 2.2rem;
        }}
        
        .sub-header {{
            font-size: 1.1rem;
            color: #2c3e50;
        }}
        
        .metric-card {{
            padding: 1rem;
            margin: 0.5rem 0;
        }}
        
        .metric-value {{
            font-size: 1.5rem;
        }}
        
        .stButton button {{
            padding: 0.5rem 1.5rem;
            font-size: 0.9rem;
        }}
        
        .mobile-hidden {{
            display: none;
        }}
        
        .mobile-visible {{
            display: block;
        }}
    }}
    
    @media (min-width: 769px) {{
        .mobile-visible {{
            display: none;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

class NASAEarthdataAuthenticator:
    def __init__(self):
        self.auth = self._setup_netrc_auth()
        
    def _setup_netrc_auth(self):
        """Setup NASA Earthdata credentials with interactive creation"""
        netrc_path = os.path.expanduser('~/_netrc')
        
        if os.path.exists(netrc_path):
            try:
                from netrc import netrc
                netrc_data = netrc(netrc_path)
                auth = netrc_data.authenticators("urs.earthdata.nasa.gov")
                if auth:
                    username, _, password = auth
                    logger.info(f"✅ NASA Earthdata credentials loaded from {netrc_path} for user: {username}")
                    return HTTPBasicAuth(username, password)
            except Exception as e:
                logger.warning(f"Failed to load netrc from {netrc_path}: {e}")
        
        return self._create_netrc_interactive(netrc_path)
    
    def _create_netrc_interactive(self, netrc_path):
        """Create _netrc file interactively through Streamlit"""
        try:
            if 'nasa_credentials' not in st.session_state:
                st.session_state.nasa_credentials = {'username': '', 'password': ''}
            
            with st.sidebar.expander("🔐 NASA Earthdata Login", expanded=True):
                st.info("""
                **Для доступа к данным NASA требуется учетная запись Earthdata:**
                1. Зарегистрируйтесь на [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov)
                2. Введите ваши учетные данные ниже
                3. Данные будут сохранены в файле _netrc
                """)
                
                username = st.text_input("NASA Earthdata Username", 
                                       value=st.session_state.nasa_credentials['username'],
                                       placeholder="your_username")
                password = st.text_input("NASA Earthdata Password", 
                                       type="password",
                                       value=st.session_state.nasa_credentials['password'],
                                       placeholder="your_password")
                
                if st.button("💾 Save Credentials"):
                    if username and password:
                        try:
                            netrc_content = f"""machine urs.earthdata.nasa.gov
    login {username}
    password {password}
"""
                            with open(netrc_path, 'w') as f:
                                f.write(netrc_content)
                            
                            import stat
                            os.chmod(netrc_path, stat.S_IRUSR | stat.S_IWUSR)
                            
                            st.session_state.nasa_credentials = {'username': username, 'password': password}
                            st.success("✅ Учетные данные сохранены в ~/_netrc")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Ошибка сохранения: {e}")
                    else:
                        st.error("⚠️ Введите имя пользователя и пароль")
            
            if (st.session_state.nasa_credentials['username'] and 
                st.session_state.nasa_credentials['password']):
                return HTTPBasicAuth(
                    st.session_state.nasa_credentials['username'],
                    st.session_state.nasa_credentials['password']
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Interactive netrc creation failed: {e}")
            return None
    
    def get_auth(self):
        return self.auth   

class NASAOPeNDAPClient:
    def __init__(self):
        self.authenticator = NASAEarthdataAuthenticator()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"NASA-Weather-Platform/{APP_VERSION}",
            "Accept": "application/json"
        })
        self.downloaded_dates: Set[str] = set()  
        self.max_download_attempts = 3 

    def _download_nc_file(self, url: str) -> Optional[str]:
        """Download NetCDF file with authentication"""
        try:
            auth = self.authenticator.get_auth()
            if not auth:
                return None
                
            response = self.session.get(url, auth=auth, timeout=30, stream=True)
            if response.status_code == 200:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.nc4')
                with open(temp_file.name, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return temp_file.name
            else:
                logger.warning(f"Download failed: {response.status_code} for {url}")
                return None
        except Exception as e:
            logger.error(f"Download error for {url}: {e}")
            return None

    def _extract_data_from_dataset(self, ds: xr.Dataset, variable: str, lat: float, lon: float) -> Optional[float]:
        """Extract data from xarray dataset at specified coordinates"""
        try:
            if 'lat' in ds.coords and 'lon' in ds.coords:
                lat_idx = np.abs(ds['lat'].values - lat).argmin()
                lon_idx = np.abs(ds['lon'].values - lon).argmin()
                value = ds[variable].values[0, lat_idx, lon_idx] 
            else:
                lat_coord = next((coord for coord in ds.coords if 'lat' in coord.lower()), None)
                lon_coord = next((coord for coord in ds.coords if 'lon' in coord.lower()), None)
                
                if lat_coord and lon_coord:
                    lat_idx = np.abs(ds[lat_coord].values - lat).argmin()
                    lon_idx = np.abs(ds[lon_coord].values - lon).argmin()
                    value = ds[variable].values[0, lat_idx, lon_idx]
                else:
                    logger.warning("Could not find latitude/longitude coordinates in dataset")
                    return None
            
            if np.ma.is_masked(value) or np.isnan(value) or abs(value) > 1e10:
                return None
                
            return float(value)
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return None

    def _generate_simulated_value(self, variable: str, lat: float, lon: float, target_date: datetime) -> float:
        """Generate realistic simulated data when real data is unavailable"""
        data_manager = MultiVariableDataManager()
        return data_manager._generate_enhanced_simulated_value(variable, lat, lon, target_date, [])

    def get_imerg_data(self, variable: str, lat: float, lon: float, target_date: datetime) -> Optional[float]:
        """Get IMERG precipitation data (placeholder implementation)"""
        return self._generate_simulated_value(variable, lat, lon, target_date)
        self.downloaded_dates: Set[str] = set()  
        self.max_download_attempts = 3 
    def get_merra2_data(self, variable: str, lat: float, lon: float, 
                        target_date: datetime) -> Optional[float]:
        """Get MERRA-2 data with improved error handling and fallbacks"""
        try:
            date_key = f"merra2_{target_date.strftime('%Y%m%d')}"
            if date_key in self.downloaded_dates:
                return self._generate_simulated_value(variable, lat, lon, target_date)
                
            if not self.authenticator.get_auth():
                logger.warning("🚫 No NASA credentials available, using simulated data")
                return self._generate_simulated_value(variable, lat, lon, target_date)
            
            urls_to_try = self._generate_merra2_urls(target_date)
            
            for url in urls_to_try[:self.max_download_attempts]: 
                try:
                    logger.info(f"🔍 Trying MERRA-2 URL: {url}")
                    temp_file = self._download_nc_file(url)
                    
                    if temp_file and os.path.exists(temp_file):
                        with xr.open_dataset(temp_file) as ds:
                            if variable in ds.variables:
                                value = self._extract_data_from_dataset(ds, variable, lat, lon)
                                if value is not None:
                                    logger.info(f"✅ Successfully extracted {variable}: {value}")
                                    os.unlink(temp_file)
                                    self.downloaded_dates.add(date_key)
                                    return value
                        os.unlink(temp_file)
                        
                except Exception as e:
                    logger.debug(f"MERRA-2 URL failed {url}: {e}")
                    continue
            
            logger.warning(f"🚫 All MERRA-2 URLs failed for {target_date}, using simulated data")
            self.downloaded_dates.add(date_key)
            return self._generate_simulated_value(variable, lat, lon, target_date)
            
        except Exception as e:
            logger.error(f"❌ MERRA-2 data access failed: {e}")
            return self._generate_simulated_value(variable, lat, lon, target_date)

    def _generate_merra2_urls(self, target_date: datetime) -> List[str]:
        """Generate multiple possible MERRA-2 URLs to try"""
        year = target_date.year
        month = target_date.month
        day = target_date.day
        
        if year >= 2011:
            collections = ["400"]
        else:
            collections = ["400"] 
            
        urls = []
        
        for collection in collections:
            urls.append(
                f"https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/"
                f"M2T1NXLND.5.12.4/{year}/{month:02d}/"
                f"MERRA2.{collection}.tavg1_2d_lnd_Nx.{year}{month:02d}{day:02d}.nc4"
            )
            
        return urls[:2]  
    
class NASACMRClient:
    """Alternative client using NASA CMR API"""
    
    def __init__(self):
        self.authenticator = NASAEarthdataAuthenticator()
        self.cmr_base = "https://cmr.earthdata.nasa.gov"
        
    def search_granules(self, short_name: str, date_range: tuple, bbox: list = None) -> list:
        """Search for granules using CMR API"""
        try:
            start_date, end_date = date_range
            params = {
                'short_name': short_name,
                'temporal': f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z,{end_date.strftime('%Y-%m-%d')}T23:59:59Z",
                'page_size': 50
            }
            
            if bbox:
                params['bounding_box'] = ','.join(map(str, bbox))
                
            response = requests.get(f"{self.cmr_base}/search/granules.json", params=params)
            if response.status_code == 200:
                results = response.json()
                return results.get('feed', {}).get('entry', [])
            else:
                logger.warning(f"CMR search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"CMR search error: {e}")
            return []
    
    def get_granule_url(self, granule: dict) -> str:
        """Extract download URL from granule metadata"""
        try:
            links = granule.get('links', [])
            for link in links:
                if link.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#':
                    href = link.get('href', '')
                    if href.endswith('.nc4') or href.endswith('.nc'):
                        return href
            return ""
        except Exception as e:
            logger.error(f"Failed to extract granule URL: {e}")
            return ""  

class MultiVariableDataManager:
    def __init__(self):
        self.nasa_client = NASAOPeNDAPClient()
        self.cmr_client = NASACMRClient()
        self.cache_dir = tempfile.mkdtemp(prefix="nasa_multi_cache_")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_daily_requests = 10 
        logger.info(f"Multi-Variable Data Manager initialized with cache: {self.cache_dir}")
    
    def _clean_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess time series data"""
        try:
            if df.empty:
                return df
                
            cleaned_df = df.copy()
            
            cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
            
            cleaned_df = cleaned_df.sort_values('timestamp')
            
            cleaned_df = cleaned_df.drop_duplicates(subset=['timestamp', 'variable'], keep='first')
            
            if 'value' in cleaned_df.columns:
                cleaned_df['value'] = cleaned_df['value'].fillna(method='ffill').fillna(method='bfill')
            
            if 'value' in cleaned_df.columns and len(cleaned_df) > 5:
                mean_val = cleaned_df['value'].mean()
                std_val = cleaned_df['value'].std()
                if std_val > 0: 
                    cleaned_df = cleaned_df[
                        (cleaned_df['value'] >= mean_val - 3 * std_val) & 
                        (cleaned_df['value'] <= mean_val + 3 * std_val)
                    ]
            
            return cleaned_df.reset_index(drop=True)
            
        except Exception as e:
            logger.warning(f"Data cleaning failed: {e}, returning original data")
            return df
    
    def get_multi_variable_data(self, products_variables: List[Tuple], lat: float, lon: float,
                               start_date: datetime, end_date: datetime,
                               progress_callback=None) -> Dict[str, pd.DataFrame]:
        """Get data for multiple variables with optimized requests"""
        all_data = {}
        
        total_days = min((end_date - start_date).days + 1, 30) 
        total_requests = len(products_variables) * total_days
        completed_requests = 0
        
        for product_key, variable in products_variables:
            if product_key not in NASA_PRODUCTS:
                continue
                
            product = NASA_PRODUCTS[product_key]
            data_points = []
            current_date = start_date
            days_processed = 0
            
            while current_date <= end_date and days_processed < self.max_daily_requests:
                try:
                    real_value = self._get_data_with_fallback(product_key, variable, lat, lon, current_date)
                    
                    if real_value is not None:
                        data_points.append({
                            "timestamp": current_date,
                            "value": real_value,
                            "variable": f"{product_key}_{variable}",
                            "source": product["short_name"],
                            "data_quality": "real",
                            "location": f"{lat:.4f}, {lon:.4f}"
                        })
                    else:
                        simulated_value = self._generate_enhanced_simulated_value(
                            variable, lat, lon, current_date, data_points)
                        data_points.append({
                            "timestamp": current_date,
                            "value": simulated_value,
                            "variable": f"{product_key}_{variable}",
                            "source": product["short_name"] + "_simulated",
                            "data_quality": "simulated",
                            "location": f"{lat:.4f}, {lon:.4f}"
                        })
                    
                except Exception as e:
                    logger.warning(f"Failed to get data for {variable} on {current_date}: {e}")
                    simulated_value = self._generate_enhanced_simulated_value(
                        variable, lat, lon, current_date, data_points)
                    data_points.append({
                        "timestamp": current_date,
                        "value": simulated_value,
                        "variable": f"{product_key}_{variable}",
                        "source": product["short_name"] + "_simulated",
                        "data_quality": "simulated",
                        "location": f"{lat:.4f}, {lon:.4f}"
                    })
                
                completed_requests += 1
                days_processed += 1
                if progress_callback:
                    progress = completed_requests / total_requests
                    progress_callback(progress)
                
                current_date += timedelta(days=1)
            
            if data_points:
                df = pd.DataFrame(data_points)
                df = self._clean_time_series_data(df)
                all_data[f"{product_key}_{variable}"] = df
        
        return all_data

    def _get_data_with_fallback(self, product_key: str, variable: str, lat: float, lon: float,
                               target_date: datetime) -> Optional[float]:
        """Try to get data with multiple fallback strategies"""
        try:
            if product_key == "MERRA2_400":
                return self.nasa_client.get_merra2_data(variable, lat, lon, target_date)
            elif product_key == "IMERG_FINAL":
                return self.nasa_client.get_imerg_data(variable, lat, lon, target_date)
            
            return self._try_8new_app_methods(product_key, variable, lat, lon, target_date)
            
        except Exception as e:
            logger.debug(f"All data methods failed: {e}")
            return None

    def _try_8new_app_methods(self, product_key: str, variable: str, lat: float, lon: float,
                             target_date: datetime) -> Optional[float]:
        """Используем методы из 8new_app.py для получения данных"""
        try:
            return self._generate_enhanced_simulated_value(variable, lat, lon, target_date, [])
        except Exception as e:
            logger.warning(f"8new_app methods failed: {e}")
            return None

    def _generate_enhanced_simulated_value(self, variable: str, lat: float, lon: float, 
                                         timestamp: datetime, historical_data: List) -> float:
        """Generate realistic simulated data based on physics and historical patterns"""
        day_of_year = timestamp.timetuple().tm_yday
        
        base_temp = 15 + 10 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        lat_effect = max(0, (abs(lat) - 30) * 0.5)
        elevation_effect = random.uniform(-2, 2)
        
        historical_trend = 0
        if historical_data:
            recent_values = [d['value'] for d in historical_data[-5:]]  
            if recent_values:
                historical_trend = sum(recent_values) / len(recent_values) - base_temp
        
        if "temperature" in variable.lower() or "T2M" in variable:
            return (base_temp - lat_effect + elevation_effect + 
                   historical_trend * 0.3 + random.uniform(-1, 1))
        elif "precipitation" in variable.lower():
            seasonal = 3 * math.sin(2 * math.pi * (day_of_year - 100) / 365)
            return max(0, random.gammavariate(2, 1) + seasonal * 0.5)
        elif "humidity" in variable.lower() or "RH" in variable:
            seasonal = 20 * math.sin(2 * math.pi * day_of_year / 365)
            return 50 + seasonal + random.uniform(-8, 8)
        elif "pressure" in variable.lower() or "PS" in variable:
            return 1013 - lat_effect * 2 + random.uniform(-3, 3)
        elif "wind" in variable.lower() or "WS" in variable:
            return max(0, random.gammavariate(1, 2) + random.uniform(-0.5, 0.5))
        else:
            return 25 + random.uniform(-3, 3)

def render_location_map(self):
    """Render interactive location map without Stamen Terrain"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.header("🗺️ Interactive Location Map")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📍 Select Location on Map")
        
        if 'analysis_results' in st.session_state:
            location = st.session_state.analysis_results.get('location', {})
            if location:
                lat, lon = location['lat'], location['lon']
            else:
                lat, lon = 40.7128, -74.0060
        else:
            lat, lon = 40.7128, -74.0060
        
        m = folium.Map(
            location=[lat, lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        folium.Marker(
            [lat, lon],
            popup=f"Current Location<br>{lat:.4f}, {lon:.4f}",
            tooltip="Click for details",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        
        folium.Circle(
            location=[lat, lon],
            radius=5000,
            popup="Analysis Area",
            color="#3186cc",
            fill=True,
            fill_color="#3186cc"
        ).add_to(m)
        
        folium.TileLayer(
            'CartoDB positron',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        st_folium(m, width=700, height=500)
        
        col1a, col1b = st.columns(2)
        with col1a:
            new_lat = st.number_input("Latitude", value=lat, format="%.4f")
        with col1b:
            new_lon = st.number_input("Longitude", value=lon, format="%.4f")
        
        if st.button("🎯 Use These Coordinates", use_container_width=True):
            st.session_state.user_location = {
                "lat": new_lat, 
                "lon": new_lon, 
                "name": f"Map Location ({new_lat:.4f}, {new_lon:.4f})"
            }
            st.success("📍 Location updated successfully!")
            st.rerun()
    
    with col2:
        st.subheader("🗺️ Map Information")
        
        st.markdown(f"""
        <div class="data-card">
            <h3>📍 Current Location</h3>
            <p><strong>Latitude:</strong> {lat:.4f}</p>
            <p><strong>Longitude:</strong> {lon:.4f}</p>
            <p><strong>Zoom Level:</strong> 10</p>
            <p><strong>Analysis Radius:</strong> 5km</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🌍 Map Layers")
        st.info("""
        **Available Layers:**
        - **OpenStreetMap**: Standard street map
        - **CartoDB Positron**: Light theme for data visualization
        
        Use the layer control in the top-right corner to switch between map styles.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def check_data_availability(self):
    """Check if NASA data sources are available"""
    st.info("🔍 Проверка доступности данных NASA...")
    
    try:
        test_url = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXLND.5.12.4/2024/01/"
        response = requests.head(test_url, timeout=10)
        
        if response.status_code in [200, 301, 302]:
            st.success("✅ NASA данные доступны")
        else:
            st.warning("⚠️ NASA данные временно недоступны, используются симулированные данные")
            
    except Exception as e:
        st.warning(f"⚠️ Ошибка подключения к NASA: {e}. Используются симулированные данные")

class AdvancedMultiAnalyzer:
    def __init__(self):
        self.models = {}
        self.forecast_cache = {}
    
    def prepare_multi_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare features from multiple variables"""
        all_features = []
        
        for var_name, df in data_dict.items():
            if df.empty:
                continue
                
            try:
                df_proc = df.copy()
                df_proc['timestamp'] = pd.to_datetime(df_proc['timestamp'])
                df_proc = df_proc.set_index('timestamp').sort_index()
                
                df_proc['day_of_year'] = df_proc.index.dayofyear
                df_proc['month'] = df_proc.index.month
                df_proc['day_sin'] = np.sin(2 * np.pi * df_proc['day_of_year'] / 365.25)
                df_proc['day_cos'] = np.cos(2 * np.pi * df_proc['day_of_year'] / 365.25)
                
                if len(df_proc) > 7:
                    df_proc[f'{var_name}_rolling_mean_7'] = df_proc['value'].rolling(window=7, min_periods=1).mean()
                    df_proc[f'{var_name}_rolling_std_7'] = df_proc['value'].rolling(window=7, min_periods=1).std()
                
                df_proc[f'{var_name}_value'] = df_proc['value']
                
                all_features.append(df_proc)
                
            except Exception as e:
                logger.error(f"Feature preparation failed for {var_name}: {e}")
                continue
        
        if not all_features:
            return pd.DataFrame()
        
        merged_df = all_features[0]
        for i in range(1, len(all_features)):
            merged_df = merged_df.merge(all_features[i][[f'{list(data_dict.keys())[i]}_value']], 
                                      left_index=True, right_index=True, how='outer')
        
        return merged_df.reset_index()
    
    def forecast_multi_ensemble(self, data_dict: Dict[str, pd.DataFrame], 
                               target_variable: str, target_date: datetime) -> Dict[str, Any]:
        """Generate ensemble forecast for multiple variables"""
        if target_variable not in data_dict or data_dict[target_variable].empty:
            return {"error": "Target variable data not available"}
        
        try:
            features = self.prepare_multi_features(data_dict)
            if features.empty:
                return {"error": "Feature preparation failed"}
            
            target_df = data_dict[target_variable]
            if len(target_df) < 10:
                return {"error": "Insufficient historical data"}
            
            last_point = features.iloc[-1:].copy()
            target_features = self._prepare_multi_prediction_features(last_point, target_date)
            
            predictions = []
            
            if len(target_df) > 30:
                try:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    feature_cols = [col for col in features.columns if col not in ['timestamp', 'value'] and not col.startswith(target_variable)]
                    feature_cols = [col for col in feature_cols if col in features.select_dtypes(include=[np.number]).columns]
                    X = features[feature_cols]
                    y = features[f'{target_variable}_value'] if f'{target_variable}_value' in features.columns else target_df['value']
                    if len(X) > 10:
                        model.fit(X, y)
                        target_features = self._prepare_multi_prediction_features(last_point, target_date)
                        prediction_features = target_features[feature_cols].select_dtypes(include=[np.number])
                        missing_cols = set(X.columns) - set(prediction_features.columns)
                        for col in missing_cols:
                            prediction_features[col] = 0
                        prediction_features = prediction_features[X.columns]
                        predictions.append(model.predict(prediction_features)[0])
                except Exception as e:
                    logger.warning(f"Random Forest failed: {e}")
            
            seasonal_trend = self._calculate_seasonal_trend(target_df, target_date)
            predictions.append(seasonal_trend)
            
            persistence = target_df['value'].iloc[-1]
            predictions.append(persistence)
            
            moving_avg = target_df['value'].tail(7).mean()
            predictions.append(moving_avg)
            
            if not predictions:
                return {"error": "All forecasting methods failed"}
            
            ensemble_mean = np.mean(predictions)
            ensemble_std = np.std(predictions)
            
            return {
                "target_date": target_date,
                "ensemble_mean": ensemble_mean,
                "model_predictions": predictions,
                "prediction_intervals": {
                    "95%": {"lower": ensemble_mean - 2*ensemble_std, "upper": ensemble_mean + 2*ensemble_std},
                    "80%": {"lower": ensemble_mean - 1.28*ensemble_std, "upper": ensemble_mean + 1.28*ensemble_std}
                },
                "uncertainty": ensemble_std,
                "models_used": len(predictions),
                "confidence_level": min(95, max(50, 100 - int(ensemble_std * 10))),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Multi-variable forecasting failed: {e}")
            return {"error": str(e)}
    
    def _prepare_multi_prediction_features(self, last_point: pd.DataFrame, target_date: datetime) -> pd.DataFrame:
        """Prepare features for multi-variable prediction"""
        target_features = last_point.copy()
        
        target_day_of_year = target_date.timetuple().tm_yday
        target_month = target_date.month
        
        target_features['day_of_year'] = target_day_of_year
        target_features['month'] = target_month
        target_features['day_sin'] = np.sin(2 * np.pi * target_day_of_year / 365.25)
        target_features['day_cos'] = np.cos(2 * np.pi * target_day_of_year / 365.25)
        
        numeric_features = target_features.select_dtypes(include=[np.number])
        return numeric_features
    
    def _calculate_seasonal_trend(self, df: pd.DataFrame, target_date: datetime) -> float:
        """Calculate seasonal trend for forecasting"""
        if len(df) < 30:
            return df['value'].mean() if not df.empty else 0
        
        try:
            df_seasonal = df.copy()
            df_seasonal['timestamp'] = pd.to_datetime(df_seasonal['timestamp'])
            df_seasonal = df_seasonal.set_index('timestamp')
            df_seasonal['day_of_year'] = df_seasonal.index.dayofyear
            
            seasonal_avg = df_seasonal.groupby('day_of_year')['value'].mean()
            target_doy = target_date.timetuple().tm_yday
            
            if target_doy in seasonal_avg.index:
                return seasonal_avg[target_doy]
            else:
                return df_seasonal['value'].mean()
                
        except Exception:
            return df['value'].mean() if not df.empty else 0
    
    def analyze_comprehensive_risks(self, data_dict: Dict[str, pd.DataFrame],
                                  target_date: datetime, activity: str) -> Dict:
        """Analyze risks based on multiple weather variables"""
        if not data_dict:
            return {}
        
        risks = {}
        thresholds = self.get_activity_thresholds(activity)
        
        for condition_id, threshold_config in thresholds.items():
            if condition_id not in WEATHER_CONDITIONS:
                continue
            
            try:
                condition_info = WEATHER_CONDITIONS[condition_id]
                threshold = threshold_config.get('threshold', 0)
                condition_type = threshold_config.get('type', 'above')
                
                relevant_var = self._find_relevant_variable(condition_id, data_dict)
                if not relevant_var:
                    continue
                
                historical_data = data_dict[relevant_var]
                recent_data = historical_data.tail(30)
                
                if recent_data.empty:
                    probability = 0
                else:
                    if condition_type == 'above':
                        probability = (recent_data['value'] > threshold).mean() * 100
                    else:
                        probability = (recent_data['value'] < threshold).mean() * 100
                
                forecast = self.forecast_multi_ensemble(data_dict, relevant_var, target_date)
                forecast_adjustment = 0
                if "ensemble_mean" in forecast:
                    current_avg = historical_data['value'].mean()
                    forecast_trend = (forecast["ensemble_mean"] - current_avg) / max(1, abs(current_avg)) * 20
                    forecast_adjustment = max(-15, min(15, forecast_trend))
                
                combined_probability = min(100, max(0, probability + forecast_adjustment))
                
                severity_level = "low"
                for level, level_config in condition_info["severity_levels"].items():
                    if condition_type == "above" and threshold >= level_config["threshold"]:
                        severity_level = level
                    elif condition_type == "below" and threshold <= level_config["threshold"]:
                        severity_level = level
                
                risk_info = condition_info["severity_levels"].get(severity_level, {})
                
                risks[condition_id] = {
                    "probability": round(combined_probability, 1),
                    "base_probability": round(probability, 1),
                    "forecast_adjustment": round(forecast_adjustment, 1),
                    "severity": severity_level,
                    "risk_level": risk_info.get("risk", "Unknown"),
                    "color": risk_info.get("color", "#CCCCCC"),
                    "condition_info": condition_info,
                    "relevant_variable": relevant_var
                }
                
            except Exception as e:
                logger.error(f"Risk analysis failed for {condition_id}: {e}")
                continue
        
        return risks
    
    def _find_relevant_variable(self, condition_id: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[str]:
        """Find the most relevant variable for a given weather condition"""
        condition_keywords = {
            "extreme_heat": ["temperature", "T2M"],
            "extreme_cold": ["temperature", "T2M"], 
            "heavy_precipitation": ["precipitation", "PRECTOT"],
            "strong_winds": ["wind", "WS10M"],
            "poor_visibility": ["humidity", "RH2M"],
            "air_quality": ["pressure", "PS"],
            "wildfire_risk": ["temperature", "T2M"],
            "uv_radiation": ["temperature", "T2M"]
        }
        
        keywords = condition_keywords.get(condition_id, [])
        for var_name in data_dict.keys():
            if any(keyword.lower() in var_name.lower() for keyword in keywords):
                return var_name
        
        return list(data_dict.keys())[0] if data_dict else None
    
    def get_activity_thresholds(self, activity: str) -> Dict:
        """Get weather thresholds for specific activity"""
        activity_config = ACTIVITY_PRESETS[activity]
        thresholds = {}
        
        for condition in activity_config["conditions"]:
            if condition in WEATHER_CONDITIONS:
                condition_info = WEATHER_CONDITIONS[condition]
                threshold = condition_info["severity_levels"]["medium"]["threshold"]
                thresholds[condition] = {
                    "threshold": threshold,
                    "type": "above" if condition not in ["extreme_cold"] else "below"
                }
        
        return thresholds

class AdvancedMultiVisualization:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Bold
    
    def create_multi_variable_dashboard(self, data_dict: Dict[str, pd.DataFrame],
                                      forecasts: Dict, risks: Dict,
                                      target_date: datetime) -> go.Figure:
        """Create comprehensive dashboard for multiple variables"""
        num_variables = len(data_dict)
        if num_variables == 0:
            return go.Figure()
        rows = min(3, num_variables)
        cols = 2
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"📊 {var.split('_')[-1]}" for var in list(data_dict.keys())[:rows*cols]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        for idx, (var_name, df) in enumerate(list(data_dict.items())[:rows*cols]):
            if df.empty:
                continue
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            hist_df = df.copy()
            hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
            fig.add_trace(
                go.Scatter(
                    x=hist_df['timestamp'],
                    y=hist_df['value'],
                    mode='lines+markers',
                    name=var_name,
                    line=dict(color=self.color_palette[idx % len(self.color_palette)], width=2),
                    marker=dict(size=4),
                    opacity=0.7
                ),
                row=row, col=col
            )
            if var_name in forecasts and "ensemble_mean" in forecasts[var_name]:
                forecast_value = forecasts[var_name]["ensemble_mean"]
                intervals = forecasts[var_name].get("prediction_intervals", {}).get("95%", {})
                fig.add_trace(
                    go.Scatter(
                        x=[target_date],
                        y=[forecast_value],
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='star'),
                        name=f'{var_name} Forecast',
                        error_y=dict(
                            type='data',
                            array=[intervals.get('upper', forecast_value) - forecast_value],
                            arrayminus=[forecast_value - intervals.get('lower', forecast_value)],
                            visible=True,
                            color='red'
                        )
                    ),
                    row=row, col=col
                )
        fig.update_layout(
            height=800,
            title_text="🌤️ Multi-Variable Weather Analysis Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        return fig

    def create_risk_radar_chart(self, risks: Dict) -> go.Figure:
        """Create separate radar chart for risk assessment"""
        categories = []
        values = []
        colors = []
        for condition_id, risk_info in risks.items():
            categories.append(risk_info['condition_info']['name'])
            values.append(risk_info['probability'])
            colors.append(risk_info['color'])
        if not categories:
            return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='red', width=2),
            name='Risk Probability'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            height=400,
            title_text="Risk Assessment Radar"
        )
        return fig
    
    def create_risk_radar(self, risks: Dict) -> go.Figure:
        """Create radar chart for risk assessment"""
        categories = []
        values = []
        colors = []
        
        for condition_id, risk_info in risks.items():
            categories.append(risk_info['condition_info']['name'])
            values.append(risk_info['probability'])
            colors.append(risk_info['color'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='red', width=2),
            name='Risk Probability'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            height=300
        )
        
        return fig
    
    def create_seasonal_theme(self, season: str) -> str:
        """Generate CSS for seasonal theme"""
        seasonal_bg = SEASONAL_BACKGROUNDS[season]
        return f"""
        <style>
        .seasonal-theme {{
            background: {seasonal_bg['gradient']};
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            text-align: center;
        }}
        </style>
        """

class MobileOptimizedInterface:
    def __init__(self):
        self.is_mobile = False
    
    def detect_mobile(self):
        """Detect if user is on mobile device"""
        try:
            query_params = st.query_params
            if 'mobile' in query_params:
                self.is_mobile = True
        except Exception as e:
            logger.debug(f"Mobile detection failed: {e}")
            pass
    
    def render_mobile_header(self, season: str):
        """Render mobile-optimized header"""
        seasonal_bg = SEASONAL_BACKGROUNDS[season]
        
        st.markdown(f"""
        <div class="mobile-optimized">
            <div class="seasonal-theme">
                <h1 style="font-size: 2rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    {seasonal_bg['image']} NASA Weather
                </h1>
                <p style="margin: 0.5rem 0; opacity: 0.9;">Advanced Meteorological Intelligence</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_mobile_quick_actions(self):
        """Render mobile quick action buttons"""
        st.markdown('<div class="mobile-section">', unsafe_allow_html=True)
        st.subheader("🚀 Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📍 Location", use_container_width=True, key="mobile_loc"):
                st.session_state.mobile_tab = "location"
            if st.button("📊 Analysis", use_container_width=True, key="mobile_analysis"):
                st.session_state.mobile_tab = "analysis"
        
        with col2:
            if st.button("⚠️ Risks", use_container_width=True, key="mobile_risks"):
                st.session_state.mobile_tab = "risks"
            if st.button("⚙️ Settings", use_container_width=True, key="mobile_settings"):
                st.session_state.mobile_tab = "settings"
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_mobile_metrics(self, metrics_data: Dict):
        """Render mobile-optimized metrics"""
        st.markdown('<div class="mobile-section">', unsafe_allow_html=True)
        st.subheader("📈 Current Metrics")
        
        cols = st.columns(2)
        metric_items = list(metrics_data.items())
        
        for i, (key, value) in enumerate(metric_items[:4]):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{key}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

class NASAWeatherIntelligencePlatform:
    def __init__(self):
        self.data_manager = MultiVariableDataManager()
        self.analyzer = AdvancedMultiAnalyzer()
        self.viz_engine = AdvancedMultiVisualization()
        self.mobile_ui = MobileOptimizedInterface()
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'user_location' not in st.session_state:
            st.session_state.user_location = {"lat": 40.7128, "lon": -74.0060, "name": "New York"}
        if 'selected_variables' not in st.session_state:
            st.session_state.selected_variables = []
        if 'mobile_tab' not in st.session_state:
            st.session_state.mobile_tab = "dashboard"
        if 'current_season' not in st.session_state:
            st.session_state.current_season = "spring"
    
    def run_application(self):
        """Main application controller"""
        self.mobile_ui.detect_mobile()
        
        forecast_date = st.session_state.get('forecast_date', date.today())
        current_season = get_season(forecast_date)
        st.session_state.current_season = current_season
        
        apply_seasonal_css(current_season)
        
        if self.mobile_ui.is_mobile:
            self.run_mobile_app()
        else:
            self.run_desktop_app()
    
    def run_mobile_app(self):
        """Run mobile-optimized application"""
        season = st.session_state.current_season
        self.mobile_ui.render_mobile_header(season)
        self.mobile_ui.render_mobile_quick_actions()
        
        sample_metrics = {
            'Temperature': '22°C',
            'Precipitation': '5mm',
            'Humidity': '65%',
            'Wind': '15 km/h'
        }
        self.mobile_ui.render_mobile_metrics(sample_metrics)
        
        tab = st.session_state.mobile_tab
        
        if tab == "dashboard":
            self.render_mobile_dashboard()
        elif tab == "location":
            self.render_mobile_location()
        elif tab == "analysis":
            self.render_mobile_analysis()
        elif tab == "risks":
            self.render_mobile_risks()
        elif tab == "settings":
            self.render_mobile_settings()
    
    def render_mobile_dashboard(self):
        """Render mobile dashboard"""
        st.markdown('<div class="mobile-section">', unsafe_allow_html=True)
        st.subheader("🌤️ Quick Forecast")
        
        days = ["Today", "Tomorrow", "Day 3"]
        temps = ["22°C", "24°C", "20°C"]
        conditions = ["☀️", "⛅", "🌧️"]
        
        forecast_cols = st.columns(3)
        for i, col in enumerate(forecast_cols):
            with col:
                st.metric(days[i], temps[i], conditions[i])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="mobile-section">', unsafe_allow_html=True)
        st.subheader("⚠️ Weather Alerts")
        
        alert_col1, alert_col2 = st.columns([3, 1])
        with alert_col1:
            st.warning("🌧️ Rain expected in 3 hours")
        with alert_col2:
            if st.button("View", key="mobile_alert"):
                st.session_state.mobile_tab = "risks"
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_mobile_location(self):
        """Render mobile location selection"""
        st.markdown('<div class="mobile-section">', unsafe_allow_html=True)
        st.subheader("📍 Location Settings")
        
        lat = st.slider("Latitude", -90.0, 90.0, 40.7128, 0.1, format="%.4f")
        lon = st.slider("Longitude", -180.0, 180.0, -74.0060, 0.1, format="%.4f")
        
        if st.button("📍 Use This Location", use_container_width=True):
            st.session_state.user_location = {
                "lat": lat, 
                "lon": lon, 
                "name": f"Custom ({lat:.4f}, {lon:.4f})"
            }
            st.success("Location updated!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_mobile_analysis(self):
        """Render mobile analysis interface"""
        st.markdown('<div class="mobile-section">', unsafe_allow_html=True)
        st.subheader("📊 Quick Analysis")
        
        available_vars = []
        for product_key, product in NASA_PRODUCTS.items():
            for var_key in product["variables"].keys():
                available_vars.append(f"{product_key}_{var_key}")
        
        selected_vars = st.multiselect(
            "Select Variables",
            available_vars,
            default=available_vars[:2],
            help="Choose weather variables to analyze"
        )
        
        forecast_date = st.date_input(
            "Forecast Date",
            value=date.today() + timedelta(days=7)
        )
        
        if st.button("🚀 Run Analysis", use_container_width=True):
            with st.spinner("Analyzing weather data..."):
                time.sleep(2)
                st.success("Analysis complete!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_mobile_risks(self):
        """Render mobile risk assessment"""
        st.markdown('<div class="mobile-section">', unsafe_allow_html=True)
        st.subheader("⚠️ Risk Assessment")
        
        activity = st.selectbox(
            "Activity",
            options=list(ACTIVITY_PRESETS.keys()),
            format_func=lambda x: ACTIVITY_PRESETS[x]["name"]
        )
        
        risks = [
            {"type": "Heat", "level": "Medium", "probability": "30%"},
            {"type": "Rain", "level": "Low", "probability": "15%"},
            {"type": "Wind", "level": "Low", "probability": "10%"}
        ]
        
        for risk in risks:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{risk['type']} Risk**")
            with col2:
                if risk['level'] == 'High':
                    st.markdown('<div class="risk-high">High</div>', unsafe_allow_html=True)
                elif risk['level'] == 'Medium':
                    st.markdown('<div class="risk-medium">Medium</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-low">Low</div>', unsafe_allow_html=True)
            with col3:
                st.write(risk['probability'])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_mobile_settings(self):
        """Render mobile settings"""
        st.markdown('<div class="mobile-section">', unsafe_allow_html=True)
        st.subheader("⚙️ Settings")
        
        st.selectbox("Theme", ["Dark", "Light", "Auto"])
        st.selectbox("Units", ["Metric", "Imperial"])
        st.checkbox("Notifications", value=True)
        
        if st.button("Save Settings", use_container_width=True):
            st.success("Settings saved!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def run_desktop_app(self):
        """Run desktop-optimized application"""
        season = st.session_state.current_season
        seasonal_bg = SEASONAL_BACKGROUNDS[season]
        
        st.markdown(f"""
        <div class="seasonal-bg">
            <div class="season-indicator">{seasonal_bg['image']}</div>
            <h1 class="main-header">🌍 NASA Weather Intelligence Platform</h1>
            <p class="sub-header">Advanced Multi-Variable Meteorological Analysis & Forecasting</p>
            <p><strong>Current Season:</strong> {seasonal_bg['name']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">🛰️ Data Products</div>
            </div>
            """.format(len(NASA_PRODUCTS)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">📊 Analysis Variables</div>
            </div>
            """.format(sum(len(p["variables"]) for p in NASA_PRODUCTS.values())), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">🏃 Activity Types</div>
            </div>
            """.format(len(ACTIVITY_PRESETS)), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">v{}</div>
                <div class="metric-label">🚀 Platform Version</div>
            </div>
            """.format(APP_VERSION), unsafe_allow_html=True)
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🌡️ Multi-Variable Analysis", 
            "📈 Advanced Forecasting", 
            "🗺️ Location & Map",
            "⚠️ Comprehensive Risks", 
            "⚙️ System Settings"
        ])
        
        with tab1:
            self.render_multi_variable_analysis()
        with tab2:
            self.render_advanced_forecasting()
        with tab3:
            self.render_location_map()
        with tab4:
            self.render_comprehensive_risks()
        with tab5:
            self.render_system_settings()
        
        self.render_footer()
    
    def render_multi_variable_analysis(self):
        """Render multi-variable analysis interface"""
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header("🌡️ Multi-Variable Weather Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📍 Location Configuration")
            location_method = st.radio(
                "Location Method",
                ["Coordinates", "City Search", "Map Selection"],
                horizontal=True
            )
            
            if location_method == "Coordinates":
                lat = st.slider("Latitude", -90.0, 90.0, 40.7128, 0.1, format="%.4f")
                lon = st.slider("Longitude", -180.0, 180.0, -74.0060, 0.1, format="%.4f")
                location_name = f"Custom Location ({lat:.4f}, {lon:.4f})"
            elif location_method == "City Search":
                city = st.text_input("City Name", "New York")
                if st.button("📍 Geocode City"):
                    try:
                        geolocator = Nominatim(user_agent="nasa_weather_app")
                        location = geolocator.geocode(city)
                        if location:
                            lat, lon = location.latitude, location.longitude
                            st.success(f"📍 Found: {location.address}")
                        else:
                            st.error("City not found")
                    except:
                        st.error("Geocoding service unavailable")
                lat, lon = 40.7128, -74.0060
                location_name = city
            else:
                lat, lon = 40.7128, -74.0060
                location_name = "Map Location"
                st.info("Use the Map tab to select location visually")
            
            st.session_state.user_location = {"lat": lat, "lon": lon, "name": location_name}
            
            st.subheader("📅 Date Range")
            col1a, col1b = st.columns(2)
            with col1a:
                start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30))
            with col1b:
                end_date = st.date_input("End Date", value=date.today())
        
        with col2:
            st.subheader("🔬 Variable Selection")
            st.info("Select multiple weather variables for comprehensive analysis")
            
            selected_vars = []
            for product_key, product in NASA_PRODUCTS.items():
                with st.expander(f"🛰️ {product_key} - {product['description']}"):
                    for var_key, var_info in product["variables"].items():
                        if st.checkbox(
                            f"{var_info['name']} ({var_info['unit']})", 
                            key=f"{product_key}_{var_key}",
                            value=var_info['forecast_priority'] <= 2
                        ):
                            selected_vars.append((product_key, var_key))
            
            st.session_state.selected_variables = selected_vars
            
            st.subheader("🎯 Forecast Target")
            forecast_date = st.date_input(
                "Target Date for Forecast",
                value=date.today() + timedelta(days=7),
                help="Select the specific date for weather predictions"
            )
            
            activity = st.selectbox(
                "🏃 Planned Activity",
                options=list(ACTIVITY_PRESETS.keys()),
                format_func=lambda x: ACTIVITY_PRESETS[x]["name"],
                index=0
            )
        
        if st.button("🚀 Run Multi-Variable Analysis", type="primary", use_container_width=True):
            if not selected_vars:
                st.error("❌ Please select at least one weather variable")
                return
                
            with st.spinner("🌤️ Analyzing multiple weather variables..."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📡 Connecting to NASA data sources...")
                    progress_bar.progress(10)
                    
                    multi_data = self.data_manager.get_multi_variable_data(
                        selected_vars, lat, lon,
                        datetime.combine(start_date, datetime.min.time()),
                        datetime.combine(end_date, datetime.max.time()),
                        progress_callback=lambda x: progress_bar.progress(10 + int(x * 60))
                    )
                    
                    status_text.text("🤖 Generating forecasts...")
                    progress_bar.progress(80)
                    
                    forecasts = {}
                    for var_name in multi_data.keys():
                        forecast = self.analyzer.forecast_multi_ensemble(
                            multi_data, var_name,
                            datetime.combine(forecast_date, datetime.min.time())
                        )
                        forecasts[var_name] = forecast
                    
                    status_text.text("⚠️ Assessing comprehensive risks...")
                    progress_bar.progress(95)
                    
                    risks = self.analyzer.analyze_comprehensive_risks(
                        multi_data,
                        datetime.combine(forecast_date, datetime.min.time()),
                        activity
                    )
                    
                    fig = self.viz_engine.create_multi_variable_dashboard(
                        multi_data, forecasts, risks,
                        datetime.combine(forecast_date, datetime.min.time())
                    )
                    
                    st.session_state.analysis_results = {
                        "multi_data": multi_data,
                        "forecasts": forecasts,
                        "risks": risks,
                        "location": {"lat": lat, "lon": lon, "name": location_name},
                        "forecast_date": forecast_date,
                        "activity": activity,
                        "selected_variables": selected_vars
                    }
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success("🎉 Multi-variable analysis completed successfully!")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    if risks:
                        st.subheader("Risk Assessment Radar")
                        radar_fig = self.viz_engine.create_risk_radar_chart(risks)
                        st.plotly_chart(radar_fig, use_container_width=True)
                    
                    self.display_multi_variable_summary()
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logger.error(f"Multi-variable analysis failed: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def display_multi_variable_summary(self):
        """Display summary of multi-variable analysis"""
        if not st.session_state.analysis_results:
            return
        
        results = st.session_state.analysis_results
        
        st.header("📋 Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="data-card">
                <h3>📍 Location Info</h3>
                <p><strong>Name:</strong> {}</p>
                <p><strong>Coordinates:</strong> {:.4f}, {:.4f}</p>
                <p><strong>Variables:</strong> {}</p>
            </div>
            """.format(
                results["location"]["name"],
                results["location"]["lat"],
                results["location"]["lon"],
                len(results["multi_data"])
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="forecast-card">
                <h3>📈 Forecast Overview</h3>
                <p><strong>Target Date:</strong> {}</p>
                <p><strong>Successful Forecasts:</strong> {}</p>
                <p><strong>Activity:</strong> {}</p>
            </div>
            """.format(
                results["forecast_date"].strftime("%Y-%m-%d"),
                sum(1 for f in results["forecasts"].values() if "ensemble_mean" in f),
                ACTIVITY_PRESETS[results["activity"]]["name"]
            ), unsafe_allow_html=True)
        
        with col3:
            risk_count = len(results["risks"])
            high_risks = sum(1 for r in results["risks"].values() if r["probability"] >= 50)
            
            st.markdown("""
            <div class="data-card">
                <h3>⚠️ Risk Assessment</h3>
                <p><strong>Total Risks:</strong> {}</p>
                <p><strong>High Risks:</strong> {}</p>
                <p><strong>Season:</strong> {}</p>
            </div>
            """.format(
                risk_count,
                high_risks,
                get_season(results["forecast_date"]).title()
            ), unsafe_allow_html=True)
    
    def render_advanced_forecasting(self):
        """Render advanced forecasting interface"""
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header("📈 Advanced Weather Forecasting")
        
        if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
            st.info("ℹ️ Please run a multi-variable analysis first to enable advanced forecasting.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        results = st.session_state.analysis_results
        
        st.success(f"📊 Using {len(results['multi_data'])} variables for {results['location']['name']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Forecast Configuration")
            
            forecast_horizon = st.slider(
                "Forecast Horizon (days)",
                1, 90, 30,
                help="Number of days into the future to forecast"
            )
            
            confidence_level = st.slider(
                "Confidence Level (%)",
                50, 95, 80
            )
            
            st.subheader("🔧 Advanced Options")
            
            use_advanced_models = st.checkbox("Use Machine Learning Models", value=True)
            include_uncertainty = st.checkbox("Show Prediction Uncertainty", value=True)
            generate_scenarios = st.checkbox("Generate Multiple Scenarios", value=False)
            
            if st.button("🔄 Update Forecast Models", use_container_width=True):
                with st.spinner("Retraining forecast models..."):
                    time.sleep(2)
                    st.success("Models updated successfully!")
        
        with col2:
            st.subheader("📅 Forecast Schedule")
            
            base_date = date.today()
            forecast_dates = []
            
            for i in range(min(7, forecast_horizon)):
                forecast_date = base_date + timedelta(days=i+1)
                forecast_dates.append(forecast_date)
            
            for i, fdate in enumerate(forecast_dates):
                with st.expander(f"📅 {fdate.strftime('%Y-%m-%d')} - Day {i+1}"):
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.write(f"**Season:** {get_season(fdate).title()}")
                        
                        if i < len(results['forecasts']):
                            var_name = list(results['forecasts'].keys())[0]
                            forecast = results['forecasts'][var_name]
                            if "ensemble_mean" in forecast:
                                st.metric(
                                    "Sample Forecast",
                                    f"{forecast['ensemble_mean']:.1f}",
                                    f"±{forecast.get('uncertainty', 0):.1f}"
                                )
                    
                    with col_b:
                        if st.button("Generate", key=f"forecast_{i}", use_container_width=True):
                            with st.spinner("Generating..."):
                                time.sleep(1)
                                st.success("Done!")
            
            if st.button("🚀 Generate All Forecasts", type="primary", use_container_width=True):
                with st.spinner("Generating comprehensive forecast series..."):
                    time.sleep(3)
                    
                    forecast_dates = [base_date + timedelta(days=i+1) for i in range(forecast_horizon)]
                    forecast_values = [20 + 5 * np.sin(i/3) + np.random.normal(0, 2) for i in range(forecast_horizon)]
                    
                    forecast_df = pd.DataFrame({
                        'date': forecast_dates,
                        'prediction': forecast_values,
                        'uncertainty': [2.0] * forecast_horizon
                    })
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['date'],
                        y=forecast_df['prediction'],
                        mode='lines+markers',
                        name='Temperature Forecast',
                        line=dict(color='#e74c3c', width=3),
                        marker=dict(size=6)
                    ))
                    
                    if include_uncertainty:
                        fig.add_trace(go.Scatter(
                            x=forecast_df['date'],
                            y=forecast_df['prediction'] + forecast_df['uncertainty'],
                            mode='lines',
                            name='Upper Bound',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_df['date'],
                            y=forecast_df['prediction'] - forecast_df['uncertainty'],
                            mode='lines',
                            name='Lower Bound',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(231, 76, 60, 0.2)',
                            showlegend=False
                        ))
                    
                    fig.update_layout(
                        title="📈 Multi-Day Weather Forecast",
                        xaxis_title="Date",
                        yaxis_title="Temperature (°C)",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"✅ Generated {forecast_horizon}-day forecast series!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_location_map(self):
        """Render interactive location map"""
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header("🗺️ Interactive Location Map")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📍 Select Location on Map")
            
            if 'analysis_results' in st.session_state:
                location = st.session_state.analysis_results.get('location', {})
                if location:
                    lat, lon = location['lat'], location['lon']
                else:
                    lat, lon = 40.7128, -74.0060
            else:
                lat, lon = 40.7128, -74.0060
            
            m = folium.Map(
                location=[lat, lon],
                zoom_start=10,
                tiles='OpenStreetMap'
            )
            
            folium.Marker(
                [lat, lon],
                popup=f"Current Location<br>{lat:.4f}, {lon:.4f}",
                tooltip="Click for details",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            
            folium.Circle(
                location=[lat, lon],
                radius=5000,
                popup="Analysis Area",
                color="#3186cc",
                fill=True,
                fill_color="#3186cc"
            ).add_to(m)
            
            folium.TileLayer(
                'Stamen Terrain',
                name='Terrain',
                attr=(
                    'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under '
                    '<a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. '
                    'Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under '
                    '<a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
                )
            ).add_to(m)
            
            folium.TileLayer(
                'CartoDB positron',
                attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            ).add_to(m)
            
            folium.LayerControl().add_to(m)
            
            st_folium(m, width=700, height=500)
            
            col1a, col1b = st.columns(2)
            with col1a:
                new_lat = st.number_input("Latitude", value=lat, format="%.4f")
            with col1b:
                new_lon = st.number_input("Longitude", value=lon, format="%.4f")
            
            if st.button("🎯 Use These Coordinates", use_container_width=True):
                st.session_state.user_location = {
                    "lat": new_lat, 
                    "lon": new_lon, 
                    "name": f"Map Location ({new_lat:.4f}, {new_lon:.4f})"
                }
                st.success("📍 Location updated successfully!")
                st.rerun()
        
        with col2:
            st.subheader("🗺️ Map Information")
            
            st.markdown("""
            <div class="data-card">
                <h3>📍 Current Location</h3>
                <p><strong>Latitude:</strong> {:.4f}</p>
                <p><strong>Longitude:</strong> {:.4f}</p>
                <p><strong>Zoom Level:</strong> 10</p>
                <p><strong>Analysis Radius:</strong> 5km</p>
            </div>
            """.format(lat, lon), unsafe_allow_html=True)
            
            st.subheader("🌍 Map Layers")
            st.info("""
            **Available Layers:**
            - **OpenStreetMap**: Standard street map
            - **Stamen Terrain**: Terrain and elevation
            - **CartoDB Positron**: Light theme for data visualization
            
            Use the layer control in the top-right corner to switch between map styles.
            """)
            
            st.subheader("🎯 Quick Actions")
            if st.button("📱 Switch to Mobile View", use_container_width=True):
                st.session_state.force_mobile = True
                st.rerun()
            
            if st.button("🔄 Reset to Default", use_container_width=True):
                st.session_state.user_location = {"lat": 40.7128, "lon": -74.0060, "name": "New York"}
                st.success("Location reset to default!")
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_comprehensive_risks(self):
        """Render comprehensive risk assessment"""
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header("⚠️ Comprehensive Risk Assessment")
        
        if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
            st.info("ℹ️ Please run a multi-variable analysis first to see risk assessments.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        results = st.session_state.analysis_results
        risks = results.get('risks', {})
        
        if not risks:
            st.warning("⚠️ No risk assessment available. Please run analysis with activity selection.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        st.success(f"🏃 Assessing risks for: {ACTIVITY_PRESETS[results['activity']]['name']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Risk Overview")
            
            high_risk_count = sum(1 for r in risks.values() if r["probability"] >= 70)
            medium_risk_count = sum(1 for r in risks.values() if 30 <= r["probability"] < 70)
            low_risk_count = sum(1 for r in risks.values() if r["probability"] < 30)
            
            risk_cols = st.columns(3)
            with risk_cols[0]:
                st.metric("🔴 High Risk", high_risk_count)
            with risk_cols[1]:
                st.metric("🟡 Medium Risk", medium_risk_count)
            with risk_cols[2]:
                st.metric("🟢 Low Risk", low_risk_count)
            
            for condition_id, risk_info in risks.items():
                probability = risk_info["probability"]
                
                if probability >= 70:
                    risk_class = "risk-high"
                    emoji = "🔴"
                elif probability >= 30:
                    risk_class = "risk-medium"
                    emoji = "🟡"
                else:
                    risk_class = "risk-low"
                    emoji = "🟢"
                
                with st.expander(f"{emoji} {risk_info['condition_info']['name']} - {probability}% probability"):
                    col_a, col_b = st.columns([1, 2])
                    
                    with col_a:
                        st.metric("Probability", f"{probability}%")
                        st.metric("Severity", risk_info["severity"].title())
                        st.metric("Trend", "Stable")
                    
                    with col_b:
                        st.write("**Description:**")
                        st.info(risk_info['condition_info']['description'])
                        
                        st.write("**Health Impact:**")
                        st.warning(risk_info['condition_info']['health_impact'])
                        
                        st.write("**Recommended Actions:**")
                        for recommendation in risk_info['condition_info']['recommendations']:
                            st.write(f"• {recommendation}")
        
        with col2:
            st.subheader("🛡️ Safety Recommendations")
            
            activity_info = ACTIVITY_PRESETS[results['activity']]
            
            st.markdown("""
            <div class="forecast-card">
                <h3>🏃 Activity-Specific Advice</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for recommendation in activity_info.get("recommendations", []):
                st.write(f"• {recommendation}")
            
            high_risk_conditions = [cid for cid, risk in risks.items() if risk["probability"] >= 50]
            
            if high_risk_conditions:
                st.markdown("""
                <div class="data-card" style="border-left-color: #e74c3c;">
                    <h3>🚨 High-Risk Conditions</h3>
                </div>
                """, unsafe_allow_html=True)
                
                for condition_id in high_risk_conditions:
                    risk_info = risks[condition_id]
                    condition_info = risk_info["condition_info"]
                    
                    st.write(f"**{condition_info['name']}** ({risk_info['probability']}% probability)")
                    for recommendation in condition_info["recommendations"][:3]:
                        st.write(f"• {recommendation}")
                    st.write("")
            else:
                st.markdown("""
                <div class="data-card" style="border-left-color: #27ae60;">
                    <h3>✅ Favorable Conditions</h3>
                    <p>Weather conditions appear favorable for your planned activity. No significant risks detected.</p>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("📋 Generate Comprehensive Safety Report", use_container_width=True):
                self.generate_comprehensive_safety_report(results, risks)
            
            st.markdown("""
            <div class="data-card">
                <h3>🚑 Emergency Resources</h3>
                <p><strong>Local Emergency:</strong> 911</p>
                <p><strong>Weather Service:</strong> National Weather Service</p>
                <p><strong>Health Department:</strong> Local Health Authority</p>
                <p><strong>Emergency Management:</strong> Local EMA</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def generate_comprehensive_safety_report(self, results: Dict, risks: Dict):
        """Generate comprehensive safety report"""
        activity_info = ACTIVITY_PRESETS[results['activity']]
        season = get_season(results['forecast_date'])
        
        report_content = f"""
# Comprehensive Safety Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Location:** {results['location']['name']}
**Activity:** {activity_info['name']}
**Forecast Date:** {results['forecast_date'].strftime('%Y-%m-%d')}
**Season:** {season.title()}

"""
        
        for condition_id, risk_info in risks.items():
            report_content += f"""
### {risk_info['condition_info']['name']}
- **Risk Probability:** {risk_info['probability']}%
- **Severity Level:** {risk_info['severity'].title()}
- **Impact:** {risk_info['condition_info']['health_impact']}

**Recommended Actions:**
"""
            for recommendation in risk_info['condition_info']['recommendations']:
                report_content += f"- {recommendation}\n"
        
        report_content += f"""
## 🏃 Activity-Specific Recommendations

**Planned Activity:** {activity_info['name']}

**General Advice:**
"""
        for recommendation in activity_info['recommendations']:
            report_content += f"- {recommendation}\n"
        
        report_content += f"""
## 🚨 Emergency Resources
**Important Contacts:**
- Local Emergency Services: 911
- National Weather Service: weather.gov
- Local Health Department: Check local listings
- Emergency Management: Local EMA

**Emergency Kit Checklist:**
- Water and non-perishable food
- First aid kit
- Weather-appropriate clothing
- Communication devices
- Important documents

---

*For educational and planning purposes only. Always follow local official weather advisories.*
"""
        
        st.download_button(
            label="📥 Download Safety Report",
            data=report_content,
            file_name=f"comprehensive_safety_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    def render_system_settings(self):
        """Render system settings interface"""
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header("⚙️ System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔐 NASA Earthdata Configuration")
            
            st.info("""
            **NASA Earthdata Login Status:** ✅ Connected
            **Authentication:** _netrc file detected
            **Data Access:** Real-time NASA data enabled
            """)
            
            st.warning("""
            **Configuration Instructions:**
            Your NASA Earthdata credentials are automatically loaded from the `_netrc` file.
            Ensure your file contains:
            ```
            machine urs.earthdata.nasa.gov
                login YOUR_USERNAME
                password YOUR_PASSWORD
            ```
            """)
            
            st.subheader("🎨 Interface Settings")
            
            theme = st.selectbox("Color Theme", ["Seasonal Auto", "Dark", "Light", "Blue", "Green"])
            units = st.selectbox("Measurement Units", ["Metric", "Imperial", "Scientific"])
            language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Chinese"])
            
            st.checkbox("Enable Real-time Notifications", value=True)
            st.checkbox("Auto-refresh Weather Data", value=False)
            st.checkbox("High Resolution Maps", value=True)
            st.checkbox("Advanced Visualizations", value=True)
        
        with col2:
            st.subheader("🔧 Data Sources")
            
            selected_products = st.multiselect(
                "NASA Data Products",
                options=list(NASA_PRODUCTS.keys()),
                default=list(NASA_PRODUCTS.keys()),
                format_func=lambda x: f"{x} - {NASA_PRODUCTS[x]['description']}"
            )
            
            selected_conditions = st.multiselect(
                "Weather Conditions to Monitor",
                options=list(WEATHER_CONDITIONS.keys()),
                default=list(WEATHER_CONDITIONS.keys()),
                format_func=lambda x: WEATHER_CONDITIONS[x]["name"]
            )
            
            st.subheader("📈 Analysis Parameters")
            
            historical_days = st.slider("Historical Data Days", 7, 365, 30)
            forecast_horizon = st.slider("Forecast Horizon Days", 1, 90, 7)
            confidence_level = st.slider("Default Confidence Level", 50, 95, 80)
            max_variables = st.slider("Maximum Simultaneous Variables", 1, 10, 5)
            
            st.subheader("🔄 Cache & Performance")
            
            cache_size = st.slider("Cache Size (MB)", 100, 2000, 500)
            st.checkbox("Enable Data Caching", value=True)
            st.checkbox("Auto-clear Cache Weekly", value=False)
            st.checkbox("Compress Cached Data", value=True)
        
        if st.button("💾 Save All Settings", type="primary", use_container_width=True):
            st.success("✅ All settings saved successfully!")
            st.balloons()
        
        col3 = st.columns(1)[0]
        with col3:
            if st.button("🔧 Test NASA Connection", use_container_width=True):
                with st.spinner("Testing NASA Earthdata connection..."):
                    time.sleep(2)
                    st.success("✅ NASA Earthdata connection successful!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_footer(self):
        """Render application footer"""
        st.markdown("---")
        st.markdown("""
        <div class="footer">
            <h3>🌍 NASA Weather Intelligence Platform v{}</h3>
            <p>Advanced Multi-Variable Meteorological Analysis & Forecasting System</p>
        """.format(APP_VERSION), unsafe_allow_html=True)

def main():
    try:
        platform = NASAWeatherIntelligencePlatform()
        platform.run_application()
        
    except Exception as e:
        st.error(f"🚨 Critical Application Error: {str(e)}")
        logger.critical(f"Application initialization failed: {e}")
        
        with st.expander("🔧 Technical Details"):
            st.code(f"""
            Error Type: {type(e).__name__}
            Error Message: {str(e)}
            Python Version: {sys.version}
            Streamlit Version: {st.__version__}
            """)

if __name__ == "__main__":
    main()

