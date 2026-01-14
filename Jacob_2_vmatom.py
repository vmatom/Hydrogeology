import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Настройка страницы должна быть первой командой
st.set_page_config(
    page_title="MSGEO - Cooper-Jacob Web",
    page_icon="💧",
    layout="wide"
)

# ==========================================
# 1. ЛОГИКА (С КЭШИРОВАНИЕМ)
# ==========================================

@st.cache_data # <-- ВАЖНО: Кэшируем загрузку данных
def load_data(uploaded_file, file_type, sheet_name=None):
    """
    Функция загрузки данных. Streamlit запомнит результат,
    пока не изменится сам загруженный файл.
    """
    try:
        if file_type == 'csv':
            # Пробуем разные кодировки
            try:
                return pd.read_csv(uploaded_file, encoding='utf-8-sig')
            except:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding='cp1251')
        else:
            # Excel
            if sheet_name:
                return pd.read_excel(uploaded_file, sheet_name=sheet_name)
            else:
                return pd.read_excel(uploaded_file)
    except Exception as e:
        return None

def get_soil_info(K_val):
    if K_val > 500: return "Large Karst / Boulders-Gravel"
    elif K_val > 100: return "Gravel / Highly Fractured Rock"
    elif K_val > 10: return "Coarse Sand / Fractured Rock"
    elif K_val > 1: return "Med. Sand / Slightly Fractured Rock"
    elif K_val > 0.1: return "Fine Sand / Fractured Sandstone"
    elif K_val > 0.005: return "Sandy Loam / Siltstone"
    elif K_val > 0.0001: return "Loam / Limestone"
    else: return "Clay / Aquiclude"

def calculate_cooper_jacob(t_arr, s_arr, Q_day, r, m):
    if len(t_arr) < 2: return None
    
    # Расчет
    log_t = np.log10(t_arr)
    slope, intercept = np.polyfit(log_t, s_arr, 1)
    
    if slope == 0: return None
    
    T = 0.183 * Q_day / abs(slope)
    K = (T / m) if m > 0 else 0
    log_t0 = -intercept / slope
    t0 = 10 ** log_t0
    S_coeff = (2.25 * T * t0) / (r ** 2)
    
    t_check = t_arr.iloc[0] if len(t_arr) > 0 else 1
    u_val = (r**2 * S_coeff) / (4 * T * t_check) if (T > 0 and t_check > 0) else 999.0
    
    return {
        "T": T, "K": K, "S": S_coeff, 
        "t0": t0, "u": u_val, 
        "slope": slope, "intercept": intercept
    }

# ==========================================
# 2. ИНТЕРФЕЙС
# ==========================================

st.title("💧 MSGEO: Cooper-Jacob Analysis")
st.markdown("---")

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("1. Well Parameters")
    q_val = st.number_input("Flow Rate (Q)", value=10.0, step=0.1)
    q_unit = st.selectbox("Unit Q", ["L/sec", "m3/hour", "m3/day"])
    
    if q_unit == "L/sec": Q_day = q_val * 86.4
    elif q_unit == "m3/hour": Q_day = q_val * 24.0
    else: Q_day = q_val
    
    st.caption(f"Calculated Q: {Q_day:.1f} m³/day")
    
    r_val = st.number_input("Radius (r), m", value=10.0, step=0.1)
    m_val = st.number_input("Thickness (m), m", value=10.0, step=0.1)

# --- ЗАГРУЗКА ---
col_upload, col_settings = st.columns([1, 2])

with col_upload:
    st.header("2. Upload Data")
    uploaded_file = st.file_uploader("Excel or CSV", type=["xlsx", "xls", "csv"])

if uploaded_file:
    # Определение типа файла и листов
    file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'excel'
    sheet_name = None
    
    if file_type == 'excel':
        # Чтобы не читать весь файл ради списка листов, используем ExcelFile
        # Это тоже можно кэшировать, но операция быстрая
        xl = pd.ExcelFile(uploaded_file)
        if len(xl.sheet_names) > 1:
            sheet_name = st.selectbox("Select Sheet:", xl.sheet_names)
        else:
            sheet_name = xl.sheet_names[0]
            
    # Загрузка данных с использованием кэширования
    df = load_data(uploaded_file, file_type, sheet_name)
    
    if df is not None:
        # --- НАСТРОЙКА КОЛОНОК ---
        with col_settings:
            st.header("3. Column Mapping")
            cols = df.columns.tolist()
            c1, c2, c3 = st.columns(3)
            
            # Вспомогательная функция поиска
            def find_col(kws):
                for c in cols:
                    if any(k in c.lower() for k in kws): return c
                return cols[0] if cols else None

            with c1:
                t_col = st.selectbox("Time Column (t)", cols, index=cols.index(find_col(['time', 'время', 't'])))
            with c2:
                t_unit = st.selectbox("Time Unit", ["Minutes", "Hours", "Days"])
            with c3:
                s_col = st.selectbox("Drawdown Column (s)", cols, index=cols.index(find_col(['s', 'draw', 'пониж'])))

        # Подготовка данных (быстрая операция, можно не кэшировать)
        tf = 1/1440.0 if t_unit == "Minutes" else (1/24.0 if t_unit == "Hours" else 1.0)
        
        try:
            df_clean = df[[t_col, s_col]].copy()
            df_clean[t_col] = pd.to_numeric(df_clean[t_col], errors='coerce')
            df_clean[s_col] = pd.to_numeric(df_clean[s_col], errors='coerce')
            df_clean = df_clean.dropna()
            
            df_clean['t_days'] = df_clean[t_col] * tf
            df_clean = df_clean[df_clean['t_days'] > 0].sort_values('t_days')
            
            # Группировка дубликатов
            df_clean = df_clean.groupby('t_days', as_index=False)[s_col].mean()

            if len(df_clean) < 2:
                st.error("Not enough valid data points.")
                st.stop()

            # --- ГРАФИК ---
            st.markdown("---")
            st.header("4. Analysis & Plot")
            
            # Слайдер на всю ширину
            range_idx = st.slider(
                "Select data range for linear approximation:",
                0, len(df_clean)-1, (int(len(df_clean)/2), len(df_clean)-1)
            )
            
            start_idx, end_idx = range_idx
            subset = df_clean.iloc[start_idx : end_idx+1]
            
            res = calculate_cooper_jacob(subset['t_days'], subset[s_col], Q_day, r_val, m_val)
            
            # Matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_clean['t_days'], df_clean[s_col], c='black', alpha=0.3, label='All Data')
            ax.scatter(subset['t_days'], subset[s_col], c='red', s=40, label='Selected Range')
            
            if res:
                x_vals = np.array([df_clean['t_days'].min(), df_clean['t_days'].max()])
                # Корректировка x_vals чтобы линия не улетала в бесконечность
                if x_vals[0] <= 0: x_vals[0] = res['t0'] if res['t0'] > 0 else 1e-5

                x_line = np.logspace(np.log10(x_vals[0]), np.log10(x_vals[1]), 100)
                y_line = res['slope'] * np.log10(x_line) + res['intercept']
                
                ax.plot(x_line, y_line, 'r--', lw=2, label='Approximation')
                
                # Показываем T и K прямо на графике
                ax.text(0.05, 0.95, f"T = {res['T']:.2f}\nK = {res['K']:.3f}", 
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xscale('log')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Drawdown (m)')
            ax.grid(True, which="both", ls='--', alpha=0.5)
            ax.legend()
            
            st.pyplot(fig)

            # --- РЕЗУЛЬТАТЫ ---
            if res:
                soil = get_soil_info(res['K'])
                
                c_res1, c_res2, c_res3, c_res4 = st.columns(4)
                c_res1.metric("Transmissivity (T)", f"{res['T']:.2f}")
                c_res2.metric("Conductivity (K)", f"{res['K']:.3f}")
                c_res3.metric("Storativity (S)", f"{res['S']:.2e}")
                c_res4.metric("u check", f"{res['u']:.3f}", delta="OK" if res['u'] < 0.1 else "High > 0.1", delta_color="inverse")
                
                st.info(f"**Geology Interpretation:** {soil}")
                
                # Текст отчета
                report = f"MSGEO REPORT\nFile: {uploaded_file.name}\nQ: {Q_day}\nT: {res['T']:.4f}\nK: {res['K']:.4f}\nS: {res['S']:.4e}\nSoil: {soil}"
                st.download_button("📥 Download Report", report, "report.txt")

        except Exception as e:
            st.error(f"Error processing data: {e}")
