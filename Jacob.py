import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# ==========================================
# 1. ЛОГИКА ГЕОЛОГИИ И РАСЧЕТОВ
# ==========================================
def get_soil_info(K_val):
    """Определяет тип грунта по коэффициенту фильтрации"""
    if K_val > 500: return "Large Karst / Boulders-Gravel"
    elif K_val > 100: return "Gravel / Highly Fractured Rock"
    elif K_val > 10: return "Coarse Sand / Fractured Rock"
    elif K_val > 1: return "Med. Sand / Slightly Fractured Rock"
    elif K_val > 0.1: return "Fine Sand / Fractured Sandstone"
    elif K_val > 0.005: return "Sandy Loam / Siltstone"
    elif K_val > 0.0001: return "Loam / Limestone"
    else: return "Clay / Aquiclude"

def calculate_cooper_jacob(t_arr, s_arr, Q_day, r, m):
    """Считает параметры T, K, S по выбранным точкам"""
    if len(t_arr) < 2:
        return None
    
    # Логарифмирование времени для линейной регрессии
    log_t = np.log10(t_arr)
    
    # Линейная регрессия (polyfit степени 1)
    # s = slope * log(t) + intercept
    slope, intercept = np.polyfit(log_t, s_arr, 1)
    
    if slope == 0: return None
    
    # Расчет гидрогеологических параметров
    # T = 0.183 * Q / delta_s (где delta_s на лог цикл это и есть slope)
    # В формуле slope_m = (s2-s1)/(logt2-logt1). Это и есть наш slope.
    # Но slope может быть отрицательным или положительным в зависимости от осей, берем abs
    
    T = 0.183 * Q_day / abs(slope)
    K = (T / m) if m > 0 else 0
    
    # t0 - пересечение с осью X (s=0) -> 0 = slope * logt0 + intercept
    # logt0 = -intercept / slope
    log_t0 = -intercept / slope
    t0 = 10 ** log_t0
    
    S_coeff = (2.25 * T * t0) / (r ** 2)
    
    # Расчет u для последней точки диапазона
    t_check = t_arr.iloc[0] if len(t_arr) > 0 else 1
    u_val = (r**2 * S_coeff) / (4 * T * t_check) if (T > 0 and t_check > 0) else 999.0
    
    return {
        "T": T, "K": K, "S": S_coeff, 
        "t0": t0, "u": u_val, 
        "slope": slope, "intercept": intercept
    }

# ==========================================
# 2. ИНТЕРФЕЙС STREAMLIT
# ==========================================
st.set_page_config(page_title="MSGEO - Cooper-Jacob Web", layout="wide")

st.title("💧 MSGEO: Интерпретация откачки (Cooper-Jacob)")

# --- БОКОВАЯ ПАНЕЛЬ (Ввод параметров) ---
with st.sidebar:
    st.header("1. Параметры скважины")
    
    q_val = st.number_input("Дебит (Q)", value=10.0, step=0.1)
    q_unit = st.selectbox("Ед. Q", ["L/sec", "m3/hour", "m3/day"])
    
    # Конвертация Q в м3/сут
    if q_unit == "L/sec": Q_day = q_val * 86.4
    elif q_unit == "m3/hour": Q_day = q_val * 24.0
    else: Q_day = q_val
    
    st.info(f"Q расчетное: {Q_day:.1f} м³/сут")
    
    r_val = st.number_input("Радиус (r), м", value=10.0, step=0.1)
    m_val = st.number_input("Мощность (m), м", value=10.0, step=0.1)

# --- ЗАГРУЗКА ФАЙЛА ---
st.header("2. Загрузка данных")
uploaded_file = st.file_uploader("Выберите Excel или CSV файл", type=["xlsx", "xls", "csv"])

df = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            # Пробуем разные кодировки, как в оригинале
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='cp1251')
                
        else:
            # Excel: Сначала получаем имена листов
            xl = pd.ExcelFile(uploaded_file)
            sheet_names = xl.sheet_names
            
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox("Выберите лист:", sheet_names)
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            else:
                df = pd.read_excel(uploaded_file)
        
        st.success(f"Файл загружен: {len(df)} строк")
        
        # --- ВЫБОР КОЛОНОК ---
        col1, col2, col3 = st.columns(3)
        cols = df.columns.tolist()
        
        # Автопоиск колонок
        def find_col(keywords):
            for c in cols:
                if any(k in c.lower() for k in keywords): return c
            return cols[0] if cols else None

        with col1:
            t_col = st.selectbox("Столбец Времени (t)", cols, index=cols.index(find_col(['time', 'время', 't'])))
        with col2:
            t_unit = st.selectbox("Ед. времени", ["Minutes", "Hours", "Days"])
        with col3:
            s_col = st.selectbox("Столбец Понижения (s)", cols, index=cols.index(find_col(['s', 'draw', 'пониж'])))
            
        # Подготовка данных
        tf = 1/1440.0 if t_unit == "Minutes" else (1/24.0 if t_unit == "Hours" else 1.0)
        
        # Очистка и конвертация
        df_clean = df[[t_col, s_col]].dropna()
        # Конвертируем в числа (force), ошибки становятся NaN
        df_clean[t_col] = pd.to_numeric(df_clean[t_col], errors='coerce')
        df_clean[s_col] = pd.to_numeric(df_clean[s_col], errors='coerce')
        df_clean = df_clean.dropna()
        
        # Перевод времени в сутки и фильтрация t > 0
        df_clean['t_days'] = df_clean[t_col] * tf
        df_clean = df_clean[df_clean['t_days'] > 0].sort_values('t_days')
        
        # Усреднение дубликатов времени
        df_clean = df_clean.groupby('t_days', as_index=False)[s_col].mean()
        
        if len(df_clean) < 2:
            st.error("Недостаточно данных для построения.")
            st.stop()

        # --- ПОСТРОЕНИЕ ГРАФИКА И ВЫБОР ДИАПАЗОНА ---
        st.header("3. Анализ (Cooper-Jacob)")
        st.write("Используйте **слайдер** ниже, чтобы выбрать линейный участок графика для аппроксимации.")
        
        # Слайдер для выбора диапазона точек (индексы)
        range_idx = st.slider(
            "Диапазон точек для прямой линии:",
            min_value=0,
            max_value=len(df_clean)-1,
            value=(int(len(df_clean)/2), len(df_clean)-1), # По умолчанию вторая половина
            format="%d"
        )
        
        start_idx, end_idx = range_idx
        
        # Данные для аппроксимации
        subset = df_clean.iloc[start_idx : end_idx+1]
        
        # Расчет
        res = calculate_cooper_jacob(subset['t_days'], subset[s_col], Q_day, r_val, m_val)
        
        # --- ОТРИСОВКА ---
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 1. Все точки
        ax.scatter(df_clean['t_days'], df_clean[s_col], color='black', alpha=0.5, label='Замеры', s=15)
        
        # 2. Выбранные точки (подсветка)
        ax.scatter(subset['t_days'], subset[s_col], color='red', s=30, label='Выбранный участок')
        
        # 3. Линия аппроксимации
        if res:
            # Строим линию чуть шире диапазона, чтобы было красиво
            x_min = df_clean['t_days'].min()
            x_max = df_clean['t_days'].max()
            
            # Y = slope * log10(X) + intercept
            # Генерируем точки для линии
            x_line = np.linspace(x_min, x_max, 100)
            y_line = res['slope'] * np.log10(x_line) + res['intercept']
            
            ax.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label='Cooper-Jacob Line')
            
            # Текст на графике
            mid_x = 10 ** ((np.log10(subset['t_days'].min()) + np.log10(subset['t_days'].max())) / 2)
            mid_y = (subset[s_col].min() + subset[s_col].max()) / 2
            ax.text(mid_x, mid_y, f"T={res['T']:.1f}\nK={res['K']:.2f}", 
                    color="darkred", fontweight="bold", backgroundcolor="#ffffffaa")

        ax.set_xscale('log')
        ax.set_xlabel("Время (сутки)")
        ax.set_ylabel("Понижение (м)")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.legend()
        
        # Показываем график в Streamlit
        st.pyplot(fig)
        
        # --- РЕЗУЛЬТАТЫ ---
        if res:
            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Transmissivity (T)", f"{res['T']:.2f} м²/сут")
            r2.metric("Conductivity (K)", f"{res['K']:.3f} м/сут")
            r3.metric("Storativity (S)", f"{res['S']:.2e}")
            
            soil_name = get_soil_info(res['K'])
            st.info(f"🛑 Предполагаемый грунт: **{soil_name}**")
            
            if res['u'] > 0.1:
                st.warning(f"⚠️ u = {res['u']:.2f} (> 0.1). Метод может быть неточен.")
            else:
                st.success(f"✅ u = {res['u']:.4f} (OK)")
                
            # Сохранение отчета
            report_text = f"""MSGEO WEB REPORT
Date: {pd.Timestamp.now()}
File: {uploaded_file.name}
Q: {Q_day:.2f} m3/day
r: {r_val} m
m: {m_val} m

Results:
T: {res['T']:.4f} m2/day
K: {res['K']:.4f} m/day
S: {res['S']:.4e}
u: {res['u']:.4f}
Soil: {soil_name}
"""
            st.download_button("💾 Скачать отчет (.txt)", report_text, file_name="report.txt")
            
    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")

else:
    st.info("Пожалуйста, загрузите файл данных, чтобы начать.")
