import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import joblib # <-- ИМПОРТ ДЛЯ ЗАГРУЗКИ "МОЗГА"

# --- 1. ЗАГРУЗКА РЕСУРСОВ ---

# --- ПУТЬ К ШРИФТУ ---
FONT_PATH = 'VERDANA.TTF' 
font_trick = ImageFont.truetype(FONT_PATH, 36)
font_score = ImageFont.truetype(FONT_PATH, 30)

# --- ЗАГРУЗКА "МОЗГА" (ИИ-МОДЕЛИ) ---
try:
    model = joblib.load('gymnastics_model.pkl')
except FileNotFoundError:
    st.error("ОШИБКА: Файл 'gymnastics_model.pkl' не найден! Убедитесь, что вы загрузили его на GitHub.")
    st.stop() # Останавливаем приложение, если "мозг" не найден

# --- Инициализация MediaPipe ---
mp_pose = mp.solutions.pose

def format_time(milliseconds):
    total_seconds = int(milliseconds / 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

# --- 2. НАША ГЛАВНАЯ ФУНКЦИЯ ОБРАБОТКИ (полностью новая логика) ---
def process_video(video_file, settings):
    
    # --- Настройки ---
    FRAMES_TO_HOLD = settings["hold_frames"]
    
    # --- Переменные ---
    total_score = 0
    protocol_entries = [] 
    
    # Флаги и счетчики кадров (как раньше)
    current_pose = "other"
    previous_pose = "other"
    pose_hold_frames = 0
    
    # Очки за трюки (теперь в одном месте)
    SCORES = {
        "arabesque": 8,
        "leg_lift": 5,
        "split": 10,
        "other": 0
    }
    
    # Названия трюков на русском
    POSE_NAMES_RU = {
        "arabesque": "ЛАСТОЧКА",
        "leg_lift": "ПОДЪЕМ НОГИ",
        "split": "ШПАГАТ",
        "other": ""
    }
    
    cap = cv2.VideoCapture(video_file)
    
    # Индикатор выполнения
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    st_frame = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Обновляем индикатор
            current_frame += 1
            progress_percent = int((current_frame / total_frames) * 100)
            progress_text.text(f"Идет обработка... Кадр {current_frame}/{total_frames} ({progress_percent}%)")
            progress_bar.progress(progress_percent)
            
            current_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_time_str = format_time(current_time_msec)
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            trick_text = ""

            try:
                landmarks = results.pose_landmarks.landmark
                
                # --- 1. ПРЕВРАЩАЕМ СКЕЛЕТ В "ОТПЕЧАТОК" ---
                # (Точно так же, как в train_model.py)
                pose_landmarks_list = []
                for landmark in landmarks:
                    pose_landmarks_list.extend([landmark.x, landmark.y, landmark.z])

                # --- 2. "МОЗГ" ДЕЛАЕТ ПРЕДСКАЗАНИЕ ---
                # Мы "спрашиваем" ИИ: "На что похожа эта поза?"
                # model.predict() ожидает 2D-массив, поэтому [pose_landmarks_list]
                prediction = model.predict([pose_landmarks_list])
                current_pose = prediction[0] # (Берем первый ответ, т.к. он один)
                
                # --- 3. ЛОГИКА ОЦЕНКИ (на основе предсказания) ---
                
                if current_pose != "other":
                    # Если ИИ увидел трюк
                    
                    if current_pose == previous_pose:
                        # Мы все еще в той же позе, продолжаем считать кадры
                        pose_hold_frames += 1
                        
                        trick_text = f"{POSE_NAMES_RU[current_pose]}! (Держать... {pose_hold_frames}/{FRAMES_TO_HOLD})"
                        
                        if pose_hold_frames == FRAMES_TO_HOLD:
                            # Ура! Мы продержали позу достаточно долго!
                            score = SCORES[current_pose]
                            label = POSE_NAMES_RU[current_pose]
                            
                            total_score += score
                            trick_text = f"{label}! +{score} БАЛЛОВ"
                            protocol_entries.append(f"{current_time_str} - {label} (+{score}б)")
                    else:
                        # Это новый трюк, сбрасываем счетчик
                        pose_hold_frames = 1
                        previous_pose = current_pose
                else:
                    # Если ИИ увидел "other", сбрасываем все
                    pose_hold_frames = 0
                    previous_pose = "other"
                    
            except Exception as e:
                # Если "скелет" не найден, сбрасываем
                pose_hold_frames = 0
                previous_pose = "other"
                pass 
            
            # --- Отрисовка ---
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            if trick_text:
                draw.text((50, 50), trick_text, font=font_trick, fill=(0, 255, 0))
            draw.text((50, 100), f"ВРЕМЯ: {current_time_str} | ИТОГО БАЛЛОВ: {total_score}", font=font_score, fill=(255, 255, 0))
            
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) 
            st_frame.image(image, channels="BGR", use_container_width=True)

        cap.release()
        
        # Очищаем индикатор
        progress_text.empty()
        progress_bar.empty()
        
        # --- Финальный отчет ---
        st.success(f"Обработка завершена! Итоговый счет: {total_score} баллов.")
        st.balloons() 
        
        st.subheader("📝 Детальный Протокол (Лог)")
        
        if not protocol_entries:
            st.warning("Ни одного трюка не засчитано.")
        else:
            for entry in protocol_entries:
                st.write(entry)
        
        # --- КНОПКА СКАЧИВАНИЯ ---
        report_text = f"""
        ==================================
        ФИНАЛЬНЫЙ ПРОТОКОЛ АНАЛИЗА (ИИ-МОДЕЛЬ)
        ==================================
        
        Итоговый счет: {total_score} баллов
        
        --- Детальный Лог Трюков ---
        """
        if not protocol_entries:
            report_text += "\nНи одного трюка не засчитано."
        else:
            for entry in protocol_entries:
                report_text += f"\n{entry}"
        
        report_text += f"""
        
        ==================================
        --- Настройки анализа ---
        Время удержания:   {settings["hold_frames"]} кадров
        """
        
        st.download_button(
            label="⬇️ Скачать протокол (.txt)",
            data=report_text,
            file_name="gymnastics_report_AI.txt",
            mime="text/plain"
        )

# --- 3. КОД "САЙТА" (Streamlit) ---
st.set_page_config(layout="wide")
st.title("🤖 ИИ-Анализатор художественной гимнастики")

st.sidebar.title("Панель Управления Судьи")
st.sidebar.write("Настройте сложность анализа:")

# (Мы убрали старые ползунки для углов, т.к. ИИ они не нужны)
hold_frames = st.sidebar.slider(
    "Время удержания позы (кадры):", 
    min_value=5, max_value=60, value=10
)

# Собираем настройки
settings = {
    "hold_frames": hold_frames,
}

st.write("Загрузите видео с выступлением, и ИИ-модель оценит трюки.")

uploaded_file = st.file_uploader("Выберите видео-файл", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Начать анализ"):
        st.info("Идет обработка... Это может занять некоторое время.")
        process_video(uploaded_file.read(), settings)
