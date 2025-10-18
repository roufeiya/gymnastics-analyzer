import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import joblib 

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
    st.stop() 

# --- Инициализация MediaPipe ---
mp_pose = mp.solutions.pose

def format_time(milliseconds):
    total_seconds = int(milliseconds / 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

# --- 2. НАША ГЛАВНАЯ ФУНКЦИЯ ОБРАБОТКИ (УБРАЛИ УДЕРЖАНИЕ) ---
def process_video(video_file): # Убрали settings, т.к. hold_frames не нужен
    
    # --- Переменные ---
    total_score = 0
    protocol_entries = [] 
    
    # --- УБРАЛИ ФЛАГИ И СЧЕТЧИКИ УДЕРЖАНИЯ ---
    # current_pose = "other" # Больше не нужно отслеживать предыдущую позу
    # previous_pose = "other"
    # pose_hold_frames = 0
    # --- КОНЕЦ УДАЛЕНИЯ ---
    
    # Флаг, чтобы не давать баллы за один и тот же трюк несколько кадров подряд
    trick_just_scored = False 
    last_scored_pose = "other"

    # Очки за трюки
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
    
    # --- Читаем видео из байтов, переданных Streamlit ---
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file)
    video_path = tfile.name
    # --- Конец чтения ---
    
    cap = cv2.VideoCapture(video_path)
    
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
            # Проверка деления на ноль, если total_frames = 0
            if total_frames > 0:
                progress_percent = int((current_frame / total_frames) * 100)
            else:
                progress_percent = 0
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
            current_pose = "other" # По умолчанию - 'other'

            try:
                landmarks = results.pose_landmarks.landmark
                
                # --- 1. ПРЕВРАЩАЕМ СКЕЛЕТ В "ОТПЕЧАТОК" ---
                pose_landmarks_list = []
                for landmark in landmarks:
                    pose_landmarks_list.extend([landmark.x, landmark.y, landmark.z])

                # --- 2. "МОЗГ" ДЕЛАЕТ ПРЕДСКАЗАНИЕ ---
                prediction = model.predict([pose_landmarks_list])
                current_pose = prediction[0] 
                
                # --- 3. ИЗМЕНЕННАЯ ЛОГИКА ОЦЕНКИ (Без удержания) ---
                
                if current_pose != "other":
                    # Если ИИ увидел трюк
                    
                    # Проверяем, не тот ли это трюк, за который мы ТОЛЬКО ЧТО дали балл
                    if not trick_just_scored or current_pose != last_scored_pose:
                        score = SCORES[current_pose]
                        label = POSE_NAMES_RU[current_pose]
                        
                        total_score += score
                        trick_text = f"{label}! +{score} БАЛЛОВ"
                        protocol_entries.append(f"{current_time_str} - {label} (+{score}б)")
                        
                        trick_just_scored = True # Ставим флаг, что балл начислен
                        last_scored_pose = current_pose
                else:
                    # Если ИИ увидел "other", сбрасываем флаг
                    trick_just_scored = False
                    last_scored_pose = "other"
                    
            except Exception as e:
                # Если "скелет" не найден, сбрасываем
                trick_just_scored = False
                last_scored_pose = "other"
                pass 
            
            # --- Отрисовка ---
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
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
        ФИНАЛЬНЫЙ ПРОТОКОЛ АНАЛИЗА (ИИ-МОДЕЛЬ v2)
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
        """
        
        st.download_button(
            label="⬇️ Скачать протокол (.txt)",
            data=report_text,
            file_name="gymnastics_report_AI_v2.txt",
            mime="text/plain"
        )

# --- 3. КОД "САЙТА" (Streamlit) ---
st.set_page_config(layout="wide")
st.title("🤖 ИИ-Анализатор художественной гимнастики")

# --- УБРАЛИ ПОЛЗУНКИ СЛОЖНОСТИ (кроме времени) ---
st.sidebar.title("Настройки") 
# hold_frames = st.sidebar.slider( # Оставляем пока, вдруг пригодится
#     "Мин. кадров для удержания (сейчас не используется):", 
#     min_value=1, max_value=60, value=5 
# )
settings = {
     "hold_frames": 1 # Просто ставим 1 кадр
}
# --- КОНЕЦ УДАЛЕНИЯ ---

st.write("Загрузите видео с выступлением, и ИИ-модель оценит трюки.")

uploaded_file = st.file_uploader("Выберите видео-файл", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Начать анализ"):
        st.info("Идет обработка... Это может занять некоторое время.")
        # Передаем байты файла напрямую
        process_video(uploaded_file.read(), settings)
