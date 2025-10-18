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

# --- 2. НАША ГЛАВНАЯ ФУНКЦИЯ ОБРАБОТКИ ---
def process_video(video_file): 
    
    # --- НОВЫЙ БЛОК: "ПЕРИОД ОХЛАЖДЕНИЯ" ---
    # Не начислять баллы в течение N кадров после трюка
    COOLDOWN_FRAMES = 30 # (30 кадров = ~1 секунда)
    frames_since_last_score = COOLDOWN_FRAMES # (Начинаем с > 30, чтобы можно было сразу засчитать)
    # --- КОНЕЦ НОВОГО БЛОКА ---
    
    # --- Переменные ---
    total_score = 0
    protocol_entries = [] 

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
    
    # --- Читаем видео из байтов ---
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file)
    video_path = tfile.name
    
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
                
            # --- ОБНОВЛЕНО: Увеличиваем счетчик "охлаждения" ---
            frames_since_last_score += 1
            
            # Обновляем индикатор
            current_frame += 1
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
            current_pose = "other"

            try:
                landmarks = results.pose_landmarks.landmark
                
                pose_landmarks_list = []
                for landmark in landmarks:
                    pose_landmarks_list.extend([landmark.x, landmark.y, landmark.z])

                prediction = model.predict([pose_landmarks_list])
                current_pose = prediction[0] 
                
                # --- ОБНОВЛЕНО: ЛОГИКА ОЦЕНКИ (С "ОХЛАЖДЕНИЕМ") ---
                
                if current_pose != "other":
                    # Если ИИ увидел трюк
                    
                    # Проверяем, прошло ли "охлаждение"
                    if frames_since_last_score > COOLDOWN_FRAMES:
                        score = SCORES[current_pose]
                        label = POSE_NAMES_RU[current_pose]
                        
                        total_score += score
                        trick_text = f"{label}! +{score} БАЛЛОВ"
                        protocol_entries.append(f"{current_time_str} - {label} (+{score}б)")
                        
                        # СБРАСЫВАЕМ СЧЕТЧИК
                        frames_since_last_score = 0 
                # --- КОНЕЦ ИЗМЕНЕНИЙ ---
                    
            except Exception as e:
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
        ФИНАЛЬНЫЙ ПРОТОКОЛ АНАЛИЗА (ИИ-МОДЕЛЬ v3)
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
            file_name="gymnastics_report_AI_v3.txt",
            mime="text/plain"
        )

# --- 3. КОД "САЙТА" (Streamlit) ---
st.set_page_config(layout="wide")
st.title("🤖 ИИ-Анализатор художественной гимнастики")

st.write("Загрузите видео с выступлением, и ИИ-модель оценит трюки.")

uploaded_file = st.file_uploader("Выберите видео-файл", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Начать анализ"):
        st.info("Идет обработка... Это может занять некоторое время.")
        process_video(uploaded_file.read())
