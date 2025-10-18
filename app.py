import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

# --- ПУТЬ К ШРИФТУ ---
FONT_PATH = 'VERDANA.TTF' 

font_trick = ImageFont.truetype(FONT_PATH, 36)
font_score = ImageFont.truetype(FONT_PATH, 30)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def format_time(milliseconds):
    total_seconds = int(milliseconds / 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

# --- Наша главная функция обработки ---
def process_video(video_file, settings):
    
    # --- Настройки со слайдеров ---
    SPLIT_THRESHOLD_MIN = settings["split_angle"]
    LEG_LIFT_THRESHOLD_ANGLE = 100 
    RING_THRESHOLD_DISTANCE = 0.12 
    KNEE_STRAIGHT_ANGLE = settings["knee_angle"]
    FRAMES_TO_HOLD = settings["hold_frames"]
    ARABESQUE_THRESHOLD_MIN = settings["arabesque_angle"]

    # --- Пороги для градуированной оценки ---
    SPLIT_THRESHOLD_GOOD = SPLIT_THRESHOLD_MIN + 5
    SPLIT_THRESHOLD_PERFECT = SPLIT_THRESHOLD_MIN + 10
    ARABESQUE_THRESHOLD_GOOD = ARABESQUE_THRESHOLD_MIN + 10
    ARABESQUE_THRESHOLD_PERFECT = ARABESQUE_THRESHOLD_MIN + 20

    # --- Переменные ---
    total_score = 0
    protocol_entries = [] 
    
    split_in_progress = False
    leg_lift_in_progress = False
    ring_in_progress = False
    arabesque_in_progress = False
    
    split_hold_frames = 0
    leg_lift_hold_frames = 0
    ring_hold_frames = 0
    arabesque_hold_frames = 0
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    
    # --- НОВЫЙ БЛОК: ИНДИКАТОР ВЫПОЛНЕНИЯ ---
    # 1. Получаем общее количество кадров в видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    # 2. Создаем "пустой" индикатор
    progress_text = st.empty()
    progress_bar = st.progress(0)
    # --- КОНЕЦ НОВОГО БЛОКА ---
    
    st_frame = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # --- ОБНОВЛЯЕМ ИНДИКАТОР ---
            current_frame += 1
            progress_percent = int((current_frame / total_frames) * 100)
            progress_text.text(f"Идет обработка... Кадр {current_frame}/{total_frames} ({progress_percent}%)")
            progress_bar.progress(progress_percent)
            # --- КОНЕЦ ОБНОВЛЕНИЯ ---
            
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
                
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                nose_pos = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]

                pelvis_center = np.mean([left_hip, right_hip], axis=0)
                split_angle = calculate_angle(left_ankle, pelvis_center, right_ankle)
                left_leg_lift_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle) 
                right_leg_lift_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle) 
                distance_to_nose = calculate_distance(left_ankle, nose_pos) 
                arabesque_angle_left_support = calculate_angle(left_shoulder, left_hip, right_ankle)
                arabesque_angle_right_support = calculate_angle(right_shoulder, right_hip, left_ankle)
                
                is_left_leg_lifted = (left_leg_lift_angle < LEG_LIFT_THRESHOLD_ANGLE and left_knee_angle > KNEE_STRAIGHT_ANGLE)
                is_right_leg_lifted = (right_leg_lift_angle < LEG_LIFT_THRESHOLD_ANGLE and right_knee_angle > KNEE_STRAIGHT_ANGLE)
                is_left_support_arabesque = (left_knee_angle > KNEE_STRAIGHT_ANGLE and right_knee_angle > KNEE_STRAIGHT_ANGLE and arabesque_angle_left_support > ARABESQUE_THRESHOLD_MIN)
                is_right_support_arabesque = (right_knee_angle > KNEE_STRAIGHT_ANGLE and left_knee_angle > KNEE_STRAIGHT_ANGLE and arabesque_angle_right_support > ARABESQUE_THRESHOLD_MIN)

                # --- ЛОГИКА ОЦЕНКИ (Градуированная + Тайм-коды) ---
                
                # --- 1. Шпагат ---
                if split_angle > SPLIT_THRESHOLD_MIN:
                    if not split_in_progress: 
                        split_hold_frames += 1
                        trick_text = f"ШПАГАТ! (Держать... {split_hold_frames}/{FRAMES_TO_HOLD})"
                        if split_hold_frames >= FRAMES_TO_HOLD:
                            if split_angle > SPLIT_THRESHOLD_PERFECT:
                                score, label = 10, "ИДЕАЛЬНЫЙ ШПАГАТ"
                            elif split_angle > SPLIT_THRESHOLD_GOOD:
                                score, label = 7, "ХОРОШИЙ ШПАГАТ"
                            else:
                                score, label = 4, "ЗАЧЕТ (ШПАГАТ)"
                            
                            total_score += score
                            split_in_progress = True 
                            trick_text = f"{label}! +{score} БАЛЛОВ"
                            protocol_entries.append(f"{current_time_str} - {label} (+{score}б)")
                
                # --- 2. Подъем ноги (вперед) ---
                elif (is_left_leg_lifted or is_right_leg_lifted) and not split_in_progress: 
                    if not leg_lift_in_progress:
                        leg_lift_hold_frames += 1
                        trick_text = f"ПОДЪЕМ НОГИ! (Держать... {leg_lift_hold_frames}/{FRAMES_TO_HOLD})"
                        if leg_lift_hold_frames >= FRAMES_TO_HOLD:
                            score, label = 5, "ПОДЪЕМ ПРЯМОЙ НОГИ"
                            total_score += score
                            leg_lift_in_progress = True
                            trick_text = f"{label}! +{score} БАЛЛОВ"
                            protocol_entries.append(f"{current_time_str} - {label} (+{score}б)")

                # --- 3. Ласточка (Арабеск) ---
                elif (is_left_support_arabesque or is_right_support_arabesque) and not leg_lift_in_progress and not split_in_progress:
                    if not arabesque_in_progress:
                        arabesque_hold_frames += 1
                        trick_text = f"ЛАСТОЧКА! (Держать... {arabesque_hold_frames}/{FRAMES_TO_HOLD})"
                        if arabesque_hold_frames >= FRAMES_TO_HOLD:
                            current_arabesque_angle = max(arabesque_angle_left_support, arabesque_angle_right_support)
                            if current_arabesque_angle > ARABESQUE_THRESHOLD_PERFECT:
                                score, label = 8, "ИДЕАЛЬНАЯ ЛАСТОЧКА"
                            elif current_arabesque_angle > ARABESQUE_THRESHOLD_GOOD:
                                score, label = 5, "ХОРОШАЯ ЛАСТОЧКА"
                            else:
                                score, label = 3, "ЗАЧЕТ (ЛАСТОЧКА)"
                                
                            total_score += score
                            arabesque_in_progress = True
                            trick_text = f"{label}! +{score} БАЛЛОВ"
                            protocol_entries.append(f"{current_time_str} - {label} (+{score}б)")
                
                # --- 4. Колечко ---
                elif distance_to_nose < RING_THRESHOLD_DISTANCE:
                    if not ring_in_progress:
                        ring_hold_frames += 1
                        trick_text = f"КОЛЕЧКО! (Держать... {ring_hold_frames}/{FRAMES_TO_HOLD})"
                        if ring_hold_frames >= FRAMES_TO_HOLD:
                            score, label = 15, "КОЛЕЧКО"
                            total_score += score
                            ring_in_progress = True
                            trick_text = f"{label}! +{score} БАЛЛОВ"
                            protocol_entries.append(f"{current_time_str} - {label} (+{score}б)")
                
                # --- 5. Сброс ---
                else:
                    split_hold_frames = 0
                    split_in_progress = False
                    leg_lift_hold_frames = 0
                    leg_lift_in_progress = False
                    arabesque_hold_frames = 0
                    arabesque_in_progress = False
                    ring_hold_frames = 0
                    ring_in_progress = False

            except Exception as e:
                pass 
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            if trick_text:
                draw.text((50, 50), trick_text, font=font_trick, fill=(0, 255, 0))
            draw.text((50, 100), f"ВРЕМЯ: {current_time_str} | ИТОГО БАЛЛОВ: {total_score}", font=font_score, fill=(255, 255, 0))
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) 

            st_frame.image(image, channels="BGR", use_container_width=True)

        cap.release()
        
        # --- Очищаем индикатор выполнения ---
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
        ФИНАЛЬНЫЙ ПРОТОКОЛ АНАЛИЗА
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
        Мин. угол шпагата: {settings["split_angle"]}°
        Мин. угол прямой ноги: {settings["knee_angle"]}°
        Мин. угол ласточки: {settings["arabesque_angle"]}°
        """
        
        st.download_button(
            label="⬇️ Скачать протокол (.txt)",
            data=report_text,
            file_name="gymnastics_report.txt",
            mime="text/plain"
        )
        # --- КОНЕЦ ---


# --- КОД "САЙТА" (Streamlit) ---

st.set_page_config(layout="wide")
st.title("🤖 Анализатор художественной гимнастики")

st.sidebar.title("Панель Управления Судьи")
st.sidebar.write("Настройте сложность анализа:")

hold_frames = st.sidebar.slider(
    "Время удержания позы (кадры):", 
    min_value=5, max_value=60, value=10
)
split_angle = st.sidebar.slider(
    "Мин. 'Зачет' для шпагата (градусы):",
    min_value=150, max_value=180, value=160
)
knee_angle = st.sidebar.slider(
    "Мин. угол прямой ноги (градусы):", 
    min_value=150, max_value=180, value=165
)
arabesque_angle = st.sidebar.slider(
    "Мин. 'Зачет' для 'Ласточки' (градусы):",
    min_value=30, max_value=90, value=45
)

settings = {
    "hold_frames": hold_frames,
    "split_angle": split_angle,
    "knee_angle": knee_angle,
    "arabesque_angle": arabesque_angle
}

st.write("Загрузите видео с выступлением, и нейросеть оценит трюки.")

uploaded_file = st.file_uploader("Выберите видео-файл", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Начать анализ"):
        # Удаляем старый отчет, если он был
        st.empty() 
        
        st.info("Идет обработка... Это может занять некоторое время.")
        process_video(uploaded_file, settings)
