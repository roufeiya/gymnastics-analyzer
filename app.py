import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import joblib
import time # <-- Импорт для получения времени нажатия кнопки

# --- 1. ЗАГРУЗКА РЕСУРСОВ ---

# --- ПУТЬ К ШРИФТУ ---
FONT_PATH = 'VERDANA.TTF'
try:
    font_trick = ImageFont.truetype(FONT_PATH, 36)
    font_score = ImageFont.truetype(FONT_PATH, 30)
except IOError:
    st.error(f"ОШИБКА: Не удалось загрузить шрифт '{FONT_PATH}'. Убедитесь, что файл шрифта загружен на GitHub.")
    st.stop()


# --- ЗАГРУЗКА "МОЗГА" (ИИ-МОДЕЛИ) ---
try:
    model = joblib.load('gymnastics_model.pkl')
except FileNotFoundError:
    st.error("ОШИБКА: Файл 'gymnastics_model.pkl' не найден! Убедитесь, что вы загрузили его на GitHub.")
    st.stop()

# --- Инициализация MediaPipe ---
mp_pose = mp.solutions.pose

def format_time_manual(timestamp):
    """Форматирует Unix timestamp (float) в строку 'ЧЧ:ММ:СС'"""
    return time.strftime('%H:%M:%S', time.localtime(timestamp))

def format_time(milliseconds):
    """Конвертирует миллисекунды видео (float) в строку 'ММ:СС'"""
    if milliseconds < 0: milliseconds = 0 # Защита от отрицательных значений
    total_seconds = int(milliseconds / 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

# --- 2. ФУНКЦИЯ ДЛЯ АВТОМАТИЧЕСКОГО (ИИ) АНАЛИЗА ---
# (Код почти не изменился, только немного перенесен)
def process_video_ai(video_bytes):
    # --- "Период Охлаждения" ---
    COOLDOWN_FRAMES = 30
    frames_since_last_score = COOLDOWN_FRAMES

    # --- Переменные ---
    total_score = 0
    protocol_entries = []

    SCORES = {"arabesque": 8, "leg_lift": 5, "split": 10, "other": 0}
    POSE_NAMES_RU = {"arabesque": "ЛАСТОЧКА", "leg_lift": "ПОДЪЕМ НОГИ", "split": "ШПАГАТ", "other": ""}

    # --- Читаем видео из байтов ---
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_bytes)
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

            frames_since_last_score += 1
            current_frame += 1
            if total_frames > 0:
                progress_percent = int((current_frame / total_frames) * 100)
            else:
                progress_percent = 0
            progress_text.text(f"Идет обработка ИИ... Кадр {current_frame}/{total_frames} ({progress_percent}%)")
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

                if current_pose != "other":
                    if frames_since_last_score > COOLDOWN_FRAMES:
                        score = SCORES[current_pose]
                        label = POSE_NAMES_RU[current_pose]
                        total_score += score
                        trick_text = f"{label}! +{score} БАЛЛОВ"
                        protocol_entries.append(f"{current_time_str} - {label} (+{score}б)")
                        frames_since_last_score = 0

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
        st.success(f"Обработка ИИ завершена! Итоговый счет: {total_score} баллов.")
        st.balloons()
        return total_score, protocol_entries # Возвращаем результат

# --- 3. ИНТЕРФЕЙС STREAMLIT ---
st.set_page_config(layout="wide")
st.title("🤖 Анализатор художественной гимнастики")

# --- Инициализация session_state для ручного режима ---
if 'manual_score' not in st.session_state:
    st.session_state.manual_score = 0
if 'manual_protocol' not in st.session_state:
    st.session_state.manual_protocol = []

# --- Боковая панель: ВЫБОР РЕЖИМА ---
st.sidebar.title("Настройки")
analysis_mode = st.sidebar.radio(
    "Выберите режим оценки:",
    ('Автоматический (ИИ)', 'Ручной')
)

st.write("Загрузите видео с выступлением.")
uploaded_file = st.file_uploader("Выберите видео-файл", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    video_bytes = uploaded_file.read() # Читаем файл один раз

    # --- Отображаем видео ПЕРЕД кнопками ---
    st.video(video_bytes)

    # --- Логика для разных режимов ---
    if analysis_mode == 'Автоматический (ИИ)':
        st.subheader("Режим: Автоматический (ИИ)")
        if st.button("Начать ИИ-анализ"):
            st.info("Идет обработка ИИ... Это может занять некоторое время.")
            ai_total_score, ai_protocol_entries = process_video_ai(video_bytes)

            # --- Отображаем ИИ-результат ---
            st.subheader("📝 Детальный Протокол (ИИ)")
            if not ai_protocol_entries:
                st.warning("ИИ не засчитал ни одного трюка.")
            else:
                for entry in ai_protocol_entries:
                    st.write(entry)

            # --- Кнопка скачивания ИИ-отчета ---
            report_text = f"ФИНАЛЬНЫЙ ПРОТОКОЛ АНАЛИЗА (ИИ)\nИтоговый счет: {ai_total_score} баллов\n\n--- Детальный Лог Трюков ---\n"
            report_text += "\n".join(ai_protocol_entries) if ai_protocol_entries else "Ни одного трюка не засчитано."
            st.download_button(
                label="⬇️ Скачать протокол ИИ (.txt)",
                data=report_text,
                file_name="gymnastics_report_AI.txt",
                mime="text/plain"
            )

    elif analysis_mode == 'Ручной':
        st.subheader("Режим: Ручной")
        st.write("Смотрите видео, ставьте на паузу и нажимайте кнопки для добавления баллов.")

        # --- Кнопки для ручного ввода ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🤸‍♀️ Шпагат (+10)"):
                st.session_state.manual_score += 10
                st.session_state.manual_protocol.append(f"{format_time_manual(time.time())} - ШПАГАТ (+10б)")
        with col2:
            if st.button("🕊️ Ласточка (+8)"):
                st.session_state.manual_score += 8
                st.session_state.manual_protocol.append(f"{format_time_manual(time.time())} - ЛАСТОЧКА (+8б)")
        with col3:
            if st.button("🦵 Подъем ноги (+5)"):
                st.session_state.manual_score += 5
                st.session_state.manual_protocol.append(f"{format_time_manual(time.time())} - ПОДЪЕМ НОГИ (+5б)")
        with col4:
            if st.button("💍 Колечко (+15)"):
                st.session_state.manual_score += 15
                st.session_state.manual_protocol.append(f"{format_time_manual(time.time())} - КОЛЕЧКО (+15б)")

        # --- Отображаем текущий ручной счет и протокол ---
        st.metric("Текущий Ручной Счет:", st.session_state.manual_score)

        st.subheader("📝 Протокол Ручной Оценки")
        if not st.session_state.manual_protocol:
            st.info("Нажмите кнопки выше, чтобы добавить трюки.")
        else:
            # Показываем протокол в обратном порядке (последние сверху)
            for entry in reversed(st.session_state.manual_protocol):
                st.write(entry)

        # --- Кнопка сброса для ручного режима ---
        if st.button("🔄 Сбросить ручной счет"):
            st.session_state.manual_score = 0
            st.session_state.manual_protocol = []
            st.rerun() # Обновляем страницу, чтобы очистить протокол

        # --- Кнопка скачивания ручного отчета ---
        if st.session_state.manual_protocol: # Показываем, только если есть что скачивать
            report_text = f"ПРОТОКОЛ РУЧНОЙ ОЦЕНКИ\nИтоговый счет: {st.session_state.manual_score} баллов\n\n--- Детальный Лог Трюков ---\n"
            report_text += "\n".join(st.session_state.manual_protocol)
            st.download_button(
                label="⬇️ Скачать ручной протокол (.txt)",
                data=report_text,
                file_name="gymnastics_report_manual.txt",
                mime="text/plain"
            )

# --- Подсказка внизу ---
st.markdown("---")
st.caption("Создано с помощью Streamlit, OpenCV и MediaPipe")
