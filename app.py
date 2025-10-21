import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import joblib
import time

# --- 1. ЗАГРУЗКА РЕСУРСОВ ---

# --- ПУТЬ К ШРИФТУ ---
FONT_PATH = 'VERDANA.TTF'
try:
    font_trick = ImageFont.truetype(FONT_PATH, 36)
    font_score = ImageFont.truetype(FONT_PATH, 30)
    font_timer = ImageFont.truetype(FONT_PATH, 96)
except IOError:
    try:
        FONT_PATH = 'C:/Windows/Fonts/Verdana.ttf'
        font_trick = ImageFont.truetype(FONT_PATH, 36)
        font_score = ImageFont.truetype(FONT_PATH, 30)
        font_timer = ImageFont.truetype(FONT_PATH, 96)
    except IOError:
        st.error(f"ОШИБКА: Не удалось загрузить шрифт '{FONT_PATH}'.")
        st.stop()

# --- ЗАГРУЗКА "МОЗГА" (ИИ-МОДЕЛИ) ---
try:
    model = joblib.load('gymnastics_model.pkl')
except FileNotFoundError:
    st.error("ОШИБКА: Файл 'gymnastics_model.pkl' не найден!")
    st.stop()

# --- Инициализация MediaPipe ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Словарь очков и названий ---
SCORES = {"arabesque": 8, "leg_lift": 5, "split": 10, "other": 0}
POSE_NAMES_RU = {"arabesque": "ЛАСТОЧКА", "leg_lift": "ПОДЪЕМ НОГИ", "split": "ШПАГАТ", "other": "ДРУГОЕ"}
# Список поз для выбора при коррекции
POSE_OPTIONS = list(POSE_NAMES_RU.keys()) # ['arabesque', 'leg_lift', 'split', 'other']

def format_time_manual(timestamp):
    return time.strftime('%H:%M:%S', time.localtime(timestamp))

def format_time(milliseconds):
    if milliseconds < 0: milliseconds = 0
    total_seconds = int(milliseconds / 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

# --- 2. ФУНКЦИЯ ДЛЯ АВТОМАТИЧЕСКОГО (ИИ) АНАЛИЗА ФАЙЛА ---
# (Без изменений)
def process_video_ai(video_bytes, settings):
    # ... (весь код process_video_ai из версии 4.2 остается здесь) ...
    COOLDOWN_FRAMES = settings["cooldown_frames"]
    FEET_ON_GROUND_THRESHOLD = 0.9
    frames_since_last_score = COOLDOWN_FRAMES
    total_score = 0; protocol_entries = []
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4'); tfile.write(video_bytes); video_path = tfile.name
    cap = cv2.VideoCapture(video_path); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); current_frame = 0
    progress_text = st.empty(); progress_bar = st.progress(0); st_frame = st.empty()
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read();
            if not ret: break
            frames_since_last_score += 1; current_frame += 1
            if total_frames > 0: progress_percent = int((current_frame / total_frames) * 100)
            else: progress_percent = 0
            progress_text.text(f"Идет обработка ИИ... Кадр {current_frame}/{total_frames} ({progress_percent}%)"); progress_bar.progress(progress_percent)
            current_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC); current_time_str = format_time(current_time_msec)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image.flags.writeable = False
            results = pose.process(image); image.flags.writeable = True; image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            trick_text = ""; current_pose = "other"; both_feet_on_ground = False
            try:
                landmarks = results.pose_landmarks.landmark; pose_landmarks_list = []
                for landmark in landmarks: pose_landmarks_list.extend([landmark.x, landmark.y, landmark.z])
                prediction = model.predict([pose_landmarks_list]); current_pose = prediction[0]
                left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y; right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                if left_ankle_y > FEET_ON_GROUND_THRESHOLD and right_ankle_y > FEET_ON_GROUND_THRESHOLD: both_feet_on_ground = True
                if current_pose != "other" and not both_feet_on_ground:
                    if frames_since_last_score > COOLDOWN_FRAMES:
                        score = SCORES[current_pose]; label = POSE_NAMES_RU[current_pose]; total_score += score
                        trick_text = f"{label}! +{score} БАЛЛОВ"; protocol_entries.append(f"{current_time_str} - {label} (+{score}б)"); frames_since_last_score = 0
            except Exception as e: pass
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); draw = ImageDraw.Draw(pil_image)
            if trick_text: draw.text((50, 50), trick_text, font=font_trick, fill=(0, 255, 0))
            draw.text((50, 100), f"ВРЕМЯ: {current_time_str} | ИТОГО БАЛЛОВ: {total_score}", font=font_score, fill=(255, 255, 0))
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR); st_frame.image(image, channels="BGR", use_container_width=True)
        cap.release(); progress_text.empty(); progress_bar.empty()
        st.success(f"Обработка ИИ завершена! Итоговый счет: {total_score} баллов."); st.balloons()
        return total_score, protocol_entries

# --- 3. ФУНКЦИЯ АНАЛИЗА ОДНОГО СНИМКА (КАДРА) ---
# (Без изменений)
def analyze_snapshot(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image.flags.writeable = False
    results = None
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose: results = pose.process(image)
    image.flags.writeable = True; image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    predicted_pose = "other"; score = 0; label = POSE_NAMES_RU["other"]
    try:
        if results and results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark; pose_landmarks_list = []
            for landmark in landmarks: pose_landmarks_list.extend([landmark.x, landmark.y, landmark.z])
            prediction = model.predict([pose_landmarks_list]); predicted_pose = prediction[0]
            score = SCORES[predicted_pose]; label = POSE_NAMES_RU[predicted_pose]
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else: st.warning("Не удалось распознать позу на снимке.")
    except Exception as e:
        st.error(f"Ошибка при анализе снимка: {e}"); predicted_pose = "error"; label = "ОШИБКА"
    return image, predicted_pose, label, score

# --- 4. ФУНКЦИЯ ДЛЯ РЕЖИМА КАМЕРЫ ---
# (Без изменений)
def camera_mode():
    # ... (весь код camera_mode из версии 4.3 остается здесь) ...
    st.subheader("Режим: Онлайн (Камера)")
    camera_action = st.radio("Выберите действие:", ('Показать видео с камеры', 'Сделать снимок с таймером'))
    if camera_action == 'Показать видео с камеры':
        st.write("Нажмите кнопку ниже, чтобы включить веб-камеру.")
        run_camera = st.button("Включить камеру"); st_frame = st.empty()
        if run_camera:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened(): st.error("Не удалось получить доступ к камере."); return
            stop_button_placeholder = st.empty(); stop_button_pressed = stop_button_placeholder.button("⏹️ Выключить камеру", key="stop_cam")
            while cap.isOpened() and not stop_button_pressed:
                ret, frame = cap.read();
                if not ret: break
                frame = cv2.flip(frame, 1); st_frame.image(frame, channels="BGR", use_container_width=True)
                stop_button_pressed = stop_button_placeholder.button("⏹️ Выключить камеру", key="stop_cam")
            cap.release(); st_frame.empty(); stop_button_placeholder.empty(); st.info("Камера выключена.")
    elif camera_action == 'Сделать снимок с таймером':
        st.write("Выберите время таймера, встаньте в позу и нажмите 'Старт'.")
        timer_duration = st.selectbox("Время таймера (секунды):", [3, 5, 10])
        start_button = st.button(f"📸 Старт ({timer_duration} сек)"); st_frame = st.empty(); result_placeholder = st.empty()
        if start_button:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened(): st.error("Не удалось получить доступ к камере."); return
            start_time = time.time(); snapshot = None
            while cap.isOpened():
                ret, frame = cap.read();
                if not ret: break
                frame = cv2.flip(frame, 1); elapsed_time = time.time() - start_time; remaining_time = timer_duration - elapsed_time
                if remaining_time > 0:
                    timer_text = str(int(np.ceil(remaining_time)))
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); draw = ImageDraw.Draw(pil_image)
                    text_width, text_height = draw.textbbox((0,0), timer_text, font=font_timer)[2:]
                    text_x = (frame.shape[1] - text_width) // 2; text_y = (frame.shape[0] - text_height) // 2
                    draw.text((text_x+5, text_y+5), timer_text, font=font_timer, fill=(0, 0, 0, 128))
                    draw.text((text_x, text_y), timer_text, font=font_timer, fill=(255, 0, 0))
                    frame_with_timer = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    st_frame.image(frame_with_timer, channels="BGR", use_container_width=True); time.sleep(0.03)
                else: snapshot = frame.copy(); break
            cap.release(); st_frame.empty()
            if snapshot is not None:
                st.info("Анализирую снимок..."); analyzed_image, pose_name, pose_label, score = analyze_snapshot(snapshot)
                st.image(analyzed_image, channels="BGR", caption="Результат анализа снимка", use_container_width=True)
                if pose_name != "error":
                    if score > 0: result_placeholder.success(f"Распознана поза: **{pose_label}** (+{score} баллов)")
                    else: result_placeholder.info(f"Распознана поза: **{pose_label}** (0 баллов)")

# --- 5. ИНТЕРФЕЙС STREAMLIT ---
st.set_page_config(layout="wide")
st.title("🤖 Анализатор художественной гимнастики")

# --- Инициализация session_state ---
if 'manual_score' not in st.session_state: st.session_state.manual_score = 0
if 'manual_protocol' not in st.session_state: st.session_state.manual_protocol = []
# Новые переменные для состояния коррекции фото
if 'photo_analyzed' not in st.session_state: st.session_state.photo_analyzed = False
if 'correction_made' not in st.session_state: st.session_state.correction_made = False

# --- Боковая панель: ВЫБОР РЕЖИМА + НАСТРОЙКА COOLDOWN ---
st.sidebar.title("Настройки")
analysis_mode = st.sidebar.radio(
    "Выберите режим:",
    ('Автоматический (ИИ)', 'Ручной', 'Онлайн (Камера)', 'Анализ Фото'),
    key='mode_select', # Добавляем ключ, чтобы сбросить состояние при смене режима
    on_change=lambda: st.session_state.update(photo_analyzed=False, correction_made=False) # Сбрасываем флаги фото
)

cooldown_frames = 30
if analysis_mode == 'Автоматический (ИИ)':
     cooldown_frames = st.sidebar.slider(
         "'Период охлаждения' ИИ (кадры):", min_value=5, max_value=90, value=30,
         help="Сколько кадров ИИ должен 'молчать' после засчитывания трюка (30 кадров ≈ 1 сек)."
     )

settings = { "cooldown_frames": cooldown_frames }

# --- Основная часть ---

if analysis_mode == 'Онлайн (Камера)':
    camera_mode()

elif analysis_mode == 'Анализ Фото':
    st.subheader("Режим: Анализ Фото")
    st.write("Загрузите одно фото для анализа позы.")
    uploaded_image = st.file_uploader(
        "Выберите фото",
        type=['png', 'jpg', 'jpeg'],
        key='photo_uploader', # Ключ для виджета
        on_change=lambda: st.session_state.update(photo_analyzed=False, correction_made=False) # Сброс при загрузке нового фото
    )

    if uploaded_image is not None:
        image_bytes = uploaded_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            # --- Запускаем анализ только один раз или если фото изменилось ---
            if not st.session_state.photo_analyzed:
                st.info("Анализирую фото...")
                analyzed_image, pose_name, pose_label, score = analyze_snapshot(frame)
                # Сохраняем результат в session_state
                st.session_state.analyzed_image = analyzed_image
                st.session_state.predicted_pose = pose_name
                st.session_state.predicted_label = pose_label
                st.session_state.predicted_score = score
                st.session_state.photo_analyzed = True # Ставим флаг, что анализ сделан
                st.session_state.correction_made = False # Сбрасываем флаг коррекции
                st.rerun() # Перезапускаем скрипт, чтобы обновить интерфейс

            # --- Отображаем результат (всегда, если фото загружено) ---
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_image, caption="Оригинальное фото", use_container_width=True)
            with col2:
                # Берем результат из session_state
                if 'analyzed_image' in st.session_state:
                     st.image(st.session_state.analyzed_image, channels="BGR", caption="Результат анализа", use_container_width=True)

            # Выводим текстовый результат
            if 'predicted_pose' in st.session_state:
                pose_name = st.session_state.predicted_pose
                pose_label = st.session_state.predicted_label
                score = st.session_state.predicted_score

                if pose_name != "error":
                    st.write(f"**Предсказание ИИ:** {pose_label} ({score} баллов)")

                    # --- НОВЫЙ БЛОК: КНОПКИ КОРРЕКЦИИ ---
                    st.write("Предсказание верное?")
                    col_corr1, col_corr2, col_corr3 = st.columns([1,1,3]) # Колонки для кнопок

                    with col_corr1:
                        if st.button("👍 Правильно", key="correct_yes", disabled=st.session_state.correction_made):
                            st.session_state.correction_made = True
                            st.success("Отлично! Спасибо за подтверждение.")
                            # Здесь можно добавить код для сохранения 'правильного' примера (пока не делаем)
                            st.rerun() # Обновляем, чтобы заблокировать кнопки

                    with col_corr2:
                         # Показываем кнопку "Неправильно" только если еще не было коррекции
                        if st.button("👎 Неправильно", key="correct_no", disabled=st.session_state.correction_made):
                            st.session_state.correction_made = True # Ставим флаг, что началась коррекция
                            # Не выводим сообщение сразу, ждем выбора правильной позы
                            st.rerun() # Обновляем, чтобы показать selectbox

                    # --- Выбор правильной позы (появляется после нажатия "Неправильно") ---
                    if st.session_state.correction_made and 'predicted_pose' in st.session_state and st.session_state.predicted_pose != "error":
                         # Проверяем, была ли нажата кнопка "Правильно", чтобы не показывать selectbox зря
                         if not st.session_state.get('correct_yes_pressed', False): # Используем get для избежания ошибки при первом запуске
                              # Ищем индекс предсказанной позы для значения по умолчанию
                              try:
                                   default_index = POSE_OPTIONS.index(st.session_state.predicted_pose)
                              except ValueError:
                                   default_index = 0 # Если предсказанной позы нет в опциях (маловероятно)

                              correct_pose = st.selectbox(
                                   "Выберите правильную позу:",
                                   options=POSE_OPTIONS,
                                   format_func=lambda x: POSE_NAMES_RU[x], # Показываем русские названия
                                   index=default_index, # Предлагаем предсказанную позу по умолчанию
                                   key='correct_pose_select'
                              )
                              if st.button("✅ Подтвердить исправление", key="confirm_correction"):
                                   st.info(f"Спасибо! Исправление записано: {POSE_NAMES_RU[correct_pose]}")
                                   # Здесь можно добавить код для сохранения 'неправильного' примера и правильного ответа
                                   # Например, записать в файл: f"{uploaded_image.name},{st.session_state.predicted_pose},{correct_pose}"

                else:
                    st.error("Во время анализа фото произошла ошибка.")
        else:
            st.error("Не удалось прочитать файл изображения.")


else: # Режимы Авто(ИИ) и Ручной
    st.write("Загрузите видео с выступлением.")
    uploaded_file = st.file_uploader("Выберите видео-файл", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        video_bytes = uploaded_file.read()
        st.video(video_bytes)

        if analysis_mode == 'Автоматический (ИИ)':
            # ... (код для Авто ИИ без изменений) ...
             st.subheader("Режим: Автоматический (ИИ)")
             if st.button("Начать ИИ-анализ"):
                 st.info("Идет обработка ИИ... Это может занять некоторое время.")
                 ai_total_score, ai_protocol_entries = process_video_ai(video_bytes, settings)
                 st.subheader("📝 Детальный Протокол (ИИ)")
                 if not ai_protocol_entries: st.warning("ИИ не засчитал ни одного трюка.")
                 else:
                     for entry in ai_protocol_entries: st.write(entry)
                 report_text = f"ФИНАЛЬНЫЙ ПРОТОКОЛ АНАЛИЗА (ИИ)\nИтоговый счет: {ai_total_score} баллов\n\n--- Детальный Лог Трюков ---\n"
                 report_text += "\n".join(ai_protocol_entries) if ai_protocol_entries else "Ни одного трюка не засчитано."
                 report_text += f"\n\n---\nНастройки: Cooldown = {settings['cooldown_frames']} кадров"
                 st.download_button(label="⬇️ Скачать протокол ИИ (.txt)",data=report_text,file_name="gymnastics_report_AI.txt",mime="text/plain")

        elif analysis_mode == 'Ручной':
            # ... (код для Ручного режима без изменений) ...
            st.subheader("Режим: Ручной")
            st.write("Смотрите видео, ставьте на паузу и нажимайте кнопки для добавления баллов.")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("🤸‍♀️ Шпагат (+10)"): st.session_state.manual_score += 10; st.session_state.manual_protocol.append(f"{format_time_manual(time.time())} - ШПАГАТ (+10б)"); st.rerun()
            with col2:
                if st.button("🕊️ Ласточка (+8)"): st.session_state.manual_score += 8; st.session_state.manual_protocol.append(f"{format_time_manual(time.time())} - ЛАСТОЧКА (+8б)"); st.rerun()
            with col3:
                if st.button("🦵 Подъем ноги (+5)"): st.session_state.manual_score += 5; st.session_state.manual_protocol.append(f"{format_time_manual(time.time())} - ПОДЪЕМ НОГИ (+5б)"); st.rerun()
            with col4:
                if st.button("💍 Колечко (+15)"): st.session_state.manual_score += 15; st.session_state.manual_protocol.append(f"{format_time_manual(time.time())} - КОЛЕЧКО (+15б)"); st.rerun()
            st.metric("Текущий Ручной Счет:", st.session_state.manual_score)
            st.subheader("📝 Протокол Ручной Оценки")
            if not st.session_state.manual_protocol: st.info("Нажмите кнопки выше, чтобы добавить трюки.")
            else:
                for entry in reversed(st.session_state.manual_protocol): st.write(entry)
            if st.button("🔄 Сбросить ручной счет"): st.session_state.manual_score = 0; st.session_state.manual_protocol = []; st.rerun()
            if st.session_state.manual_protocol:
                report_text = f"ПРОТОКОЛ РУЧНОЙ ОЦЕНКИ\nИтоговый счет: {st.session_state.manual_score} баллов\n\n--- Детальный Лог Трюков ---\n"
                report_text += "\n".join(st.session_state.manual_protocol)
                st.download_button(label="⬇️ Скачать ручной протокол (.txt)",data=report_text,file_name="gymnastics_report_manual.txt",mime="text/plain")

st.markdown("---")
st.caption("Создано с помощью Streamlit, OpenCV и MediaPipe")
