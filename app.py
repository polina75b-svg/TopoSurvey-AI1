import math
import io
import csv

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pytesseract
import fitz

from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(
    page_title="TopoSurvey AI",
    page_icon="📐",
    layout="wide"
)

st.title("📐 TopoSurvey AI")
st.write(
    "Анализ топографических съёмок: "
    "распознавание чисел, масштаб, площадь и периметр."
)

st.warning(
    "Это экспериментальная версия. "
    "Проверяйте распознанные высоты, масштаб и границы "
    "по исходному чертежу."
)


# ----------------------------------------
# 1. Загрузка файла
# ----------------------------------------

uploaded_file = st.file_uploader(
    "Загрузите топографическую съёмку",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):

        document = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        if len(document) == 0:
            st.error("PDF не содержит страниц.")
            st.stop()

        page_number = st.selectbox(
            "Выберите страницу PDF",
            options=list(range(len(document))),
            format_func=lambda x: f"Страница {x + 1}"
        )

        page = document.load_page(page_number)

        # Увеличиваем разрешение страницы для OCR
        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False
        )

        image = Image.open(
            io.BytesIO(pix.tobytes("png"))
        ).convert("RGB")

    else:

        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

    st.subheader("Исходный чертёж")

    st.write(
        f"Размер изображения: "
        f"{image.width} × {image.height} пикселей"
    )
    st.image(image, caption="Исходный чертёж", use_container_width=True)
    # ----------------------------------------
    # 2. Распознавание чисел
    # ----------------------------------------

    st.subheader("Распознавание высотных отметок")

    if st.button("Распознать текст и числа"):

        image_array = np.array(image)

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        # Увеличиваем контраст для OCR
        processed = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        ).apply(gray)

        try:
            ocr_data = pytesseract.image_to_data(
                processed,
                lang="rus+eng",
                config="--psm 11",
                output_type=pytesseract.Output.DATAFRAME
            )

            ocr_data = ocr_data.dropna(
                subset=["text"]
            )

            ocr_data["text"] = (
                ocr_data["text"].astype(str).str.strip()
            )

            ocr_data = ocr_data[
                (ocr_data["text"] != "") &
                (ocr_data["conf"] >= 0)
            ].copy()

            st.session_state["ocr_data"] = ocr_data

            st.success(
                f"Найдено текстовых фрагментов: "
                f"{len(ocr_data)}"
            )

        except Exception as error:
            st.error(f"Ошибка OCR: {error}")

    if "ocr_data" in st.session_state:

        ocr_data = st.session_state["ocr_data"]

        st.dataframe(
            ocr_data[
                ["text", "conf", "left", "top"]
            ].rename(
                columns={
                    "text": "Распознанный текст",
                    "conf": "Уверенность OCR",
                    "left": "X, пиксели",
                    "top": "Y, пиксели"
                }
            ),
            use_container_width=True
        )

        st.caption(
            "OCR распознаёт текстовые фрагменты, "
            "но не определяет автоматически, "
            "какие числа являются высотными отметками."
        )

        csv_data = ocr_data.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Скачать результаты OCR (CSV)",
            data=csv_data,
            file_name="toposurvey_ocr.csv",
            mime="text/csv"
        )

    # ----------------------------------------
    # 3. Масштаб и измерения
    # ----------------------------------------

    st.divider()
    st.subheader("Масштаб и площадь участка")

    st.write(
        "Для измерений укажите две точки с известным "
        "расстоянием между ними. Затем отметьте вершины "
        "границы участка на изображении."
    )

    st.info(
        "Для первой версии желательно использовать "
        "чёткий чертёж с известными размерами."
    )

    # Уменьшаем изображение для удобства разметки.
    # Все координаты далее относятся к этому изображению.
    max_width = 1000

    if image.width > max_width:
        display_height = round(
            image.height * max_width / image.width
        )

        display_image = image.resize(
            (max_width, display_height)
        )
    else:
        display_image = image

    display_width, display_height = display_image.size

    st.caption(
        f"Размер изображения для разметки: "
        f"{display_width} × {display_height} пикселей"
    )

    if "scale_points" not in st.session_state:
        st.session_state["scale_points"] = []

    if "boundary_points" not in st.session_state:
        st.session_state["boundary_points"] = []

    if "current_click" not in st.session_state:
        st.session_state["current_click"] = None

    st.write("### Выберите действие")

    mode = st.radio(
        "Что вы хотите отметить?",
        [
            "Точка масштаба A",
            "Точка масштаба B",
            "Вершина границы участка"
        ],
        horizontal=True
    )

    clicked = streamlit_image_coordinates(
        display_image,
        key="survey_image"
    )

    if clicked is not None:
        st.session_state["current_click"] = (
            float(clicked["x"]),
            float(clicked["y"])
        )

    if st.session_state["current_click"] is not None:

        x, y = st.session_state["current_click"]

        st.write(
            f"Выбранная точка: X = {x:.1f}, Y = {y:.1f}"
        )

        if st.button("Сохранить выбранную точку"):

            point = (x, y)

            if mode == "Точка масштаба A":

                points = st.session_state["scale_points"]

                if len(points) == 0:
                    points.append(point)
                else:
                    points[0] = point

                st.session_state["scale_points"] = points

            elif mode == "Точка масштаба B":

                points = st.session_state["scale_points"]

                if len(points) == 0:
                    points.append(point)
                    st.warning(
                        "Точка A пока не задана. "
                        "Отметьте её отдельно."
                    )
                elif len(points) == 1:
                    points.append(point)
                else:
                    points[1] = point

                st.session_state["scale_points"] = points

            else:

                st.session_state["boundary_points"].append(
                    point
                )

            st.rerun()

    scale_points = st.session_state["scale_points"]

    st.write(
        f"Точек масштаба: {len(scale_points)} из 2"
    )

    known_distance = st.number_input(
        "Расстояние между точками A и B в метрах",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    meters_per_pixel = None

    if len(scale_points) == 2:

        ax, ay = scale_points[0]
        bx, by = scale_points[1]

        pixel_distance = math.hypot(
            bx - ax,
            by - ay
        )

        if pixel_distance > 0 and known_distance > 0:

            meters_per_pixel = (
                known_distance / pixel_distance
            )

            st.success(
                f"Масштаб измерения: "
                f"{meters_per_pixel:.6f} м/пиксель"
            )

    st.write("### Граница участка")

    boundary_points = st.session_state["boundary_points"]

    st.write(
        f"Отмечено вершин: {len(boundary_points)}"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Удалить последнюю вершину"):
            if boundary_points:
                boundary_points.pop()
                st.session_state["boundary_points"] = (
                    boundary_points
                )
                st.rerun()

    with col2:
        if st.button("Очистить границу"):
            st.session_state["boundary_points"] = []
            st.rerun()

    if len(boundary_points) >= 3:

        if meters_per_pixel is not None:

            # Формула площади многоугольника
            # по координатам вершин (формула Гаусса).
            area_pixels = 0.0

            for i in range(len(boundary_points)):

                x1, y1 = boundary_points[i]

                x2, y2 = boundary_points[
                    (i + 1) % len(boundary_points)
                ]

                area_pixels += x1 * y2 - x2 * y1

            area_pixels = abs(area_pixels) / 2

            area_m2 = (
                area_pixels * meters_per_pixel ** 2
            )

            perimeter_pixels = 0.0

            for i in range(len(boundary_points)):

                x1, y1 = boundary_points[i]

                x2, y2 = boundary_points[
                    (i + 1) % len(boundary_points)
                ]

                perimeter_pixels += math.hypot(
                    x2 - x1,
                    y2 - y1
                )

            perimeter_m = (
                perimeter_pixels * meters_per_pixel
            )

            st.metric(
                "Площадь участка",
                f"{area_m2:.2f} м²"
            )

            st.metric(
                "Периметр участка",
                f"{perimeter_m:.2f} м"
            )

            st.metric(
                "Площадь в гектарах",
                f"{area_m2 / 10000:.4f} га"
            )

            st.caption(
                "Расчёт выполнен по отмеченному многоугольнику "
                "и заданному масштабу. Точность зависит от "
                "правильности разметки и масштаба."
            )

        else:

            st.warning(
                "Сначала отметьте две точки масштаба "
                "и укажите расстояние между ними."
            )

    else:

        st.info(
            "Для расчёта площади отметьте не менее "
            "трёх вершин границы участка."
        )

else:

    st.info(
        "Загрузите PDF или изображение, чтобы начать анализ."
    )

st.divider()

st.caption(
    "TopoSurvey AI | Экспериментальный инструмент "
    "для анализа топографических материалов"
)
