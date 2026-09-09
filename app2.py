import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import pandas as pd
import io


# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------

st.set_page_config(
    page_title="AI Object Detection Agent",
    page_icon="🔍",
    layout="wide"
)


# ----------------------------------------------------------
# LOAD YOLO MODEL
# ----------------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


try:
    model = load_model()
except Exception:
    st.error(
        "AI model could not be loaded. "
        "Please make sure yolo11n.pt is in the same folder as app2.py."
    )
    st.stop()


# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------

st.title("🔍 AI Object Detection Agent")

st.write(
    "Upload an image or take a photo. "
    "The AI model will detect objects and show their confidence."
)

st.divider()


# ----------------------------------------------------------
# SIDEBAR SETTINGS
# ----------------------------------------------------------

st.sidebar.header("⚙️ Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05,
    help="Only objects above this confidence level will be shown."
)

all_class_names = sorted(model.names.values())

selected_classes = st.sidebar.multiselect(
    "Object Filter",
    options=all_class_names,
    default=[],
    help="Leave empty to detect all supported objects."
)

st.sidebar.divider()

st.sidebar.write("**AI Model:** YOLO11n")
st.sidebar.write("**Task:** Object Detection")


# ----------------------------------------------------------
# MODE SELECTION
# ----------------------------------------------------------

mode = st.radio(
    "Select Input Method",
    ["📷 Upload Image", "📸 Camera"],
    horizontal=True
)


# ----------------------------------------------------------
# GET CLASS IDS
# ----------------------------------------------------------

def get_class_ids(selected_names):

    if not selected_names:
        return None

    name_to_id = {
        name: class_id
        for class_id, name in model.names.items()
    }

    return [
        name_to_id[name]
        for name in selected_names
    ]


# ----------------------------------------------------------
# DETECTION FUNCTION
# ----------------------------------------------------------

def detect_objects(image):

    class_ids = get_class_ids(selected_classes)

    results = model.predict(
        image,
        conf=confidence_threshold,
        classes=class_ids,
        verbose=False
    )

    detected_objects = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = model.names[class_id]

            detected_objects.append({
                "name": object_name,
                "confidence": confidence
            })

    detected_image = results[0].plot(
        conf=True,
        labels=True,
        boxes=True
    )

    return detected_image, detected_objects


# ----------------------------------------------------------
# DISPLAY RESULTS
# ----------------------------------------------------------

def display_results(original_image, detected_image, objects):

    total_objects = len(objects)

    unique_objects = len(
        set(
            item["name"]
            for item in objects
        )
    )

    if total_objects > 0:

        average_confidence = (
            sum(
                item["confidence"]
                for item in objects
            )
            / total_objects
        )

    else:

        average_confidence = 0


    # ------------------------------------------------------
    # OVERVIEW
    # ------------------------------------------------------

    st.subheader("📊 Detection Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Objects Detected",
            total_objects
        )

    with col2:
        st.metric(
            "Object Types",
            unique_objects
        )

    with col3:
        st.metric(
            "Average Confidence",
            f"{average_confidence * 100:.1f}%"
        )

    with col4:
        st.metric(
            "AI Model",
            "YOLO11n"
        )


    st.divider()


    # ------------------------------------------------------
    # IMAGE RESULTS
    # ------------------------------------------------------

    st.subheader("🖼️ Visual Results")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Original Image**")

        st.image(
            original_image,
            use_container_width=True
        )

    with col2:

        st.write("**AI Detection Result**")

        st.image(
            detected_image,
            use_container_width=True
        )


    st.divider()


    # ------------------------------------------------------
    # OBJECT RESULTS
    # ------------------------------------------------------

    st.subheader("🎯 Detected Objects")


    if not objects:

        st.warning(
            "No objects were detected. "
            "Try lowering the confidence threshold."
        )

        return


    object_counts = Counter(
        item["name"]
        for item in objects
    )


    # ------------------------------------------------------
    # CLEAN OBJECT SUMMARY
    # ------------------------------------------------------

    summary_data = []

    for name, count in object_counts.items():

        confidence_values = [
            item["confidence"]
            for item in objects
            if item["name"] == name
        ]

        best_confidence = max(confidence_values)

        summary_data.append({
            "Object": name.capitalize(),
            "Quantity": count,
            "Confidence": f"{best_confidence * 100:.1f}%"
        })


    summary_df = pd.DataFrame(summary_data)

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )


    # ------------------------------------------------------
    # INDIVIDUAL DETECTIONS
    # ------------------------------------------------------

    st.subheader("📋 Detection Details")

    details_data = []

    for item in objects:

        details_data.append({
            "Object": item["name"].capitalize(),
            "Confidence": f"{item['confidence'] * 100:.1f}%"
        })


    details_df = pd.DataFrame(details_data)

    st.dataframe(
        details_df,
        use_container_width=True,
        hide_index=True
    )


    # ------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------

    st.subheader("📈 Confidence")

    for item in objects:

        object_name = item["name"].capitalize()
        confidence = item["confidence"]

        st.write(
            f"{object_name} — {confidence * 100:.1f}%"
        )

        st.progress(confidence)


    st.divider()


    # ------------------------------------------------------
    # DOWNLOAD RESULT
    # ------------------------------------------------------

    st.subheader("⬇️ Export Result")

    buffer = io.BytesIO()

    Image.fromarray(
        detected_image
    ).save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    st.download_button(
        label="Download Detection Image",
        data=buffer,
        file_name="AI_Detection_Result.png",
        mime="image/png"
    )


# ----------------------------------------------------------
# IMAGE UPLOAD MODE
# ----------------------------------------------------------

if mode == "📷 Upload Image":

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )


    if uploaded_file is None:

        st.info(
            "Upload a JPG, JPEG, or PNG image to start detection."
        )

    else:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception:

            st.error(
                "The uploaded file is not a valid image."
            )

            st.stop()


        with st.spinner("Analyzing image..."):

            try:

                detected_image, detected_objects = detect_objects(
                    image
                )

            except Exception:

                st.error(
                    "The image could not be analyzed. "
                    "Please try another image."
                )

                st.stop()


        display_results(
            image,
            detected_image,
            detected_objects
        )


# ----------------------------------------------------------
# CAMERA MODE
# ----------------------------------------------------------

else:

    st.subheader("📸 Camera")

    camera_photo = st.camera_input(
        "Take a picture"
    )


    if camera_photo is None:

        st.info(
            "Take a picture using your camera to start detection."
        )

    else:

        try:

            image = Image.open(
                camera_photo
            ).convert("RGB")

        except Exception:

            st.error(
                "The camera image could not be processed."
            )

            st.stop()


        with st.spinner("Analyzing camera image..."):

            try:

                detected_image, detected_objects = detect_objects(
                    image
                )

            except Exception:

                st.error(
                    "The camera image could not be analyzed."
                )

                st.stop()


        display_results(
            image,
            detected_image,
            detected_objects
        )
