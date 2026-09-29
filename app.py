import streamlit as st
from PIL import Image
from transformers import pipeline
import torch


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="WasteWise AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 20px;
        color: #666;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }

    .result-title {
        font-size: 32px;
        font-weight: 700;
    }

    .confidence {
        font-size: 19px;
        margin-top: 8px;
    }

    .recommendation {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #ddd;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "microsoft/resnet-50"


# ============================================================
# WASTE INFORMATION
# ============================================================

WASTE_INFO = {

    "cardboard": {
        "category": "Recyclable",
        "action": "Place clean and dry cardboard in the recycling stream.",
        "tip": "Flatten cardboard boxes and remove plastic or food contamination."
    },

    "paper": {
        "category": "Recyclable",
        "action": "Place clean paper in the paper recycling stream.",
        "tip": "Avoid recycling heavily contaminated or wet paper."
    },

    "plastic": {
        "category": "Potentially Recyclable",
        "action": "Check your local recycling rules before disposal.",
        "tip": "Rinse containers when required and separate caps where local rules require it."
    },

    "glass": {
        "category": "Potentially Recyclable",
        "action": "Use the appropriate glass-recycling collection where available.",
        "tip": "Do not mix broken glass with ordinary recycling unless your local system allows it."
    },

    "metal": {
        "category": "Potentially Recyclable",
        "action": "Place accepted metal packaging in the appropriate recycling stream.",
        "tip": "Empty and rinse food or drink containers when required."
    },

    "organic": {
        "category": "Organic Waste",
        "action": "Place suitable food or garden waste in an organic-waste stream where available.",
        "tip": "Follow your local composting or organic-waste rules."
    },

    "trash": {
        "category": "General Waste",
        "action": "Dispose of it through the appropriate general-waste stream.",
        "tip": "Check whether the item has a specialized recycling or collection option."
    }
}


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_model():

    device = 0 if torch.cuda.is_available() else -1

    model = pipeline(
        "image-classification",
        model=MODEL_NAME,
        device=device
    )

    return model


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("WasteWise AI")

    st.write(
        "AI-powered waste identification and recycling awareness assistant."
    )

    st.divider()

    st.subheader("AI Model")

    st.code(MODEL_NAME)

    st.divider()

    st.subheader("Supported Input")

    st.write("JPG")
    st.write("JPEG")
    st.write("PNG")
    st.write("WEBP")

    st.divider()

    st.caption(
        "This application provides an experimental AI prediction. "
        "Always follow your local waste-management rules."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">WasteWise AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Waste Classification & Recycling Assistant'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading AI model..."):

    try:

        classifier = load_model()

    except Exception as error:

        st.error("Unable to load the AI model.")

        st.exception(error)

        st.stop()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Analyze Waste")

input_method = st.radio(
    "Choose image source",
    [
        "Upload Image",
        "Use Camera"
    ],
    horizontal=True
)


image = None


# ============================================================
# UPLOAD IMAGE
# ============================================================

if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


# ============================================================
# CAMERA
# ============================================================

else:

    camera_image = st.camera_input(
        "Take a picture of the waste"
    )

    if camera_image:

        image = Image.open(
            camera_image
        ).convert("RGB")


# ============================================================
# ANALYZE
# ============================================================

if image is not None:

    st.divider()

    col1, col2 = st.columns(
        [1, 1]
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with col1:

        st.subheader("Input Image")

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------------------------------
    # BUTTON
    # --------------------------------------------------------

    with col2:

        st.subheader("AI Analysis")

        analyze = st.button(
            "Analyze Waste",
            type="primary",
            use_container_width=True
        )

        if analyze:

            with st.spinner(
                "Analyzing image..."
            ):

                try:

                    predictions = classifier(
                        image,
                        top_k=5
                    )

                except Exception as error:

                    st.error(
                        "Image analysis failed."
                    )

                    st.exception(error)

                    st.stop()


            # =================================================
            # BEST RESULT
            # =================================================

            best = predictions[0]

            label = best["label"]

            confidence = (
                best["score"] * 100
            )


            # =================================================
            # DISPLAY RESULT
            # =================================================

            st.markdown(
                f"""
                <div class="result-card">

                <div class="result-title">
                {label}
                </div>

                <div class="confidence">
                Confidence: {confidence:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(
                    best["score"],
                    1.0
                )
            )


            # =================================================
            # WASTE CATEGORY MATCH
            # =================================================

            normalized_label = label.lower()

            matched_info = None

            for waste_type, info in WASTE_INFO.items():

                if waste_type in normalized_label:

                    matched_info = info

                    break


            if matched_info:

                st.subheader(
                    "Waste Management Recommendation"
                )

                st.markdown(
                    f"""
                    <div class="recommendation">

                    <strong>Category:</strong>
                    {matched_info["category"]}

                    <br><br>

                    <strong>Recommended Action:</strong>
                    {matched_info["action"]}

                    <br><br>

                    <strong>Tip:</strong>
                    {matched_info["tip"]}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.info(
                    "The AI identified the object, but it could not "
                    "reliably map the prediction to a WasteWise waste category. "
                    "Check local recycling guidance before disposal."
                )


            # =================================================
            # TOP PREDICTIONS
            # =================================================

            st.subheader(
                "Top AI Predictions"
            )

            for prediction in predictions:

                prediction_label = prediction[
                    "label"
                ]

                prediction_score = (
                    prediction["score"] * 100
                )

                st.write(
                    f"{prediction_label} — "
                    f"{prediction_score:.2f}%"
                )

                st.progress(
                    min(
                        prediction["score"],
                        1.0
                    )
                )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.header("How WasteWise AI Works")

step1, step2, step3 = st.columns(3)

with step1:

    st.subheader("1. Upload")

    st.write(
        "Upload a photo of a waste item or capture one using your camera."
    )

with step2:

    st.subheader("2. AI Analysis")

    st.write(
        "The computer-vision model analyzes the image and returns predictions."
    )

with step3:

    st.subheader("3. Guidance")

    st.write(
        "WasteWise provides recycling and disposal awareness guidance."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    Important: WasteWise AI is a prototype. Image classification can be
    incorrect, and recyclability depends on material composition,
    contamination, local facilities, and municipal rules. Do not use
    the prediction as the sole basis for hazardous-waste disposal.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"WasteWise AI | Model: {MODEL_NAME}"
)

