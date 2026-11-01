# ============================================================
# app.py
# HateSpeechAI - AI Content Moderation Dashboard
# ============================================================

# Import Streamlit for the web interface.
import streamlit as st

# Import the moderation engine.
from moderation import moderate_text


# ============================================================
# PAGE CONFIGURATION
# ============================================================

# Configure the Streamlit page.
st.set_page_config(
    page_title="HateSpeechAI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

# Add custom styling to make the application look professional.
st.markdown(
    """
    <style>

    /* Main application title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    /* Application subtitle */
    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    /* Result card */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-top: 20px;
    }

    /* Decision text */
    .decision {
        font-size: 30px;
        font-weight: 700;
    }

    /* Small information text */
    .small-text {
        color: #666666;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# APPLICATION HEADER
# ============================================================

# Display the main application title.
st.markdown(
    '<div class="main-title">🛡️ HateSpeechAI</div>',
    unsafe_allow_html=True
)


# Display the application description.
st.markdown(
    """
    <div class="subtitle">
    AI-powered hate speech and offensive content moderation
    using a fine-tuned RoBERTa model.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

# Create the sidebar.
with st.sidebar:

    # Sidebar heading.
    st.header("🤖 AI Model")

    # Display model information.
    st.write("**Model:** RoBERTa")

    st.write(
        "**Task:** 3-class text classification"
    )

    # Display the supported classes.
    st.subheader("Classes")

    st.write("🔴 Hate Speech")

    st.write("🟠 Offensive")

    st.write("🟢 Neither")

    # Display moderation rules.
    st.subheader("Moderation Rules")

    st.write(
        "🔴 High-confidence hate speech → REMOVE"
    )

    st.write(
        "🟠 Potentially harmful content → REVIEW"
    )

    st.write(
        "🟢 Safe content → ALLOW"
    )

    # Divider.
    st.divider()

    # Display project information.
    st.subheader("About")

    st.write(
        """
        HateSpeechAI analyzes submitted text,
        predicts its content category, calculates
        confidence scores, and recommends an
        appropriate moderation action.
        """
    )


# ============================================================
# MAIN INPUT SECTION
# ============================================================

# Create a section heading.
st.header("🔍 Analyze Social Media Content")


# Explain what the user should do.
st.write(
    "Enter a social media post below and click **Analyze Content**."
)


# Create the text input area.
text = st.text_area(
    "Social Media Post",
    height=180,
    placeholder=(
        "Example: What a beautiful sunset today!"
    )
)


# Create the analysis button.
analyze_button = st.button(
    "🔎 Analyze Content",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

# Run the model only after the button is pressed.
if analyze_button:

    # Check whether the input is empty.
    if not text.strip():

        # Display an error message.
        st.error(
            "Please enter some text before analyzing."
        )

    else:

        # Show a loading indicator while the model runs.
        with st.spinner(
            "Analyzing content with RoBERTa..."
        ):

            try:

                # Send the text to the moderation engine.
                result = moderate_text(
                    text
                )

            except Exception as error:

                # Display an error if something unexpected occurs.
                st.error(
                    f"An error occurred: {error}"
                )

                # Stop further processing.
                st.stop()


        # ====================================================
        # EXTRACT RESULTS
        # ====================================================

        # Get the predicted classification.
        classification = result[
            "classification"
        ]

        # Get model confidence.
        confidence = result[
            "confidence"
        ]

        # Get moderation decision.
        decision = result[
            "decision"
        ]

        # Get explanation.
        reason = result[
            "reason"
        ]

        # Get class probabilities.
        probabilities = result[
            "probabilities"
        ]


        # ====================================================
        # RESULT HEADER
        # ====================================================

        st.divider()

        st.header("📊 Moderation Result")


        # ====================================================
        # TOP METRICS
        # ====================================================

        # Create three columns.
        col1, col2, col3 = st.columns(3)


        # Classification metric.
        with col1:

            st.metric(
                "Classification",
                classification
            )


        # Confidence metric.
        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


        # Decision metric.
        with col3:

            st.metric(
                "Decision",
                decision
            )


        # ====================================================
        # DECISION MESSAGE
        # ====================================================

        if decision == "REMOVE":

            st.error(
                f"🚫 **REMOVE** — {reason}"
            )

        elif decision == "REVIEW":

            st.warning(
                f"⚠️ **REVIEW** — {reason}"
            )

        else:

            st.success(
                f"✅ **ALLOW** — {reason}"
            )


        # ====================================================
        # PROBABILITY SECTION
        # ====================================================

        st.subheader(
            "Class Probabilities"
        )


        # Create three probability columns.
        prob1, prob2, prob3 = st.columns(3)


        # Hate Speech probability.
        with prob1:

            hate_probability = probabilities[
                "Hate Speech"
            ]

            st.metric(
                "🔴 Hate Speech",
                f"{hate_probability * 100:.2f}%"
            )

            st.progress(
                hate_probability
            )


        # Offensive probability.
        with prob2:

            offensive_probability = probabilities[
                "Offensive"
            ]

            st.metric(
                "🟠 Offensive",
                f"{offensive_probability * 100:.2f}%"
            )

            st.progress(
                offensive_probability
            )


        # Neither probability.
        with prob3:

            neither_probability = probabilities[
                "Neither"
            ]

            st.metric(
                "🟢 Neither",
                f"{neither_probability * 100:.2f}%"
            )

            st.progress(
                neither_probability
            )


        # ====================================================
        # ANALYZED TEXT
        # ====================================================

        st.subheader(
            "Analyzed Content"
        )

        # Display the submitted text.
        st.info(
            text
        )


        # ====================================================
        # MODERATION EXPLANATION
        # ====================================================

        st.subheader(
            "🤖 AI Moderation Explanation"
        )

        st.write(
            reason
        )


# ============================================================
# EXAMPLE POSTS
# ============================================================

# Add a divider.
st.divider()


# Display examples section.
st.header("🧪 Try Example Posts")


# Create example posts.
examples = {

    "Safe Content":
        "What a beautiful sunset today!",

    "Positive Message":
        "Everyone deserves respect regardless of who they are.",

    "Potentially Harmful":
        "You are such an idiot.",

    "Hateful Example":
        "I hate people from that group and they should not exist."
}


# Display examples as buttons.
for example_name, example_text in examples.items():

    # Create an expandable section.
    with st.expander(
        example_name
    ):

        # Display example text.
        st.write(
            example_text
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center">

    <p>
    🛡️ <b>HateSpeechAI</b>
    </p>

    <p class="small-text">
    AI-powered content moderation using fine-tuned RoBERTa.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)