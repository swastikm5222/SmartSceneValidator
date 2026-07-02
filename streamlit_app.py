import streamlit as st
import requests

st.set_page_config(
    page_title="SMC AI Image Validator",
    page_icon="📷",
    layout="wide"
)

st.title("📷 SMC AI Image Validator")

label = st.selectbox(
    "Select Label",
    [
        "Selfie with Person Met",
        "Front Image With Name Board",
        "Kitchen Photograph",
        "Front View of Property",
        "Approach To Property",
        "Interior Property"
    ]
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

API_ENDPOINTS = {
    "Selfie with Person Met": "http://127.0.0.1:8000/validate/selfie",
    "Front Image With Name Board": "http://127.0.0.1:8000/validate/name-board",
    "Kitchen Photograph": "http://127.0.0.1:8000/validate/kitchen",
    "Front View of Property": "http://127.0.0.1:8000/validate/property-front",
    "Approach To Property": "http://127.0.0.1:8000/validate/approach-property",
    "Interior Property": "http://127.0.0.1:8000/validate/interior-property",
}


def _safe_call_api(api_url, files):
    
    try:
        response = requests.post(api_url, files=files, timeout=30)
    except requests.exceptions.ConnectionError:
        return None, (
            "Could not reach the validation API. Is the FastAPI server "
            "running at http://127.0.0.1:8000?"
        )
    except requests.exceptions.Timeout:
        return None, "The validation API took too long to respond (timed out)."
    except requests.exceptions.RequestException as exc:
        return None, f"Request to the validation API failed: {exc}"

    if response.status_code != 200:
       
        try:
            body = response.json()
            detail = body.get("reason") or body.get("detail") or response.text
        except ValueError:
            detail = response.text or f"HTTP {response.status_code}"
        return None, f"API returned an error (HTTP {response.status_code}): {detail}"

    try:
        return response.json(), None
    except ValueError:
        return None, "API returned a response that wasn't valid JSON."


if uploaded_file:

    api_url = API_ENDPOINTS[label]

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(
            uploaded_file,
            caption="Uploaded Image",
            width=250
        )

    with col2:
        with st.spinner("Validating..."):
            result, error = _safe_call_api(api_url, files)

        st.subheader("Validation Result")

        if error:
            st.error(error)
        elif result is None:
            st.error("No result returned from the API.")
        else:
            status = result.get("status")
            reason = result.get("reason", "No reason provided.")

            if status == "VALID":
                st.success(reason)
            elif status == "ERROR":
                st.warning(f"Validator error: {reason}")
            else:
                st.error(reason)

            if "faces_detected" in result:
                st.metric(
                    "Faces Detected",
                    result["faces_detected"]
                )

            with st.expander("Raw API Response"):
                st.json(result)
