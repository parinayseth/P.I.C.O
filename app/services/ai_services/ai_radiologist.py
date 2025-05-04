import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import json
import random
import tempfile
import re
from google import genai
from google.genai import types
from PIL import Image, ImageColor, ImageDraw, ImageFont


import subprocess
import sys

def install_missing_imports(*modules):
    """
    Checks if the given Python modules are installed and attempts to install
    them using pip if they are not found.

    Args:
        *modules: A variable number of strings, where each string is the
                  name of a Python module to check and potentially install.
    """
    for module in modules:
        try:
            __import__(module)
            print(f"'{module}' is already installed.")
        except ImportError:
            print(f"'{module}' is not installed. Attempting to install...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                print(f"Successfully installed '{module}'.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing '{module}': {e}")
            except FileNotFoundError:
                print("Error: pip command not found. Please ensure pip is installed and in your system's PATH.")


# install_missing_imports("dicom2jpg", "pydicom")
import dicom2jpg
import pydicom
# Function to load .dcm files from the folder
def load_dicom_images(folder_path):
    if not os.path.isdir(folder_path):
        st.error(f"The folder at {folder_path} does not exist. Please check the path.")
        return [], []

    dcm_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]

    if not dcm_files:
        st.warning(f"No DICOM files found in the folder: {folder_path}")
        return [], []

    dcm_files.sort()
    dicom_images = [pydicom.dcmread(os.path.join(folder_path, f)) for f in dcm_files]
    images = [dcm.pixel_array for dcm in dicom_images]
    return images, dcm_files

# Function to call the LLM and process bounding boxes
def call_llm(img: Image, prompt: str) -> str:
    system_prompt = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
    If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
    Output a json list where each entry contains the 2D bounding box in "box_2d" and a text label in "label".
    """

    client = genai.Client(api_key="")
    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=[prompt, img],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.5,
            safety_settings=[types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            )],
        ),
    )
    return response.text

def parse_json(json_input: str) -> str:
    match = re.search(r"```json\n(.*?)```", json_input, re.DOTALL)
    json_input = match.group(1) if match else ""
    return json_input

def plot_bounding_boxes(img: Image, bounding_boxes: str) -> Image:
    width, height = img.size
    colors = [colorname for colorname in ImageColor.colormap.keys()]
    draw = ImageDraw.Draw(img)

    bounding_boxes = parse_json(bounding_boxes)

    for bounding_box in json.loads(bounding_boxes):
        color = random.choice(colors)

        abs_y1 = int(bounding_box["box_2d"][0] / 1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1] / 1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2] / 1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
        draw.text(
            (abs_x1 + 8, abs_y1 + 6),
            bounding_box["label"],
            fill=color,
            font=ImageFont.truetype("/home/warmachine/codes/gitdir/travis-scripts/fonts/Arial.ttf", size=14),
        )

    return img

# Streamlit Application
st.set_page_config(page_title="DICOM Viewer and Bounding Box Detection")
st.markdown("<h1 style='text-align: center;'>DICOM Image Viewer & Bounding Box Detection</h1>", unsafe_allow_html=True)

# Sidebar selection between DICOM Viewer and AI Bounding Box Detection
option = st.sidebar.radio(
    "Select Functionality",
    ["DICOM Viewer", "AI Bounding Box Detection"]
)

# **DICOM Viewer** UI
if option == "DICOM Viewer":
    st.sidebar.header("DICOM Image Viewer")

    # Use Streamlit's column layout to center the folder path input
    col1, col2, col3 = st.columns([1, 2, 1])

    # Folder path input for DICOM viewer
    with col2:
        folder_path = st.text_input("Enter the folder path containing DICOM files", "")

    if folder_path:
        images, dcm_files = load_dicom_images(folder_path)

        if images:
            # Slider for selecting the DICOM image index
            image_index = st.slider('Select Image Index:', 0, len(images) - 1, 0)

            # Display selected DICOM image
            selected_image = images[image_index]
            selected_file = dcm_files[image_index]

            fig, ax = plt.subplots()
            ax.imshow(selected_image, cmap="gray")
            ax.set_title(f"Image: {selected_file}")
            st.pyplot(fig)

            # Add buttons for image conversion
            st.subheader("DICOM Image Conversion")

            # Convert selected DICOM to JPG
            if st.button("Convert to JPG"):
                dicom_img = dcm_files[image_index]  # Get the file path for the selected DICOM
                try:
                    dicom2jpg.dicom2jpg(folder_path + dicom_img, )  # Convert to JPG format
                    #print(output)
                    st.success(f"Image {folder_path + dicom_img} successfully converted to JPG.")
                except Exception as e:
                    st.error(f"Error converting {dicom_img} to JPG: {e}")

            # Convert all DICOM files in folder to BMP format
            if st.button("Convert All to BMP"):
                try:
                    dicom2jpg.dicom2bmp(folder_path, target_root=".")  # specify export location
                    st.success("All DICOM files successfully converted to BMP.")
                except Exception as e:
                    st.error(f"Error converting DICOM files to BMP: {e}")

            # Convert all DICOM files in folder to PNG format
            if st.button("Convert All to PNG"):
                try:
                    dicom2jpg.dicom2png(folder_path)
                    st.success("All DICOM files successfully converted to PNG.")
                except Exception as e:
                    st.error(f"Error converting DICOM files to PNG: {e}")

            # Convert a single DICOM file to numpy.ndarray for further use
            if st.button("Convert to NumPy Array"):
                dicom_img = dcm_files[image_index]  # Get the selected DICOM image
                try:
                    img_data = dicom2jpg.dicom2img(dicom_img)
                    st.success(f"Image {dicom_img} successfully converted to NumPy array.")
                    # Optionally, you can display or process img_data here as needed
                except Exception as e:
                    st.error(f"Error converting {dicom_img} to NumPy array: {e}")

# **AI Bounding Box Detection** UI
elif option == "AI Bounding Box Detection":
    st.sidebar.header("AI Bounding Box Detection")

    # Image upload for bounding box detection
    uploaded_image = st.sidebar.file_uploader("Upload an image for bounding box detection", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Allow user to input prompt for object detection
        prompt = st.text_input("Enter prompt for object detection")
        if st.button("Run Bounding Box Detection") and prompt:
            # Save the uploaded image to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_image.getbuffer())
            image_path = temp_file.name

            img = Image.open(image_path)
            width, height = img.size
            resized_image = img.resize((1024, int(1024 * height / width)), Image.Resampling.LANCZOS)

            os.unlink(image_path)

            with st.spinner("Running object detection..."):
                response = call_llm(resized_image, prompt)
                plotted_image = plot_bounding_boxes(resized_image, response)
            st.image(plotted_image)
