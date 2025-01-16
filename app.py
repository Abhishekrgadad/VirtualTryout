from flask import Flask, render_template, request
from transformers import pipeline
from PIL import Image, ImageDraw
import numpy as np
import os
import mediapipe as mp

app = Flask(__name__)

# Load Segformer model
print("Loading Segformer model...")
pipe = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes")
print("Model loaded!")

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Ensure the static directory exists for saving images
os.makedirs("static", exist_ok=True)


def overlay_cloth_on_person(user_img, cloth_img):
    """
    Use Mediapipe to find body landmarks on the user image,
    and fit the cloth image accurately based on the landmarks.
    """
    # Convert user image to numpy array
    user_np = np.array(user_img)

    # Detect body landmarks
    results = pose.process(user_np)

    if not results.pose_landmarks:
        return "static/result_image.jpg"  # No landmarks found

    # Draw pose landmarks for visualization
    draw = ImageDraw.Draw(user_img)
    for landmark in results.pose_landmarks.landmark:
        x = int(landmark.x * user_img.width)
        y = int(landmark.y * user_img.height)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="red")

    # Placeholder cloth overlay logic
    cloth_np = np.array(cloth_img)
    cloth_resized = cloth_img.resize((200, 300))  # Adjust size for testing
    user_img.paste(cloth_resized, (100, 150), cloth_resized)  # Example placement

    # Save result image
    result_path = "static/result_image.jpg"
    user_img.save(result_path)
    return result_path


@app.route("/", methods=["GET", "POST"])
def index():
    user_path = cloth_path = result_path = None

    if request.method == "POST":
        # Get uploaded files
        user_image = request.files.get("user_image")
        cloth_image = request.files.get("cloth_image")

        if user_image:
            user_img = Image.open(user_image).convert("RGB")
            user_path = "static/user_uploaded.jpg"
            user_img.save(user_path)

        if cloth_image:
            cloth_img = Image.open(cloth_image).convert("RGBA")
            cloth_path = "static/cloth_uploaded.png"
            cloth_img.save(cloth_path)

        # Process images if both are uploaded
        if user_image and cloth_image:
            result_path = overlay_cloth_on_person(user_img, cloth_img)

    return render_template(
        "index.html",
        user_image=user_path,
        cloth_image=cloth_path,
        result_image=result_path,
    )


if __name__ == "__main__":
    app.run(debug=True)
