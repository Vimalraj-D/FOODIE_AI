from flask import Flask, request, render_template, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import cv2
from ultralytics import YOLO
import base64
from io import BytesIO
import numpy as np

app = Flask(__name__)

model_path = "backup_save"
tokenizer = AutoTokenizer.from_pretrained(model_path)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"  
)
model.eval()

yolo_model = YOLO("runs/detect/train/weights/best.pt")

def ask_model(question):
    print("Generating response for:", question)
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")

    with torch.no_grad():
        output = model.generate(**inputs, max_length=1024, temperature=0.7, top_p=0.9, do_sample=True)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.split("### Response:")[-1].strip()
    print(f"Generated response (length: {len(response)} chars):", response)
    return response

def detect_food_from_image(img):
    results = yolo_model(img)
    detected_food = None
    
    for result in results:
        if result.boxes: 
            labels = result.names  
            class_idx = int(result.boxes.cls[0])  
            detected_food = labels[class_idx]
            break  
    
    return detected_food

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    food_name = request.form.get('food_name')  
    file = request.files.get('food_image')     
    print("Received food_name:", food_name)
    if file:
        print("Received image file:", file.filename)

    bot_response = None
    img_result = None

    if food_name:  
        bot_response = ask_model(f"Provide detailed information about {food_name}")
    elif file:  
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        detected_food = detect_food_from_image(img)
        print("Detected food from image:", detected_food)
        
        if detected_food:
            bot_response = ask_model(f"Provide detailed information about {detected_food}")
        else:
            bot_response = "Sorry, I couldn't identify the food in the image."
    else:
        bot_response = "Please enter a food name or upload an image."

    if file:
        file.seek(0)
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        results = yolo_model(img)
        for result in results:
            img = result.plot()  

        _, buffer = cv2.imencode('.jpg', img) 
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_result = f"data:image/jpeg;base64,{img_base64}"

    print(f"Sending bot_response (length: {len(bot_response)} chars):", bot_response)
    return jsonify({
        'bot_response': bot_response,
        'image_result': img_result
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7860)
