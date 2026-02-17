from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
from ultralytics import RTDETR
import os
import io

app = Flask(__name__)
CORS(app)  

MODEL_DIR = 'model'
foodtray_model = None
menu_model = None

# Warna untuk bounding box tiap kategori
BOX_COLORS = {
    'foodtray': (0, 200, 255),     # cyan
    'menu': (255, 80, 80),          # merah
}

# Palet warna untuk berbagai class
CLASS_COLORS = [
    (255, 80, 80),    # merah
    (80, 255, 80),    # hijau
    (80, 80, 255),    # biru
    (255, 255, 80),   # kuning
    (255, 80, 255),   # magenta
    (80, 255, 255),   # cyan
    (255, 160, 80),   # oranye
    (160, 80, 255),   # ungu
    (80, 255, 160),   # teal
    (255, 80, 160),   # pink
]


def draw_detections(image, detections, label_prefix='', color=None):
    """Draw bounding boxes dan label pada gambar."""
    draw = ImageDraw.Draw(image)

    # Coba load font, fallback ke default
    try:
        font_size = max(14, min(image.size) // 40)
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
        font_size = 12

    for i, det in enumerate(detections):
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        cls_name = det['class']
        conf = det['confidence']

        # Pilih warna
        if color:
            box_color = color
        else:
            box_color = CLASS_COLORS[det.get('class_id', i) % len(CLASS_COLORS)]

        # Gambar bounding box (ketebalan 3px)
        for thickness in range(3):
            draw.rectangle(
                [x1 - thickness, y1 - thickness, x2 + thickness, y2 + thickness],
                outline=box_color
            )

        # Label text
        label = f"{label_prefix}{cls_name} {conf*100:.1f}%"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        # Background rectangle untuk label
        label_y = max(y1 - th - 6, 0)
        draw.rectangle([x1, label_y, x1 + tw + 8, label_y + th + 6], fill=box_color)
        draw.text((x1 + 4, label_y + 2), label, fill=(255, 255, 255), font=font)

    return image


def image_to_response(image, fmt='JPEG'):
    """Convert PIL Image ke Flask response (image/jpeg) — Postman auto-preview."""
    buf = io.BytesIO()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buf, format=fmt, quality=85)
    buf.seek(0)
    response = make_response(send_file(buf, mimetype='image/jpeg'))
    response.headers['Content-Type'] = 'image/jpeg'
    return response


def load_models():
    global foodtray_model, menu_model
    
    try:
        foodtray_path = os.path.join(MODEL_DIR, 'foodtray.pt')
        menu_path = os.path.join(MODEL_DIR, 'menu.pt')
        
        if not os.path.exists(foodtray_path):
            print(f"Warning: {foodtray_path} not found")
            return False
        
        if not os.path.exists(menu_path):
            print(f"Warning: {menu_path} not found")
            return False
        
        print("Loading foodtray model...")
        foodtray_model = RTDETR(foodtray_path)
        print(f"Foodtray model loaded! Classes: {list(foodtray_model.names.values())}")
        
        print("Loading menu model...")
        menu_model = RTDETR(menu_path)
        print(f"Menu model loaded! Classes: {list(menu_model.names.values())}")
        
        return True
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

@app.route('/')
def hello_world():
    return jsonify({
        'message': 'RT-DETR Food Detection API - MBG Project',
        'version': '1.0.0',
        'status': 'running',
        'models_loaded': {
            'foodtray': foodtray_model is not None,
            'menu': menu_model is not None
        },
        'endpoints': {
            'health': 'GET /api/health',
            'models_info': 'GET /api/models',
            'detect': 'POST /api/detect (JSON)',
            'detect_preview': 'POST /api/detect/preview (image preview)',
            'foodtray_only': 'POST /api/detect/foodtray-only (JSON)',
            'foodtray_preview': 'POST /api/detect/foodtray-only/preview (image preview)',
            'menu_only': 'POST /api/detect/menu-only (JSON)',
            'menu_preview': 'POST /api/detect/menu-only/preview (image preview)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models': {
            'foodtray': 'loaded' if foodtray_model else 'not loaded',
            'menu': 'loaded' if menu_model else 'not loaded'
        }
    })

@app.route('/api/models', methods=['GET'])
def get_models_info():
    info = {
        'foodtray': None,
        'menu': None
    }
    
    if foodtray_model:
        info['foodtray'] = {
            'classes': foodtray_model.names,
            'num_classes': len(foodtray_model.names)
        }
    
    if menu_model:
        info['menu'] = {
            'classes': menu_model.names,
            'num_classes': len(menu_model.names)
        }
    
    return jsonify(info)

@app.route('/api/detect', methods=['POST'])
def detect_food():
    try:
        if not foodtray_model or not menu_model:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please restart server.'
            }), 500
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided. Use key "image" in form-data.'
            }), 400
        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        try:
            image = Image.open(image_file.stream)
            print(f"Image loaded: {image.size} - {image.mode}")
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid image file: {str(e)}'
            }), 400
        
        print("Step 1: Detecting foodtray...")
        foodtray_results = foodtray_model(image)
        
        foodtray_detections = []
        for result in foodtray_results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'class': result.names[int(box.cls[0])],
                    'class_id': int(box.cls[0]),
                    'confidence': round(float(box.conf[0]), 4),
                    'bbox': {
                        'x1': round(float(box.xyxy[0][0]), 2),
                        'y1': round(float(box.xyxy[0][1]), 2),
                        'x2': round(float(box.xyxy[0][2]), 2),
                        'y2': round(float(box.xyxy[0][3]), 2)
                    }
                }
                foodtray_detections.append(detection)
        print(f"Found {len(foodtray_detections)} foodtray sections")
        print("Step 2: Detecting menu items...")
        menu_results = menu_model(image)
        
        menu_detections = []
        for result in menu_results:
            boxes = result.boxes
            for box in boxes:
                bbox_coords = box.xyxy[0]
                
                # Hitung ukuran bounding box
                width = float(bbox_coords[2] - bbox_coords[0])
                height = float(bbox_coords[3] - bbox_coords[1])
                area = width * height
                
                detection = {
                    'class': result.names[int(box.cls[0])],
                    'class_id': int(box.cls[0]),
                    'confidence': round(float(box.conf[0]), 4),
                    'bbox': {
                        'x1': round(float(bbox_coords[0]), 2),
                        'y1': round(float(bbox_coords[1]), 2),
                        'x2': round(float(bbox_coords[2]), 2),
                        'y2': round(float(bbox_coords[3]), 2),
                        'width': round(width, 2),
                        'height': round(height, 2),
                        'area': round(area, 2)
                    }
                }
                menu_detections.append(detection)
        
        print(f"Found {len(menu_detections)} food items")
        
        # 6. Prepare response JSON
        response = {
            'success': True,
            'timestamp': None,  
            'image_info': {
                'width': image.size[0],
                'height': image.size[1],
                'mode': image.mode
            },
            'foodtray': {
                'detected': len(foodtray_detections) > 0,
                'count': len(foodtray_detections),
                'detections': foodtray_detections
            },
            'menu': {
                'detected': len(menu_detections) > 0,
                'count': len(menu_detections),
                'detections': menu_detections
            },
            'summary': {
                'total_detections': len(foodtray_detections) + len(menu_detections),
                'foodtray_types': list(set([d['class'] for d in foodtray_detections])),
                'food_items': list(set([d['class'] for d in menu_detections]))
            }
        }
        
        print("Detection completed successfully!")
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/detect/preview', methods=['POST'])
def detect_food_preview():
    """Deteksi foodtray + menu, return gambar anotasi langsung."""
    try:
        if not foodtray_model or not menu_model:
            return jsonify({'success': False, 'error': 'Models not loaded'}), 500
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided. Use key "image" in form-data.'}), 400

        image = Image.open(request.files['image'].stream).convert('RGB')

        # Deteksi foodtray
        foodtray_dets = []
        for result in foodtray_model(image):
            for box in result.boxes:
                foodtray_dets.append({
                    'class': result.names[int(box.cls[0])],
                    'class_id': int(box.cls[0]),
                    'confidence': round(float(box.conf[0]), 4),
                    'bbox': {
                        'x1': round(float(box.xyxy[0][0]), 2),
                        'y1': round(float(box.xyxy[0][1]), 2),
                        'x2': round(float(box.xyxy[0][2]), 2),
                        'y2': round(float(box.xyxy[0][3]), 2)
                    }
                })

        # Deteksi menu
        menu_dets = []
        for result in menu_model(image):
            for box in result.boxes:
                c = box.xyxy[0]
                menu_dets.append({
                    'class': result.names[int(box.cls[0])],
                    'class_id': int(box.cls[0]),
                    'confidence': round(float(box.conf[0]), 4),
                    'bbox': {
                        'x1': round(float(c[0]), 2), 'y1': round(float(c[1]), 2),
                        'x2': round(float(c[2]), 2), 'y2': round(float(c[3]), 2)
                    }
                })

        # Gambar bounding box
        annotated = image.copy()
        annotated = draw_detections(annotated, foodtray_dets, label_prefix='[tray] ', color=BOX_COLORS['foodtray'])
        annotated = draw_detections(annotated, menu_dets, label_prefix='', color=None)

        print(f"Preview: {len(foodtray_dets)} foodtray, {len(menu_dets)} menu items")
        return image_to_response(annotated)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detect/foodtray-only/preview', methods=['POST'])
def detect_foodtray_only_preview():
    try:
        if not foodtray_model:
            return jsonify({'success': False, 'error': 'Foodtray model not loaded'}), 500
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        image = Image.open(request.files['image'].stream).convert('RGB')
        detections = []
        for result in foodtray_model(image):
            for box in result.boxes:
                detections.append({
                    'class': result.names[int(box.cls[0])],
                    'class_id': int(box.cls[0]),
                    'confidence': round(float(box.conf[0]), 4),
                    'bbox': {
                        'x1': round(float(box.xyxy[0][0]), 2),
                        'y1': round(float(box.xyxy[0][1]), 2),
                        'x2': round(float(box.xyxy[0][2]), 2),
                        'y2': round(float(box.xyxy[0][3]), 2)
                    }
                })

        annotated = image.copy()
        annotated = draw_detections(annotated, detections, label_prefix='[tray] ', color=BOX_COLORS['foodtray'])
        return image_to_response(annotated)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detect/menu-only/preview', methods=['POST'])
def detect_menu_only_preview():
    """Deteksi menu saja, return gambar anotasi."""
    try:
        if not menu_model:
            return jsonify({'success': False, 'error': 'Menu model not loaded'}), 500
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        image = Image.open(request.files['image'].stream).convert('RGB')
        detections = []
        for result in menu_model(image):
            for box in result.boxes:
                c = box.xyxy[0]
                detections.append({
                    'class': result.names[int(box.cls[0])],
                    'class_id': int(box.cls[0]),
                    'confidence': round(float(box.conf[0]), 4),
                    'bbox': {
                        'x1': round(float(c[0]), 2), 'y1': round(float(c[1]), 2),
                        'x2': round(float(c[2]), 2), 'y2': round(float(c[3]), 2)
                    }
                })

        annotated = image.copy()
        annotated = draw_detections(annotated, detections)
        return image_to_response(annotated)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detect/foodtray-only', methods=['POST'])
def detect_foodtray_only():
    try:
        if not foodtray_model:
            return jsonify({
                'success': False, 
                'error': 'Foodtray model not loaded'
            }), 500
        
        if 'image' not in request.files:
            return jsonify({
                'success': False, 
                'error': 'No image provided'
            }), 400
        
        image = Image.open(request.files['image'].stream)
        print(f"Detecting foodtray in image: {image.size}")
        
        results = foodtray_model(image)
        
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'class': result.names[int(box.cls[0])],
                    'class_id': int(box.cls[0]),
                    'confidence': round(float(box.conf[0]), 4),
                    'bbox': {
                        'x1': round(float(box.xyxy[0][0]), 2),
                        'y1': round(float(box.xyxy[0][1]), 2),
                        'x2': round(float(box.xyxy[0][2]), 2),
                        'y2': round(float(box.xyxy[0][3]), 2)
                    }
                })
        
        return jsonify({
            'success': True,
            'count': len(detections),
            'detections': detections,
            'types': list(set([d['class'] for d in detections]))
        })
    
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/api/detect/menu-only', methods=['POST'])
def detect_menu_only():
    try:
        if not menu_model:
            return jsonify({
                'success': False, 
                'error': 'Menu model not loaded'
            }), 500
        
        if 'image' not in request.files:
            return jsonify({
                'success': False, 
                'error': 'No image provided'
            }), 400
        
        image = Image.open(request.files['image'].stream)
        print(f"Detecting food items in image: {image.size}")
        
        results = menu_model(image)
        
        detections = []
        for result in results:
            for box in result.boxes:
                bbox_coords = box.xyxy[0]
                width = float(bbox_coords[2] - bbox_coords[0])
                height = float(bbox_coords[3] - bbox_coords[1])
                
                detections.append({
                    'class': result.names[int(box.cls[0])],
                    'class_id': int(box.cls[0]),
                    'confidence': round(float(box.conf[0]), 4),
                    'bbox': {
                        'x1': round(float(bbox_coords[0]), 2),
                        'y1': round(float(bbox_coords[1]), 2),
                        'x2': round(float(bbox_coords[2]), 2),
                        'y2': round(float(bbox_coords[3]), 2),
                        'width': round(width, 2),
                        'height': round(height, 2)
                    }
                })
        
        return jsonify({
            'success': True,
            'count': len(detections),
            'detections': detections,
            'food_items': list(set([d['class'] for d in detections]))
        })
    
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("RT-DETR Food Detection Server - MBG Project")
    print("=" * 60)
    print("Project: Estimasi Nutrisi Makanan dengan RT-DETR")
    print("Purpose: Deteksi foodtray dan jenis makanan")
    print("=" * 60)
    
    # Load models saat startup
    if load_models():
        print("\nAll models loaded successfully!")
        print("Starting Flask server on http://localhost:5000")
        print("API accessible at http://0.0.0.0:5000")
        print("=" * 60)
        print("\n Available endpoints:")
        print("  - GET  /")
        print("  - GET  /api/health")
        print("  - GET  /api/models")
        print("  - POST /api/detect              (JSON)")
        print("  - POST /api/detect/preview       (image preview)")
        print("  - POST /api/detect/foodtray-only  (JSON)")
        print("  - POST /api/detect/foodtray-only/preview (image)")
        print("  - POST /api/detect/menu-only      (JSON)")
        print("  - POST /api/detect/menu-only/preview     (image)")
        print("\n Server starting...\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n Failed to load models!")
        print("=" * 60)
        print("Troubleshooting:")
        print("  1. Check if 'model/' folder exists")
        print("  2. Check if model files exist:")
        print("     - model/foodtray.pt")
        print("     - model/menu.pt")
        print("  3. Verify ultralytics is installed:")
        print("     pip install ultralytics")
        print("=" * 60)  