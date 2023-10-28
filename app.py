import cv2
import io
import os
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
from fer import FER
from fer.classes import Video
from concurrent.futures import ThreadPoolExecutor
import uuid

app = Flask(__name__)
executor = ThreadPoolExecutor()

# Create a dictionary to track task statuses and results
task_results = {}

def analyze_video(video_file, mtcnn, task_id):
    video = Video(str(video_file))  # Convert video_path to a string
    detector = FER(mtcnn=mtcnn)

    # Output list of dictionaries
    raw_data = video.analyze(detector, display=False)

    # Convert to pandas for analysis
    df = video.to_pandas(raw_data)
    df = video.get_first_face(df)
    df = video.get_emotions(df)

    # Store the result in the task_results dictionary
    task_results[task_id] = df.to_dict(orient='records')

    # Delete the input video file after analysis is completed
    try:
        os.remove(video_file)
    except Exception as e:
        print(f"Error deleting input video file: {str(e)}")

@app.route('/analyzevideo', methods=['POST'])
def analyze_video_endpoint():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'})

    video_file = request.files['video']
    mtcnn = bool(int(request.form.get('mtcnn', 0)))

    if not video_file:
        return jsonify({'error': 'Invalid video file'})

    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Save the video file to a temporary location
    video_path = Path('input\\' + task_id + '.mp4')
    video_file.save(video_path)

    try:
        # Submit the video analysis task and pass the task ID
        future = executor.submit(analyze_video, video_path, mtcnn, task_id)

        return jsonify({'task_id': task_id, 'message': 'Video analysis started.'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/getresult/<task_id>', methods=['GET'])
def get_task_result(task_id):
    if task_id in task_results:
        result = task_results[task_id]

        # Calculate average emotions
        average_emotions = {}
        for entry in result:
            for emotion, value in entry.items():
                if emotion in average_emotions:
                    average_emotions[emotion] += value
                else:
                    average_emotions[emotion] = value

        total_entries = len(result)
        for emotion in average_emotions:
            average_emotions[emotion] /= total_entries

        # Convert scores to percentages
        for emotion, value in average_emotions.items():
            average_emotions[emotion] = round(value * 100, 2)

        # Delete the output file after the result is obtained
        try:
            os.remove(f'input\\{task_id}.mp4')
        except Exception as e:
            print(f"Error deleting input result file: {str(e)}")

        return jsonify({'task_id': task_id, 'emotions': result, 'average_emotions': average_emotions})
    else:
        return jsonify({'task_id': task_id, 'message': 'Task not found or not completed yet.'})

if __name__ == "__main__":
    app.run(debug=True)
