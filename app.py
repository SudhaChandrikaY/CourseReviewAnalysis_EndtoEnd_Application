from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sqlite3
import torch
import os
import pandas as pd
import re  # For review validation

app = Flask(__name__)

# Load the trained model and tokenizer
model_path = "./DeBert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Database configuration
DATABASE = "reviews.db"

def init_db():
    """Initialize the database with tables."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS courses (
            course_code TEXT PRIMARY KEY,
            course_name TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_code TEXT,
            review TEXT,
            sentiment INTEGER,
            FOREIGN KEY(course_code) REFERENCES courses(course_code)
        )
    ''')
    conn.commit()
    conn.close()

def populate_courses_from_csv():
    """Populate courses table with data from CSV."""
    file_path = "EECS_Course_Data.csv"  # Ensure this file exists in your project directory
    if os.path.exists(file_path):
        course_data = pd.read_csv(file_path, encoding='latin1')
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        for _, row in course_data.iterrows():
            c.execute('INSERT OR IGNORE INTO courses (course_code, course_name) VALUES (?, ?)', (row['Course Code'], row['Course Name']))
        conn.commit()
        conn.close()

# Initialize database and populate courses
init_db()
populate_courses_from_csv()

# Sentiment analysis prediction
def predict_sentiment(review):
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction + 1  # Return sentiment as a score (1-5)

# Validate review content
def is_valid_review(review):
    """Check if the review is valid (not gibberish)."""
    if len(review) < 10:  # Too short
        return False
    if not re.search(r'[a-zA-Z]', review):  # No meaningful words
        return False
    return True

@app.after_request
def add_header(response):
    """Add headers to prevent caching."""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Routes
@app.route('/')
def home():
    """Redirect to the course overview page."""
    return course_overview()

#Get course reviews in main page

@app.route('/course_overview', methods=['GET'])
def course_overview():
    """Fetch all courses with their average rating and number of reviews."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    # Fetch course data with average rating and review count
    c.execute('''
        SELECT 
            courses.course_code, 
            courses.course_name, 
            AVG(CAST(reviews.sentiment AS FLOAT)) as avg_rating,
            COUNT(reviews.id) as review_count
        FROM courses
        LEFT JOIN reviews ON courses.course_code = reviews.course_code
        GROUP BY courses.course_code, courses.course_name
    ''')
    courses_with_ratings = c.fetchall()
    conn.close()

    # Map numeric sentiments to average ratings for template rendering
    def format_rating(avg_rating):
        return round(avg_rating, 1) if avg_rating is not None else 0

    courses_with_ratings = [
        (course[0], course[1], format_rating(course[2]), course[3])  # Add formatted rating and count
        for course in courses_with_ratings
    ]

    return render_template('course_overview.html', courses=courses_with_ratings)


#To view all the course reviews

@app.route('/view_all_course_reviews', methods=['GET'])
def view_all_course_reviews():
    """Display all course reviews with detailed metrics."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    # Fetch course details along with review metrics
    c.execute('''
        SELECT 
            courses.course_code, 
            courses.course_name, 
            COUNT(reviews.id) as total_reviews,
            SUM(CASE WHEN sentiment IN (4, 5) THEN 1 ELSE 0 END) as positive_reviews,
            SUM(CASE WHEN sentiment IN (1, 2) THEN 1 ELSE 0 END) as negative_reviews
        FROM courses
        LEFT JOIN reviews ON courses.course_code = reviews.course_code
        GROUP BY courses.course_code, courses.course_name
    ''')
    courses_with_metrics = c.fetchall()
    conn.close()

    # Calculate positive and negative rates
    enhanced_courses = []
    for course in courses_with_metrics:
        course_code, course_name, total_reviews, positive_reviews, negative_reviews = course
        positive_rate = (positive_reviews / total_reviews * 100) if total_reviews > 0 else 0
        negative_rate = (negative_reviews / total_reviews * 100) if total_reviews > 0 else 0
        enhanced_courses.append(
            (course_code, course_name, total_reviews, positive_reviews, negative_reviews, positive_rate, negative_rate)
        )

    # Debug: Print all course metrics
    #print("Courses with Metrics:", enhanced_courses)

    # Pass 'enumerate' to the template
    return render_template('view_all_course_reviews.html', courses=enhanced_courses, enumerate=enumerate)



#To fetch all the reviews of selected course

@app.route('/get_course_reviews/<course_code>', methods=['GET'])
def get_course_reviews(course_code):
    """Fetch all reviews for a specific course."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        SELECT review, sentiment
        FROM reviews
        WHERE course_code = ?
    ''', (course_code,))
    reviews = c.fetchall()
    conn.close()


    return jsonify({"reviews": reviews})

@app.route('/get_visualization_data', methods=['GET'])
def get_visualization_data():
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()

        # Fetch undergrad and grad positive/negative reviews
        c.execute('''
            SELECT 
                CASE 
                    WHEN CAST(SUBSTR(courses.course_code, INSTR(courses.course_code, ' ') + 1) AS INTEGER) < 500 THEN 'undergrad'
                    ELSE 'grad'
                END AS course_type,
                SUM(CASE WHEN sentiment IN (4, 5) THEN 1 ELSE 0 END) AS positive_reviews,
                SUM(CASE WHEN sentiment IN (1, 2) THEN 1 ELSE 0 END) AS negative_reviews
            FROM courses
            LEFT JOIN reviews ON courses.course_code = reviews.course_code
            GROUP BY course_type
        ''')
        reviews_by_type = {row[0]: {"positive_reviews": row[1], "negative_reviews": row[2]} for row in c.fetchall()}

        # Fetch overall sentiment distribution
        c.execute('''
            SELECT 
                COALESCE(SUM(CASE WHEN sentiment IN (4, 5) THEN 1 ELSE 0 END), 0) AS positive_reviews,
                COALESCE(SUM(CASE WHEN sentiment IN (1, 2) THEN 1 ELSE 0 END), 0) AS negative_reviews
            FROM reviews
        ''')
        sentiment_counts = c.fetchone()
        positive_reviews = sentiment_counts[0] or 0
        negative_reviews = sentiment_counts[1] or 0

        conn.close()

        return jsonify({
            "undergrad": reviews_by_type.get("undergrad", {"positive_reviews": 0, "negative_reviews": 0}),
            "grad": reviews_by_type.get("grad", {"positive_reviews": 0, "negative_reviews": 0}),
            "sentiments": {
                "positive": positive_reviews,
                "negative": negative_reviews
            }
        })
    except Exception as e:
        print(f"Error fetching visualization data: {str(e)}")
        return jsonify({"error": "An error occurred while fetching visualization data."}), 500


@app.route('/get_course_visualization_data/<course_code>', methods=['GET'])
def get_course_visualization_data(course_code):
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()

        # Fetch sentiment data for the course
        c.execute('''
            SELECT 
                SUM(CASE WHEN sentiment IN (4, 5) THEN 1 ELSE 0 END) as positive_reviews,
                SUM(CASE WHEN sentiment IN (1, 2) THEN 1 ELSE 0 END) as negative_reviews
            FROM reviews
            WHERE course_code = ?
        ''', (course_code,))
        sentiment_counts = c.fetchone()
        positive_reviews = sentiment_counts[0] or 0
        negative_reviews = sentiment_counts[1] or 0

        # Fetch course name
        c.execute('SELECT course_name FROM courses WHERE course_code = ?', (course_code,))
        course_name = c.fetchone()[0]

        conn.close()

        return jsonify({
            "course_name": course_name,
            "positive_reviews": positive_reviews,
            "negative_reviews": negative_reviews
        })
    except Exception as e:
        print(f"Error fetching course visualization data: {str(e)}")
        return jsonify({"error": "An error occurred while fetching course-specific visualization data."}), 500


@app.route('/get_courses', methods=['GET'])
def get_courses():
    """Fetch all courses from the database."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT course_code, course_name FROM courses')
    courses = c.fetchall()
    conn.close()
    return jsonify({"courses": [{"course_code": course[0], "course_name": course[1]} for course in courses]})

@app.route('/submit_review_page', methods=['GET'])
def submit_review_page():
    """Serve the review submission page."""
    return render_template('submit_review.html')

@app.route('/submit_review', methods=['POST'])
def submit_review():
    """Submit a review and store it in the database."""
    data = request.json
    course_code = data['course_code']
    review = data['review']

    # Validate the review
    if not is_valid_review(review):
        return jsonify({"message": "Invalid review. Please write a meaningful review."}), 400

    # Predict sentiment
    sentiment = predict_sentiment(review)

    # Store the review and sentiment in the database as an integer
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('INSERT INTO reviews (course_code, review, sentiment) VALUES (?, ?, ?)', (course_code, review, sentiment))
    conn.commit()
    conn.close()

    # Debug: Confirm storage (this will print to the server console but not the client)
    print(f"Stored: Course Code: {course_code}, Sentiment: {sentiment}, Review: {review}")

    # Return success message without exposing sentiment
    return jsonify({"message": "Review submitted successfully!"})


if __name__ == '__main__':
    app.run(debug=True)
