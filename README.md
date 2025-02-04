# CourseReview

This repository implements a Course Review Analysis Application using NLP techniques and the BERT model. The project collects and analyzes course reviews, leveraging pre-trained transformers for sentiment classification and generating insights.

________________________________________
**Overview**
1. BERT Model for Sentiment Analysis:
    - Trained on a dataset of 100K course reviews for sentiment classification.
    - Analyzes and classifies reviews into predefined categories.
2. Visualization and Analysis:
    - Generates insights from the training data using visualizations (label distributions, word clouds).
    - Compares BERT's performance against other models.
3. Flask Web Application:
    - Uses the trained BERT model to predict sentiments for new course reviews.
    - Displays results through a user-friendly interface using HTML, CSS, and JavaScript.
4. Model Comparison:
    - Compares BERT with other models.
    - BERT outperforms alternatives and is chosen for production.
________________________________________
**Tech Stack**

Frontend: HTML, CSS, and JavaScript for the UI
Backend: Python with Flask for API and data handling
Database: SQLite for managing course data and reviews
Model: DeBERTa (variant of BERT) for Sentiment Analysis
Hosting/Development
Model training: Google Colab Pro (A100 GPU)
Application: Local (VS Code)
________________________________________
**Features**
- Train BERT Model: Processes 100K course reviews for sentiment classification.
- Data Visualization: Displays label distributions and word clouds for analysis.
- Flask Integration: Runs a web application to handle and classify new course reviews.
- Interactive UI: Developed using HTML, CSS, and JavaScript to show predictions.
- Model Comparison: Evaluates multiple models, highlighting BERT's superior accuracy.

________________________________________
**Flow Chart**
![image](https://github.com/user-attachments/assets/24eb0aef-c428-41b6-84fd-f9fa6c594db0)
________________________________________
**Requirements**
- Python 3.x
- Libraries:
    - transformers
    - torch
    - flask
    - pandas
    - matplotlib
    - seaborn
    - wordcloud
Install dependencies using:
```
pip install transformers torch flask pandas matplotlib seaborn wordcloud
```
________________________________________
Setup and Execution
1. Create and Activate Virtual Environment:
```
python -m venv course_venv
source course_venv/Scripts/activate        # For Windows
. course_venv/bin/activate                # For Unix/MacOS
```
2. Train and Analyze Data:
Run the BERT model and visualize results:
```
python main.py
```
   - Generates label distributions and word clouds.
   - Saves BERT model outputs for review analysis.
3. Run Flask Application:
Use the trained model to classify new course reviews:
```
python app.py
```
4. Access the Web Interface:
   - Open your browser and go to http://127.0.0.1:5000.
   - Submit new reviews and view their sentiment predictions.
________________________________________
**Project Structure**
- main.py:
  - Trains the BERT model on 100K course reviews.
  - Generates visualizations (label distributions, word clouds).
  - Saves and analyzes model outputs.
- app.py:
  - Flask web server that uses the trained BERT model to classify new reviews.
  - Displays predictions using HTML, CSS, and JavaScript.
- modelscomparision.py:
  - Compares BERT with other models (e.g., logistic regression, SGD, DT).
  - Demonstrates BERT's superior performance.
- EECS_Course_Data.csv: Preloaded course data for analysis.
- reviews.csv: Input review dataset for training.
________________________________________
**Key Visualizations**
- Label Distribution: Pie charts showing the proportion of sentiment labels.
- Word Clouds: Visual representation of frequently occurring words in reviews.
- Model Performance: Comparison charts for accuracy, precision, and recall.
________________________________________
**Model Comparison**
We evaluated multiple models to classify course reviews:
1.	Logistic Regression
2.	SGD (Stochastic Gradient Descent)
3.	Decision Tree
4.	BERT (Bidirectional Encoder Representations from Transformers)
BERT outperformed all other models in terms of accuracy and precision, making it the final choice for deployment.
________________________________________
Web Interface Demo
- Frontend: Built using HTML, CSS, and JavaScript.
- Backend: Flask API processes review inputs and returns predictions using the trained BERT model.
________________________________________
Images

![image](https://github.com/user-attachments/assets/5c212f3f-1e8e-4e84-bb93-99c1b6192f83)

![image](https://github.com/user-attachments/assets/b700d246-3527-4d22-ace4-63568ae59b00)

![image](https://github.com/user-attachments/assets/2e8dce13-665a-46fb-806c-7efa80c78177)

________________________________________
**Author**

SudhaChandrikaY




