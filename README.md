# Potato Disease Prediction Project

This project aims to predict the disease level of a potato plant by analyzing its leaf image. It utilizes machine learning techniques, specifically a Convolutional Neural Network (CNN) model, to classify the disease level based on the input image.

## Usage and Installation

### Cloning the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/potato-disease-prediction.git
```

### Backend Setup
1.Navigate to the `api` directory:
```sh
cd api
```
2.Install Python dependencies from `requirements.txt`:
```sh
pip install -r requirements.txt
```
3.Run the server using either `main.py` or `main-tf-serving.py`:
```sh
python main.py
```
Alternatively:
```sh
python main-tf-serving.py
```

### Frontend Setup
1.Navigate to the `frontend` directory:
```sh
cd frontend
```
2.Install npm dependencies:
```sh
npm install
```
3.Start the local server:
```sh
npm start
```
4.Access the site in your browser at http://localhost:3000.

### Environment Configuration
If you want to deploy the project to another location, modify the environment variables in the .env file accordingly.

## Frontend Development
The frontend of this project was developed using modern web technologies, including:
- React.js for building the user interface components
- Material-UI for styling and UI components
- Material-UI Dropzone for drag-and-drop file upload functionality
- Integration with backend API for seamless communication



