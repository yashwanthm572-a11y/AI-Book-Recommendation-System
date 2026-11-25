# AI BOOK RECOMMENDATION SYSTEM

## 1. PROJECT OVERVIEW

The AI Book Recommendation System is a Python-based web application developed using Streamlit. It helps users discover books and receive personalized recommendations based on their interests, genres, and preferences.

The application provides an interactive interface for exploring books, managing wishlists, and discovering new reading choices.

## 2. KEY FEATURES

### User Features
- User Registration and Login
- Secure Password Hashing
- Personalized Book Recommendations
- Explore Books Collection
- Book Details and Ratings
- Personal Wishlist Management
- AI Book Assistant
- Analytics Dashboard

### Admin Features
- Admin Dashboard
- Add New Books
- Edit Existing Book Details
- Delete Books
- View Registered Users
- Monitor System Statistics

### Book Covers
- 20 Book Cover Images
- Local Image Storage
- Automatic Placeholder for Missing Covers

## 3. TECHNOLOGIES USED

- Python
- Streamlit
- Pandas
- SQLite
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- HTML and CSS

## 4. PROJECT FOLDER STRUCTURE

```text
BOOK_RECOMMENDATION_SYSTEM/
│
├── app.py
├── README.txt
├── requirements.txt
│
├── data/
│   ├── books.csv
│   └── book_recommendation.db
│
├── modules/
│   ├── __init__.py
│   └── recommendation.py
│
└── assets/
    └── covers/
        ├── atomic_habits.jpg
        ├── cosmos.jpg
        ├── harry_potter.jpg
        ├── lord_of_the_rings.jpg
        ├── psychology_of_money.jpg
        └── 15 additional book cover images
```

## 5. INSTALLATION AND SETUP

### Step 1: Open Project Folder

Open the `BOOK_RECOMMENDATION_SYSTEM` folder in VS Code.

IMPORTANT: Select the inner project folder that contains `app.py`, not its parent folder.

### Step 2: Open Terminal

Open a new terminal in VS Code.

### Step 3: Navigate to the Correct Project Folder

If the terminal opens in the parent folder, execute this command:

```powershell
cd "C:\Users\yashw\Downloads\BOOK_RECOMMENDATION_SYSTEM_Updated\BOOK_RECOMMENDATION_SYSTEM"
```

### Step 4: Verify Project Location

Run this command to check whether `app.py` exists:

```powershell
dir app.py
```

If `app.py` is displayed, you are in the correct project folder.

### Step 5: Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 6: Run the Application

```powershell
python -m streamlit run app.py
```

### Step 7: Access the Application

After running the command, Streamlit will display a local URL.

Default URL:

```text
http://localhost:8501
```

Open the URL in your web browser to access the application.

## 6. HOW TO RUN THE PROJECT IN THE FUTURE

To avoid the folder path error every time:

1. Open VS Code.
2. Select File → Open Folder.
3. Open the inner `BOOK_RECOMMENDATION_SYSTEM` folder containing `app.py`.
4. Open a new terminal.
5. Run the following command:

```powershell
python -m streamlit run app.py
```

You do not need to execute the `cd` command every time if VS Code opens the correct project folder.

## 7. RECOMMENDATION SYSTEM

The recommendation module uses TF-IDF Vectorization and Cosine Similarity to identify books with similar descriptions and recommend relevant titles to users.

The recommendation algorithm helps users discover books based on their reading interests.

## 8. DATABASE

SQLite is used to store user account information and wishlist data.

The application uses secure password hashing for user authentication.

## 9. DATASET

The project includes a CSV dataset containing 20 books with the following details:

- Book ID
- Title
- Author
- Genre
- Rating
- Description

## 10. BOOK COVER MANAGEMENT

The application stores book cover images locally in:

```text
assets/covers/
```

The project includes 20 book cover image files in JPG format.

The image filenames are mapped to their respective book titles in the application.

An internet connection is not required to display the supplied local book cover images.

## 11. FUTURE ENHANCEMENTS

- Advanced Personalized Recommendations
- More Books and Categories
- Improved AI Book Assistant
- Cloud Database Integration
- Online Deployment

## 12. PROJECT INFORMATION

Project Name: AI Book Recommendation System

Project Type: Python Web Application

Framework: Streamlit

Programming Language: Python

Database: SQLite

Domain: Artificial Intelligence and Recommendation Systems

Project Status: Successfully Developed
