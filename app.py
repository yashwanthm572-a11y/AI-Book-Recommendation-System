import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import hashlib
import hmac
import secrets
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modules.recommendation import BookRecommender

# -------------------- PAGE SETTINGS --------------------
st.set_page_config(
    page_title="AI Book Recommendation",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM DESIGN --------------------
st.markdown("""
<style>
.stApp { background: #0b1020; }
[data-testid="stSidebar"] { background: #111a2e; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1450px; }
.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 22px;
    margin-bottom: 1.4rem;
    background: linear-gradient(120deg, #182b50 0%, #3a246b 55%, #6b356b 100%);
    border: 1px solid rgba(255,255,255,.12);
}
.hero h1 { color: white; font-size: 2.5rem; margin: 0 0 .5rem 0; }
.hero p { color: #e0e7ff; font-size: 1.05rem; margin: 0; }
.section-note { color: #aab7d2; margin-top: -8px; }
.book-title { font-size: 1rem; font-weight: 700; min-height: 2.5rem; }
div[data-testid="stMetric"] {
    background: #151f35;
    border: 1px solid #293958;
    padding: 16px;
    border-radius: 14px;
}
div.stButton > button { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# -------------------- PROJECT PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
DATA_PATH = DATA_DIR / "books.csv"
COVERS_DIR = ASSETS_DIR / "covers"
DATABASE_PATH = DATA_DIR / "book_recommendation.db"

# Keep all writable paths anchored to this file, never to the terminal's
# current working directory. Create runtime folders before SQLite/CSV writes.
DATA_DIR.mkdir(parents=True, exist_ok=True)
COVERS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------- SQLITE DATABASE --------------------
def get_db_connection():
    return sqlite3.connect(DATABASE_PATH)



def hash_password(password, salt=None):
    """Hash a password with PBKDF2; never store plain-text passwords."""
    salt = salt or secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, 200_000
    )
    return salt.hex(), password_hash.hex()


def verify_password(password, salt_hex, expected_hash):
    salt = bytes.fromhex(salt_hex)
    _, actual_hash = hash_password(password, salt)
    return hmac.compare_digest(actual_hash, expected_hash)


def initialize_database():
    """Create account tables and migrate the earlier shared wishlist safely."""
    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        user_columns = [row[1] for row in connection.execute("PRAGMA table_info(users)")]
        if "role" not in user_columns:
            connection.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")

        existing = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='wishlist'"
        ).fetchone()

        if existing:
            columns = [
                row[1] for row in connection.execute("PRAGMA table_info(wishlist)")
            ]
            if "user_id" not in columns:
                connection.execute("ALTER TABLE wishlist RENAME TO wishlist_legacy")

        connection.execute("""
            CREATE TABLE IF NOT EXISTS wishlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                book_title TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, book_title),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        connection.commit()


def create_user(username, password, role="user"):
    username = username.strip()
    salt, password_hash = hash_password(password)
    try:
        with get_db_connection() as connection:
            cursor = connection.execute(
                "INSERT INTO users (username, salt, password_hash, role) VALUES (?, ?, ?, ?)",
                (username, salt, password_hash, role)
            )
            user_id = cursor.lastrowid

            # If the old app had a shared wishlist, transfer it to the first
            # account created so those saved books are not lost.
            user_count = connection.execute(
                "SELECT COUNT(*) FROM users"
            ).fetchone()[0]
            legacy_exists = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='wishlist_legacy'"
            ).fetchone()
            if user_count == 1 and legacy_exists:
                old_titles = connection.execute(
                    "SELECT book_title FROM wishlist_legacy"
                ).fetchall()
                connection.executemany(
                    "INSERT OR IGNORE INTO wishlist (user_id, book_title) VALUES (?, ?)",
                    [(user_id, row[0]) for row in old_titles]
                )
            connection.commit()
        return user_id, None
    except sqlite3.IntegrityError:
        return None, "That username is already taken. Please choose another."


def authenticate_user(username, password):
    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT id, salt, password_hash FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),)
        ).fetchone()

    if row and verify_password(password, row[1], row[2]):
        return row[0]
    return None


def get_user_role(user_id):
    with get_db_connection() as connection:
        row = connection.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
    return row[0] if row else "user"


def activate_admin(user_id):
    with get_db_connection() as connection:
        connection.execute("UPDATE users SET role = 'admin' WHERE id = ?", (user_id,))
        connection.commit()


def get_user_list():
    with get_db_connection() as connection:
        return pd.read_sql_query(
            "SELECT id, username, role, created_at FROM users ORDER BY created_at DESC",
            connection
        )


def get_total_wishlist_items():
    with get_db_connection() as connection:
        return connection.execute("SELECT COUNT(*) FROM wishlist").fetchone()[0]


def save_books_to_csv(frame):
    frame.to_csv(DATA_PATH, index=False)
    load_books.clear()


def get_wishlist(user_id):
    with get_db_connection() as connection:
        rows = connection.execute(
            "SELECT book_title FROM wishlist WHERE user_id = ? ORDER BY added_at DESC",
            (user_id,)
        ).fetchall()
    return [row[0] for row in rows]


def add_to_wishlist(user_id, book_title):
    with get_db_connection() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO wishlist (user_id, book_title) VALUES (?, ?)",
            (user_id, book_title)
        )
        connection.commit()


def remove_from_wishlist(user_id, book_title):
    with get_db_connection() as connection:
        connection.execute(
            "DELETE FROM wishlist WHERE user_id = ? AND book_title = ?",
            (user_id, book_title)
        )
        connection.commit()


initialize_database()

# Local cover filenames for the five images you added.
# Other books will show a placeholder until you add their covers.

BOOK_COVERS = {
    "Harry Potter": "harry_potter.jpg",
    "Lord of the Rings": "lord_of_the_rings.jpg",
    "The Hobbit": "the_hobbit.jpg",
    "The Alchemist": "the_alchemist.jpg",
    "Atomic Habits": "atomic_habits.jpg",
    "Rich Dad Poor Dad": "rich_dad_poor_dad.jpg",
    "Think and Grow Rich": "think_and_grow_rich.jpg",
    "The Psychology of Money": "psychology_of_money.jpg",
    "The Silent Patient": "the_silent_patient.jpg",
    "Gone Girl": "gone_girl.jpg",
    "Sherlock Holmes": "sherlock_holmes.jpg",
    "Murder on the Orient Express": "murder_on_the_orient_express.jpg",
    "The Fault in Our Stars": "the_fault_in_our_stars.jpg",
    "Me Before You": "me_before_you.jpg",
    "The Notebook": "the_notebook.jpg",
    "Sapiens": "sapiens.jpg",
    "Brief History of Time": "brief_history_of_time.jpg",
    "Cosmos": "cosmos.jpg",
    "The Art of War": "the_art_of_war.jpg",
    "Deep Work": "deep_work.jpg",
}


# -------------------- LOAD DATASET --------------------
@st.cache_data
def load_books():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Book dataset not found at {DATA_PATH}. "
            "Restore data/books.csv inside the project folder."
        )
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "book_id", "title", "author", "genre", "rating", "description"
    ]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns in books.csv: {missing_columns}")

    for column in ["title", "author", "genre", "description"]:
        df[column] = df[column].fillna("").astype(str)

    df["rating"] = pd.to_numeric(
        df["rating"], errors="coerce"
    ).fillna(0)

    return df


def show_cover(title, author=""):
    """Display a local JPG cover, or a placeholder if no cover is available."""
    image_filename = BOOK_COVERS.get(title)
    image_path = COVERS_DIR / image_filename if image_filename else None

    if image_path and image_path.is_file():
        st.image(str(image_path), use_container_width=True)
    else:
        st.markdown(
            f"""
            <div style="
                height:230px;
                border-radius:12px;
                background:linear-gradient(145deg,#243b63,#583c78);
                display:flex;
                flex-direction:column;
                align-items:center;
                justify-content:center;
                text-align:center;
                padding:16px;
                color:white;
                font-size:1rem;
                font-weight:700;
                border:1px solid #52658b;">
                <div style="font-size:2rem;">📚</div>
                <div>{title}</div>
                <div style="font-size:.75rem;font-weight:400;margin-top:8px;">
                    {author}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


try:
    books = load_books()
except Exception as error:
    st.error(f"Dataset Error: {error}")
    st.stop()

# Keep the existing TF-IDF + cosine similarity recommendation algorithm.
try:
    model = BookRecommender(books)
except Exception as error:
    st.error(f"Recommendation Model Error: {error}")
    st.stop()


# -------------------- AI BOOK ASSISTANT HELPERS --------------------
def get_assistant_recommendations(user_prompt, frame, count=5):
    """Find books matching a natural-language request using local TF-IDF search.

    This assistant runs entirely from the local book catalogue; it does not call
    a hosted LLM or require an API key.
    """
    query = str(user_prompt).strip()
    if not query or frame.empty:
        return pd.DataFrame()

    corpus = (
        frame["title"].fillna("").astype(str) + " " +
        frame["author"].fillna("").astype(str) + " " +
        frame["genre"].fillna("").astype(str) + " " +
        frame["description"].fillna("").astype(str)
    ).tolist()

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(corpus + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    except ValueError:
        scores = [0.0] * len(frame)

    ranked = frame.copy()
    ranked["match_score"] = scores
    prompt_words = {word.lower().strip(".,!?;:") for word in query.split() if len(word) > 2}
    # Give a small boost when the user's words explicitly match a genre or author.
    for idx, row in ranked.iterrows():
        genre_words = set(str(row["genre"]).lower().split())
        author_words = set(str(row["author"]).lower().split())
        if prompt_words & genre_words:
            ranked.loc[idx, "match_score"] += 0.18
        if prompt_words & author_words:
            ranked.loc[idx, "match_score"] += 0.12

    ranked = ranked.sort_values(["match_score", "rating"], ascending=[False, False])
    ranked = ranked.head(min(count, len(ranked))).copy()
    ranked["match_percent"] = (ranked["match_score"].clip(0, 1) * 100).round(1)
    return ranked


def explain_book_match(book, prompt):
    """Create a transparent explanation from the book's stored catalogue fields."""
    prompt_terms = {word.lower().strip(".,!?;:") for word in str(prompt).split() if len(word) > 2}
    matched = []
    for label, value in [("genre", book["genre"]), ("author", book["author"]), ("description", book["description"])]:
        value_terms = set(str(value).lower().replace(",", " ").replace(".", " ").split())
        if prompt_terms & value_terms:
            matched.append(label)
    if matched:
        return "Matches your request through its " + ", ".join(matched) + "."
    return f"Suggested from your request using its {book['genre']} genre and {float(book['rating']):.1f}/5 catalogue rating."


# -------------------- LOGIN / REGISTRATION --------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None

if st.session_state.user_id is None:
    st.markdown("""
    <div class="hero">
        <h1>📚 Welcome to AI Book Explorer</h1>
        <p>Sign in or create an account to manage your own wishlist.</p>
    </div>
    """, unsafe_allow_html=True)

    login_tab, register_tab = st.tabs(["Login", "Create Account"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login", type="primary")

        if login_submitted:
            if not login_username.strip() or not login_password:
                st.error("Please enter both username and password.")
            else:
                authenticated_id = authenticate_user(login_username, login_password)
                if authenticated_id is None:
                    st.error("Invalid username or password.")
                else:
                    st.session_state.user_id = authenticated_id
                    st.session_state.username = login_username.strip()
                    st.rerun()

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Create password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            admin_code = st.text_input("Admin setup code (optional)", type="password",
                                       help="For the project demo, use the setup code shared in README.txt.")
            register_submitted = st.form_submit_button("Create Account", type="primary")

        if register_submitted:
            clean_username = new_username.strip()
            if len(clean_username) < 3:
                st.error("Username must contain at least 3 characters.")
            elif len(new_password) < 8:
                st.error("Password must contain at least 8 characters.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                requested_role = "admin" if admin_code == "BOOKADMIN2026" else "user"
                new_user_id, error_message = create_user(clean_username, new_password, requested_role)
                if error_message:
                    st.error(error_message)
                else:
                    st.session_state.user_id = new_user_id
                    st.session_state.username = clean_username
                    st.success("Account created successfully!")
                    st.rerun()

    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.markdown("## 📚 Book Explorer")
st.sidebar.caption("Discover your next favourite read")

# Read this user's persistent wishlist from SQLite.
current_user_id = st.session_state.user_id
wishlist_titles = get_wishlist(current_user_id)

current_role = get_user_role(current_user_id)
st.sidebar.caption(f"Signed in as **{st.session_state.username}** · {current_role.title()}")
if current_role != "admin":
    with st.sidebar.expander("Admin setup"):
        setup_code = st.text_input("Admin setup code", type="password", key="admin_setup_code")
        if st.button("Activate Admin Access", key="activate_admin_btn"):
            if setup_code == "BOOKADMIN2026":
                activate_admin(current_user_id)
                st.success("Admin access activated.")
                st.rerun()
            else:
                st.error("Incorrect admin setup code.")
if st.sidebar.button("Log out"):
    st.session_state.user_id = None
    st.session_state.username = None
    st.rerun()

navigation = ["Home", "Recommendations", "AI Book Assistant", "Explore Books", "Book Details", "Wishlist", "Analytics"]
if current_role == "admin":
    navigation.append("Admin Dashboard")
page = st.sidebar.radio("Navigation", navigation)

st.sidebar.divider()
st.sidebar.caption(f"📖 {len(books)} books in your library")
st.sidebar.caption(f"💛 {len(wishlist_titles)} saved favourites")

# -------------------- HOME PAGE --------------------
if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>📚 Find your next great read</h1>
        <p>Explore books, discover similar stories and get recommendations
        powered by machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("📚 Total Books", len(books))
    col2.metric("⭐ Average Rating", f"{books['rating'].mean():.2f} / 5")
    col3.metric("🏷️ Total Genres", books["genre"].nunique())

    st.markdown("### ⭐ Top Rated Books")
    st.markdown(
        '<p class="section-note">Reader favourites from your collection</p>',
        unsafe_allow_html=True
    )

    top_books = books.sort_values("rating", ascending=False).head(5)
    columns = st.columns(5)

    for column, (_, book) in zip(columns, top_books.iterrows()):
        with column:
            with st.container(border=True):
                show_cover(book["title"], book["author"])
                st.markdown(
                    f'<div class="book-title">{book["title"]}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"⭐ **{book['rating']:.1f}**")
                st.caption(book["author"])
                st.caption(book["genre"])

    st.markdown("### 🧭 Browse by Genre")
    genre_counts = books["genre"].value_counts()
    genre_columns = st.columns(4)

    for index, (genre_name, total) in enumerate(genre_counts.items()):
        with genre_columns[index % 4]:
            st.info(f"**{genre_name}**  \n{total} books")

# -------------------- RECOMMENDATIONS PAGE --------------------
elif page == "Recommendations":
    st.markdown("""
    <div class="hero">
        <h1>✨ Your personal book finder</h1>
        <p>Select a book you enjoy and discover titles with similar themes,
        authors and descriptions.</p>
    </div>
    """, unsafe_allow_html=True)

    selected_book = st.selectbox(
        "Choose a book you like",
        books["title"].tolist()
    )

    selected_row = books.loc[
        books["title"] == selected_book
    ].iloc[0]

    left, right = st.columns([1, 3])

    with left:
        show_cover(selected_row["title"], selected_row["author"])
        st.caption(
            f"⭐ {selected_row['rating']:.1f} · {selected_row['genre']}"
        )

    with right:
        st.subheader(selected_row["title"])
        st.write(f"**Author:** {selected_row['author']}")
        st.write(selected_row["description"])

        count = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=min(10, len(books) - 1),
            value=min(5, len(books) - 1)
        )

    if st.button(
        "✨ Recommend Similar Books",
        type="primary",
        use_container_width=True
    ):
        results = model.recommend(selected_book, count)

        st.markdown("### 📚 Recommended for You")

        if results.empty:
            st.warning("No recommendations found for this title.")
        else:
            result_columns = st.columns(3)

            for index, (_, book) in enumerate(results.iterrows()):
                with result_columns[index % 3]:
                    with st.container(border=True):
                        show_cover(book["title"], book["author"])
                        st.markdown(f"**{book['title']}**")
                        st.caption(book["author"])
                        st.write(
                            f"⭐ {book['rating']:.1f} · {book['genre']}"
                        )
                        similarity = float(book["similarity"])
                        st.progress(min(similarity / 100, 1.0))
                        st.caption(f"Similarity: {similarity:.1f}%")
                        st.write(book["description"])

# -------------------- AI BOOK ASSISTANT PAGE --------------------
elif page == "AI Book Assistant":
    st.markdown("""
    <div class="hero">
        <h1>🤖 AI Book Assistant</h1>
        <p>Describe the kind of book you want and get suggestions from your library.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("This assistant uses local TF-IDF text matching on your book catalogue. It works offline and does not need an API key.")
    st.markdown("**Try asking:** *I want a motivational self-help book* · *Suggest fantasy adventure* · *Books about money and finance*")

    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = [{
            "role": "assistant",
            "content": "Hi! 📚 Tell me what you enjoy reading — a genre, topic, author, mood, or a few keywords. I'll find matching books in your library."
        }]

    for message in st.session_state.assistant_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("results"):
                result_frame = pd.DataFrame(message["results"])
                result_cols = st.columns(min(3, len(result_frame)))
                for result_index, (_, result) in enumerate(result_frame.iterrows()):
                    with result_cols[result_index % len(result_cols)]:
                        with st.container(border=True):
                            show_cover(result["title"], result["author"])
                            st.markdown(f"**{result['title']}**")
                            st.caption(f"{result['author']} · {result['genre']}")
                            st.write(f"⭐ {float(result['rating']):.1f} / 5")
                            st.caption(result["reason"])

    user_prompt = st.chat_input("Describe the book you are looking for...")
    if user_prompt:
        st.session_state.assistant_messages.append({"role": "user", "content": user_prompt})
        matched_books = get_assistant_recommendations(user_prompt, books, count=5)
        if matched_books.empty:
            answer = "I couldn't find a matching book in the current library. Try another genre, topic, author, or keyword."
            result_records = []
        else:
            best_score = float(matched_books.iloc[0]["match_score"])
            if best_score <= 0:
                answer = "I couldn't find a strong text match, so here are some of the highest-rated books in your library. Try adding a genre or topic for more focused suggestions."
            else:
                answer = "Here are the books that best match your request. I ranked them using their title, author, genre and description, with rating used as a tie-breaker."
            result_records = []
            for _, book in matched_books.iterrows():
                record = book.to_dict()
                record["reason"] = explain_book_match(book, user_prompt)
                result_records.append(record)
        st.session_state.assistant_messages.append({
            "role": "assistant", "content": answer, "results": result_records
        })
        st.rerun()

    if st.button("🧹 Clear chat", key="clear_assistant_chat"):
        st.session_state.assistant_messages = [{
            "role": "assistant",
            "content": "Chat cleared! Tell me what kind of book you would like to read."
        }]
        st.rerun()

# -------------------- EXPLORE BOOKS PAGE --------------------
elif page == "Explore Books":
    st.markdown("""
    <div class="hero">
        <h1>🔎 Explore the library</h1>
        <p>Search by title or author and filter the collection by genre.</p>
    </div>
    """, unsafe_allow_html=True)

    search = st.text_input(
        "Search books or authors",
        placeholder="e.g. Harry Potter, J.K. Rowling"
    )

    genres = ["All"] + sorted(books["genre"].dropna().unique().tolist())
    selected_genre = st.selectbox("Filter by genre", genres)

    filtered_books = books.copy()

    if search:
        filtered_books = filtered_books[
            filtered_books["title"].str.contains(
                search, case=False, na=False
            )
            |
            filtered_books["author"].str.contains(
                search, case=False, na=False
            )
        ]

    if selected_genre != "All":
        filtered_books = filtered_books[
            filtered_books["genre"] == selected_genre
        ]

    st.caption(f"Showing {len(filtered_books)} of {len(books)} books")

    if filtered_books.empty:
        st.warning("No books match your search. Try another title or genre.")
    else:
        for start in range(0, len(filtered_books), 4):
            row_books = filtered_books.iloc[start:start + 4]
            columns = st.columns(4)

            for column, (_, book) in zip(columns, row_books.iterrows()):
                with column:
                    with st.container(border=True):
                        show_cover(book["title"], book["author"])
                        st.markdown(f"**{book['title']}**")
                        st.caption(book["author"])
                        st.write(f"⭐ {book['rating']:.1f}")
                        st.caption(book["genre"])

                        with st.expander("About this book"):
                            st.write(book["description"])


# -------------------- BOOK DETAILS PAGE --------------------
elif page == "Book Details":
    st.markdown("""
    <div class="hero">
        <h1>📖 Book Details</h1>
        <p>Explore a book's cover, author, genre, rating and description.</p>
    </div>
    """, unsafe_allow_html=True)

    detail_title = st.selectbox(
        "Choose a book",
        books["title"].tolist(),
        key="details_book_select"
    )
    detail_book = books.loc[books["title"] == detail_title].iloc[0]

    image_col, info_col = st.columns([1, 2])

    with image_col:
        show_cover(detail_book["title"], detail_book["author"])

    with info_col:
        st.title(detail_book["title"])
        st.subheader(detail_book["author"])
        st.write(f"**Genre:** {detail_book['genre']}")
        st.write(f"**Rating:** ⭐ {detail_book['rating']:.1f} / 5")
        st.write("**About this book**")
        st.write(detail_book["description"])

        if detail_title in get_wishlist(current_user_id):
            if st.button("💛 Remove from Wishlist", type="secondary"):
                remove_from_wishlist(current_user_id, detail_title)
                st.rerun()
        else:
            if st.button("🤍 Add to Wishlist", type="primary"):
                add_to_wishlist(current_user_id, detail_title)
                st.rerun()

# -------------------- WISHLIST PAGE --------------------
elif page == "Wishlist":
    st.markdown("""
    <div class="hero">
        <h1>💛 My Wishlist</h1>
        <p>Your saved books, all in one place.</p>
    </div>
    """, unsafe_allow_html=True)

    saved_titles = get_wishlist(current_user_id)

    if not saved_titles:
        st.info("Your wishlist is empty. Open Book Details and add books you like.")
    else:
        st.write(f"**{len(saved_titles)} saved book(s)**")
        for start in range(0, len(saved_titles), 3):
            row_titles = saved_titles[start:start + 3]
            columns = st.columns(3)

            for column, title in zip(columns, row_titles):
                matched = books.loc[books["title"] == title]
                if matched.empty:
                    continue
                saved_book = matched.iloc[0]

                with column:
                    with st.container(border=True):
                        show_cover(saved_book["title"], saved_book["author"])
                        st.markdown(f"**{saved_book['title']}**")
                        st.caption(saved_book["author"])
                        st.write(
                            f"⭐ {saved_book['rating']:.1f} · {saved_book['genre']}"
                        )
                        if st.button(
                            "Remove",
                            key=f"remove_wishlist_{saved_book['book_id']}"
                        ):
                            remove_from_wishlist(current_user_id, title)
                            st.rerun()

# -------------------- ANALYTICS PAGE --------------------
elif page == "Analytics":
    st.markdown("""
    <div class="hero">
        <h1>📊 Book Analytics</h1>
        <p>A quick overview of ratings and genres in your collection.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        genre_count = books["genre"].value_counts().reset_index()
        genre_count.columns = ["Genre", "Total Books"]

        fig1 = px.bar(
            genre_count,
            x="Genre",
            y="Total Books",
            title="Books by Genre",
            template="plotly_dark",
            color="Total Books"
        )
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            books,
            x="rating",
            nbins=10,
            title="Book Rating Distribution",
            template="plotly_dark"
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🏆 Top 10 Highest Rated Books")
    top10 = books.nlargest(10, "rating")[
        ["title", "author", "genre", "rating"]
    ]
    st.dataframe(
        top10,
        use_container_width=True,
        hide_index=True
    )

# -------------------- ADMIN DASHBOARD --------------------
elif page == "Admin Dashboard":
    # Re-check role on every rerun; ordinary users cannot open admin tools.
    if get_user_role(current_user_id) != "admin":
        st.error("Admin access required.")
        st.stop()

    st.markdown("""
    <div class="hero">
        <h1>👑 Admin Dashboard</h1>
        <p>Manage books, registered users and library data.</p>
    </div>
    """, unsafe_allow_html=True)

    user_frame = get_user_list()
    total_wishlist = get_total_wishlist_items()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📚 Total Books", len(books))
    m2.metric("👥 Total Users", len(user_frame))
    m3.metric("💛 Total Wishlist Items", total_wishlist)
    m4.metric("⭐ Average Rating", f"{books['rating'].mean():.2f}" if len(books) else "0.00")

    manage_tab, users_tab, stats_tab = st.tabs(["📚 Manage Books", "👥 Manage Users", "📊 System Stats"])

    with manage_tab:
        st.subheader("Manage Books")
        search_admin = st.text_input("Search by title, author or genre", key="admin_book_search")
        admin_view = books.copy()
        if search_admin:
            mask = (
                admin_view["title"].str.contains(search_admin, case=False, na=False) |
                admin_view["author"].str.contains(search_admin, case=False, na=False) |
                admin_view["genre"].str.contains(search_admin, case=False, na=False)
            )
            admin_view = admin_view[mask]
        st.dataframe(admin_view[["book_id", "title", "author", "genre", "rating"]],
                     use_container_width=True, hide_index=True)

        with st.expander("➕ Add New Book", expanded=False):
            with st.form("admin_add_book_form", clear_on_submit=True):
                new_title = st.text_input("Book title")
                new_author = st.text_input("Author")
                new_genre = st.text_input("Genre")
                new_rating = st.number_input("Rating (0–5)", min_value=0.0, max_value=5.0,
                                             value=4.0, step=0.1)
                new_description = st.text_area("Description")
                add_submitted = st.form_submit_button("Add Book", type="primary")
            if add_submitted:
                if not all([new_title.strip(), new_author.strip(), new_genre.strip()]):
                    st.error("Please enter title, author and genre.")
                elif books["title"].str.casefold().eq(new_title.strip().casefold()).any():
                    st.error("A book with this title already exists.")
                else:
                    new_id = int(pd.to_numeric(books["book_id"], errors="coerce").max()) + 1 if len(books) else 1
                    new_row = {
                        "book_id": new_id, "title": new_title.strip(),
                        "author": new_author.strip(), "genre": new_genre.strip(),
                        "rating": float(new_rating), "description": new_description.strip()
                    }
                    updated = pd.concat([books, pd.DataFrame([new_row])], ignore_index=True)
                    save_books_to_csv(updated)
                    st.success("Book added successfully.")
                    st.rerun()

        st.markdown("#### Edit or Delete a Book")
        if len(books):
            selected_id = st.selectbox(
                "Select book", books["book_id"].tolist(),
                format_func=lambda value: f"{value} — {books.loc[books['book_id'] == value, 'title'].iloc[0]}",
                key="admin_selected_book"
            )
            selected = books.loc[books["book_id"] == selected_id].iloc[0]
            with st.form("admin_edit_book_form"):
                edit_title = st.text_input("Title", value=str(selected["title"]))
                edit_author = st.text_input("Author", value=str(selected["author"]))
                edit_genre = st.text_input("Genre", value=str(selected["genre"]))
                edit_rating = st.number_input("Rating (0–5)", min_value=0.0, max_value=5.0,
                                              value=float(selected["rating"]), step=0.1)
                edit_description = st.text_area("Description", value=str(selected["description"]))
                save_edit = st.form_submit_button("Save Changes", type="primary")
            if save_edit:
                updated = books.copy()
                row_index = updated.index[updated["book_id"] == selected_id][0]
                updated.loc[row_index, ["title", "author", "genre", "rating", "description"]] = [
                    edit_title.strip(), edit_author.strip(), edit_genre.strip(),
                    float(edit_rating), edit_description.strip()
                ]
                save_books_to_csv(updated)
                st.success("Book updated.")
                st.rerun()

            confirm_delete = st.checkbox(f"I confirm deletion of '{selected['title']}'",
                                          key=f"confirm_delete_{selected_id}")
            if st.button("🗑️ Delete Selected Book", type="secondary", disabled=not confirm_delete):
                updated = books.loc[books["book_id"] != selected_id].copy()
                save_books_to_csv(updated)
                with get_db_connection() as connection:
                    connection.execute("DELETE FROM wishlist WHERE book_title = ?", (selected["title"],))
                    connection.commit()
                st.success("Book deleted.")
                st.rerun()

    with users_tab:
        st.subheader("Registered Users")
        st.caption("Password details are never displayed.")
        if user_frame.empty:
            st.info("No registered users yet.")
        else:
            st.dataframe(user_frame, use_container_width=True, hide_index=True)

    with stats_tab:
        st.subheader("Library Statistics")
        stat_left, stat_right = st.columns(2)
        with stat_left:
            genre_stats = books["genre"].value_counts().rename_axis("Genre").reset_index(name="Books")
            st.plotly_chart(px.bar(genre_stats, x="Genre", y="Books", title="Books by Genre",
                                   template="plotly_dark"), use_container_width=True)
        with stat_right:
            if len(books):
                rating_stats = books.copy()
                st.plotly_chart(px.histogram(rating_stats, x="rating", nbins=10,
                                             title="Book Rating Distribution",
                                             template="plotly_dark"), use_container_width=True)
        st.caption("Statistics update when the dashboard reruns.")

# -------------------- FOOTER --------------------
st.divider()
st.caption(
    "AI-Based Book Recommendation System | Python · Streamlit · "
    "TF-IDF · Cosine Similarity · AI Book Assistant · User Accounts · SQLite Wishlist · Admin Dashboard"
)
