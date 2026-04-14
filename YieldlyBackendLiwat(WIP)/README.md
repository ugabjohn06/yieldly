Crop disease detection app for Philippine Eggplant Farmers.  

---

# Step by Step Setup

# Step 1 — Prerequisites

Make sure you have the following installed:

- **Python 3.9 or higher** → https://www.python.org/downloads/
- **pip** (comes with Python)
- A terminal / command prompt

To verify, run the following:
```
python --version
pip --version
```

---

# Step 2 — Create a Project Folder

```
Ex. YielydlyLikod
```

Copy `app.py` and `requirements.txt` into this folder.

---

# Step 3 — Create a Virtual Environment (Recommended)

```
python -m venv venv
```

Activate it:

- **Windows:**  `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

---

# Step 4 — Install Dependencies

```
pip install -r requirements.txt
```

---

# Step 5 — Run the App


streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

```
# Step 6 — Test It
```
1. The app launches in **Demo Mode** (mock inference).
2. Upload any `.jpg` or `.png` photo of an eggplant plant.
3. You'll see a simulated disease diagnosis, confidence score, action steps, and prevention tips.
```
---

# Disease Classes (Model Output)

| Index | Class              | Cause                    |
|-------|--------------------|--------------------------|
| 0     | Bacterial Wilt     | Ralstonia solanacearum   |
| 1     | Phomopsis Blight   | Phomopsis vexans         |
| 2     | Cercospora Leaf Spot | Cercospora melongenae  |
| 3     | Fruit and Shoot Borer | Leucinodes orbonalis  |
| 4     | Healthy Plant      | Good Caring              |

---

# Project Team

- Brent Dela Cruz  
- Ethan Tyler Razalan  
- Gabriel John Uy  

Yieldy v0.1 
