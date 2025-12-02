# 🏏 IPL Win Probability Predictor

यह एक **Streamlit आधारित Machine Learning वेब ऐप** है जो लाइव मैच डेटा के आधार पर यह अनुमान लगाता है कि **Batting Team जीत सकती है या नहीं**। यह ऐप प्रशिक्षित ML मॉडल (`model.pkl`) का उपयोग करता है।

---

## 🚀 Features

* ✅ Batting vs Bowling team चयन
* ✅ Match city आधारित prediction
* ✅ Target, runs left, balls left, wickets आधारित अनुमान
* ✅ CRR और RRR का उपयोग
* ✅ Real‑time Win Probability (%)
* ✅ Attractive UI with custom CSS

---

## 🗂 Project Structure

```
project-folder/
│
├── app.py          # Streamlit main application
├── model.pkl       # Trained machine learning model
├── README.md       # Project documentation
```

---

## ⚙️ Requirements

Python 3.8+ और नीचे दिए गए packages:

```bash
pip install streamlit scikit-learn pandas numpy
```

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Browser में automatically app open हो जाएगा।

---

## 🧠 How Prediction Works

Model निम्न features पर trained है:

* Batting Team
* Bowling Team
* Match City
* Total Target Runs
* Runs Left
* Balls Left
* Wickets Left
* Current Run Rate (CRR)
* Required Run Rate (RRR)

Model `predict_proba()` का उपयोग करके जीत की probability निकालता है।

---

## ❗ Common Errors

### model.pkl not found

```text
FileNotFoundError: model.pkl
```

✔ Solution: `model.pkl` को `app.py` वाली directory में रखें।

### Prediction error

✔ Check करें कि सभी numeric inputs सही format में हों।

---

## 🚀 Future Improvements

* Team logos add करना
* Live match API integration
* Toss factor जोड़ना
* Over by over prediction
* Mobile UI optimization

---

## 📜 License

यह प्रोजेक्ट educational और portfolio use के लिए बनाया गया है। आप इसे freely modify कर सकते हैं।

---

## 👨‍💻 Developer

**Ankit Kumar**  [Instagram](https://www.instagram.com/__ankit._.op_/)

Python | Machine Learning | Streamlit

---

✅ यह README.md GitHub repository के लिए ready है। Direct paste करके use करें।
