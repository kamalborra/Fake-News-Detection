from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
def getPrediction(news):
    db = pickle.load(open('C:\\Users\\kamal\\OneDrive\\Desktop\\Fake News Detection\\pickles\\pre_training.pkl', 'rb'))
    tfidf_test=db.transform(news)

    db = pickle.load(open('C:\\Users\\kamal\\OneDrive\\Desktop\\Fake News Detection\\pickles\\model.pkl', 'rb'))
    res = db.predict(tfidf_test)
    if res[0]=='REAL':
        return "Entered news is real"
    return "Entered news is fake"