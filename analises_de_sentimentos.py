import pandas as pd
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def carregar_dados_json(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)
    return pd.DataFrame(dados)

caminho_json = "dados_sentimentos.json"
df = carregar_dados_json(caminho_json)


def preprocessar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", texto)
    return texto

df["Texto"] = df["Texto"].apply(preprocessar_texto)

X_train, X_test, y_train, y_test = train_test_split(df["Texto"], df["Sentimento"], test_size=0.2, random_state=42)

modelo = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("naive_bayes", MultinomialNB())
])


modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")


def analisar_sentimento_interativo():
    print("Olá, sou seu assistente emocional. Como você está se sentindo hoje?")
    while True:
        texto = input("Você: ")
        if texto.lower() in ["sair", "tchau", "fim"]:
            print("Psicobot: Foi um prazer conversar com você. Cuide-se!")
            break
        texto_processado = preprocessar_texto(texto)
        sentimento = modelo.predict([texto_processado])[0]
        
        if sentimento == "positivo":
            print("Psicobot: Isso é maravilhoso! Fico feliz por você. Continue assim!!")
        elif sentimento == "negativo":
            print("Psicobot: vejo que as coisas não vão nada bem.")
            print("Psicobot: Sinto muito por isso. Mas Relaxe coisas boas estão por vir")
        return

analisar_sentimento_interativo()