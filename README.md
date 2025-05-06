
# 🤖 Psicobot - Analisador de Sentimentos com Machine Learning

O **Psicobot** é um assistente emocional simples que utiliza **Machine Learning** para identificar sentimentos (positivos ou negativos) com base em textos fornecidos pelo usuário. Ele foi desenvolvido em Python usando `scikit-learn` e um modelo Naive Bayes treinado com dados simulados.

---

## 🚀 Funcionalidades

- Análise automática de sentimento (positivo ou negativo) com base em frases livres.
- Interface interativa no terminal.
- Treinamento de modelo com `TfidfVectorizer` + `MultinomialNB`.
- Dados de entrada armazenados em JSON.

---

## 🧠 Tecnologias e Bibliotecas

- Python 3.10+
- pandas
- scikit-learn
- json
- re

---

## 📁 Estrutura do Projeto

```
psicobot_sentimentos/
│
├── dados_sentimentos.json        # Base de dados simulada com frases e sentimentos
├── psicobot.py                   # Script principal com o treinamento e interação
├── README.md                     # Documentação do projeto
```

---

## ▶️ Como Rodar o Projeto

1. Clone o repositório:
```bash
git clone https://github.com/felipe-migui2308/psicobot_sentimentos.git
cd psicobot_sentimentos
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o script:
```bash
python psicobot.py
```

---

## 💬 Exemplo de Uso

```
Olá, sou seu assistente emocional. Como você está se sentindo hoje?
Você: Ganhei uma promoção no trabalho!
Psicobot: Isso é maravilhoso! Fico feliz por você. Continue assim!!
```

---

## ✍️ Autor

- Luiz Felipe Miguel Custódio  
- 📘 Estudante de Ciência da Computação (6º semestre)  
- 🔗 [Meu GitHub](https://github.com/felipe-migui2308)

---

## 📌 Observações

Este é um projeto de estudo e aprendizado em Machine Learning aplicado a Processamento de Linguagem Natural (NLP).  
A base de dados utilizada é pequena e didática, podendo ser expandida para melhorar a acurácia do modelo.
