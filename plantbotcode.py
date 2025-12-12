import streamlit as st
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re

st.title("PlantBot – Intent Classifier Trainer")

def tokenize(sentence):
    return re.findall(r"\w+|[^\w\s]", sentence.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [w.lower() for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

uploaded_file = st.file_uploader("Upload Intents JSON File", type=["json"])

if uploaded_file:
    intents = json.load(uploaded_file)

    all_words = []
    tags = []
    xy = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    tags = sorted(set(tags))
    ignore_tokens = ['?', '!', '.', ',', "'", '"']
    all_words = [w for w in all_words if w not in ignore_tokens]
    all_words = sorted(set(all_words))

    X = []
    y = []

    for (pattern_tokens, tag) in xy:
        bow = bag_of_words(pattern_tokens, all_words)
        X.append(bow)
        y.append(tags.index(tag))

    X = np.array(X)
    y = np.array(y, dtype=np.int64)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, shuffle=True
    )

    class ChatDataset(Dataset):
        def __init__(self, X, y):
            self.x_data = torch.from_numpy(X)
            self.y_data = torch.from_numpy(y).long()
            self.n_samples = X.shape[0]

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    batch_size = 32

    train_dataset = ChatDataset(X_train, y_train)
    val_dataset = ChatDataset(X_val, y_val)
    test_dataset = ChatDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X.shape[1]
    hidden_size = 192
    output_size = len(tags)

    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            self.l1 = nn.Linear(input_size, hidden_size)
            self.l2 = nn.Linear(hidden_size, hidden_size)
            self.l3 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.l1(x)
            out = self.relu(out)
            out = self.l2(out)
            out = self.relu(out)
            out = self.l3(out)
            return out

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def compute_accuracy(data_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for words, labels in data_loader:
                words = words.to(device)
                labels = labels.to(device)
                outputs = model(words)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        model.train()
        return 100.0 * correct / total if total > 0 else 0.0

    epochs = st.number_input("Epochs", min_value=1, max_value=500, value=50)

    if st.button("Train Model"):
        progress = st.progress(0)
        for epoch in range(epochs):
            for words, labels in train_loader:
                words = words.to(device)
                labels = labels.to(device)
                outputs = model(words)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            progress.progress((epoch + 1) / epochs)

        st.success("Training Completed")

        train_acc = compute_accuracy(train_loader)
        val_acc = compute_accuracy(val_loader)
        test_acc = compute_accuracy(test_loader)

        st.write("Train Accuracy:", train_acc)
        st.write("Validation Accuracy:", val_acc)
        st.write("Test Accuracy:", test_acc)

        data = {
            "model_state": model.state_dict(),
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "all_words": all_words,
            "tags": tags
        }

        FILE = "plant_chatbot_data.pth"
        torch.save(data, FILE)
        st.download_button("Download Model (.pth)", data=open(FILE, "rb"), file_name=FILE)

        st.subheader("Chat with PlantBot")
        user_input = st.text_input("You:", "")

        if user_input:
            tokens = tokenize(user_input)
            X_test_single = bag_of_words(tokens, all_words)
            X_test_single = torch.from_numpy(X_test_single).float().unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                output = model(X_test_single)
                probs = torch.softmax(output, dim=1)
                _, predicted = torch.max(probs, dim=1)

            tag = tags[predicted.item()]

            for intent in intents["intents"]:
                if intent["tag"] == tag:
                    reply = np.random.choice(intent["responses"])
                    break

            st.write("PlantBot:", reply)
