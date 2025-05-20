import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# --- Configuration ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'abstract_classifier.pt')  # Chemin  vers le modèle
print(f"Tentative de chargement du modèle depuis : {MODEL_SAVE_PATH}")
print(f"Le fichier existe : {os.path.exists(MODEL_SAVE_PATH)}")
TOKENIZER_NAME = 'bert-base-uncased'  
NUM_LABELS = 6  
MAX_LENGTH = 512  
DEVICE = torch.device("cpu")  

# Noms de classes pour l'affichage
CLASS_NAMES = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
# --- Fin de la configuration ---

# 1. Chargement du modèle et du tokenizer
def load_model_and_tokenizer(model_path, tokenizer_name, num_labels, device):
    if not os.path.exists(model_path):
        print(f"Erreur : Fichier modèle non trouvé à l'emplacement '{model_path}'.")
        print("Veuillez vous assurer d'avoir entraîné le modèle et de l'avoir enregistré à cet emplacement.")
        return None, None
        
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = BertForSequenceClassification.from_pretrained(tokenizer_name, num_labels=num_labels)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Erreur lors du chargement de l'état du modèle : {e}")
        print("Assurez-vous que l'architecture du modèle dans ce script correspond au modèle enregistré.")
        return None, None
        
    model.to(device)
    model.eval()  # Définir le modèle en mode évaluation
    print("Modèle et tokenizer chargés avec succès.")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(MODEL_SAVE_PATH, TOKENIZER_NAME, NUM_LABELS, DEVICE)

# 2. Fonction de prédiction
def classify_abstract(abstract_text):
    if model is None or tokenizer is None:
        return "Erreur : Modèle ou tokenizer non chargé. Voir les détails dans la console."
    if not abstract_text or not abstract_text.strip():
        return {"Erreur": "Veuillez saisir du texte dans le champ résumé."}

    inputs = tokenizer.encode_plus(
        abstract_text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(NUM_LABELS)}
    
    return confidences

# 3. Interface Gradio
iface = gr.Interface(
    fn=classify_abstract,
    inputs=gr.Textbox(lines=10, placeholder="Saisir le texte du résumé ici...", label="Texte du résumé"),
    outputs=gr.Label(num_top_classes=NUM_LABELS, label="Probabilités de catégorie prédites"),
    title="Classificateur de résumés scientifiques",
    description=(
        f"Saisir un résumé académique pour le classer dans l'une des catégories suivantes : "
        f"{', '.join(CLASS_NAMES)}. "
        f"Le modèle affichera la probabilité pour chaque catégorie."
    ),
    examples=[
        ["Cet article présente un algorithme novateur pour le traitement efficace des images à l'aide de réseaux de neurones convolutionnels. Nous démontrons des performances supérieures sur les jeux de données de référence tout en réduisant la complexité de calcul."],
        ["Nous étudions les propriétés mécaniques quantiques des trous noirs et leurs implications pour la théorie de l'information. Nos résultats suggèrent une résolution du paradoxe de l'information des trous noirs."]
    ],
    allow_flagging='never'
)

if __name__ == '__main__':
    print("Lancement de l'interface Gradio...")
    if model is None or tokenizer is None:
        print("Avertissement : L'interface sera lancée mais les prédictions risquent de ne pas fonctionner en raison de problèmes de chargement du modèle.")
    iface.launch()
