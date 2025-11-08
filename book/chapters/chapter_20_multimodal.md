# Chapitre 20: Modèles Multimodaux - Vision et Au-delà

## Introduction au Multimodal

Les **modèles multimodaux** peuvent traiter et générer plusieurs types de données: texte, images, audio, vidéo. Ils permettent des applications impossibles avec du texte seul.

```python
"""
MULTIMODAL = Modèles qui comprennent PLUSIEURS modalités

Évolution:
  2017-2020: Unimodal
    • GPT-2, GPT-3: Texte seulement
    • ResNet, ViT: Images seulement
    • Whisper: Audio seulement

  2021: Vision-Language (VL)
    • CLIP (OpenAI): Encode texte + images dans même espace
    • DALL-E: Génère images depuis texte
    • Applications: Recherche d'images, classification zero-shot

  2022-2023: LLMs multimodaux
    • Flamingo (DeepMind): Vision + Language LLM
    • GPT-4V (OpenAI): GPT-4 avec vision
    • LLaVA: LLM + Vision adapter
    • Applications: VQA, OCR, image understanding

  2024+: Truly multimodal
    • Gemini: Natif multimodal (texte, image, audio, vidéo)
    • GPT-4o: Omni-modal (texte, vision, audio en temps réel)
    • Applications: Agents multimodaux, assistants vocaux avancés

Modalités courantes:
  📝 Texte: Tokens, embeddings
  🖼️  Image: Pixels → Vision Transformer → embeddings
  🎵 Audio: Waveform → Spectrogram → embeddings
  🎬 Vidéo: Séquence d'images + audio
  📊 Structured data: Tables, graphs

Défis:
  ❌ Alignement entre modalités
  ❌ Coûts computationnels (images = beaucoup de tokens)
  ❌ Training data (paires texte-image rares)
  ❌ Évaluation (comment mesurer quality?)

Applications:
  ✅ Visual Question Answering (VQA)
  ✅ Image captioning
  ✅ OCR et document understanding
  ✅ Medical imaging + diagnostics
  ✅ Autonomous vehicles
  ✅ Content moderation (images + contexte)
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from PIL import Image
import base64
from io import BytesIO


# ============================================================================
# CLIP: CONTRASTIVE LANGUAGE-IMAGE PRE-TRAINING
# ============================================================================

"""
CLIP (OpenAI, 2021) = Modèle qui comprend texte ET images

Architecture:
  1. Image Encoder (Vision Transformer)
     Image → Patches → ViT → Image embedding

  2. Text Encoder (Transformer)
     Texte → Tokens → Transformer → Text embedding

  3. Contrastive Learning
     • Aligne les embeddings image et texte
     • Image de chat + "a cat" → embeddings proches
     • Image de chat + "a dog" → embeddings éloignés

Training:
  • Dataset: 400M paires (image, caption) du web
  • Objectif: Maximiser similarité des paires correctes
  • Résultat: Zero-shot classification, recherche d'images

Formule (Contrastive Loss):
  L = -log(exp(sim(I, T+) / τ) / Σ exp(sim(I, Tj) / τ))

  Où:
    I = image embedding
    T+ = text embedding correct
    Tj = tous les text embeddings du batch
    sim = cosine similarity
    τ = température (hyperparamètre)

Capacités:
  ✅ Zero-shot image classification
  ✅ Image search par texte
  ✅ Similarité image-texte
  ✅ Transfer learning

Limitations:
  ❌ Pas de génération de texte
  ❌ Pas de VQA (questions-réponses)
  ❌ Comprend, mais ne raisonne pas
"""

@dataclass
class CLIPOutput:
    """Sortie d'un modèle CLIP"""
    image_embeddings: np.ndarray    # (batch, dim)
    text_embeddings: np.ndarray     # (batch, dim)
    logits_per_image: np.ndarray    # (batch_img, batch_text)
    logits_per_text: np.ndarray     # (batch_text, batch_img)


class SimpleCLIP:
    """
    Implémentation simplifiée de CLIP

    En production, utiliser:
      - transformers: CLIPModel, CLIPProcessor
      - open_clip: Implémentation open-source avec + de modèles

    Exemple:
        >>> clip = SimpleCLIP()
        >>> scores = clip.zero_shot_classification(
        ...     image,
        ...     labels=["cat", "dog", "bird"]
        ... )
        >>> # scores = [0.8, 0.15, 0.05] → cat!
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Args:
            model_name: Nom du modèle CLIP sur HuggingFace
                - openai/clip-vit-base-patch32 (151M params)
                - openai/clip-vit-large-patch14 (428M params)
        """
        self.model_name = model_name
        self.embedding_dim = 512  # Dimension des embeddings

        print(f"Initialisation CLIP: {model_name}")
        print(f"Dimension embeddings: {self.embedding_dim}")

        # En production: charger le vrai modèle
        # from transformers import CLIPModel, CLIPProcessor
        # self.model = CLIPModel.from_pretrained(model_name)
        # self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode une image en embedding

        Args:
            image: Image PIL

        Returns:
            Embedding normalisé de dimension (embedding_dim,)
        """
        # En production:
        # inputs = self.processor(images=image, return_tensors="pt")
        # with torch.no_grad():
        #     image_features = self.model.get_image_features(**inputs)
        #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # return image_features.numpy()

        # Simulation
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)  # Normaliser
        return embedding

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode un texte en embedding

        Args:
            text: Texte à encoder

        Returns:
            Embedding normalisé
        """
        # En production:
        # inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        # with torch.no_grad():
        #     text_features = self.model.get_text_features(**inputs)
        #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # return text_features.numpy()

        # Simulation
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def compute_similarity(
        self,
        image_emb: np.ndarray,
        text_emb: np.ndarray
    ) -> float:
        """
        Calcule similarité cosinus entre image et texte

        Formule:
            similarity = (image · text) / (||image|| × ||text||)

        Comme les embeddings sont normalisés:
            similarity = image · text

        Returns:
            Score entre -1 et 1 (plus haut = plus similaire)
        """
        return float(np.dot(image_emb, text_emb))

    def zero_shot_classification(
        self,
        image: Image.Image,
        labels: List[str],
        return_scores: bool = True
    ) -> Union[str, Dict[str, float]]:
        """
        Classification zero-shot d'une image

        Processus:
          1. Encode l'image
          2. Encode chaque label en texte
          3. Calcule similarité image-texte pour chaque label
          4. Retourne label avec meilleure similarité

        Args:
            image: Image à classifier
            labels: Liste de classes possibles
            return_scores: Si True, retourne tous les scores

        Returns:
            Label prédit ou dict {label: score}

        Exemple:
            >>> scores = clip.zero_shot_classification(
            ...     cat_image,
            ...     labels=["a photo of a cat", "a photo of a dog"]
            ... )
            >>> # {"a photo of a cat": 0.92, "a photo of a dog": 0.08}
        """
        print("\n" + "="*80)
        print("ZERO-SHOT CLASSIFICATION")
        print("="*80)
        print(f"Labels: {labels}")

        # 1. Encoder l'image
        image_emb = self.encode_image(image)
        print(f"\n✅ Image encodée: shape {image_emb.shape}")

        # 2. Encoder chaque label
        text_embs = []
        for label in labels:
            text_emb = self.encode_text(label)
            text_embs.append(text_emb)

        text_embs = np.stack(text_embs)  # (n_labels, dim)
        print(f"✅ {len(labels)} labels encodés: shape {text_embs.shape}")

        # 3. Calculer similarités
        # Similarité = image_emb · text_embs^T
        similarities = image_emb @ text_embs.T  # (n_labels,)

        print(f"\n📊 Similarités brutes:")
        for label, sim in zip(labels, similarities):
            print(f"  {label}: {sim:.4f}")

        # 4. Appliquer softmax pour obtenir probabilités
        # scores = exp(sim * scale) / Σ exp(sim * scale)
        temperature = 100.0  # CLIP utilise 100
        logits = similarities * temperature
        exp_logits = np.exp(logits - np.max(logits))  # Stabilité numérique
        probs = exp_logits / exp_logits.sum()

        print(f"\n📊 Probabilités (après softmax):")
        scores_dict = {}
        for label, prob in zip(labels, probs):
            scores_dict[label] = float(prob)
            print(f"  {label}: {prob:.4f} ({prob*100:.1f}%)")

        if return_scores:
            return scores_dict
        else:
            # Retourner label avec meilleur score
            best_idx = np.argmax(probs)
            return labels[best_idx]

    def image_search(
        self,
        query: str,
        image_database: List[Tuple[str, Image.Image]],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Recherche d'images par requête texte

        Args:
            query: Requête texte (ex: "a red car")
            image_database: Liste de (id, image)
            top_k: Nombre de résultats à retourner

        Returns:
            Liste de (image_id, score) triée par score

        Exemple:
            >>> database = [
            ...     ("img1", cat_image),
            ...     ("img2", dog_image),
            ...     ("img3", car_image)
            ... ]
            >>> results = clip.image_search("a cute animal", database, top_k=2)
            >>> # [("img1", 0.92), ("img2", 0.87)]
        """
        print("\n" + "="*80)
        print("IMAGE SEARCH")
        print("="*80)
        print(f"Query: '{query}'")
        print(f"Database: {len(image_database)} images")

        # 1. Encoder la requête
        query_emb = self.encode_text(query)

        # 2. Encoder toutes les images
        image_embs = []
        image_ids = []

        for img_id, image in image_database:
            emb = self.encode_image(image)
            image_embs.append(emb)
            image_ids.append(img_id)

        image_embs = np.stack(image_embs)  # (n_images, dim)

        # 3. Calculer similarités
        similarities = query_emb @ image_embs.T  # (n_images,)

        # 4. Trier par score descendant
        sorted_indices = np.argsort(similarities)[::-1]

        # 5. Retourner top-k
        results = []
        print(f"\n📊 Top {top_k} résultats:")

        for rank, idx in enumerate(sorted_indices[:top_k], 1):
            img_id = image_ids[idx]
            score = float(similarities[idx])
            results.append((img_id, score))
            print(f"  {rank}. {img_id}: {score:.4f}")

        return results


# ============================================================================
# LLaVA: LARGE LANGUAGE AND VISION ASSISTANT
# ============================================================================

"""
LLaVA = LLM (Vicuna/Llama) + Vision Encoder (CLIP)

Architecture:
  1. Vision Encoder (CLIP ViT):
     Image → Patches → ViT → Visual features

  2. Projection Layer:
     Visual features → LLM embedding space
     (Aligne les dimensions vision et langage)

  3. Language Model (Vicuna/Llama):
     Visual tokens + Text tokens → Generate response

Training (2 étapes):
  1. Pre-training (alignment):
     • Freeze Vision Encoder et LLM
     • Train seulement le Projection layer
     • Dataset: Image captioning (COCO, etc.)
     • Objectif: Aligner vision et langage

  2. Fine-tuning (instruction):
     • Train Projection + LLM (vision encoder freeze)
     • Dataset: Instruction-following (GPT-4 generated)
     • Objectif: Suivre instructions visuelles

Exemple de conversation:
  User: [Image of a pizza] What do you see?
  LLaVA: I see a delicious pizza with cheese, tomatoes, and basil.

  User: What are the ingredients?
  LLaVA: The pizza has mozzarella cheese, fresh tomatoes, basil leaves,
         and appears to be on a thin crust.

Avantages vs GPT-4V:
  ✅ Open-source
  ✅ Peut être fine-tuné
  ✅ Moins cher

Désavantages:
  ❌ Moins performant
  ❌ Plus petit (7B-13B vs GPT-4 large)
"""

@dataclass
class VisionLanguageInput:
    """Input pour modèle vision-language"""
    images: List[Image.Image]
    text: str
    max_new_tokens: int = 512
    temperature: float = 0.7


@dataclass
class VisionLanguageOutput:
    """Output d'un modèle vision-language"""
    text: str
    visual_tokens: Optional[np.ndarray] = None


class SimpleLLaVA:
    """
    Implémentation simplifiée de LLaVA

    En production:
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf"
        )
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    """

    def __init__(self, model_name: str = "llava-1.5-7b"):
        self.model_name = model_name
        print(f"\n📦 Chargement LLaVA: {model_name}")
        print("Composants:")
        print("  • Vision Encoder: CLIP ViT-L/14")
        print("  • Projection: Linear layer (1024 → 4096)")
        print("  • LLM: Vicuna-7B")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 512
    ) -> str:
        """
        Génère une réponse basée sur image + prompt

        Args:
            image: Image PIL
            prompt: Requête utilisateur
            max_tokens: Nombre max de tokens à générer

        Returns:
            Réponse générée

        Exemple:
            >>> llava = SimpleLLaVA()
            >>> response = llava.generate(
            ...     image,
            ...     "Describe this image in detail"
            ... )
        """
        print("\n" + "="*80)
        print("GÉNÉRATION LLAVA")
        print("="*80)
        print(f"Prompt: {prompt}")

        # En production:
        # inputs = processor(
        #     text=prompt,
        #     images=image,
        #     return_tensors="pt"
        # )
        #
        # with torch.no_grad():
        #     outputs = model.generate(
        #         **inputs,
        #         max_new_tokens=max_tokens
        #     )
        #
        # response = processor.decode(outputs[0], skip_special_tokens=True)

        # Simulation
        response = f"""Basé sur l'image fournie, je vois [description détaillée].
L'image montre [éléments principaux]. Les couleurs dominantes sont [couleurs].
Cette scène suggère [contexte/interprétation]."""

        print(f"\n💬 Réponse: {response}")
        return response

    def visual_question_answering(
        self,
        image: Image.Image,
        question: str
    ) -> str:
        """
        Répond à une question sur une image

        Exemple:
            >>> answer = llava.visual_question_answering(
            ...     image,
            ...     "How many people are in this image?"
            ... )
            >>> # "There are 3 people in this image."
        """
        prompt = f"Question: {question}\nAnswer:"
        return self.generate(image, prompt, max_tokens=128)

    def image_captioning(
        self,
        image: Image.Image,
        style: str = "detailed"
    ) -> str:
        """
        Génère une caption pour une image

        Args:
            image: Image
            style: "short", "detailed", or "creative"

        Returns:
            Caption générée
        """
        prompts = {
            "short": "Provide a brief caption for this image.",
            "detailed": "Describe this image in detail.",
            "creative": "Write a creative and engaging description of this image."
        }

        prompt = prompts.get(style, prompts["detailed"])
        return self.generate(image, prompt, max_tokens=256)

    def ocr_with_understanding(
        self,
        image: Image.Image,
        task: str = "extract_and_summarize"
    ) -> str:
        """
        OCR + compréhension du texte dans l'image

        Args:
            task: "extract_only", "extract_and_summarize", "extract_and_translate"

        Returns:
            Texte extrait et traité
        """
        prompts = {
            "extract_only": "Extract all text visible in this image.",
            "extract_and_summarize": "Extract the text from this image and summarize the main points.",
            "extract_and_translate": "Extract the text and translate it to French."
        }

        prompt = prompts.get(task, prompts["extract_only"])
        return self.generate(image, prompt, max_tokens=512)


# ============================================================================
# GPT-4 VISION (GPT-4V)
# ============================================================================

class GPT4Vision:
    """
    Client pour GPT-4 Vision (GPT-4V)

    GPT-4V est le modèle vision de OpenAI, intégré dans GPT-4.

    Capacités:
      ✅ Visual Question Answering
      ✅ OCR et document understanding
      ✅ Object detection et counting
      ✅ Scene understanding
      ✅ Code generation from UI screenshots
      ✅ Math problems from images

    Limitations:
      ❌ Pas de génération d'images
      ❌ Pas d'édition d'images
      ❌ Peut halluciner des détails
      ❌ Coûteux ($0.01/image)
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def analyze_image(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        max_tokens: int = 512
    ) -> str:
        """
        Analyse une image avec GPT-4V

        Args:
            image: URL d'image ou objet PIL Image
            prompt: Question/instruction
            max_tokens: Tokens max dans la réponse

        Returns:
            Réponse générée

        Exemple:
            >>> gpt4v = GPT4Vision(api_key="sk-...")
            >>> response = gpt4v.analyze_image(
            ...     "https://example.com/chart.png",
            ...     "Explain the trends shown in this chart"
            ... )
        """
        # Convertir Image PIL en base64 si nécessaire
        if isinstance(image, Image.Image):
            image_url = self._image_to_base64_url(image)
        else:
            image_url = image

        print("\n" + "="*80)
        print("GPT-4 VISION ANALYSIS")
        print("="*80)
        print(f"Prompt: {prompt}")

        # En production:
        # import openai
        # response = openai.chat.completions.create(
        #     model="gpt-4-vision-preview",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": prompt},
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": image_url}
        #                 }
        #             ]
        #         }
        #     ],
        #     max_tokens=max_tokens
        # )
        # return response.choices[0].message.content

        # Simulation
        response = f"[GPT-4V Analysis] {prompt}\n\nRéponse détaillée basée sur l'image..."
        print(f"\n💬 Réponse: {response}")
        return response

    def _image_to_base64_url(self, image: Image.Image) -> str:
        """Convertit une image PIL en data URL base64"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"


# ============================================================================
# DÉMONSTRATIONS
# ============================================================================

def demo_clip():
    """Démo CLIP zero-shot classification"""
    print("="*80)
    print("DÉMONSTRATION: CLIP ZERO-SHOT CLASSIFICATION")
    print("="*80)

    # Créer modèle
    clip = SimpleCLIP()

    # Créer une image fictive
    image = Image.new('RGB', (224, 224), color='red')

    # Test 1: Classification simple
    print("\n\nTEST 1: Classification d'animaux")
    print("-" * 80)

    labels = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a fish"
    ]

    scores = clip.zero_shot_classification(image, labels)

    # Test 2: Classification plus complexe
    print("\n\nTEST 2: Classification de scènes")
    print("-" * 80)

    scene_labels = [
        "a beautiful sunset over the ocean",
        "a crowded city street",
        "a peaceful forest",
        "a modern office space"
    ]

    scores2 = clip.zero_shot_classification(image, scene_labels)


def demo_llava():
    """Démo LLaVA"""
    print("\n\n" + "="*80)
    print("DÉMONSTRATION: LLaVA VISION-LANGUAGE")
    print("="*80)

    llava = SimpleLLaVA()

    # Image fictive
    image = Image.new('RGB', (512, 512), color='blue')

    # Test 1: Image captioning
    print("\n\nTEST 1: Image Captioning")
    print("-" * 80)

    caption = llava.image_captioning(image, style="detailed")

    # Test 2: Visual QA
    print("\n\nTEST 2: Visual Question Answering")
    print("-" * 80)

    answer = llava.visual_question_answering(
        image,
        "What colors do you see in this image?"
    )

    # Test 3: OCR
    print("\n\nTEST 3: OCR with Understanding")
    print("-" * 80)

    ocr_result = llava.ocr_with_understanding(
        image,
        task="extract_and_summarize"
    )


if __name__ == "__main__":
    # Démos
    demo_clip()
    demo_llava()

    print("\n\n" + "="*80)
    print("KEY TAKEAWAYS - PARTIE 1")
    print("="*80)
    print("""
1. CLIP (2021)
   • Aligne texte et images dans même espace
   • Zero-shot classification sans fine-tuning
   • Applications: Image search, classification
   • Limitation: Pas de génération de texte

2. LLaVA (2023)
   • LLM + Vision = Conversations visuelles
   • Architecture: CLIP ViT + Projection + Vicuna
   • Training: Alignment → Instruction following
   • Open-source et fine-tunable

3. GPT-4 VISION
   • Vision intégrée dans GPT-4
   • Meilleur modèle vision-language (2024)
   • Coûteux mais très performant
   • API simple à utiliser

4. CAPACITÉS MULTIMODALES
   ✅ Visual Question Answering
   ✅ Image captioning
   ✅ OCR et document understanding
   ✅ Object detection
   ✅ Scene understanding

PROCHAINE PARTIE: Audio, Vidéo et Génération multimodale
    """)
