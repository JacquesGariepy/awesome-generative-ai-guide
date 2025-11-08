# Chapitre 20 - Partie 3: Intégration Multimodale et Projet Complet

## Modèles Vraiment Multimodaux

Les modèles de dernière génération (2024+) sont **nativement multimodaux** : ils traitent texte, vision, audio de manière unifiée plutôt que via des modules séparés.

```python
"""
ÉVOLUTION VERS LE MULTIMODAL NATIF

Génération 1 (2022-2023): Modular
  Architecture:
    Vision Encoder → Projection → LLM
    Audio Encoder → Projection → LLM

  Problèmes:
    ❌ Modules entraînés séparément
    ❌ Alignement imparfait
    ❌ Latence (multiple forward passes)

Génération 2 (2024+): Natif
  Architecture:
    Unified Transformer qui traite ALL modalities

  Exemples:
    • GPT-4o (OpenAI): Omni-modal
    • Gemini 1.5 (Google): Natif multimodal
    • Claude 3.5 (Anthropic): Vision native

  Avantages:
    ✅ Meilleure compréhension inter-modalités
    ✅ Plus rapide (1 forward pass)
    ✅ Raisonnement sur plusieurs modalités simultanément

GPT-4o (Omni):
  • Entrées: Texte, Image, Audio
  • Sorties: Texte, Audio (temps réel)
  • Latence: ~320ms audio-to-audio (human-like)
  • Capacités: Conversations vocales naturelles

Gemini 1.5:
  • Context: 1M tokens (incluant images/vidéo)
  • Natif multimodal dès l'entraînement
  • Peut analyser 1h de vidéo + raisonner dessus
"""

from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from pathlib import Path
import json


# ============================================================================
# GPT-4o: OMNIMODAL MODEL
# ============================================================================

@dataclass
class MultiModalMessage:
    """Message multimodal pour GPT-4o"""
    role: Literal["user", "assistant", "system"]
    content: List[Dict[str, Any]]  # Peut mixer texte, images, audio

    @classmethod
    def from_text(cls, role: str, text: str) -> 'MultiModalMessage':
        """Créer message texte simple"""
        return cls(
            role=role,
            content=[{"type": "text", "text": text}]
        )

    @classmethod
    def from_image_and_text(
        cls,
        role: str,
        text: str,
        image_url: str
    ) -> 'MultiModalMessage':
        """Message avec texte + image"""
        return cls(
            role=role,
            content=[
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
        )


class GPT4Omni:
    """
    Interface pour GPT-4o (Omnimodal)

    GPT-4o = GPT-4 optimized pour multimodal

    Capacités:
      ✅ Vision (comme GPT-4V)
      ✅ Audio input/output (nouveau!)
      ✅ Temps réel (320ms latency)
      ✅ 128k context
      ✅ Cheaper que GPT-4

    Pricing (vs GPT-4):
      • Input: $5/1M tokens (vs $30)
      • Output: $15/1M tokens (vs $60)
      → 50% moins cher

    Use cases:
      • Customer support vocal
      • Real-time translation
      • Voice assistants
      • Accessibility
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "gpt-4o"

    def chat(
        self,
        messages: List[MultiModalMessage],
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> str:
        """
        Conversation multimodale

        Args:
            messages: Liste de messages (peut inclure images)
            max_tokens: Tokens max
            temperature: 0-2

        Returns:
            Réponse texte

        Exemple:
            >>> gpt4o = GPT4Omni(api_key="sk-...")
            >>> messages = [
            ...     MultiModalMessage.from_image_and_text(
            ...         "user",
            ...         "What's in this image?",
            ...         "https://example.com/photo.jpg"
            ...     )
            ... ]
            >>> response = gpt4o.chat(messages)
        """
        print("\n" + "="*80)
        print("GPT-4o CHAT")
        print("="*80)

        # Préparer les messages pour l'API
        api_messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]

        print(f"Messages: {len(messages)}")
        print(f"Modalités: ", end="")

        # Détecter modalités présentes
        modalities = set()
        for msg in messages:
            for content_item in msg.content:
                modalities.add(content_item["type"])
        print(", ".join(modalities))

        # En production:
        # import openai
        # client = openai.OpenAI(api_key=self.api_key)
        # response = client.chat.completions.create(
        #     model=self.model,
        #     messages=api_messages,
        #     max_tokens=max_tokens,
        #     temperature=temperature
        # )
        # return response.choices[0].message.content

        # Simulation
        response = "Réponse multimodale basée sur les inputs fournis..."
        print(f"\n💬 Réponse: {response}")
        return response

    def analyze_video(
        self,
        video_frames: List[str],
        question: str
    ) -> str:
        """
        Analyse une vidéo (séquence d'images)

        Args:
            video_frames: URLs des frames (sample de la vidéo)
            question: Question sur la vidéo

        Returns:
            Réponse

        Exemple:
            >>> frames = [
            ...     "frame_001.jpg",
            ...     "frame_030.jpg",
            ...     "frame_060.jpg"
            ... ]
            >>> answer = gpt4o.analyze_video(
            ...     frames,
            ...     "What is happening in this video?"
            ... )
        """
        print("\n" + "="*80)
        print("ANALYSE VIDÉO")
        print("="*80)
        print(f"Frames: {len(video_frames)}")
        print(f"Question: {question}")

        # Construire message avec toutes les frames
        content = [{"type": "text", "text": question}]

        for frame_url in video_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": frame_url}
            })

        messages = [
            MultiModalMessage(role="user", content=content)
        ]

        return self.chat(messages)


# ============================================================================
# GEMINI: MULTIMODAL NATIF
# ============================================================================

class Gemini15:
    """
    Interface Gemini 1.5 (Google)

    Gemini 1.5 Pro:
      • Context: 1M tokens (2M en preview)
      • Multimodal natif (texte, image, audio, vidéo)
      • Peut traiter: 1h vidéo, 11h audio, 700k mots
      • "Needle in haystack": Trouve info dans 1M tokens

    Pricing:
      • Input (<128k): $3.50/1M tokens
      • Input (>128k): $7/1M tokens
      • Output: $10.50/1M tokens

    Capacités uniques:
      ✅ Contexte massif (1M tokens)
      ✅ Analyse vidéo complète (pas juste frames)
      ✅ Code execution
      ✅ Grounding avec Google Search
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_content(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        videos: Optional[List[str]] = None,
        audio: Optional[List[str]] = None
    ) -> str:
        """
        Génère du contenu multimodal

        Args:
            prompt: Prompt texte
            images: Liste d'images PIL
            videos: Liste de chemins vidéo
            audio: Liste de chemins audio

        Returns:
            Réponse générée

        Exemple:
            >>> gemini = Gemini15(api_key="...")
            >>> response = gemini.generate_content(
            ...     prompt="Résume cette vidéo",
            ...     videos=["lecture.mp4"]
            ... )
        """
        print("\n" + "="*80)
        print("GEMINI 1.5 GÉNÉRATION")
        print("="*80)
        print(f"Prompt: {prompt}")

        modalities = []
        if images:
            modalities.append(f"{len(images)} images")
        if videos:
            modalities.append(f"{len(videos)} vidéos")
        if audio:
            modalities.append(f"{len(audio)} audios")

        if modalities:
            print(f"Inputs: {', '.join(modalities)}")

        # En production:
        # import google.generativeai as genai
        # genai.configure(api_key=self.api_key)
        # model = genai.GenerativeModel('gemini-1.5-pro')
        #
        # # Upload files
        # uploaded_files = []
        # for video in videos or []:
        #     file = genai.upload_file(video)
        #     uploaded_files.append(file)
        #
        # response = model.generate_content([prompt] + uploaded_files)
        # return response.text

        # Simulation
        response = f"Analyse multimodale de {', '.join(modalities) if modalities else 'texte'}..."
        print(f"\n💬 Réponse: {response}")
        return response

    def count_tokens(self, content: List[Any]) -> int:
        """
        Compte les tokens pour du contenu multimodal

        Gemini token counting:
          • Texte: ~4 chars = 1 token
          • Image: ~258 tokens (pour 1024x1024)
          • Vidéo: ~258 tokens par seconde
          • Audio: ~32 tokens par seconde
        """
        # En production:
        # model = genai.GenerativeModel('gemini-1.5-pro')
        # response = model.count_tokens(content)
        # return response.total_tokens

        # Estimation
        total = 0
        for item in content:
            if isinstance(item, str):
                total += len(item) // 4
            # TODO: add image/video/audio counting

        return total


# ============================================================================
# PROJET COMPLET: ASSISTANT MULTIMODAL INTELLIGENT
# ============================================================================

class MultiModalAssistant:
    """
    Assistant intelligent multimodal

    Capacités:
      • Comprend texte, images, audio
      • Génère texte, images, audio
      • Workflows complexes multi-étapes
      • Mémoire conversationnelle

    Use cases:
      1. Customer support:
         - User envoie screenshot → Assistant analyse et répond

      2. Content creation:
         - User: "Créer une vidéo sur les océans"
         - Assistant: génère script → images → audio → vidéo

      3. Data analysis:
         - User envoie graphique → Assistant analyse tendances

      4. Accessibility:
         - Describe images pour malvoyants
         - Transcribe audio pour malentendants
    """

    def __init__(
        self,
        gpt4o_key: Optional[str] = None,
        gemini_key: Optional[str] = None
    ):
        # Initialiser les modèles disponibles
        self.gpt4o = GPT4Omni(gpt4o_key) if gpt4o_key else None
        self.gemini = Gemini15(gemini_key) if gemini_key else None

        # Historique conversationnel
        self.history: List[Dict[str, Any]] = []

        print("="*80)
        print("🤖 ASSISTANT MULTIMODAL INITIALISÉ")
        print("="*80)
        print(f"Modèles disponibles:")
        if self.gpt4o:
            print("  ✅ GPT-4o (texte, vision, audio)")
        if self.gemini:
            print("  ✅ Gemini 1.5 (texte, vision, vidéo, audio)")

    def process_message(
        self,
        text: Optional[str] = None,
        images: Optional[List[Any]] = None,
        audio: Optional[str] = None,
        video: Optional[str] = None
    ) -> str:
        """
        Traite un message multimodal

        Args:
            text: Texte de l'utilisateur
            images: Images jointes
            audio: Fichier audio
            video: Fichier vidéo

        Returns:
            Réponse de l'assistant
        """
        print("\n" + "="*80)
        print("📥 NOUVEAU MESSAGE")
        print("="*80)

        # Déterminer les modalités
        modalities = []
        if text:
            modalities.append("texte")
        if images:
            modalities.append(f"{len(images)} image(s)")
        if audio:
            modalities.append("audio")
        if video:
            modalities.append("vidéo")

        print(f"Modalités: {', '.join(modalities)}")

        # Choisir le modèle approprié
        if video and self.gemini:
            # Gemini pour vidéo
            print("🎯 Utilisation: Gemini 1.5 (vidéo)")
            response = self.gemini.generate_content(
                prompt=text or "Analyse ce contenu",
                videos=[video] if video else None
            )

        elif images and self.gpt4o:
            # GPT-4o pour images
            print("🎯 Utilisation: GPT-4o (vision)")
            messages = [
                MultiModalMessage.from_image_and_text(
                    "user",
                    text or "Qu'y a-t-il dans cette image?",
                    "image_url"  # Simplification
                )
            ]
            response = self.gpt4o.chat(messages)

        else:
            response = "Traitement multimodal de votre demande..."

        # Sauvegarder dans l'historique
        self.history.append({
            "user": {
                "text": text,
                "modalities": modalities
            },
            "assistant": response
        })

        return response

    def create_video_content(self, topic: str) -> Dict[str, Any]:
        """
        Workflow complet: Créer une vidéo sur un sujet

        Étapes:
          1. Générer script (LLM)
          2. Générer images pour chaque scène (SD)
          3. Générer narration audio (TTS)
          4. Assembler vidéo

        Args:
            topic: Sujet de la vidéo

        Returns:
            Métadonnées de la vidéo créée
        """
        print("\n" + "="*80)
        print("🎬 CRÉATION DE VIDÉO AUTOMATIQUE")
        print("="*80)
        print(f"Sujet: {topic}")

        # Étape 1: Générer script
        print("\n📝 Étape 1/4: Génération du script")
        print("-" * 80)

        script_prompt = f"""Écris un script de 1 minute pour une vidéo sur: {topic}

Format:
[SCÈNE 1]
Texte: ...
Image: Description de l'image à générer

[SCÈNE 2]
...
"""

        # Utiliser GPT-4o ou Gemini
        if self.gpt4o:
            script = self.gpt4o.chat([
                MultiModalMessage.from_text("user", script_prompt)
            ])
        else:
            script = "[SIMULATION] Script généré automatiquement..."

        print(f"Script généré: {len(script)} caractères")

        # Étape 2: Générer images
        print("\n🎨 Étape 2/4: Génération des images")
        print("-" * 80)

        # Parser le script pour extraire descriptions d'images
        # (simplifié ici)
        image_prompts = [
            f"Scene for video about {topic}, style: cinematic, 4k"
        ]

        print(f"{len(image_prompts)} images à générer")
        # En production: utiliser Stable Diffusion

        # Étape 3: Générer audio
        print("\n🎤 Étape 3/4: Génération de la narration")
        print("-" * 80)

        # Extraire texte à narrer du script
        narration_text = f"Ceci est une vidéo sur {topic}..."
        print(f"Texte à narrer: {len(narration_text)} caractères")
        # En production: utiliser TTS

        # Étape 4: Assembler
        print("\n🎞️  Étape 4/4: Assemblage de la vidéo")
        print("-" * 80)
        print("Combinaison images + audio → MP4")

        result = {
            "script": script,
            "num_scenes": len(image_prompts),
            "duration_seconds": 60,
            "output_path": f"video_{topic.replace(' ', '_')}.mp4"
        }

        print(f"\n✅ Vidéo créée: {result['output_path']}")
        return result

    def accessibility_helper(
        self,
        image: Optional[Any] = None,
        audio: Optional[str] = None,
        mode: Literal["describe_image", "transcribe_audio"] = "describe_image"
    ) -> str:
        """
        Aide à l'accessibilité

        Args:
            image: Image à décrire (pour malvoyants)
            audio: Audio à transcrire (pour malentendants)
            mode: Type d'aide

        Returns:
            Description ou transcription
        """
        print("\n" + "="*80)
        print("♿ ACCESSIBILITÉ")
        print("="*80)

        if mode == "describe_image" and image:
            print("Mode: Description d'image pour malvoyants")

            if self.gpt4o:
                messages = [
                    MultiModalMessage.from_image_and_text(
                        "user",
                        "Décris cette image en détail pour une personne malvoyante. "
                        "Sois précis sur les objets, couleurs, positions, et l'atmosphère.",
                        "image_url"
                    )
                ]
                description = self.gpt4o.chat(messages)
            else:
                description = "Description détaillée de l'image..."

            return description

        elif mode == "transcribe_audio" and audio:
            print("Mode: Transcription audio pour malentendants")

            # Utiliser Whisper
            # whisper = SimpleWhisper("large-v3")
            # result = whisper.transcribe(audio)
            # return result.text

            return "Transcription de l'audio..."

        return "Mode ou input invalide"


# ============================================================================
# BEST PRACTICES MULTIMODAL
# ============================================================================

def print_best_practices():
    """Guide des best practices multimodal"""
    print("\n" + "="*80)
    print("📚 BEST PRACTICES - MULTIMODAL AI")
    print("="*80)

    print("""
1. CHOIX DU MODÈLE
   ┌────────────────┬──────────────┬─────────────┬──────────────┐
   │ Use Case       │ GPT-4o       │ Gemini 1.5  │ Open-source  │
   ├────────────────┼──────────────┼─────────────┼──────────────┤
   │ Vision simple  │ ✅ Excellent │ ✅ Excellent│ ⚠️  LLaVA    │
   │ Long context   │ ⚠️  128k     │ ✅ 1M       │ ❌           │
   │ Vidéo complète │ ❌           │ ✅ Natif    │ ❌           │
   │ Audio temps    │ ✅ 320ms     │ ⚠️  Batch   │ ❌           │
   │   réel         │              │             │              │
   │ Prix           │ ⚠️  $$$      │ ✅ $$       │ ✅ Gratuit   │
   │ Fine-tuning    │ ❌           │ ⚠️  Limité  │ ✅ Total     │
   └────────────────┴──────────────┴─────────────┴──────────────┘

2. OPTIMISATION DES COÛTS
   Images:
     • Resize avant envoi (max 2048px)
     • Utiliser "low" detail mode si possible
     • Batch processing pour volume

   Vidéos:
     • Sample frames (ex: 1 frame/sec) au lieu de vidéo complète
     • Utiliser GPT-4o pour frames, Gemini pour vidéo complète

   Audio:
     • Compresser en MP3 ou Opus
     • Utiliser Whisper local pour transcription
     • Envoyer transcription au LLM (moins cher que audio)

3. QUALITÉ ET PRÉCISION
   ✅ DO:
     • Prompts spécifiques et détaillés
     • Fournir contexte avec les images
     • Valider outputs critiques
     • Tester sur données diverses

   ❌ DON'T:
     • Assumer précision 100% (hallucinations)
     • Utiliser pour décisions médicales/légales sans validation
     • Ignorer biais dans les datasets

4. LATENCE ET PERFORMANCE
   • Vision: 1-5s par image (selon modèle)
   • Audio transcription: ~0.1x realtime (Whisper)
   • TTS: ~0.5s pour phrase courte
   • Image generation: 3-10s (Stable Diffusion)

   Optimisations:
     • Cache résultats fréquents
     • Paralléliser quand possible
     • Utiliser modèles plus petits si suffisant

5. SÉCURITÉ ET PRIVACY
   ⚠️  Images peuvent contenir:
     • Données personnelles (visages, plaques)
     • Informations sensibles (documents)
     • Contenu inapproprié

   Mesures:
     ✅ Vérifier data retention policies
     ✅ Anonymiser données avant envoi si nécessaire
     ✅ Content moderation pour UGC
     ✅ Compliance (GDPR, HIPAA, etc.)

6. FALLBACKS ET ERREURS
   • Timeout: Images trop grandes, vidéos trop longues
   • Refusals: Contenu interdit
   • Hallucinations: Descriptions incorrectes

   Stratégies:
     → Multiple modèles (fallback)
     → Validation croisée
     → Confidence scores
     → Human review pour cas critiques

7. MÉTRIQUES ET ÉVALUATION
   Vision:
     • Accuracy: Correct descriptions
     • Hallucination rate: Fausses informations
     • User satisfaction

   Audio:
     • WER (Word Error Rate)
     • MOS (Mean Opinion Score) pour TTS

   Génération:
     • FID (Fréchet Inception Distance)
     • CLIP score (alignement texte-image)
     • Human evaluation
    """)


# ============================================================================
# DÉMONSTRATION FINALE
# ============================================================================

def demo_complete_assistant():
    """Démo complète de l'assistant multimodal"""
    print("="*80)
    print("🎯 DÉMONSTRATION: ASSISTANT MULTIMODAL COMPLET")
    print("="*80)

    # Créer assistant
    assistant = MultiModalAssistant(
        gpt4o_key="demo",  # En production: vraie clé
        gemini_key="demo"
    )

    # Scénario 1: Analyse d'image
    print("\n\n" + "="*80)
    print("SCÉNARIO 1: CUSTOMER SUPPORT - ANALYSE D'IMAGE")
    print("="*80)

    response = assistant.process_message(
        text="Mon produit ne fonctionne pas, voici une photo du problème",
        images=["error_screen.jpg"]
    )

    # Scénario 2: Création de contenu
    print("\n\n" + "="*80)
    print("SCÉNARIO 2: CRÉATION DE CONTENU VIDÉO")
    print("="*80)

    video_result = assistant.create_video_content(
        topic="Les énergies renouvelables"
    )

    # Scénario 3: Accessibilité
    print("\n\n" + "="*80)
    print("SCÉNARIO 3: ACCESSIBILITÉ")
    print("="*80)

    # Pour malvoyants
    description = assistant.accessibility_helper(
        image="photo.jpg",
        mode="describe_image"
    )

    # Pour malentendants
    transcription = assistant.accessibility_helper(
        audio="podcast.mp3",
        mode="transcribe_audio"
    )


if __name__ == "__main__":
    # Best practices
    print_best_practices()

    # Démo complète
    demo_complete_assistant()

    print("\n\n" + "="*80)
    print("🎉 CHAPITRE 20 COMPLÉTÉ: MULTIMODAL AI")
    print("="*80)
    print("""
RÉCAPITULATIF:

PARTIE 1: Vision et Vision-Language
  • CLIP: Alignement texte-image
  • LLaVA: LLM + Vision
  • GPT-4V: Vision SOTA

PARTIE 2: Audio et Génération
  • Whisper: STT universel
  • TTS: Bark, VALL-E, ElevenLabs
  • Stable Diffusion / DALL-E 3: Génération d'images

PARTIE 3: Intégration
  • GPT-4o: Omnimodal temps réel
  • Gemini 1.5: Context massif (1M tokens)
  • Projet: Assistant multimodal complet

COMPÉTENCES ACQUISES:
  ✅ Utiliser CLIP pour classification zero-shot
  ✅ Implémenter VQA avec LLaVA
  ✅ Transcrire audio avec Whisper
  ✅ Générer images avec Stable Diffusion
  ✅ Construire assistant multimodal complet
  ✅ Best practices production

APPLICATIONS PRATIQUES:
  • Customer support intelligent
  • Création de contenu automatisée
  • Accessibilité (malvoyants, malentendants)
  • Data analysis (graphiques, documents)
  • Education (tuteurs multimodaux)

PROCHAINS CHAPITRES:
  • Ch.21: Long Context (100k+ tokens)
  • Ch.22: Chain-of-Thought et Reasoning
  • Ch.23-25: Projets pratiques complets
    """)
