# Chapitre 20 - Partie 2: Audio, Speech et Génération Multimodale

## Audio et Speech Recognition

Les modèles audio permettent de transcrire, comprendre et générer de la parole et de la musique.

```python
"""
AUDIO AI = Traitement et génération de signal audio

Tâches principales:
  1. Speech-to-Text (STT):
     Audio → Texte
     Exemple: Whisper, Wav2Vec2

  2. Text-to-Speech (TTS):
     Texte → Audio
     Exemple: Bark, VALL-E, ElevenLabs

  3. Audio Understanding:
     Audio → Classification/Analyse
     Exemple: Audio event detection, music genre

  4. Audio Generation:
     Prompt → Audio/Music
     Exemple: AudioLM, MusicGen

Défis:
  ❌ Variabilité (accents, bruit, qualité)
  ❌ Temps réel (latence)
  ❌ Multilingue
  ❌ Émotions et prosodie

État de l'art (2024):
  • Whisper (OpenAI): STT SOTA, 99 langues
  • GPT-4o: Audio natif + vision + texte
  • ElevenLabs: TTS ultra-réaliste
  • MusicGen (Meta): Génération de musique
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path


# ============================================================================
# WHISPER: SPEECH-TO-TEXT UNIVERSEL
# ============================================================================

"""
WHISPER (OpenAI, 2022) = Speech-to-Text SOTA

Architecture:
  Encoder-Decoder Transformer
  • Encoder: Mel spectrogram → Hidden states
  • Decoder: Hidden states → Tokens texte

Training:
  • 680,000 heures d'audio multilingue du web
  • 99 langues
  • Tasks: Transcription, translation, language detection

Tailles de modèles:
  • tiny: 39M params, ~32x faster
  • base: 74M params
  • small: 244M params
  • medium: 769M params
  • large-v3: 1550M params (meilleur)

Performance (WER - Word Error Rate):
  • Anglais: 2-3% (human-level ~5%)
  • Multilingue: 4-8%
  • Robust au bruit

Capacités spéciales:
  ✅ Transcription + timestamps
  ✅ Translation vers anglais
  ✅ Language detection automatique
  ✅ Voice Activity Detection (VAD)
  ✅ Robust au bruit de fond
"""

@dataclass
class WhisperOutput:
    """Sortie de Whisper"""
    text: str                           # Transcription complète
    language: str                       # Langue détectée
    segments: List[Dict[str, Any]]      # Segments avec timestamps
    duration: float                     # Durée audio en secondes


class SimpleWhisper:
    """
    Interface simplifiée pour Whisper

    En production:
        import whisper

        model = whisper.load_model("large-v3")
        result = model.transcribe("audio.mp3")

    Ou avec Transformers:
        from transformers import pipeline

        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3"
        )
        result = pipe("audio.mp3")
    """

    def __init__(self, model_size: str = "base"):
        """
        Args:
            model_size: tiny, base, small, medium, large-v3
        """
        self.model_size = model_size

        # Specs des modèles
        model_specs = {
            "tiny": {"params": "39M", "vram": "1GB", "speed": "32x"},
            "base": {"params": "74M", "vram": "1GB", "speed": "16x"},
            "small": {"params": "244M", "vram": "2GB", "speed": "6x"},
            "medium": {"params": "769M", "vram": "5GB", "speed": "2x"},
            "large-v3": {"params": "1550M", "vram": "10GB", "speed": "1x"}
        }

        specs = model_specs.get(model_size, model_specs["base"])

        print(f"\n📦 Chargement Whisper-{model_size}")
        print(f"   Paramètres: {specs['params']}")
        print(f"   VRAM: {specs['vram']}")
        print(f"   Vitesse relative: {specs['speed']}")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> WhisperOutput:
        """
        Transcrit un fichier audio

        Args:
            audio_path: Chemin vers fichier audio (mp3, wav, m4a, etc.)
            language: Code langue (fr, en, es, etc.) ou None pour auto-detect
            task: "transcribe" ou "translate" (vers anglais)

        Returns:
            WhisperOutput avec transcription et métadonnées

        Exemple:
            >>> whisper = SimpleWhisper("large-v3")
            >>> result = whisper.transcribe("meeting.mp3", language="fr")
            >>> print(result.text)
            >>> # "Bonjour à tous, bienvenue à cette réunion..."
        """
        print("\n" + "="*80)
        print("TRANSCRIPTION WHISPER")
        print("="*80)
        print(f"Fichier: {audio_path}")
        print(f"Langue: {language or 'auto-detect'}")
        print(f"Tâche: {task}")

        # En production:
        # import whisper
        # model = whisper.load_model(self.model_size)
        # result = model.transcribe(
        #     audio_path,
        #     language=language,
        #     task=task
        # )

        # Simulation
        result = WhisperOutput(
            text="Ceci est une transcription simulée de l'audio. "
                 "Le système Whisper analyse le signal audio et le convertit en texte. "
                 "Cette technologie est utile pour les sous-titres, la transcription de réunions, etc.",
            language=language or "fr",
            segments=[
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 3.5,
                    "text": "Ceci est une transcription simulée de l'audio."
                },
                {
                    "id": 1,
                    "start": 3.5,
                    "end": 7.2,
                    "text": "Le système Whisper analyse le signal audio et le convertit en texte."
                }
            ],
            duration=10.5
        )

        print(f"\n✅ Transcription complétée")
        print(f"   Langue détectée: {result.language}")
        print(f"   Durée: {result.duration:.1f}s")
        print(f"   Segments: {len(result.segments)}")
        print(f"\n📝 Texte:")
        print(f"   {result.text}")

        return result

    def transcribe_with_timestamps(
        self,
        audio_path: str,
        word_level: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Transcription avec timestamps détaillés

        Args:
            audio_path: Fichier audio
            word_level: Si True, timestamp par mot (sinon par phrase)

        Returns:
            Liste de segments avec start, end, text

        Exemple:
            >>> segments = whisper.transcribe_with_timestamps("audio.mp3")
            >>> for seg in segments:
            ...     print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}")
            [0.0s - 3.5s]: Première phrase
            [3.5s - 7.2s]: Deuxième phrase
        """
        result = self.transcribe(audio_path)

        print(f"\n📍 TIMESTAMPS {'(mot)' if word_level else '(segment)'}")
        print("-" * 80)

        for segment in result.segments:
            start = segment['start']
            end = segment['end']
            text = segment['text']
            print(f"[{start:6.1f}s - {end:6.1f}s]: {text}")

        return result.segments


# ============================================================================
# TEXT-TO-SPEECH (TTS)
# ============================================================================

"""
TEXT-TO-SPEECH = Synthèse vocale

Évolution:
  2010s: Concatenative TTS
    • Assemblage de morceaux pré-enregistrés
    • Robotique

  2016+: Neural TTS
    • Tacotron, WaveNet
    • Plus naturel

  2023+: Foundation TTS
    • VALL-E (Microsoft): Clone voice avec 3s d'audio
    • Bark (Suno): Open-source, émotions
    • ElevenLabs: Ultra-réaliste, commercial

Composants:
  1. Text Analysis:
     Texte → Phonèmes

  2. Acoustic Model:
     Phonèmes → Mel spectrogram

  3. Vocoder:
     Mel spectrogram → Audio waveform

Métriques:
  • MOS (Mean Opinion Score): 1-5, humain évaluation
  • WER: Word Error Rate (si re-transcrit)
  • Naturalness, Intelligibility, Prosody
"""

@dataclass
class TTSOutput:
    """Sortie TTS"""
    audio: np.ndarray       # Waveform
    sample_rate: int        # Hz (ex: 24000)
    duration: float         # Secondes


class SimpleTTS:
    """
    Système Text-to-Speech

    En production:
        # Bark (open-source)
        from transformers import AutoProcessor, BarkModel

        processor = AutoProcessor.from_pretrained("suno/bark")
        model = BarkModel.from_pretrained("suno/bark")

        inputs = processor("Hello, how are you?")
        audio = model.generate(**inputs)

        # OU ElevenLabs (API)
        from elevenlabs import generate, play

        audio = generate(
            text="Hello, how are you?",
            voice="Bella"
        )
        play(audio)
    """

    def __init__(self, voice: str = "default"):
        self.voice = voice
        print(f"\n🎤 Initialisation TTS")
        print(f"   Voix: {voice}")

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0
    ) -> TTSOutput:
        """
        Synthétise de l'audio depuis du texte

        Args:
            text: Texte à synthétiser
            speed: Vitesse (0.5 = lent, 2.0 = rapide)
            pitch: Hauteur (0.5 = grave, 2.0 = aigu)

        Returns:
            TTSOutput avec waveform

        Exemple:
            >>> tts = SimpleTTS(voice="fr-male-1")
            >>> audio = tts.synthesize("Bonjour tout le monde!")
            >>> # Sauvegarder: soundfile.write("output.wav", audio.audio, audio.sample_rate)
        """
        print("\n" + "="*80)
        print("SYNTHÈSE VOCALE")
        print("="*80)
        print(f"Texte: {text}")
        print(f"Voix: {self.voice}")
        print(f"Vitesse: {speed}x")
        print(f"Pitch: {pitch}x")

        # Simulation
        sample_rate = 24000
        duration = len(text.split()) * 0.5 / speed  # ~0.5s par mot
        num_samples = int(sample_rate * duration)

        # Générer waveform fictive
        audio = np.random.randn(num_samples) * 0.1  # Bruit simulé

        result = TTSOutput(
            audio=audio,
            sample_rate=sample_rate,
            duration=duration
        )

        print(f"\n✅ Audio généré")
        print(f"   Durée: {duration:.1f}s")
        print(f"   Échantillons: {num_samples:,}")
        print(f"   Sample rate: {sample_rate} Hz")

        return result


# ============================================================================
# GÉNÉRATION MULTIMODALE: DALL-E, STABLE DIFFUSION
# ============================================================================

"""
GÉNÉRATION D'IMAGES DEPUIS TEXTE

Modèles principaux:
  1. DALL-E 2/3 (OpenAI):
     • Haute qualité
     • API payante
     • $0.02-0.08 par image

  2. Stable Diffusion (Stability AI):
     • Open-source
     • Gratuit (self-hosted)
     • Très personnalisable

  3. Midjourney:
     • Meilleure qualité artistique
     • Discord bot
     • Subscription

Architecture (Stable Diffusion):
  1. Text Encoder (CLIP):
     Prompt → Text embedding

  2. Diffusion Model (UNet):
     Noise + Text embedding → Image (progressivement)
     Processus:
       Random noise → Moins de noise → Moins de noise → Image finale
       (50-100 steps)

  3. Decoder (VAE):
     Latent space → High-res image

Techniques avancées:
  • LoRA: Fine-tuning efficace
  • ControlNet: Contrôle de la composition
  • IP-Adapter: Style transfer
  • Inpainting: Édition localisée
"""

@dataclass
class ImageGenerationOutput:
    """Sortie de génération d'image"""
    images: List[Any]  # PIL Images
    prompt: str
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None


class SimpleStableDiffusion:
    """
    Interface Stable Diffusion

    En production:
        from diffusers import StableDiffusionPipeline
        import torch

        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")

        image = pipe(
            prompt="a beautiful sunset over mountains",
            negative_prompt="blurry, low quality",
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
    """

    def __init__(self, model: str = "sd-2.1"):
        self.model = model
        print(f"\n🎨 Chargement Stable Diffusion: {model}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted",
        num_images: int = 1,
        steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> ImageGenerationOutput:
        """
        Génère des images depuis un prompt texte

        Args:
            prompt: Description de l'image désirée
            negative_prompt: Ce qu'on ne veut PAS dans l'image
            num_images: Nombre d'images à générer
            steps: Nombre d'étapes de diffusion (20-100)
            guidance_scale: Force du guidage texte (1-20, 7.5 optimal)
            seed: Seed pour reproductibilité

        Returns:
            ImageGenerationOutput

        Exemple:
            >>> sd = SimpleStableDiffusion()
            >>> result = sd.generate(
            ...     prompt="a cute cat wearing a wizard hat, digital art",
            ...     negative_prompt="blurry, realistic, photo",
            ...     num_images=4,
            ...     steps=50
            ... )
            >>> result.images[0].save("cat_wizard.png")
        """
        print("\n" + "="*80)
        print("GÉNÉRATION D'IMAGE - STABLE DIFFUSION")
        print("="*80)
        print(f"Prompt: {prompt}")
        print(f"Negative: {negative_prompt}")
        print(f"Images: {num_images}")
        print(f"Steps: {steps}")
        print(f"Guidance: {guidance_scale}")
        print(f"Seed: {seed or 'random'}")

        # Estimation du temps
        time_per_step = 0.05  # 50ms par step sur GPU
        total_time = steps * time_per_step * num_images
        print(f"\n⏱️  Temps estimé: {total_time:.1f}s")

        # Simulation
        from PIL import Image
        images = [
            Image.new('RGB', (512, 512), color=(100+i*20, 150, 200))
            for i in range(num_images)
        ]

        result = ImageGenerationOutput(
            images=images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed
        )

        print(f"\n✅ {num_images} image(s) générée(s)")
        return result

    def img2img(
        self,
        image: Any,
        prompt: str,
        strength: float = 0.75
    ):
        """
        Transforme une image existante selon un prompt

        Args:
            image: Image PIL source
            prompt: Comment modifier l'image
            strength: Force de la transformation (0-1)

        Exemple:
            >>> sd = SimpleStableDiffusion()
            >>> result = sd.img2img(
            ...     photo,
            ...     prompt="turn into oil painting",
            ...     strength=0.7
            ... )
        """
        print("\n" + "="*80)
        print("IMAGE-TO-IMAGE")
        print("="*80)
        print(f"Prompt: {prompt}")
        print(f"Strength: {strength}")
        print("\n💡 L'image source sera transformée selon le prompt")


class DALLE3:
    """
    Interface DALL-E 3 (OpenAI)

    DALL-E 3 > DALL-E 2:
      ✅ Meilleure qualité
      ✅ Meilleur respect du prompt
      ✅ Meilleur texte dans images
      ✅ Plus de styles

    Pricing:
      • Standard (1024×1024): $0.040/image
      • HD (1024×1792): $0.080/image
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1
    ) -> List[str]:
        """
        Génère des images avec DALL-E 3

        Args:
            prompt: Description (max 4000 chars)
            size: "1024x1024", "1024x1792", "1792x1024"
            quality: "standard" ou "hd"
            n: Nombre d'images (1-10)

        Returns:
            Liste d'URLs d'images

        Exemple:
            >>> dalle = DALLE3(api_key="sk-...")
            >>> urls = dalle.generate(
            ...     prompt="A serene landscape with mountains and a lake at sunset",
            ...     quality="hd"
            ... )
            >>> print(urls[0])
        """
        print("\n" + "="*80)
        print("DALL-E 3 GÉNÉRATION")
        print("="*80)
        print(f"Prompt: {prompt}")
        print(f"Size: {size}")
        print(f"Quality: {quality}")

        # En production:
        # import openai
        # response = openai.images.generate(
        #     model="dall-e-3",
        #     prompt=prompt,
        #     size=size,
        #     quality=quality,
        #     n=n
        # )
        # return [img.url for img in response.data]

        # Simulation
        urls = [f"https://example.com/dalle3_image_{i}.png" for i in range(n)]
        print(f"\n✅ {n} image(s) générée(s)")
        for i, url in enumerate(urls, 1):
            print(f"   {i}. {url}")

        return urls


# ============================================================================
# DÉMONSTRATIONS
# ============================================================================

def demo_whisper():
    """Démo Whisper transcription"""
    print("="*80)
    print("DÉMONSTRATION: WHISPER SPEECH-TO-TEXT")
    print("="*80)

    # Test différentes tailles
    print("\n📊 COMPARAISON DES MODÈLES")
    print("-" * 80)

    for size in ["tiny", "base", "small", "medium", "large-v3"]:
        whisper = SimpleWhisper(model_size=size)

    # Transcription
    print("\n\n🎤 TRANSCRIPTION")
    print("-" * 80)

    whisper = SimpleWhisper("large-v3")
    result = whisper.transcribe(
        "meeting.mp3",
        language="fr"
    )

    # Timestamps
    print("\n\n📍 TRANSCRIPTION AVEC TIMESTAMPS")
    print("-" * 80)

    segments = whisper.transcribe_with_timestamps("meeting.mp3")


def demo_tts():
    """Démo Text-to-Speech"""
    print("\n\n" + "="*80)
    print("DÉMONSTRATION: TEXT-TO-SPEECH")
    print("="*80)

    tts = SimpleTTS(voice="fr-male-1")

    # Test 1: Synthèse simple
    print("\n🔊 Test 1: Synthèse simple")
    audio = tts.synthesize(
        "Bonjour, bienvenue dans ce tutoriel sur les modèles multimodaux!"
    )

    # Test 2: Variations
    print("\n🔊 Test 2: Vitesse et pitch")
    audio_fast = tts.synthesize(
        "Texte rapide!",
        speed=1.5,
        pitch=1.2
    )


def demo_image_generation():
    """Démo génération d'images"""
    print("\n\n" + "="*80)
    print("DÉMONSTRATION: GÉNÉRATION D'IMAGES")
    print("="*80)

    # Stable Diffusion
    print("\n🎨 STABLE DIFFUSION")
    print("-" * 80)

    sd = SimpleStableDiffusion()

    result = sd.generate(
        prompt="a majestic lion in a futuristic cyberpunk city, neon lights, digital art, highly detailed",
        negative_prompt="blurry, low quality, realistic photo, watermark",
        num_images=4,
        steps=50,
        guidance_scale=7.5
    )

    # Image-to-image
    print("\n\n🔄 IMAGE-TO-IMAGE")
    print("-" * 80)

    from PIL import Image
    source_img = Image.new('RGB', (512, 512))

    sd.img2img(
        source_img,
        prompt="transform into impressionist painting style",
        strength=0.7
    )

    # DALL-E 3
    print("\n\n🎨 DALL-E 3")
    print("-" * 80)

    # dalle = DALLE3(api_key="sk-...")
    # urls = dalle.generate(
    #     prompt="A peaceful zen garden with cherry blossoms",
    #     quality="hd"
    # )


def demo_multimodal_workflow():
    """Démo workflow multimodal complet"""
    print("\n\n" + "="*80)
    print("WORKFLOW MULTIMODAL COMPLET")
    print("="*80)

    print("""
Scénario: Création de contenu vidéo automatique

1. INPUT: Script texte
   "Créer une vidéo sur les océans"

2. GÉNÉRATION IMAGE (Stable Diffusion):
   → Scène 1: "beautiful coral reef underwater, 4k"
   → Scène 2: "whale swimming in deep blue ocean"
   → Scène 3: "sunset over calm ocean waves"

3. GÉNÉRATION AUDIO (TTS):
   → Narration: "Les océans couvrent 70% de notre planète..."

4. ASSEMBLAGE VIDÉO:
   → Combiner images + audio → MP4

5. SOUS-TITRES (Whisper):
   → Transcrire audio → SRT file

RÉSULTAT: Vidéo complète générée automatiquement!
    """)


if __name__ == "__main__":
    # Démos
    demo_whisper()
    demo_tts()
    demo_image_generation()
    demo_multimodal_workflow()

    print("\n\n" + "="*80)
    print("KEY TAKEAWAYS - PARTIE 2")
    print("="*80)
    print("""
1. WHISPER (Speech-to-Text)
   • SOTA pour transcription (99 langues)
   • Robust au bruit
   • Timestamps automatiques
   • Translation vers anglais

2. TEXT-TO-SPEECH
   • Bark: Open-source, émotions
   • VALL-E: Voice cloning 3s
   • ElevenLabs: Ultra-réaliste (commercial)

3. GÉNÉRATION D'IMAGES
   Stable Diffusion:
     ✅ Open-source, gratuit
     ✅ Personnalisable (LoRA, ControlNet)
     ❌ Qualité variable

   DALL-E 3:
     ✅ Meilleure qualité
     ✅ Meilleur respect du prompt
     ❌ Payant ($0.04/image)

4. APPLICATIONS
   • Transcription automatique (réunions, podcasts)
   • Génération de contenu (images, vidéos)
   • Accessibilité (TTS pour malvoyants)
   • Création artistique

PROCHAINE PARTIE: Intégration multimodale et projets
    """)
