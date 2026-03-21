"""
Interactive VQA testing: pick a random image, ask a question, get answers from all models.

Usage in notebook:
    from interactive import InteractiveVQA
    ivqa = InteractiveVQA(loaded_models, vocab, vocab, VAL_IMAGE_DIR, device=DEVICE)
    ivqa.pick_image()           # display a random image
    ivqa.ask("What color is the cat?")  # all 4 models answer
"""

import os
import sys
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(__file__))
from inference import (
    greedy_decode, greedy_decode_with_attention,
    beam_search_decode, beam_search_decode_with_attention,
)

MODEL_DESCRIPTIONS = {
    'A': 'SimpleCNN, No Attention',
    'B': 'ResNet101, No Attention',
    'C': 'SimpleCNN, Dual Attn + Coverage',
    'D': 'ResNet101, Dual Attn + Coverage',
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class InteractiveVQA:
    """Interactive VQA session: random image + user question -> 4 model answers."""

    def __init__(self, models, vocab, vocab, image_dir,
                 device='cpu', beam_width=3, no_repeat_ngram=3):
        """
        Args:
            models: dict {model_type: model} e.g. {'A': modelA, 'B': modelB, ...}
            vocab: Vocabulary for questions
            vocab: Vocabulary for answers
            image_dir: path to val2014 image directory
            device: torch device
            beam_width: beam search width
            no_repeat_ngram: n-gram blocking size for beam search
        """
        self.models = models
        self.vocab = vocab
        self.vocab = vocab
        self.image_dir = image_dir
        self.device = device
        self.beam_width = beam_width
        self.no_repeat_ngram = no_repeat_ngram

        self.all_images = sorted(glob.glob(os.path.join(image_dir, "COCO_val2014_*.jpg")))
        if not self.all_images:
            raise FileNotFoundError(f"No images found in {image_dir}")

        self.current_img_path = None
        self.current_img_pil = None

    def pick_image(self, img_path=None):
        """Pick and display a random image (or a specific one).

        Args:
            img_path: optional specific image path. If None, picks randomly.
        """
        if img_path is not None:
            self.current_img_path = img_path
        else:
            self.current_img_path = random.choice(self.all_images)

        self.current_img_pil = Image.open(self.current_img_path).convert("RGB")

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(self.current_img_pil)
        ax.axis('off')
        ax.set_title(os.path.basename(self.current_img_path),
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

        print(f"Image ready: {os.path.basename(self.current_img_path)}")
        print("Call .ask('your question') to get model responses.")

    def ask(self, question, show_plot=True):
        """Ask a question about the current image. All loaded models respond.

        Args:
            question: str, the question to ask
            show_plot: if True, display side-by-side image + answer table

        Returns:
            dict {model_type: {'greedy': str, 'beam': str}}
        """
        if self.current_img_pil is None:
            print("No image selected. Call .pick_image() first.")
            return {}

        if not question or not question.strip():
            print("Empty question. Please provide a question string.")
            return {}

        question = question.strip()

        # Preprocess
        img_tensor = TRANSFORM(self.current_img_pil)
        q_tensor = torch.tensor(
            self.vocab.numericalize(question), dtype=torch.long
        )

        print(f"Image   : {os.path.basename(self.current_img_path)}")
        print(f"Question: {question}")
        print(f"Tokens  : {self.vocab.tokenize(question)}")
        print()

        # Run inference
        results = {}
        with torch.no_grad():
            for m in sorted(self.models.keys()):
                model = self.models[m]
                use_attn = m in ('C', 'D')

                # Greedy
                if use_attn:
                    g = greedy_decode_with_attention(
                        model, img_tensor, q_tensor, self.vocab, device=self.device)
                else:
                    g = greedy_decode(
                        model, img_tensor, q_tensor, self.vocab, device=self.device)

                # Beam search
                if use_attn:
                    b = beam_search_decode_with_attention(
                        model, img_tensor, q_tensor, self.vocab,
                        device=self.device, beam_width=self.beam_width,
                        no_repeat_ngram_size=self.no_repeat_ngram)
                else:
                    b = beam_search_decode(
                        model, img_tensor, q_tensor, self.vocab,
                        device=self.device, beam_width=self.beam_width,
                        no_repeat_ngram_size=self.no_repeat_ngram)

                results[m] = {'greedy': g, 'beam': b}

        # Print text results
        print("=" * 70)
        print("  MODEL RESPONSES")
        print("=" * 70)
        for m in sorted(results.keys()):
            desc = MODEL_DESCRIPTIONS.get(m, '')
            print(f"\n  Model {m} -- {desc}")
            print(f"  {'-' * 64}")
            print(f"  Greedy : {results[m]['greedy']}")
            print(f"  Beam({self.beam_width}): {results[m]['beam']}")
        print("\n" + "=" * 70)

        # Visual comparison
        if show_plot:
            self._plot_comparison(question, results)

        return results

    def _plot_comparison(self, question, results):
        """Display side-by-side image + answer table."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].imshow(self.current_img_pil)
        axes[0].axis('off')
        axes[0].set_title(f"Q: {question}", fontsize=11, fontweight='bold', wrap=True)

        axes[1].axis('off')
        cell_text = []
        row_labels = []
        for m in sorted(results.keys()):
            desc = MODEL_DESCRIPTIONS.get(m, m)
            row_labels.append(f"Model {m}\n({desc[:25]})")
            cell_text.append([results[m]['greedy'], results[m]['beam']])

        table = axes[1].table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=['Greedy', f'Beam (w={self.beam_width})'],
            cellLoc='left',
            loc='center',
            colWidths=[0.45, 0.45],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 2.2)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#4c72b0')
                cell.set_text_props(color='white', fontweight='bold')
            elif col == -1:
                cell.set_facecolor('#f0f0f0')
                cell.set_text_props(fontweight='bold', fontsize=8)

        axes[1].set_title('Model Responses', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
