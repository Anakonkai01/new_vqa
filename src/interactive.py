"""Interactive VQA session — pick a random image, ask questions, compare models.

Designed to run inside a Jupyter notebook.  Loads any subset of models
(A–E) and provides a ``pick_image`` / ``ask`` interface.

Usage in notebook
-----------------
::

    from interactive import InteractiveVQA
    from inference import load_model_from_checkpoint
    from vocab import Vocabulary

    vocab_q = Vocabulary(); vocab_q.load("data/processed/vocab_questions.json")
    vocab_a = Vocabulary(); vocab_a.load("data/processed/vocab_answers.json")

    models = {
        "E": load_model_from_checkpoint("E", "checkpoints/model_e_best.pth",
                                        len(vocab_q), len(vocab_a), device=DEVICE),
    }

    ivqa = InteractiveVQA(models, vocab_q, vocab_a,
                          image_dir="data/raw/val2014", device=DEVICE)
    ivqa.pick_image()                    # display a random image
    ivqa.ask("What color is the cat?")   # all loaded models answer
"""

from __future__ import annotations

import glob
import os
import random
import sys
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(__file__))

from inference import (
    beam_search_decode,
    beam_search_decode_with_attention,
    greedy_decode,
    greedy_decode_with_attention,
)
from vocab import Vocabulary


# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_DESCRIPTIONS: Dict[str, str] = {
    "A": "SimpleCNN, No Attention",
    "B": "ResNet101, No Attention",
    "C": "SimpleCNN, Dual Attn + Coverage",
    "D": "ResNet101, Dual Attn + Coverage",
    "E": "CLIP ViT-B/32, FiLM + Dual Attn",
}

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── InteractiveVQA ────────────────────────────────────────────────────────────

class InteractiveVQA:
    """Interactive VQA session for Jupyter notebooks.

    Displays a randomly selected (or user-specified) image and runs all loaded
    models against a user-supplied question, showing both greedy and beam-search
    answers in a formatted table.

    Args:
        models: Dict mapping model type string (``'A'``–``'E'``) to
            a loaded, eval-mode ``nn.Module``.
        vocab_q: Question vocabulary.
        vocab_a: Answer vocabulary.
        image_dir: Path to the directory containing JPEG images
            (e.g. ``data/raw/val2014``).
        device: PyTorch device.
        beam_width: Number of beams for beam-search decode (default 3).
        no_repeat_ngram: N-gram blocking size for beam search (default 3).
    """

    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        vocab_q: Vocabulary,
        vocab_a: Vocabulary,
        image_dir: str,
        device: Union[torch.device, str] = "cpu",
        beam_width: int = 3,
        no_repeat_ngram: int = 3,
    ) -> None:
        self.models          = models
        self.vocab_q         = vocab_q
        self.vocab_a         = vocab_a
        self.image_dir       = image_dir
        self.device          = device
        self.beam_width      = beam_width
        self.no_repeat_ngram = no_repeat_ngram

        self.all_images: list[str] = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg"))
        )
        if not self.all_images:
            raise FileNotFoundError(f"No .jpg images found in {image_dir!r}")

        self.current_img_path: Optional[str] = None
        self.current_img_pil:  Optional[Image.Image] = None

    # ── Image selection ───────────────────────────────────────────────────────

    def pick_image(self, img_path: Optional[str] = None) -> None:
        """Pick and display an image.

        Args:
            img_path: Specific image path.  If ``None``, selects randomly.
        """
        self.current_img_path = img_path if img_path is not None else random.choice(self.all_images)
        self.current_img_pil  = Image.open(self.current_img_path).convert("RGB")

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(self.current_img_pil)
        ax.axis("off")
        ax.set_title(os.path.basename(self.current_img_path),
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

        print(f"Image ready: {os.path.basename(self.current_img_path)}")
        print("Call .ask('your question') to get model responses.")

    # ── Question answering ────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        show_plot: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        """Answer a question about the current image with all loaded models.

        Args:
            question: Natural-language question string.
            show_plot: If ``True``, display a side-by-side image + answer table.

        Returns:
            Dict ``{model_type: {'greedy': str, 'beam': str}}``.
        """
        if self.current_img_pil is None:
            print("No image selected. Call .pick_image() first.")
            return {}

        question = question.strip()
        if not question:
            print("Empty question. Please provide a question string.")
            return {}

        img_tensor = _TRANSFORM(self.current_img_pil)
        q_tensor   = torch.tensor(self.vocab_q.numericalize(question), dtype=torch.long)

        print(f"Image   : {os.path.basename(self.current_img_path)}")
        print(f"Question: {question}")
        print(f"Tokens  : {self.vocab_q.tokenize(question)}")
        print()

        results: Dict[str, Dict[str, str]] = {}
        with torch.no_grad():
            for m in sorted(self.models.keys()):
                model     = self.models[m]
                use_attn  = m in ("C", "D", "E")

                # Greedy decode
                if use_attn:
                    g = greedy_decode_with_attention(
                        model, img_tensor, q_tensor, self.vocab_a, device=self.device
                    )
                else:
                    g = greedy_decode(
                        model, img_tensor, q_tensor, self.vocab_a, device=self.device
                    )

                # Beam search decode
                if use_attn:
                    b = beam_search_decode_with_attention(
                        model, img_tensor, q_tensor, self.vocab_a,
                        device=self.device,
                        beam_width=self.beam_width,
                        no_repeat_ngram_size=self.no_repeat_ngram,
                    )
                else:
                    b = beam_search_decode(
                        model, img_tensor, q_tensor, self.vocab_a,
                        device=self.device,
                        beam_width=self.beam_width,
                        no_repeat_ngram_size=self.no_repeat_ngram,
                    )

                results[m] = {"greedy": g, "beam": b}

        # ── Text output ───────────────────────────────────────────────────────
        print("=" * 70)
        print("  MODEL RESPONSES")
        print("=" * 70)
        for m in sorted(results.keys()):
            desc = MODEL_DESCRIPTIONS.get(m, "")
            print(f"\n  Model {m} — {desc}")
            print(f"  {'-' * 64}")
            print(f"  Greedy : {results[m]['greedy']}")
            print(f"  Beam({self.beam_width}): {results[m]['beam']}")
        print("\n" + "=" * 70)

        if show_plot:
            self._plot_comparison(question, results)

        return results

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _plot_comparison(
        self,
        question: str,
        results: Dict[str, Dict[str, str]],
    ) -> None:
        """Render a side-by-side image + model-response table.

        Args:
            question: The question that was asked.
            results: Dict ``{model_type: {'greedy': str, 'beam': str}}``.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].imshow(self.current_img_pil)
        axes[0].axis("off")
        axes[0].set_title(f"Q: {question}", fontsize=11, fontweight="bold", wrap=True)

        axes[1].axis("off")
        cell_text:  list[list[str]] = []
        row_labels: list[str]       = []
        for m in sorted(results.keys()):
            desc = MODEL_DESCRIPTIONS.get(m, m)
            row_labels.append(f"Model {m}\n({desc[:25]})")
            cell_text.append([results[m]["greedy"], results[m]["beam"]])

        table = axes[1].table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=["Greedy", f"Beam (w={self.beam_width})"],
            cellLoc="left",
            loc="center",
            colWidths=[0.45, 0.45],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 2.2)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#4c72b0")
                cell.set_text_props(color="white", fontweight="bold")
            elif col == -1:
                cell.set_facecolor("#f0f0f0")
                cell.set_text_props(fontweight="bold", fontsize=8)

        axes[1].set_title("Model Responses", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()
