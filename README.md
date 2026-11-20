# LoRA from scratch : fine-tuning efficace d'un Transformer

Notebook unique : `lora_from_scratch.ipynb` (à lancer sur Google Colab avec un GPU T4, ~20 min).

Implémentation de **LoRA** à la main en PyTorch (sans `peft`), appliquée à **DistilBERT** sur **SST-2**. Comparaison avec l'entraînement de la seule tête de classification et le fine-tuning complet.

## Résultats (GPU T4, une graine, 20 000 phrases, 3 époques)

| Méthode | Param. entraînables | % du modèle | Précision val. |
|---|---|---|---|
| Head only | 592 130 | 0,88 % | 0,8326 |
| LoRA r=2 (meilleur LoRA) | 628 994 | 0,94 % | 0,8922 |
| LoRA r=32 | 1 181 954 | 1,75 % | 0,8911 |
| Full fine-tuning | 66 955 010 | 100 % | 0,9060 |

LoRA récupère ~80 % de l'écart entre la tête seule et le fine-tuning complet en n'ajoutant que 0,06 % de paramètres (r = 2). Les écarts entre rangs (87,5 % à 89,2 %) sont inférieurs à l'incertitude statistique (±1,1 point sur 872 exemples). Tableau complet et analyse dans le notebook.
