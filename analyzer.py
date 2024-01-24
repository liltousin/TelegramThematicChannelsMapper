import os

from dotenv import load_dotenv
from transformers import (XLMRobertaForSequenceClassification,
                          XLMRobertaTokenizer, pipeline)

# Ignore transformer warning


def initialize_classification_model():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN") or ""

    tokenizer = XLMRobertaTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
    model = XLMRobertaForSequenceClassification.from_pretrained(
        "joeddav/xlm-roberta-large-xnli"
    )

    return pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        token=hf_token,
    )


def classify_text_by_theme(classifier, text, theme_label):
    candidate_labels = [f"не {theme_label}", theme_label]

    result = classifier(text, candidate_labels)
    predicted_label = result["labels"][0]
    return predicted_label == theme_label, result["scores"][0]


if __name__ == "__main__":
    # pip install protobuf

    sequences_to_classify = [
        "🚀 Новости из мира майнинга! Сегодня обсуждаем последние изменения в сложности добычи, а также делимся опытом по настройке эффективного майнинг-рига. 💻🛠️ #Майнинг #Криптовалюта",
        "⛏️ В этот вечер обсудим перспективы майнинга новой криптовалюты. Какие алгоритмы стоит рассмотреть и как выбрать правильное оборудование? Приходите делиться своим опытом! 🤔💡 #МайнингТренды #Крипта",
        "🔥 Свежие результаты майнингового пула! Сегодняшний день принес новые рекорды. Обсудим лучшие методы оптимизации хэшрейта и как увеличить прибыльность майнинга. 💰📈 #МайнингПул #Майнеры",
        "🔧 Технический момент: как улучшить охлаждение майнинг-фермы? Обсуждаем новые решения и тестирование эффективности. Делитесь своими секретами в комментариях! ❄️🌡️ #ОхлаждениеФермы #МайнингТехника",
        "💡 Сегодня в фокусе — майнинг альтернативных криптовалют. Где можно найти перспективные проекты, и какие аспекты стоит учесть при выборе? Обсудим вместе! 🌐🤑 #АльтМайнинг #Инвестиции",
        ".",
    ]
    theme = "майнинг"

    classifier_model = initialize_classification_model()

    for seq in sequences_to_classify:
        is_mining, score = classify_text_by_theme(classifier_model, seq, theme)
        print(f"\n\nТекст относится к теме '{theme}': {is_mining} score: {score}\n\n")
