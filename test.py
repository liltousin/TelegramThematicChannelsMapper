import os

from dotenv import load_dotenv
from transformers import pipeline


def classify_text_by_theme(text, theme_label):
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN") or ""

    classifier = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        token=hf_token,
    )
    not_theme_label = f"не {theme_label}"
    candidate_labels = [not_theme_label, theme_label]

    result = classifier(text, candidate_labels)
    predicted_label = result["labels"][0]
    return predicted_label == theme_label


if __name__ == "__main__":
    # Пример использования функции
    sequences_to_classify = [
        "🚀 Новости из мира майнинга! Сегодня обсуждаем последние изменения в сложности добычи, а также делимся опытом по настройке эффективного майнинг-рига. 💻🛠️ #Майнинг #Криптовалюта",
        "⛏️ В этот вечер обсудим перспективы майнинга новой криптовалюты. Какие алгоритмы стоит рассмотреть и как выбрать правильное оборудование? Приходите делиться своим опытом! 🤔💡 #МайнингТренды #Крипта",
        "🔥 Свежие результаты майнингового пула! Сегодняшний день принес новые рекорды. Обсудим лучшие методы оптимизации хэшрейта и как увеличить прибыльность майнинга. 💰📈 #МайнингПул #Майнеры",
        "🔧 Технический момент: как улучшить охлаждение майнинг-фермы? Обсуждаем новые решения и тестирование эффективности. Делитесь своими секретами в комментариях! ❄️🌡️ #ОхлаждениеФермы #МайнингТехника",
        "💡 Сегодня в фокусе — майнинг альтернативных криптовалют. Где можно найти перспективные проекты, и какие аспекты стоит учесть при выборе? Обсудим вместе! 🌐🤑 #АльтМайнинг #Инвестиции",
        "Проводница выбросила кота из поезда на 30-градусный мороз. Хозяин ищет его больше недели
        Твикс возвращался с хозяином после операции. В Кирове кот выскочил из переноски и его заметила проводница. Решив, что кот бродячий, женщина выкинула его в снег. Хозяин         заметил пропажу только когда поезд тронулся.
        Сейчас котика ищет весь Киров, а полиция в действиях проводницы нарушений не нашла"
    ]
    theme = "майнинг"
    for seq in sequences_to_classify:
        is_mining = classify_text_by_theme(seq, theme)
        print(f"Текст относится к теме '{theme}': {is_mining}")