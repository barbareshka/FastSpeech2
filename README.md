# FastSpeech2
Моя версия text-to-speech системы из статьи [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://github.com/xcmyz/FastSpeech). Этот проект основан на версиях [xcmyz's](https://github.com/xcmyz/FastSpeech) и [mingo24](https://github.com/ming024/FastSpeech2/tree/d4e79eb52e8b01d24703b2dfc0385544092958f3), поэтому код похож на оригинальный.

# Novigation
Краткое описание того, что находится в файлах и папках:
- model содержит класс модели FastSpeech2, VarianceAdaptor, VariancePredictor, LengthRegulator и Conv
- text содержит функции, работающие с символами и их звучанием в (английском)[http://www.speech.cs.cmu.edu/cgi-bin/cmudict] и (китайском)[https://www.pinyin-dictionary.com/?pinyin=wo+men] языке
- transformer содержит энкодер и декодер
- utils содержит вспомогательные функции (для работы непосредственно модели нужны лишь три - pad, get_mask_from_lengths и to_device) 


# How to run?

[mingo24](https://github.com/ming024/FastSpeech2/tree/d4e79eb52e8b01d24703b2dfc0385544092958f3) написал многоуровневую структуру с генерализацией, обработкой архивов и прочим. Поэтому, чтобы не копировать его код (и воспользовавшись тем, что нужно было написать лишь саму модель), я решила сделать так - написать версию, которая легко интегрируется в его версию запуска путем замены соответствующих файлов (то есть файлов с совпадающими путями). Поэтому запуск выглядит следующим образом: для начала нужно клонировать его репозиторий (PyCharm or VSCode), после чего заменить следующие файлы:. Как только это сделано - запустить на машине (работает где-то 1-2 часа) по инструкции.

Вопрос - почему так можно?
Написанный мною код - это модель + все функции, участвующие в её создании и работе. Все остальное (написанное mingo24), является некоторыми дополнениями, работающими только на запуск модели (обработка архива, например)

# How to run?
Я запускала код 4 раза (по 2 раза на 2 разных датасетах, английском LJSpeech и китайском AISHELL-3 - с первым на обучение ушел час, со вторым - три, поэтому поставим, что среднее время работы около 2 часов). 

В данном случае я анализировала лишь качество аудио, которое генерировала модель. Анализ второго не улучшил мое понимание происходящего - очень сложно было понять качество работы модели, учитывая незнание китайского и оценивая его только прогоняя те же слова через переводчик (использовался Google Translator). Анализ работы с первым датасетом привел к следующему выводу - на коротких записях модель не выдает ошибок (все слова четкие, нет странных пауз или "бликов" - сторонних звуков), но чем длиннее запись, тем выше вероятность (и частота) встреч с различными аномалиями (Особенно часто встречаются блики - какие-то помехи, по большей части они были в конце записи). [Образцы - здесь](https://drive.google.com/drive/u/0/folders/1hRo3W0j7qF-Zjk0uzK7rv1fac9ysqplQ)



# Referencies
1) [Версия FastSpeech от xcmyz's](https://github.com/xcmyz/FastSpeech)
2) [Версия FastSpeech2 от mingo24](https://github.com/ming024/FastSpeech2/tree/d4e79eb52e8b01d24703b2dfc0385544092958f3)
