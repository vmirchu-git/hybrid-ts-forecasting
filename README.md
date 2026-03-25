# Hybrid Neural Architecture for Financial Time Series Forecasting

---

## 🇬🇧 English version
**Tech stack:** Python • PyTorch • pandas • statistics • time series analysis • deep learning  

### Context  
Financial time series forecasting is fundamentally constrained by weak predictability of returns, high noise levels, non-stationarity, and heavy-tailed distributions. Classical statistical models fail to capture nonlinear dependencies, while standalone neural networks often overfit noise without extracting stable signal.  

The core challenge of this project was not just to build a model, but to rigorously evaluate whether any predictive signal exists under realistic, leakage-free conditions.

### Goal  
- Design a hybrid neural architecture capable of capturing both local and long-term dependencies.  
- Build a statistically rigorous experimental pipeline.  
- Compare multiple problem formulations (regression, classification, probabilistic forecasting).  
- Evaluate whether deep learning provides value beyond naive baselines.  

### Approach  
- **Dataset:** S&P 500 daily data (2005–2025), covering multiple market regimes  
- **Target:** log-returns (to remove trend and stabilize variance)  

- **Feature engineering:**  
  - Intraday dynamics (open-close returns)  
  - Volatility features (rolling std)  
  - Trend indicators (moving averages)  
  - Market activity (log-volume)  

- **Data preparation:**  
  - Sliding window approach (30 days)  
  - Supervised learning formulation  

- **Architecture:**  
  - CNN → local pattern extraction  
  - LSTM → temporal dependency modeling  
  - Fully connected layer → output  

- **Modeling paradigms:**  
  1. Regression → predict return value  
  2. Classification → predict direction  
  3. Quantile regression → model uncertainty  

- **Validation strategy:**  
  - Strict chronological split (no data leakage)  
  - Train / validation / test separation  
  - Early stopping on validation  
  - Benchmarking against naive models  

- **Metrics:**  
  - Regression: MAE, RMSE  
  - Classification: Accuracy, Precision, Recall, ROC-AUC  
  - Probabilistic: interval coverage, interval width, Kupiec test  

### Results  
- Demonstrated that **predictability of mean returns is extremely limited**.  
- Neural networks **do not significantly outperform naive baselines in regression tasks**.  
- Classification results remain close to random baseline.  

- However:  
  - Strong signal identified in **volatility dynamics**  
  - Quantile models successfully adapt to market regimes  
  - Prediction intervals reflect volatility clustering  

**Key insight:**  
> Financial markets are more predictable in terms of **risk (variance)** than **returns (mean)**  

### Business Impact  
- Prevents overinvestment in ineffective predictive models  
- Provides realistic expectations for ML in finance  
- Enables shift toward:  
  - risk modeling  
  - volatility forecasting  
  - VaR estimation  

### Key Skills Highlighted  
- Time series modeling  
- Hybrid neural architectures (CNN + LSTM)  
- Statistical validation and hypothesis testing  
- Leakage-free experimental design  
- Feature engineering for financial data  
- Translating research into practical insights  

### Additional Notes  
- Fully reproducible pipeline (fixed seeds, deterministic training)  
- Emphasis on scientific rigor over metric optimization  
- Demonstrates ability to critically evaluate ML applicability  

---

## 🇷🇺 Русский вариант
**Технологии:** Python • PyTorch • pandas • статистика • временные ряды • глубокое обучение  

### Контекст  
Финансовые временные ряды характеризуются слабой предсказуемостью, высоким уровнем шума, нестационарностью и тяжёлыми хвостами распределения. Классические модели не учитывают нелинейные зависимости, а нейросети часто переобучаются на шуме.  

Ключевая задача проекта — не просто построить модель, а строго проверить наличие предсказуемого сигнала.

### Цель  
- Разработать гибридную архитектуру (CNN + LSTM)  
- Построить корректный ML-пайплайн без утечки данных  
- Сравнить разные постановки задачи  
- Оценить наличие реального сигнала  

### Подход  
- **Данные:** индекс S&P 500 (2005–2025)  
- **Целевая переменная:** логарифмическая доходность  

- **Фичи:**  
  - внутридневная динамика  
  - волатильность  
  - скользящие средние  
  - объём торгов  

- **Подготовка данных:**  
  - скользящее окно 30 дней  
  - преобразование в supervised learning  

- **Архитектура:**  
  - CNN — локальные паттерны  
  - LSTM — временные зависимости  
  - линейный слой — прогноз  

- **Постановки задачи:**  
  1. Регрессия  
  2. Классификация  
  3. Квантильная регрессия  

- **Валидация:**  
  - временное разбиение  
  - отсутствие data leakage  
  - early stopping  
  - сравнение с базовыми моделями  

### Результаты  
- Показано, что **средняя доходность практически не предсказуема**  
- Модель **не даёт значимого улучшения над наивными подходами**  
- Классификация близка к случайному угадыванию  

- При этом:  
  - обнаружен сигнал в **волатильности**  
  - модель адаптируется к рыночным режимам  
  - интервалы отражают риск  

**Ключевой вывод:**  
> Предсказуемость присутствует в **риске**, а не в **доходности**  

### Бизнес-эффект  
- Позволяет избежать неэффективных ML-решений  
- Смещает фокус на:  
  - риск-менеджмент  
  - прогнозирование волатильности  
  - VaR  

### Ключевые навыки  
- Анализ временных рядов  
- Архитектуры глубокого обучения  
- Статистическая проверка гипотез  
- Работа с шумными данными  
- Построение корректных экспериментов  

### Дополнительно  
- Воспроизводимый эксперимент  
- Акцент на научной строгости  
- Критическая оценка применимости ML  
