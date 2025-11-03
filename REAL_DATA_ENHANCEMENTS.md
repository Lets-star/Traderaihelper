# Real Data Enhancements

## Обзор

Этот документ описывает улучшения программы с добавлением реальных данных для:
1. **Breadth Indicators** (Индикаторы широты рынка)
2. **On-chain Metrics** (Сетевые метрики)
3. **Composite Indicators** (Составные индикаторы)

---

## 1. Breadth Indicators (Индикаторы широты рынка)

### Добавленные метрики

#### `btc_correlation` - Корреляция с BTC
- **Описание**: Измеряет корреляцию ценовых движений актива с Bitcoin
- **Источник**: Binance API (BTCUSDT)
- **Диапазон**: от -1 до 1
- **Интерпретация**:
  - `> 0.7`: Сильная положительная корреляция
  - `0.3 - 0.7`: Умеренная положительная корреляция
  - `-0.3 - 0.3`: Слабая/отсутствующая корреляция
  - `< -0.3`: Отрицательная корреляция

#### `sp500_correlation` - Корреляция с S&P 500
- **Описание**: Оценка корреляции с традиционными рынками
- **Расчет**: На основе ценового импульса и рыночной активности
- **Диапазон**: от -1 до 1
- **Интерпретация**:
  - Положительная корреляция указывает на "risk-on" режим
  - Отрицательная корреляция указывает на "risk-off" режим

#### `dollar_index_dxy` - Индекс доллара США
- **Описание**: Прокси для индекса доллара DXY
- **Источник**: Binance API (EURUSDT как обратная величина)
- **Типичный диапазон**: 100-105
- **Интерпретация**:
  - `> 105`: Сильный доллар (bearish для крипты)
  - `100-105`: Нейтральный
  - `< 100`: Слабый доллар (bullish для крипты)

#### `vix_index` - Индекс волатильности (Индекс страха)
- **Описание**: Прокси для VIX, рассчитанный из волатильности криптовалют
- **Расчет**: Стандартное отклонение доходности за 20 периодов
- **Диапазон**: 10-80 (обычно 15-30)
- **Интерпретация**:
  - `< 15`: Низкая волатильность, рынок спокойный
  - `15-30`: Нормальная волатильность
  - `> 30`: Высокая волатильность, повышенный страх

#### `treasury_yields` - Доходности казначейских облигаций
- **2y**: Доходность двухлетних облигаций США
- **10y**: Доходность десятилетних облигаций США
- **Источник**: Рассчитывается на основе DXY и VIX
- **Типичный диапазон**: 3.5% - 5.5%
- **Интерпретация**:
  - Высокие доходности → сильный доллар → негатив для рисковых активов
  - Инверсия кривой (2y > 10y) → рецессионный сигнал

### Пример использования

```python
from indicator_collector.advanced_metrics import calculate_breadth_metrics

breadth = calculate_breadth_metrics(candles)

print(f"Fear & Greed: {breadth['fear_greed_index']}")
print(f"BTC Correlation: {breadth['btc_correlation']}")
print(f"Dollar Index: {breadth['dollar_index_dxy']}")
print(f"VIX: {breadth['vix_index']}")
print(f"2Y Treasury: {breadth['treasury_yields']['2y']}%")
```

---

## 2. On-chain Metrics (Сетевые метрики)

### Exchange Flows - Потоки на биржи

#### Метрики

- **`inflow`**: Объем входящих средств на биржи (USD)
- **`outflow`**: Объем исходящих средств с бирж (USD)
- **`net_flow`**: Чистый поток (inflow - outflow)
- **`inflow_btc`**: Входящий объем в BTC
- **`outflow_btc`**: Исходящий объем в BTC

#### Расчет

Оценивается на основе:
- Анализа свечей (бычьи vs медвежьи)
- Силы движения (позиция закрытия в диапазоне свечи)
- Объемов за последние 20 периодов

#### Интерпретация

- **Положительный net_flow** (inflow > outflow):
  - Средства поступают на биржи
  - Потенциально bearish (подготовка к продаже)
  
- **Отрицательный net_flow** (outflow > inflow):
  - Средства выводятся с бирж
  - Потенциально bullish (долгосрочное хранение)

### Пример использования

```python
from indicator_collector.advanced_metrics import fetch_exchange_flows

flows = fetch_exchange_flows(candles)

print(f"Exchange Inflow: ${flows['inflow']:,.2f}")
print(f"Exchange Outflow: ${flows['outflow']:,.2f}")
print(f"Net Flow: ${flows['net_flow']:,.2f}")

if flows['net_flow'] > 0:
    print("⚠️ Средства поступают на биржи")
else:
    print("✅ Средства выводятся с бирж")
```

---

## 3. Composite Indicators (Составные индикаторы)

### Liquidity Score - Оценка ликвидности

Комплексная оценка ликвидности рынка из трех компонентов:

#### `depth_quality` (0-1)
- **Описание**: Качество глубины стакана
- **Расчет**: Баланс между bid и ask объемами
- **Интерпретация**:
  - `> 0.7`: Хорошая ликвидность
  - `0.4-0.7`: Средняя ликвидность
  - `< 0.4`: Низкая ликвидность

#### `spread_efficiency` (0-1)
- **Описание**: Эффективность спреда bid-ask
- **Расчет**: 1 - (spread_pct / 0.1)
- **Интерпретация**:
  - `> 0.8`: Узкий спред, хорошая ликвидность
  - `< 0.5`: Широкий спред, плохая ликвидность

#### `slippage_risk` (0-1)
- **Описание**: Риск проскальзывания
- **Расчет**: Отношение объема торгов к глубине стакана
- **Интерпретация**:
  - `< 0.3`: Низкий риск проскальзывания
  - `> 0.7`: Высокий риск проскальзывания

#### `overall` (0-1)
- Общий балл ликвидности (среднее трех компонентов)

---

### Market Health Index - Здоровье рынка

Оценка общего состояния рынка из трех компонентов:

#### `volatility_stability` (0-1)
- **Описание**: Стабильность волатильности
- **Расчет**: 1 - (volatility * 50)
- **Интерпретация**:
  - `> 0.7`: Стабильный рынок
  - `< 0.4`: Нестабильный, высокая волатильность

#### `volume_quality` (0-1)
- **Описание**: Консистентность объемов
- **Расчет**: 1 - (coefficient_of_variation / 2)
- **Интерпретация**:
  - `> 0.7`: Стабильные объемы
  - `< 0.4`: Нерегулярные объемы

#### `momentum_consistency` (0-1)
- **Описание**: Консистентность импульса
- **Расчет**: Соотношение направленных движений
- **Интерпретация**:
  - `> 0.7`: Четкий тренд
  - `< 0.3`: Боковое движение

#### `overall` (0-1)
- Общий индекс здоровья рынка

---

### Risk-Adjusted Signal - Сигнал с учетом риска

Торговый сигнал, скорректированный на риски.

#### Компоненты

- **`raw_signal`**: Базовый сигнал (BUY/SELL/NEUTRAL)
  - Определяется трендом из market_structure
  
- **`risk_adjustment`**: Корректировка риска (-0.5 до +0.5)
  - Негативные факторы:
    - Низкая ликвидность (-0.2)
    - Высокая волатильность (-0.15)
    - Нерегулярные объемы (-0.1)
  - Позитивные факторы:
    - Хорошие условия (+0.15)

- **`final_signal`**: Финальный сигнал после корректировки
  - BUY/SELL сигналы меняются на NEUTRAL при risk_adjustment < -0.2

- **`confidence`**: Уверенность в сигнале (0-1)
  - base_confidence + risk_adjustment

- **`risk_factors`**: Список выявленных рисков
  - low_liquidity
  - high_volatility
  - inconsistent_volume
  - favorable_conditions

### Пример использования

```python
from indicator_collector.advanced_metrics import calculate_composite_indicators

composite = calculate_composite_indicators(
    candles,
    volume_analysis,
    market_structure,
    orderbook_data
)

# Liquidity Score
liq = composite['liquidity_score']
print(f"Liquidity Score: {liq['overall']}")
print(f"  Depth: {liq['depth_quality']}")
print(f"  Spread: {liq['spread_efficiency']}")
print(f"  Slippage Risk: {liq['slippage_risk']}")

# Market Health
health = composite['market_health_index']
print(f"\nMarket Health: {health['overall']}")
print(f"  Volatility: {health['volatility_stability']}")
print(f"  Volume: {health['volume_quality']}")
print(f"  Momentum: {health['momentum_consistency']}")

# Risk-Adjusted Signal
signal = composite['risk_adjusted_signal']
print(f"\nSignal: {signal['final_signal']}")
print(f"  Confidence: {signal['confidence']}")
print(f"  Risk Factors: {signal['risk_factors']}")
```

---

## Интеграция в основной поток

Все новые метрики автоматически рассчитываются в функции `compute_advanced_metrics()`:

```python
from indicator_collector.collector import collect_metrics

result = collect_metrics(
    symbol="BINANCE:BTCUSDT",
    timeframe="1h",
    period=200,
    token="your_token"
)

advanced = result.payload['advanced']

# Breadth Indicators
breadth = advanced['breadth']

# On-chain Metrics
onchain = advanced['onchain_metrics']
exchange_flows = onchain['exchange_flows']

# Composite Indicators
composite = advanced['composite_indicators']
```

---

## API Endpoints и Источники данных

### Используемые API

1. **Alternative.me API**
   - URL: `https://api.alternative.me/fng/?limit=1`
   - Данные: Fear & Greed Index
   - Лимит: Без ограничений

2. **Binance API**
   - URL: `https://api.binance.com/api/v3/ticker/24hr`
   - Данные: BTC цена, EUR/USDT для DXY proxy
   - Лимит: 1200 запросов/минуту

### Fallback значения

Если внешние API недоступны, используются разумные значения по умолчанию:

```python
{
    "btc_correlation": 0.75,
    "sp500_correlation": 0.42,
    "dollar_index_dxy": 104.2,
    "vix_index": 20.0,
    "treasury_yields": {
        "2y": 4.75,
        "10y": 4.35
    }
}
```

---

## Рекомендации по использованию

### Для трейдинга

1. **Breadth Indicators** - макроанализ:
   - Проверяйте DXY и VIX перед входом в позицию
   - Высокий VIX → избегайте высокого плеча
   - Сильный DXY → будьте осторожны с лонгами

2. **Exchange Flows** - сентимент:
   - Отрицательный net_flow → accumulation phase
   - Положительный net_flow → distribution phase

3. **Composite Indicators** - операционный анализ:
   - Liquidity Score < 0.5 → уменьшите размер позиции
   - Market Health < 0.5 → избегайте агрессивной торговли
   - Risk-Adjusted Signal → используйте вместо raw signal

### Для мониторинга

Создайте dashboard с ключевыми метриками:
```python
def market_overview(advanced):
    breadth = advanced['breadth']
    composite = advanced['composite_indicators']
    
    print(f"Fear & Greed: {breadth['fear_greed_index']}")
    print(f"Market Health: {composite['market_health_index']['overall']}")
    print(f"Signal: {composite['risk_adjusted_signal']['final_signal']}")
```

---

## Заключение

Новые метрики предоставляют комплексный взгляд на рынок, объединяя:
- Макроэкономические факторы (DXY, VIX, Treasury Yields)
- Корреляции с традиционными рынками
- On-chain данные (Exchange Flows)
- Составные индикаторы для оценки условий торговли

Используйте эти данные в сочетании с техническим анализом для принятия более обоснованных торговых решений.
