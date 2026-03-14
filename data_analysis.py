import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print('=' * 60)
print('АНАЛИЗ ДАННЫХ РАБОТЫ ТЕХНОЛОГИЧЕСКОЙ УСТАНОВКИ')
print('=' * 60)

df = pd.read_csv('plant_data.csv')
print(f'\nЗагружено записей: {len(df)}')
print('\nПервые 5 записей:')
print(df.head())

print('\n' + '-' * 60)
print('ИНФОРМАЦИЯ О ДАННЫХ:')
print('-' * 60)
print(f"Период данных: с {df['date'].min()} по {df['date'].max()}")
print(f"Продукты: {df['product'].unique()}")
print(f"Диапазон температур: {df['temperature'].min():.0f} - {df['temperature'].max():.0f} °C")
print(f"Диапазон давлений: {df['pressure'].min():.1f} - {df['pressure'].max():.1f} атм")

print('\n' + '-' * 60)
print('СТАТИСТИКА ПО ПРОДУКТАМ:')
print('-' * 60)
for product in df['product'].unique():
    product_data = df[df['product'] == product]
    print(f'\nПродукт: {product.upper()}')
    print(f'  Количество записей: {len(product_data)}')
    print(f"  Средняя температура: {product_data['temperature'].mean():.1f} °C")
    print(f"  Среднее давление: {product_data['pressure'].mean():.2f} атм")
    print(f"  Средняя загрузка сырья: {product_data['feed_rate'].mean():.0f} тонн")
    print(f"  Средний выход продукта: {product_data['output'].mean():.0f} тонн")
    print(f"  Средний выход в %: {product_data['yield_percent'].mean():.1f}%")
    print(f"  Максимальный выход %: {product_data['yield_percent'].max():.1f}%")
    print(f"  Минимальный выход %: {product_data['yield_percent'].min():.1f}%")

print('\n' + '-' * 60)
print('КОРРЕЛЯЦИОННЫЙ АНАЛИЗ (для бензина):')
print('-' * 60)
benzine_data = df[df['product'] == 'бензин']
corr_temp_output = benzine_data['temperature'].corr(benzine_data['output'])
corr_pressure_output = benzine_data['pressure'].corr(benzine_data['output'])
corr_feed_output = benzine_data['feed_rate'].corr(benzine_data['output'])
print(f'Корреляция температура-выход: {corr_temp_output:.3f}')
print(f'Корреляция давление-выход: {corr_pressure_output:.3f}')
print(f'Корреляция загрузка-выход: {corr_feed_output:.3f}')
if abs(corr_temp_output) > 0.7:
    print('✓ Сильная зависимость выхода от температуры')
elif abs(corr_temp_output) > 0.3:
    print('✓ Средняя зависимость выхода от температуры')
else:
    print('✓ Слабая зависимость выхода от температуры')

print('\n' + '-' * 60)
print('ОПТИМАЛЬНЫЕ РЕЖИМЫ РАБОТЫ:')
print('-' * 60)
best_records = df.nlargest(3, 'yield_percent')
print('Топ-3 дня с максимальным выходом продукта:')
for _, row in best_records.iterrows():
    print(f"  {row['date']} | {row['product']} | Выход: {row['yield_percent']:.1f}% | T={row['temperature']}°C | P={row['pressure']} атм")

print('\n' + '-' * 60)
print('СОЗДАНИЕ ГРАФИКОВ...')
print('-' * 60)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Анализ данных технологической установки', fontsize=16)
axes[0, 0].plot(benzine_data['date'], benzine_data['yield_percent'], marker='o')
axes[0, 0].set_title('Выход бензина по дням')
axes[0, 0].set_xlabel('Дата')
axes[0, 0].set_ylabel('Выход, %')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 1].scatter(benzine_data['temperature'], benzine_data['yield_percent'], alpha=0.8)
axes[0, 1].set_title('Зависимость выхода от температуры')
axes[0, 1].set_xlabel('Температура, °C')
axes[0, 1].set_ylabel('Выход, %')
axes[0, 1].grid(True, alpha=0.3)
z = np.polyfit(benzine_data['temperature'], benzine_data['yield_percent'], 1)
p = np.poly1d(z)
axes[0, 1].plot(benzine_data['temperature'], p(benzine_data['temperature']), '--', alpha=0.8, label=f'Тренд: {z[0]:.2f}%/°C')
axes[0, 1].legend()
products_avg = df.groupby('product')['yield_percent'].mean()
axes[1, 0].bar(products_avg.index, products_avg.values)
axes[1, 0].set_title('Средний выход по продуктам')
axes[1, 0].set_xlabel('Продукт')
axes[1, 0].set_ylabel('Выход, %')
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 1].scatter(benzine_data['feed_rate'], benzine_data['output'], alpha=0.8)
axes[1, 1].set_title('Зависимость выхода от загрузки сырья')
axes[1, 1].set_xlabel('Загрузка сырья, тонн')
axes[1, 1].set_ylabel('Выход продукта, тонн')
axes[1, 1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_results.png', dpi=150)
print("\nГрафики сохранены в файл 'analysis_results.png'")

print('\n' + '-' * 60)
print('СОХРАНЕНИЕ РЕЗУЛЬТАТОВ:')
print('-' * 60)
summary = pd.DataFrame({
    'Продукт': products_avg.index,
    'Средний выход %': products_avg.values,
    'Макс выход %': [df[df['product'] == p]['yield_percent'].max() for p in products_avg.index],
    'Мин выход %': [df[df['product'] == p]['yield_percent'].min() for p in products_avg.index],
})
summary.to_csv('analysis_summary.csv', index=False, encoding='utf-8-sig')
print("Сводные результаты сохранены в 'analysis_summary.csv'")
print('\n' + '=' * 60)
print('АНАЛИЗ ЗАВЕРШЕН')
print('=' * 60)
