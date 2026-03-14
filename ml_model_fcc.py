import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

print('=' * 70)
print('ПРОГНОЗИРОВАНИЕ ВЫХОДА БЕНЗИНА НА УСТАНОВКЕ КАТАЛИТИЧЕСКОГО КРЕКИНГА')
print('=' * 70)
print('\n[1] ЗАГРУЗКА ДАННЫХ')
print('-' * 50)

df = pd.read_csv('fcc_data.csv')
print(f'Загружено записей: {len(df)}')
print('\nПервые 5 строк данных:')
print(df.head())

print('\n[2] АНАЛИЗ ДАННЫХ')
print('-' * 50)
print('\nСтатистика по параметрам:')
print(df.describe())
print('\nКорреляция параметров с выходом бензина:')
correlations = df.corr(numeric_only=True)['gasoline_yield'].sort_values(ascending=False)
print(correlations)

print('\n[3] ПОДГОТОВКА ДАННЫХ')
print('-' * 50)
X = df.drop('gasoline_yield', axis=1)
y = df['gasoline_yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Обучающая выборка: {len(X_train)} записей')
print(f'Тестовая выборка: {len(X_test)} записей')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('\n[4] ОБУЧЕНИЕ МОДЕЛИ')
print('-' * 50)
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)
print('Модель успешно обучена!')

print('\n[5] ОЦЕНКА МОДЕЛИ')
print('-' * 50)
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f'R² на обучающей выборке: {train_r2:.4f}')
print(f'R² на тестовой выборке: {test_r2:.4f}')
print(f'MAE (средняя абсолютная ошибка): {test_mae:.4f} %')
print(f'RMSE (среднеквадратичная ошибка): {test_rmse:.4f} %')
if test_r2 > 0.9:
    quality = 'Отличное качество модели'
elif test_r2 > 0.7:
    quality = 'Хорошее качество модели'
elif test_r2 > 0.5:
    quality = 'Удовлетворительное качество модели'
else:
    quality = 'Низкое качество модели; требуется больше данных'
print(quality)

print('\n[6] АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ')
print('-' * 50)
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f} ({row['importance']*100:.1f}%)")

print('\n[7] ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ')
print('-' * 50)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Анализ модели прогнозирования выхода бензина', fontsize=16)
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.8)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Идеальная линия')
axes[0, 0].set_xlabel('Реальный выход, %')
axes[0, 0].set_ylabel('Предсказанный выход, %')
axes[0, 0].set_title('Предсказания vs реальность')
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
errors = y_test - y_test_pred
axes[0, 1].hist(errors, bins=6, alpha=0.8, edgecolor='black')
axes[0, 1].axvline(0, linestyle='--')
axes[0, 1].set_title('Распределение ошибок')
axes[0, 1].set_xlabel('Ошибка, %'); axes[0, 1].set_ylabel('Количество'); axes[0, 1].grid(True, alpha=0.3)
axes[1, 0].barh(feature_importance['feature'], feature_importance['importance'])
axes[1, 0].set_title('Важность признаков')
axes[1, 0].grid(True, alpha=0.3, axis='x')
metrics = ['R² train', 'R² test', 'MAE/10', 'RMSE/10']
values = [train_r2, test_r2, test_mae/10, test_rmse/10]
axes[1, 1].bar(metrics, values)
axes[1, 1].set_title('Метрики качества модели')
axes[1, 1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('ml_results.png', dpi=150)
print("\nГрафики сохранены в файл 'ml_results.png'")

print('\n[8] ПРИМЕР ПРОГНОЗИРОВАНИЯ ДЛЯ НОВОГО РЕЖИМА')
print('-' * 50)
new_mode = pd.DataFrame({'temperature': [515], 'pressure': [2.2], 'catalyst_rate': [88], 'feed_density': [0.91], 'feed_rate': [1280]})
print('Новый технологический режим:')
print(new_mode)
new_mode_scaled = scaler.transform(new_mode)
predicted_yield = model.predict(new_mode_scaled)[0]
print(f'\nПрогнозируемый выход бензина: {predicted_yield:.2f}%')
print('\n' + '=' * 70)
print('АНАЛИЗ ЗАВЕРШЕН')
print('=' * 70)
